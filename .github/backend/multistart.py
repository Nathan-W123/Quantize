import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from backend.quantize import MolecularOptimizer, _kabsch_align
from backend.registry import get_backend

#: Two starts are near-tied when their quantum energies at their answers differ
#: by no more than this (Hartree), and far apart when, aligned, some atom
#: differs by more than NEAR_TIE_DISTANCE_ANG. The energy is the fit's own
#: quadratic model at the answer (free): measured through `cli run` on 30
#: starts (T-23) it is within 5.6e-6 Ha (median) and 2.5e-5 (90th percentile)
#: of the true energy there; the tie is twice that. 2 mA leaves out the
#: 0.15-1.2 mA pairs that differ only by where each fit stopped.
NEAR_TIE_ENERGY_HA = 5e-5
NEAR_TIE_DISTANCE_ANG = 2e-3


def _spectral_isotopologue_snapshot(opt):
    """Copy optimizer spectral targets after run (e.g. ORCA rovib-updated alphas)."""
    snap = []
    for iso in opt.spectral.isotopologues:
        snap.append(
            {
                "name": str(iso["name"]),
                "masses": np.asarray(iso["masses"], dtype=float).tolist(),
                "obs_constants": np.asarray(iso["obs_constants"], dtype=float).tolist(),
                "sigma_constants": np.asarray(iso["sigma_constants"], dtype=float).tolist(),
                "alpha_constants": np.asarray(iso["alpha_constants"], dtype=float).tolist(),
                "component_indices": np.asarray(iso["component_indices"], dtype=int).tolist(),
                "torsion_sensitive": bool(iso.get("torsion_sensitive", False)),
            }
        )
    return snap


def _run_one_start(payload):
    idx = payload["idx"]
    coords = np.asarray(payload["coords"], dtype=float)
    elems = payload["elems"]
    isotopologues = payload["isotopologues"]
    kwargs = dict(payload["optimizer_kwargs"])

    base_workdir = kwargs.pop("base_workdir", ".")
    job_name = kwargs.pop("job_name", "multistart")
    start_dir = os.path.join(base_workdir, "multistart_runs", job_name, f"start_{idx:02d}")
    os.makedirs(start_dir, exist_ok=True)
    kwargs["workdir"] = start_dir
    kwargs.setdefault("psi4_output_file", os.path.join(start_dir, "psi4.out"))

    opt = MolecularOptimizer(coords=coords, elems=elems, isotopologues=isotopologues, **kwargs)
    final_coords = opt.run()
    # One more quantum Hessian at the answer, and re-converge on it (D-146):
    # the fit computes its Hessian on its own schedule, usually near its
    # start, and the answer is fitted on that curvature and the quantum
    # weight it sets. Before anything below reads the fit.
    hessian_refresh = None
    if payload.get("refresh_hessian"):
        try:
            hessian_refresh = opt.refresh_hessian_at_answer()
        except Exception as exc:  # a failed refresh must not lose the fit
            hessian_refresh = {"error": f"{type(exc).__name__}: {exc}"}
        if hessian_refresh is not None and "error" not in hessian_refresh:
            final_coords = np.asarray(opt.coords, dtype=float).copy()
    last = opt.history[-1] if opt.history else {}
    # The quantum energy at the start's last quantum point, not at
    # final_coords -- see select_best_result.
    _e = last.get("energy", np.inf)
    _fr = last.get("freq_rms", np.inf)
    # The energy at the answer itself, from the fit's own quadratic model of
    # the quantum surface (no quantum call); near_tied_starts compares these.
    _e_answer = None
    if not opt.spectral_only and opt.quantum is not None:
        try:
            _e_answer = float(opt.quantum_prior_energy(np.asarray(final_coords, dtype=float),
                                                       use_proxy=True))
        except Exception:  # a diagnostic must not lose the fit
            _e_answer = None

    # ── Acceptance-rate diagnostics ───────────────────────────────────────────
    n_iters = len(opt.history)
    n_accepted = sum(1 for h in opt.history if h.get("accepted", False))
    accept_rate = float(n_accepted) / max(1, n_iters)

    # ── SVD diagnostics (final iteration that has singular-value data) ────────
    svd_summary: dict = {}
    last_svs = None
    final_rank = None
    for h in reversed(opt.history):
        if h.get("internal_singular_values") is not None:
            last_svs = h["internal_singular_values"]
            final_rank = h.get("internal_rank") or h.get("rank")
            break
    if last_svs is None:
        for h in reversed(opt.history):
            if h.get("rank") is not None:
                final_rank = h["rank"]
                break
    if last_svs is not None and final_rank is not None:
        sv = np.asarray(last_svs, dtype=float)
        sv_pos = sv[sv > 0]
        cond = float(sv_pos[0] / sv_pos[-1]) if sv_pos.size > 1 else float("inf")
        sv_gap = (
            float(sv[final_rank - 1] / sv[final_rank])
            if (final_rank > 0 and final_rank < len(sv) and sv[final_rank] > 0)
            else None
        )
        svd_summary = {
            "final_rank": int(final_rank),
            "condition_number": cond,
            "sv_max": float(sv[0]),
            "sv_min_kept": float(sv[final_rank - 1]) if final_rank > 0 else None,
            "sv_gap": sv_gap,
        }

    # The fit's own posterior, while the optimizer that produced it still
    # exists: once this returns only the result dict crosses back, and a
    # posterior rebuilt from that dict -- which is what the run report used to
    # do -- describes whatever objective the rebuild assumes, not the fit's.
    try:
        posterior = opt.internal_coordinate_posterior()
    except Exception as exc:  # a reporting failure must not lose the fit
        posterior = {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "idx": idx,
        "coords": final_coords,
        "posterior": posterior,
        "start_coords": np.asarray(coords, dtype=float).copy(),
        "freq_rms": float(_fr if _fr is not None else np.inf),
        "energy": float(_e if _e is not None else np.inf),
        "energy_at_answer_model": _e_answer,
        "history": opt.history,
        "workdir": start_dir,
        "spectral_isotopologues_snapshot": _spectral_isotopologue_snapshot(opt),
        "conformer_diagnostics": opt.spectral.conformer_diagnostics(),
        "conformer_summary": opt.conformer_summary,
        "n_iterations": n_iters,
        "n_accepted": n_accepted,
        "accept_rate": accept_rate,
        "svd_summary": svd_summary,
        "hessian_refresh": hessian_refresh,
    }


def _shared_dihedral_choice(starts, elems, isotopologues, kwargs):
    """Resolve ic_use_dihedrals="auto" once for every start: dihedrals for all
    of them when any one start's bonds and angles miss a direction dihedrals
    reach (D-148), bonds and angles alone otherwise.

    Each start used to decide for itself, at its own geometry, so a planar
    molecule without imposed symmetry gave the exact start 24 dihedrals and
    the jittered starts none (fluorobenzene, T-41 verifier): the reported
    coordinate set then depended on which start won. One set for every start
    keeps that fixed. It was then decided on the first start alone, so a
    later start whose bonds and angles missed a direction the first start's
    reached ran without dihedrals: a water dimer with a Cs start first and a
    start with no point group second, which ended 0.44 widths from its
    optimum (N30). Absent from the keyword arguments, the setting is the
    constructor's default, "auto".
    """
    mode = str(kwargs.get("coordinate_mode", "internal")).strip().lower()
    dih = kwargs.get("ic_use_dihedrals", "auto")
    if mode != "internal" or not (isinstance(dih, str) and dih.strip().lower() == "auto") or not starts:
        return kwargs
    import contextlib
    import io
    probe_kw = {k: v for k, v in kwargs.items() if k not in ("base_workdir", "job_name")}
    for k, start in enumerate(starts):
        with contextlib.redirect_stdout(io.StringIO()):
            probe = MolecularOptimizer(coords=np.asarray(start, dtype=float), elems=list(elems),
                                       isotopologues=isotopologues, **probe_kw)
        if probe._ic_use_dihedrals:
            where = "" if len(starts) == 1 else f" (start {k + 1})"
            print(f"[multistart] dihedrals added for every start{where}: {probe._ic_dihedrals_reason}")
            return {**kwargs, "ic_use_dihedrals": True}
    return {**kwargs, "ic_use_dihedrals": False}


def run_multistart(starts, elems, isotopologues, optimizer_kwargs, max_workers=1, job_name="job",
                   refresh_hessian=False):
    """
    ``refresh_hessian``: after each start's fit, compute the quantum Hessian at
    its answer and re-converge on it (MolecularOptimizer
    .refresh_hessian_at_answer); each result carries what it did under
    "hessian_refresh". ``quantize run`` asks for it (D-146).

    Parallel workers run independent ORCA/Psi4 jobs. Many ORCA installations (especially
    academic licenses) allow only **one** ORCA process at a time; parallel runs then exit
    without writing ``.engrad`` / ``.hess``. By default we force ``max_workers=1`` when
    ``quantum_backend='orca'``. Set ``QUANTIZE_ALLOW_PARALLEL_ORCA=1`` to keep preset parallelism.
    """
    kw = optimizer_kwargs or {}
    # MolecularOptimizer's own default when the key is absent.
    qb = str(kw.get("quantum_backend", "Psi4")).strip().lower()
    spectral_only = bool(kw.get("spectral_only", False))
    allow_parallel_orca = os.environ.get("QUANTIZE_ALLOW_PARALLEL_ORCA", "").lower() in (
        "1",
        "true",
        "yes",
        "y",
    )
    # Ask the backend class whether it supports parallel workers (e.g. ORCA does not).
    try:
        backend_cls = get_backend(qb) if qb else None
    except ValueError:
        backend_cls = None
    backend_supports_parallel = backend_cls.supports_parallel if backend_cls is not None else True
    if not backend_supports_parallel and not spectral_only and max_workers > 1 and not allow_parallel_orca:
        print(
            f"[multistart] quantum_backend={qb!r}: using max_workers=1 (backend does not support "
            "parallel workers — parallel jobs often fail with single-seat licenses). "
            "Export QUANTIZE_ALLOW_PARALLEL_ORCA=1 to allow "
            f"parallel workers (requested {max_workers})."
        )
        max_workers = 1

    start_kwargs = {**optimizer_kwargs, "job_name": job_name}
    start_kwargs = _shared_dihedral_choice(starts, elems, isotopologues, start_kwargs)
    payloads = [
        {
            "idx": i + 1,
            "coords": np.asarray(c, dtype=float),
            "elems": list(elems),
            "isotopologues": isotopologues,
            "optimizer_kwargs": start_kwargs,
            "refresh_hessian": bool(refresh_hessian),
        }
        for i, c in enumerate(starts)
    ]
    if max_workers <= 1:
        return [_run_one_start(p) for p in payloads]

    if not spectral_only:
        from backend.registry import fork_hang_warning
        why = fork_hang_warning(qb, max_workers,
                                backend_threads=kw.get("psi4_num_threads") if qb == "psi4" else None)
        if why:
            print(f"[multistart] warning: {why}")
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(_run_one_start, p): p["idx"] for p in payloads}
        for fut in as_completed(fut_map):
            results.append(fut.result())
    results.sort(key=lambda r: r["idx"])
    return results


def multistart_seed_metrics(result):
    """
    Per-start diagnostics: Cartesian RMS displacement from `start_coords`, and
    change in reported spectral RMS from first to last optimizer history entry.

    Returns
    -------
    dict with keys:
        coord_rms_disp_ang, spectral_gain_mhz, freq_rms_first, freq_rms_last
    """
    start = result.get("start_coords")
    final = np.asarray(result.get("coords"), dtype=float)
    if start is not None:
        start = np.asarray(start, dtype=float)
        coord_rms = float(np.sqrt(np.mean((final - start) ** 2)))
    else:
        coord_rms = float("nan")
    hist = result.get("history") or []
    if len(hist) >= 1:
        f_first = float(hist[0].get("freq_rms", np.inf))
        f_last = float(hist[-1].get("freq_rms", np.inf))
        gain = f_first - f_last
    else:
        f_first = f_last = float(result.get("freq_rms", np.inf))
        gain = 0.0
    return {
        "coord_rms_disp_ang": coord_rms,
        "spectral_gain_mhz": gain,
        "freq_rms_first": f_first,
        "freq_rms_last": f_last,
    }


def _effective_spectral_rms_for_selection(
    result,
    penalize_stagnant,
    stagnant_coord_rms_tol_ang,
    stagnant_max_spectral_gain_mhz,
    stagnant_penalty_mhz,
):
    """Internal rank key: penalize runs that barely moved and barely improved the spectrum."""
    base = float(result.get("freq_rms", np.inf))
    if not penalize_stagnant:
        return base
    m = multistart_seed_metrics(result)
    cr = m["coord_rms_disp_ang"]
    gain = m["spectral_gain_mhz"]
    if not np.isfinite(cr):
        return base
    stagnant = (cr < float(stagnant_coord_rms_tol_ang)) and (
        gain < float(stagnant_max_spectral_gain_mhz)
    )
    return base + (float(stagnant_penalty_mhz) if stagnant else 0.0)


def select_best_result(
    results,
    spectral_gate_abs=0.01,
    spectral_gate_rel=2.0,
    *,
    primary_objective="energy",
    penalize_stagnant=False,
    stagnant_coord_rms_tol_ang=5e-6,
    stagnant_max_spectral_gain_mhz=0.05,
    stagnant_penalty_mhz=500.0,
):
    """
    Select best multistart result.

    Parameters
    ----------
    primary_objective : {"energy", "spectral"}
        * ``energy`` (default): within the spectral gate, pick **lowest ORCA energy**
          (original behaviour).
        * ``spectral``: pick **lowest effective spectral RMS** (optionally after
          ``penalize_stagnant``), then lowest energy as tie-breaker.

    penalize_stagnant : bool
        If True, add ``stagnant_penalty_mhz`` to the comparison RMS when the run
        moved less than ``stagnant_coord_rms_tol_ang`` Å (RMS over atoms) **and**
        improved spectral RMS by less than ``stagnant_max_spectral_gain_mhz`` from
        the first to the last history record — i.e. reward starts that actually
        explore geometry or improve the fit.

    Strategy (spectral gate unchanged):
    1) Find best spectral RMS across starts.
    2) Keep starts with freq_rms <= max(spectral_gate_abs, spectral_gate_rel * best_freq_rms).
    3) Rank within the gated set using ``primary_objective`` / stagnation penalty.

    Each result's ``energy`` is the quantum energy at that start's last
    quantum point, not at its answer (_run_single_start reads the last history
    record). Measured through `cli run` (HF/6-31G; formyl fluoride,
    chlorofluoromethane, vinyl and acetyl fluoride, fluoroethane, each from
    its RHF minimum and from a 0.03 A-displaced start; 30 starts) it was
    8.6e-5 Ha (median) and up to 1.6e-3 Ha from the energy at the answer,
    against differences between starts of ~1e-5. It still put the start with
    the lowest energy at its answer first in 7 of the 10 runs; every miss was
    between starts within 1.4e-6 Ha of each other, 0.15-12 mA apart along
    flat directions such as a methyl torsion, which no energy ranking
    separates -- the fit's own model energy at the answer (free) got 8 of 10.
    Left as it is (decisions D-65).
    """
    if not results:
        raise ValueError("No multistart results to select from.")
    best_freq = min(float(r.get("freq_rms", np.inf)) for r in results)
    gate = max(float(spectral_gate_abs), float(spectral_gate_rel) * best_freq)
    gated = [r for r in results if float(r.get("freq_rms", np.inf)) <= gate]
    if not gated:
        gated = list(results)

    po = str(primary_objective).strip().lower()
    if po == "spectral":

        def sort_key(r):
            eff = _effective_spectral_rms_for_selection(
                r,
                penalize_stagnant,
                stagnant_coord_rms_tol_ang,
                stagnant_max_spectral_gain_mhz,
                stagnant_penalty_mhz,
            )
            e = float(r.get("energy", np.inf))
            if not np.isfinite(e):
                e = np.inf
            return (eff, e)

        return min(gated, key=sort_key)

    # Legacy: lowest energy within gate, then spectral RMS fallback.
    finite_energy = [r for r in gated if np.isfinite(float(r.get("energy", np.inf)))]
    if finite_energy:
        return min(finite_energy, key=lambda r: float(r.get("energy", np.inf)))
    return min(gated, key=lambda r: float(r.get("freq_rms", np.inf)))


def near_tied_starts(results, best, spectral_gate_abs=0.05, spectral_gate_rel=2.0,
                     tie_ha=NEAR_TIE_ENERGY_HA, distance_ang=NEAR_TIE_DISTANCE_ANG):
    """The other starts the selection could just as well have reported.

    A start counts when it passes the same spectral gate select_best_result
    applies, its model energy at its answer (``energy_at_answer_model``) is
    within ``tie_ha`` of the selected start's, and its answer is more than
    ``distance_ang`` from the selected one at some atom after alignment. Such a
    pair is not ranked by any energy these fits carry (T-23: every misrank
    was between starts within 1.4e-6 Ha), so the reported geometry is one of
    several the fit does not decide between -- along a flat direction such
    as a methyl torsion or a CH2 twist. Returns dicts (idx, energy difference,
    distance, freq_rms, and max_shift_in_sigma / max_shift_coordinate: the
    largest change in a reported coordinate over its width, from the two
    starts' posteriors, None without them), farthest first; empty when there
    is nothing to say (also when a start's model energy could not be formed).

    The model energy can miss by more than the tie: checked on ten CLI runs
    against true SCF energies, every start listed was near-tied and 8.6-20.6
    mA away, and one more would have been listed on the true energies
    (acetyl fluoride from its RHF minimum, start 3: model +6.3e-5 Ha, true
    +1.0e-5, 2.06 mA) -- a borderline pair can go unmentioned.
    """
    e_best = best.get("energy_at_answer_model")
    if e_best is None or not np.isfinite(float(e_best)):
        return []
    best_freq = min(float(r.get("freq_rms", np.inf)) for r in results)
    gate = max(float(spectral_gate_abs), float(spectral_gate_rel) * best_freq)
    x_best = np.asarray(best["coords"], dtype=float)
    out = []
    for r in results:
        if r is best or r.get("idx") == best.get("idx"):
            continue
        e = r.get("energy_at_answer_model")
        if e is None or not np.isfinite(float(e)) or float(r.get("freq_rms", np.inf)) > gate:
            continue
        de = float(e) - float(e_best)
        if abs(de) > float(tie_ha):
            continue
        x = np.asarray(r["coords"], dtype=float)
        if x.shape != x_best.shape:
            continue
        dist = float(np.linalg.norm(_kabsch_align(x, x_best) - x_best, axis=1).max())
        if dist > float(distance_ang):
            z, name = _reported_shift_in_sigma(best.get("posterior"), r.get("posterior"))
            out.append({"idx": r.get("idx"), "energy_difference_ha": de,
                        "max_atom_distance_ang": dist,
                        "freq_rms": float(r.get("freq_rms", np.inf)),
                        "max_shift_in_sigma": z, "max_shift_coordinate": name})
    return sorted(out, key=lambda d: -d["max_atom_distance_ang"])


def _reported_shift_in_sigma(post_best, post_other):
    """How far another start's answer is from the reported one, in the
    reported coordinates and in their own widths: the largest
    |q_other - q_best| / sigma_best over the bonds, angles and dihedrals both
    posteriors carry, and the coordinate where it falls. (None, None) when
    either posterior is missing or failed. A coordinate without a finite
    positive width on the selected start is left out of the maximum.

    The atom distance alone overstates what a near-tie changes. Measured
    through `cli run` (T-31: ten runs at HF/6-31G, the seven near-tied pairs
    8.6-20.6 mA apart at the largest atom): the reported bonds and angles moved
    by at most 0.29 of their sigma (median over the pairs 0.13) -- 2.9 mA on a 10.2 mA bond,
    0.35 deg on a 1.83 deg angle -- since the spread is along torsions, which
    move hydrogens. That is the number a reader of the structure needs.
    """
    try:                                   # None, or {"error": ...}, lands in except
        other = {n: float(v) for n, v in zip(post_other["names"], post_other["values"])}
        best_z, best_name = None, None
        for n, k, v, s in zip(post_best["names"], post_best["kinds"],
                              post_best["values"], post_best["std_err"]):
            s = float(s)
            if n not in other or not np.isfinite(s) or s <= 0.0:
                continue
            dq = other[n] - float(v)
            if k == "dihedral":                       # radians, periodic
                dq = (dq + np.pi) % (2.0 * np.pi) - np.pi
            z = abs(dq) / s
            if np.isfinite(z) and (best_z is None or z > best_z):
                best_z, best_name = float(z), str(n)
        return best_z, best_name
    except (KeyError, TypeError, ValueError):
        return None, None


def near_tie_note(near):
    """One line for the report, or None."""
    if not near:
        return None
    def _shift(d):
        z = d.get("max_shift_in_sigma")
        if z is None:
            return ""
        return f", {z:.2f} sigma at most in a reported coordinate ({d.get('max_shift_coordinate')})"
    parts = ", ".join(f"start {d['idx']} ({d['energy_difference_ha']:+.1e} Eh, "
                      f"{1000.0 * d['max_atom_distance_ang']:.1f} mA{_shift(d)})" for d in near)
    explain = (". The sigma figure is the largest difference in a reported bond, angle "
               "or dihedral, over this geometry's own width there"
               if any(d.get("max_shift_in_sigma") is not None for d in near) else "")
    return (f"near-tied starts: {parts} -- within {NEAR_TIE_ENERGY_HA:.0e} Eh of this one "
            f"in the fit's model energy at their answers but more than "
            f"{1000.0 * NEAR_TIE_DISTANCE_ANG:.0f} mA away (largest atom, aligned), so the "
            f"fit does not decide between them; this geometry is one of them{explain}")


def _infer_internal_dof(coords):
    """Estimate internal DOF from geometry (3N-6 nonlinear, 3N-5 linear)."""
    arr = np.asarray(coords, dtype=float)
    n_atoms = int(arr.shape[0])
    if n_atoms <= 1:
        return 0
    if n_atoms == 2:
        return 1
    centered = arr - arr.mean(axis=0, keepdims=True)
    geom_rank = int(np.linalg.matrix_rank(centered, tol=1e-8))
    linear = geom_rank <= 1
    dof = 3 * n_atoms - (5 if linear else 6)
    return max(dof, 1)


def underconstrained_success_score(results, best_result, isotopologues, internal_dof=None):
    """
    Compute hybrid-inversion success metrics.

    Designed for underconstrained problems where spectral data alone does not
    fully determine the geometry — the quantum prior fills in the null space.
    Low spectral rank is expected and is NOT penalized.

    Score components (sum to 100):
      50% — multi-start geometric stability  (do all starts agree?)
      35% — spectral fit quality             (RMS vs experimental sigma)
      15% — quantum null-space quality       (is the null-space at an energy minimum?)

    Multiplied by health penalties for guardrail violations and energy spikes.
    """
    if not results:
        raise ValueError("No multistart results available.")
    if best_result is None:
        raise ValueError("best_result is required.")
    if internal_dof is None:
        internal_dof = _infer_internal_dof(best_result["coords"])
    internal_dof = max(int(internal_dof), 1)

    history = best_result.get("history", [])
    last = history[-1] if history else {}
    # Use spectral_rows (observations before priors are stacked) as the rank ceiling so
    # that prior-inflated full-J rank doesn't exceed internal_dof and show >100%.
    spectral_rows = int(last.get("spectral_rows", last.get("rank", 0)))
    constrained_rank = min(int(last.get("rank", 0)), spectral_rows, internal_dof)
    rank_fraction = min(1.0, max(0.0, constrained_rank / float(internal_dof)))

    # ── Multi-start geometric stability ───────────────────────────────────────
    # How consistent is the recovered geometry across independent starts?
    # Tolerances: 0.01 Å for bonds, 0.2° for angles (last metric assumed angular).
    metric_arrays = [np.asarray(r.get("metrics", []), dtype=float) for r in results if "metrics" in r]
    if metric_arrays:
        all_metrics = np.vstack(metric_arrays)
        metric_std = np.std(all_metrics, axis=0)
    else:
        metric_std = np.array([], dtype=float)
    if metric_std.size > 0:
        denom = np.full(metric_std.shape, 0.01, dtype=float)
        denom[-1] = 0.2
        stability_penalty = float(np.linalg.norm(metric_std / denom))
        stability_score = float(np.exp(-stability_penalty))
    else:
        stability_score = 0.5

    # ── Spectral fit quality ───────────────────────────────────────────────────
    # How well does the final geometry reproduce observed rotational constants?
    # Measured in units of experimental sigma — fit within ~5σ scores near 1.
    sigma_all = np.concatenate([np.asarray(iso["sigma_constants"], dtype=float) for iso in isotopologues])
    sigma_scale = float(np.sqrt(np.mean(sigma_all ** 2))) if sigma_all.size else 1.0
    freq_rms = float(best_result.get("freq_rms", np.inf))
    sigma_ratio = freq_rms / max(sigma_scale, 1e-12)
    spectral_agreement_score = float(np.exp(-sigma_ratio / 5.0))

    # ── Quantum null-space quality ─────────────────────────────────────────────
    # For directions invisible to the spectral data, the quantum prior steers
    # the geometry toward the energy minimum. A small final null-space gradient
    # norm (|g_n|) means the null-space directions are properly converged.
    # We use the median |g_n| over the last 20% of iterations as the signal.
    g_null_vals = [float(h["g_null_norm"]) for h in history if h.get("g_null_norm") is not None and np.isfinite(float(h["g_null_norm"]))]
    if g_null_vals:
        tail = g_null_vals[max(0, len(g_null_vals) - max(1, len(g_null_vals) // 5)):]
        g_null_final = float(np.median(tail))
        quantum_quality_score = float(np.exp(-g_null_final / 0.1))
    else:
        quantum_quality_score = 0.5

    # ── Health penalties (multiplicative) ─────────────────────────────────────
    guardrail_violation_rate = 0.0
    if history:
        flags = [1.0 if int(h.get("guardrail_violations", 0)) > 0 else 0.0 for h in history]
        guardrail_violation_rate = float(np.mean(flags))
    guardrail_score = float(np.exp(-4.0 * guardrail_violation_rate))

    delta_e = [float(h["delta_energy"]) for h in history if h.get("delta_energy") is not None and np.isfinite(float(h["delta_energy"]))]
    if delta_e:
        e_scale = float(np.median(np.abs(delta_e)) + 1e-12)
        e_spike = float(np.max(np.abs(delta_e)) / e_scale)
        energy_stability_score = float(np.exp(-0.2 * max(0.0, e_spike - 1.0)))
    else:
        energy_stability_score = 1.0

    prior_wrms = [float(h["prior_wrms"]) for h in history if h.get("prior_wrms") is not None]
    prior_stability_score = 1.0
    if prior_wrms:
        p_med = float(np.median(prior_wrms))
        p_span = float(max(prior_wrms) - min(prior_wrms))
        prior_stability_score = float(np.exp(-p_span / max(1.0, p_med)))

    # ── Final score ───────────────────────────────────────────────────────────
    score = 100.0 * (
        0.50 * stability_score +
        0.35 * spectral_agreement_score +
        0.15 * quantum_quality_score
    )
    score *= (0.85 + 0.15 * guardrail_score)
    score *= (0.90 + 0.10 * energy_stability_score)
    score *= (0.95 + 0.05 * prior_stability_score)

    conf_weight_stability = 1.0
    conf_entries = [h.get("conformer_weights") for h in history if h.get("conformer_weights") is not None]
    if conf_entries:
        arr = np.asarray(conf_entries, dtype=float)
        if arr.ndim == 2 and arr.shape[0] > 1:
            std = np.std(arr, axis=0)
            conf_weight_stability = float(np.exp(-5.0 * float(np.mean(std))))
        score *= (0.95 + 0.05 * conf_weight_stability)

    return {
        "score": float(score),
        "constrained_rank": constrained_rank,
        "internal_dof": internal_dof,
        "rank_fraction": float(rank_fraction),
        "stability_score": float(stability_score),
        "spectral_agreement_score": float(spectral_agreement_score),
        "quantum_quality_score": float(quantum_quality_score),
        "sigma_scale": sigma_scale,
        "sigma_ratio": float(sigma_ratio),
        "guardrail_violation_rate": float(guardrail_violation_rate),
        "guardrail_score": float(guardrail_score),
        "energy_stability_score": float(energy_stability_score),
        "prior_stability_score": float(prior_stability_score),
        "prior_wrms_median": float(np.median(prior_wrms)) if prior_wrms else None,
        "conformer_weight_stability": float(conf_weight_stability),
        "conformer_weights_final": (
            [float(v) for v in np.asarray(conf_entries[-1], dtype=float)] if conf_entries else None
        ),
        "metric_std": metric_std.tolist(),
    }
