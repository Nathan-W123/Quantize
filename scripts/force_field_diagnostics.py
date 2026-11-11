"""Is this correction's force field trustworthy? Two cheap checks.

WHY THIS EXISTS. On 2026-11-10 the benchmark replaced HF/6-31G corrections
with MP2/cc-pVTZ ones -- unambiguously the better level of theory by any
general measure -- and ozone's hybrid result went from 1.70 mA to 7.86 mA, 4.6
times worse. The cause was not subtle: MP2 puts ozone's antisymmetric stretch
at 2212 cm^-1 against a measured 1103, exactly double, and alpha carries
1/omega, so the correction inherited it. The RMS correction error went from 117
MHz at HF to 794 MHz at MP2, with the SIGN wrong on B and C.

Nothing in the engine noticed. It computed a force field, computed alpha from
it, applied it, fitted a structure and reported a number to two decimal places.
"Better level of theory" is not a scalar, and until now there was no way for a
result to carry that caveat.

TWO SIGNALS, because they fail for different reasons.

1. MULTIREFERENCE CHARACTER, from one level. Ozone's ground state has
   substantial biradical character: its O-O-O pi system is not well described
   by a single determinant. MP2 is perturbation theory ON TOP of that single
   determinant, so it amplifies the error instead of fixing it. The symptom is
   a large double-excitation amplitude -- MP2 is being asked to do
   non-perturbative work. Measured across the whole reference set, max|t2|:

       ozone               0.1920 (6-31g)  0.1794 (cc-pVDZ)  0.1697 (cc-pVTZ)
       vinyl fluoride      0.0969          0.0876            0.0797
       formyl fluoride                     0.0799
       fluoroacetylene                     0.0689
       isocyanic acid                      0.0661
       acetyl fluoride                     0.0647
       water               0.0478          0.0471            0.0448
       chlorofluoromethane                 0.0299
       fluoroethane                        0.0209

   Ozone is more than double everything else, and the ordering holds across
   three bases, so the threshold sits in an empty gap rather than being tuned.

   Two other candidates were measured and rejected. The HOMO-LUMO gap does
   flag ozone (12.4 eV, the smallest in the set) but only 1.2x below vinyl
   fluoride's 15.2 eV, too narrow to threshold. |E_corr|/|E_SCF| does not
   discriminate at all: every molecule in the set lands at 0.0026 +/- 0.0002.

2. CROSS-LEVEL DISAGREEMENT, when two levels are available. Two force fields
   that disagree badly cannot both be right, and the engine often has both --
   every base/corr benchmark pair computes two. Largest relative disagreement
   over modes:

       ozone      HF [730, 851, 1267] vs MP2 [748, 1171, 2212]   74.6%
       water      HF [1737, 3989, 4145] vs MP2 [1650, 3871, 3992]  5.3%

   This catches a bad force field that the multireference check misses, since
   a level can be wrong for reasons other than multireference character.

WHERE THIS LIVES, AND WHAT THAT MEANS. This is benchmark tooling, next to the
benchmark that uses it, not part of the shipped engine. The consequence is
worth stating plainly rather than leaving implicit: a caller who uses
``build_correction_table_from_hessian`` directly gets NO force-field warning.
Only runs that go through accuracy_upgrades_benchmark do.

That was a deliberate choice over wiring it into the correction path. The
multireference probe costs an MP2 calculation -- seconds for a four-atom
molecule and not seconds for a large one -- so putting it in the engine's hot
path would have meant an opt-in flag there anyway, and the engine does not
otherwise reach for electronic structure it was not asked for. If a library
user ever needs the warning, the functions are importable as they stand; the
move to make then is to call them from the correction path, not to rewrite
them.

WHAT THESE DO NOT DO. Neither says which level is RIGHT. A flag means the
correction is not to be trusted at face value, not that the other level is
better. For ozone the cheaper HF/6-31G happens to score better, but its own nu3
is 15% high and its alpha errors partly cancel -- luck, not quality. Deciding
the right level needs a reference, which is what the r_e structures are for.
"""

from __future__ import annotations

import numpy as np

#: max|t2| above which a force field is reported as multireference-suspect.
#: 0.12 sits between the largest well-behaved value measured (vinyl fluoride,
#: 0.097 at 6-31g) and the smallest ozone value (0.170 at cc-pVTZ). See the
#: module docstring for the full table.
MULTIREFERENCE_T2_WARN = 0.12

#: Largest relative frequency disagreement between two levels, above which
#: neither force field is trusted. 0.25 sits between water's 5.3% and ozone's
#: 74.6%.
FREQUENCY_DISAGREEMENT_WARN = 0.25

#: Basis for the multireference probe. Small deliberately: the diagnostic is
#: about the WAVEFUNCTION's character, which the table above shows is
#: basis-insensitive, so paying for a large basis buys nothing.
MULTIREFERENCE_PROBE_BASIS = "cc-pvdz"


def multireference_diagnostic(elements, coords_ang, basis: str = MULTIREFERENCE_PROBE_BASIS,
                              charge: int = 0, spin: int = 0,
                              t2_warn: float = MULTIREFERENCE_T2_WARN) -> dict:
    """Does this molecule need more than one determinant?

    Runs RHF then MP2 at ``basis`` and reports the largest double-excitation
    amplitude, which is the cheapest standard indicator of multireference
    character. A flagged molecule is one where single-reference correlation
    methods -- MP2 above all, but CCSD(T) too, to a lesser degree -- should not
    be trusted to give a force field good enough for a vibration-rotation
    correction.

    Costs one MP2 calculation, seconds for a small molecule and not free for a
    large one, which is why callers opt in.

    Returns a dict with 'max_t2', 'homo_lumo_ev', 'e_corr_fraction',
    'flagged', 'basis' and 'reason'. On failure it returns the same shape with
    'max_t2' None and the exception text in 'reason', because a diagnostic that
    raises would take down the run it was meant to qualify.
    """
    out = {"max_t2": None, "homo_lumo_ev": None, "e_corr_fraction": None,
           "flagged": False, "basis": str(basis), "reason": ""}
    try:
        from pyscf import gto, mp, scf
    except ImportError as exc:  # pragma: no cover - pyscf is a hard dep in practice
        out["reason"] = f"pyscf unavailable: {exc}"
        return out

    coords = np.asarray(coords_ang, dtype=float)
    try:
        mol = gto.M(atom=[(str(e), tuple(map(float, xyz)))
                          for e, xyz in zip(elements, coords)],
                    basis=str(basis), charge=int(charge), spin=int(spin),
                    unit="Angstrom", verbose=0)
        mf = scf.RHF(mol)
        mf.max_cycle = max(int(getattr(mf, "max_cycle", 50)), 200)
        mf.kernel()
        if not mf.converged:
            out["reason"] = "SCF did not converge, so no diagnostic"
            return out
        n_occ = mol.nelectron // 2
        if 0 < n_occ < len(mf.mo_energy):
            out["homo_lumo_ev"] = float(
                (mf.mo_energy[n_occ] - mf.mo_energy[n_occ - 1]) * 27.211386245988)
        pt = mp.MP2(mf)
        pt.kernel()
        t2 = getattr(pt, "t2", None)
        if t2 is None:
            out["reason"] = "MP2 returned no amplitudes"
            return out
        out["max_t2"] = float(np.abs(np.asarray(t2)).max())
        if mf.e_tot:
            out["e_corr_fraction"] = float(abs(pt.e_corr) / abs(mf.e_tot))
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not kill the run
        out["reason"] = f"{type(exc).__name__}: {exc}"
        return out

    if out["max_t2"] > float(t2_warn):
        out["flagged"] = True
        out["reason"] = (
            f"max|t2| = {out['max_t2']:.4f} exceeds {float(t2_warn):.2f}: this "
            f"molecule has substantial multireference character, so a "
            f"single-reference force field -- MP2 especially -- may be badly "
            f"wrong. Measured precedent: ozone sits at 0.18 and its MP2 "
            f"antisymmetric stretch is 101% high, which made its corrected "
            f"result 4.6x worse than a cheaper level."
        )
    else:
        out["reason"] = (f"max|t2| = {out['max_t2']:.4f}, within the "
                         f"single-reference range for this set")
    return out


#: Cross-level angle disagreement, in degrees, above which a coordinate is
#: not merely imprecise but qualitatively different between the two levels.
#:
#: Measured over the 10-molecule reference set, HF/6-31G against
#: B3LYP/6-31G(d), 61 angles:
#:
#:     median  0.437    p90  1.364    p99  13.70    max  29.19
#:
#:     29.194   isocyanic acid   H-N-C     <- HF puts it 30.34 deg from the
#:      3.377   isocyanic acid   N-C-O        published reference, B3LYP 1.14
#:      1.984   chlorofluoromethane H-C-H
#:      1.831   vinyl fluoride   H4-C1=C2
#:      1.786   ozone            O-O-O
#:
#: The real failure is 8.6x the next worst and nothing at all lies between
#: 3.4 and 29.2, so 10.0 sits in an empty gap: it flags that coordinate and
#: no other on this set. HNCO is quasi-linear at H-N-C and HF simply finds a
#: different minimum there.
GEOMETRY_ANGLE_DISAGREEMENT_WARN_DEG = 10.0

#: The same for bonds, in Angstrom -- and this one is NOT calibrated against
#: an observed failure, because there wasn't one. Over the same set, 52 bonds:
#:
#:     median  0.01439   p90  0.02982   p99  0.03891   max  0.04285
#:
#: a smooth distribution with its maximum only 3x the median, which is what
#: two levels of theory differing systematically looks like. 0.15 is set at
#: roughly 3.5x the observed maximum, so it means "no bond should ever move
#: this far between levels" rather than "this is where failure begins". It has
#: never fired on real data and should not be read as validated.
GEOMETRY_BOND_DISAGREEMENT_WARN_ANG = 0.15


def geometry_disagreement(coords_a, coords_b, elems, bonds=None, angles=None,
                          label_a: str = "A", label_b: str = "B",
                          angle_warn: float = GEOMETRY_ANGLE_DISAGREEMENT_WARN_DEG,
                          bond_warn: float = GEOMETRY_BOND_DISAGREEMENT_WARN_ANG) -> dict:
    """Do two levels of theory agree on the SHAPE, not just the energy?

    The point of this one is that it needs no reference structure. The engine
    exists for molecules that have none, and a published structure is exactly
    what is unavailable when a qualitative optimisation failure happens -- so
    a check that requires one cannot run where it is needed.

    HF/6-31G puts isocyanic acid's H-N-C angle 30.34 deg away from the
    published value while B3LYP/6-31G(d) is 1.14 deg away. Nothing in the
    engine noticed. The two levels disagree by 29.19 deg on that coordinate,
    which is visible without knowing the answer, and is 8.6x the next worst
    disagreement anywhere on the reference set.

    Coordinates are taken from the `bonds` and `angles` maps so a coordinate
    lands in the same place the rest of the harness puts it. Returns
    'max_angle_dev', 'max_bond_dev', 'worst_angle', 'worst_bond',
    'flagged_angles', 'flagged_bonds', 'flagged' and 'reason'. A failure to
    compare returns the same shape with the reason filled in, because a
    diagnostic that raises takes down the run it was meant to qualify.
    """
    out = {"max_angle_dev": None, "max_bond_dev": None,
           "worst_angle": None, "worst_bond": None,
           "flagged_angles": [], "flagged_bonds": [],
           "flagged": False, "reason": "",
           "label_a": str(label_a), "label_b": str(label_b)}
    try:
        xa = np.asarray(coords_a, dtype=float)
        xb = np.asarray(coords_b, dtype=float)
    except (TypeError, ValueError) as exc:
        out["reason"] = f"coordinates not numeric: {exc}"
        return out
    if xa.shape != xb.shape or xa.ndim != 2 or xa.shape[1] != 3:
        out["reason"] = (f"cannot compare geometries of shape {xa.shape} and "
                         f"{xb.shape}")
        return out
    if xa.shape[0] != len(list(elems)):
        out["reason"] = (f"{xa.shape[0]} atoms against {len(list(elems))} "
                         f"element labels")
        return out

    def blen(x, i, j):
        return float(np.linalg.norm(x[i] - x[j]))

    def bang(x, i, j, k):
        u = x[i] - x[j]; v = x[k] - x[j]
        nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        if nu <= 0.0 or nv <= 0.0:
            return float("nan")
        return float(np.degrees(np.arccos(np.clip((u / nu) @ (v / nv), -1.0, 1.0))))

    worst_b, worst_a = (-1.0, None), (-1.0, None)
    for label, pairs in (bonds or {}).items():
        for (i, j) in pairs:
            d = abs(blen(xa, i, j) - blen(xb, i, j))
            name = f"{label}({i},{j})"
            if d > worst_b[0]:
                worst_b = (d, name)
            if d > float(bond_warn):
                out["flagged_bonds"].append((name, d))
    for label, trips in (angles or {}).items():
        for (i, j, k) in trips:
            d = abs(bang(xa, i, j, k) - bang(xb, i, j, k))
            if not np.isfinite(d):
                continue
            name = f"{label}({i},{j},{k})"
            if d > worst_a[0]:
                worst_a = (d, name)
            if d > float(angle_warn):
                out["flagged_angles"].append((name, d))

    if worst_b[1] is not None:
        out["max_bond_dev"] = worst_b[0]
        out["worst_bond"] = worst_b[1]
    if worst_a[1] is not None:
        out["max_angle_dev"] = worst_a[0]
        out["worst_angle"] = worst_a[1]
    if not out["flagged_bonds"] and not out["flagged_angles"]:
        if worst_a[1] is None and worst_b[1] is None:
            out["reason"] = "no bonds or angles given to compare"
        return out

    out["flagged"] = True
    bits = []
    for name, d in out["flagged_angles"]:
        bits.append(f"angle {name} differs by {d:.1f} deg")
    for name, d in out["flagged_bonds"]:
        bits.append(f"bond {name} differs by {d:.4f} Ang")
    out["reason"] = (
        f"{label_a} and {label_b} do not agree on the shape: "
        + "; ".join(bits)
        + ". A disagreement this large is a different minimum rather than a "
          "difference of precision, so the cheaper level's geometry should "
          "not be trusted for this coordinate."
    )
    return out


def frequency_disagreement(omega_a, omega_b, label_a: str = "A",
                           label_b: str = "B",
                           warn: float = FREQUENCY_DISAGREEMENT_WARN) -> dict:
    """How badly do two levels' harmonic frequencies disagree?

    Compared mode by mode after sorting, relative to the SMALLER of each pair
    so the measure is symmetric in the two levels and a doubling reads as 100%
    rather than 50%. Modes are matched by frequency order, which is adequate
    for the gross disagreement this is looking for and would not be for a
    careful mode correspondence.

    Returns 'max_rel', 'worst_index', 'worst_pair', 'flagged', 'reason'.
    """
    a = np.sort(np.asarray(omega_a, dtype=float))
    b = np.sort(np.asarray(omega_b, dtype=float))
    out = {"max_rel": None, "worst_index": None, "worst_pair": None,
           "flagged": False, "reason": ""}
    if a.size != b.size or a.size == 0:
        out["reason"] = (f"cannot compare {a.size} modes against {b.size}; "
                         f"the two levels are not describing the same molecule")
        return out

    denom = np.minimum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom > 0.0, np.abs(a - b) / denom, 0.0)
    k = int(np.argmax(rel))
    out["max_rel"] = float(rel[k])
    out["worst_index"] = k
    out["worst_pair"] = (float(a[k]), float(b[k]))

    if out["max_rel"] > float(warn):
        out["flagged"] = True
        out["reason"] = (
            f"{label_a} and {label_b} disagree by {100 * out['max_rel']:.0f}% "
            f"on mode {k} ({a[k]:.0f} vs {b[k]:.0f} cm^-1), above the "
            f"{100 * float(warn):.0f}% threshold. They cannot both be right, "
            f"and alpha carries 1/omega, so neither correction should be "
            f"trusted until the disagreement is resolved."
        )
    else:
        out["reason"] = (f"{label_a} and {label_b} agree to "
                         f"{100 * out['max_rel']:.0f}% on every mode")
    return out
