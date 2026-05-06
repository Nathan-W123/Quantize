"""
RAM-lite: Reduced-Axis-Method torsion-rotation Hamiltonian scaffold.

This module implements a 1D RAM-lite Hamiltonian used as a first-pass
torsional model. It is intentionally scoped:
  - 1D Fourier basis: |m> states with m = -n_basis, ..., +n_basis
  - Periodic potential V(alpha) as cosine/sine Fourier series
  - Diagonal kinetic coupling via F*(m - rho*K)^2 (and optional F4/F6 terms)
  - Optional C3 symmetry labeling via residue decomposition
  - Not a full global RAM/IAM tensor-operator treatment

n_basis convention
------------------
``n_basis`` is the **half-range** integer. The basis has size ``2*n_basis + 1``
and m values: [-n_basis, -n_basis+1, ..., 0, ..., n_basis-1, n_basis].

Relation to hindered_rotor.py's ``basis_size``:
  basis_size (full odd size) = 2 * n_basis + 1
  n_basis                    = (basis_size - 1) // 2

Potential convention
--------------------
V(alpha) = V0 + sum_n [ Vcos_n * cos(n*alpha) + Vsin_n * sin(n*alpha) ]

The hindered_rotor module uses a different RAM-style convention:
  V(phi) = sum_n (V_n / 2) * (1 - cos(n*phi))
which maps as: V0 = sum_n V_n/2, Vcos_n = -V_n/2.

Later phases will add full torsion-rotation tensor coupling, assignment,
and global fitting capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TorsionFourierPotential:
    """
    Periodic torsional potential

    V(alpha) = V0 + sum_n [Vcos_n cos(n alpha) + Vsin_n sin(n alpha)]
    """

    v0: float = 0.0
    vcos: dict[int, float] = field(default_factory=dict)
    vsin: dict[int, float] = field(default_factory=dict)
    units: str = "cm-1"


@dataclass
class TorsionHamiltonianSpec:
    """
    RAM-lite Hamiltonian specification for a single block (J, K).

    Operator model (diagonal kinetic/coupling extension):
      H = E_rot(J,K)
          + F * x^2          [or F(alpha)*x^2 if F_alpha is set]
          + F4 * x^4
          + F6 * x^6
          + c_mk * m * K
          + c_k2 * K^2
          + V(alpha)
      where x = (m - rho*K), m from basis.

    Basis (n_basis convention):
      m values: [-n_basis, ..., +n_basis], matrix size = 2*n_basis + 1.
      Equivalent hindered_rotor basis_size = 2*n_basis + 1.

    Units:
      - all energies and rotational constants in cm^-1
      - F, F4, F6, c_mk, c_k2 in cm^-1
      - rho dimensionless

    This is a diagonal-in-|m> approximation, not a full RAM/IAM tensor treatment.
    """

    F: float
    rho: float = 0.0
    F4: float = 0.0
    F6: float = 0.0
    c_mk: float = 0.0
    c_k2: float = 0.0
    A: float = 0.0
    B: float = 0.0
    C: float = 0.0
    potential: TorsionFourierPotential = field(default_factory=TorsionFourierPotential)
    F_alpha: Optional["TorsionEffectiveConstantFourier"] = None
    n_basis: int = 7
    units: str = "cm-1"
    warnings: list[str] = field(default_factory=list)


@dataclass
class TorsionEffectiveConstantFourier:
    """
    Coordinate-dependent effective torsion constant:

      F(alpha) = f0 + sum_n [fcos_n cos(n alpha) + fsin_n sin(n alpha)]
    """

    f0: float
    fcos: dict[int, float] = field(default_factory=dict)
    fsin: dict[int, float] = field(default_factory=dict)
    units: str = "cm-1"


def basis_m_values(n_basis: int) -> np.ndarray:
    """Return symmetric integer m values: [-n_basis, ..., +n_basis]."""
    n = int(n_basis)
    if n < 0:
        raise ValueError("n_basis must be >= 0.")
    return np.arange(-n, n + 1, dtype=int)


def _validate_units(units: str) -> None:
    if str(units).strip().lower() != "cm-1":
        raise ValueError("Phase-1 torsion_hamiltonian currently supports only units='cm-1'.")


def rotational_baseline_cm1(J: int, K: int, A: float, B: float) -> float:
    """
    Symmetric-top-like baseline:
      E_rot = B * J(J+1) + (A - B) * K^2

    This is a practical first-pass baseline for block energies.
    """
    j = int(J)
    k = int(K)
    return float(B) * j * (j + 1) + (float(A) - float(B)) * (k * k)


def fourier_potential_matrix(m_vals: np.ndarray, potential: TorsionFourierPotential) -> np.ndarray:
    """
    Matrix of V(alpha) in |m> basis:
      <m|cos(n alpha)|m'> = 1/2 [delta_{m',m+n} + delta_{m',m-n}]
      <m|sin(n alpha)|m'> = 1/(2i) [delta_{m',m+n} - delta_{m',m-n}]

    Returns real symmetric matrix when coefficients are real.
    """
    _validate_units(potential.units)
    m = np.asarray(m_vals, dtype=int).ravel()
    n = m.size
    V = np.zeros((n, n), dtype=complex)
    idx = {int(mm): i for i, mm in enumerate(m)}

    # constant term
    V += float(potential.v0) * np.eye(n)

    # cosine terms
    for k, amp in potential.vcos.items():
        kk = int(k)
        if kk <= 0:
            continue
        a = float(amp)
        for i, mi in enumerate(m):
            j1 = idx.get(int(mi + kk))
            j2 = idx.get(int(mi - kk))
            if j1 is not None:
                V[i, j1] += 0.5 * a
            if j2 is not None:
                V[i, j2] += 0.5 * a

    # sine terms
    for k, amp in potential.vsin.items():
        kk = int(k)
        if kk <= 0:
            continue
        a = float(amp)
        for i, mi in enumerate(m):
            j1 = idx.get(int(mi + kk))
            j2 = idx.get(int(mi - kk))
            if j1 is not None:
                V[i, j1] += -0.5j * a
            if j2 is not None:
                V[i, j2] += +0.5j * a

    # enforce hermiticity (numerical guard)
    V = 0.5 * (V + V.conj().T)
    return V


def evaluate_effective_torsion_constant_on_grid(
    effective_F: TorsionEffectiveConstantFourier,
    alpha_grid: np.ndarray,
) -> np.ndarray:
    """
    Evaluate F(alpha) on a grid (radians).
    """
    _validate_units(effective_F.units)
    a = np.asarray(alpha_grid, dtype=float).ravel()
    out = np.full(a.shape, float(effective_F.f0), dtype=float)
    for k, amp in effective_F.fcos.items():
        kk = int(k)
        if kk <= 0:
            continue
        out += float(amp) * np.cos(kk * a)
    for k, amp in effective_F.fsin.items():
        kk = int(k)
        if kk <= 0:
            continue
        out += float(amp) * np.sin(kk * a)
    return out


def effective_torsion_constant_matrix(
    m_vals: np.ndarray,
    effective_F: TorsionEffectiveConstantFourier,
) -> np.ndarray:
    """
    Convert F(alpha) Fourier coefficients to |m> basis matrix.
    """
    pot_like = TorsionFourierPotential(
        v0=float(effective_F.f0),
        vcos={int(k): float(v) for k, v in effective_F.fcos.items()},
        vsin={int(k): float(v) for k, v in effective_F.fsin.items()},
        units=effective_F.units,
    )
    return fourier_potential_matrix(m_vals, pot_like)


def build_ram_lite_hamiltonian(spec: TorsionHamiltonianSpec, J: int = 0, K: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build RAM-lite Hamiltonian block for given (J, K) in |m> basis:

      H = E_rot(J,K) + F*x^2 + F4*x^4 + F6*x^6 + c_mk*m*K + c_k2*K^2 + V(alpha)
      where x = (m - rho*K).

    Returns
    -------
    H : (N,N) ndarray
    m_vals : (N,) ndarray
    warnings : list[str]
    """
    _validate_units(spec.units)
    m_vals = basis_m_values(spec.n_basis)
    warnings = list(spec.warnings or [])

    if abs(spec.rho) > 1.0:
        warnings.append("rho magnitude > 1 is unusual; verify reduced-axis parameters.")
    if spec.n_basis < 3:
        warnings.append("Very small n_basis may be insufficient for converged torsion levels.")
    if float(spec.F) <= 0.0:
        warnings.append("Non-positive scalar F is unusual; verify torsional kinetic constant.")
    if (abs(float(spec.F4)) > 0.0) or (abs(float(spec.F6)) > 0.0) or (abs(float(spec.c_mk)) > 0.0) or (abs(float(spec.c_k2)) > 0.0):
        warnings.append(
            "Using RAM-lite completeness terms (F4/F6/c_mk/c_k2) in a diagonal approximation; "
            "this is not a full coupled operator treatment."
        )

    E_rot = rotational_baseline_cm1(J, K, spec.A, spec.B)
    k = int(K)
    x = m_vals - float(spec.rho) * k
    if spec.F_alpha is not None:
        if (abs(float(spec.F4)) > 0.0) or (abs(float(spec.F6)) > 0.0):
            warnings.append(
                "F_alpha is used with F4/F6; higher-order kinetic terms remain diagonal approximations."
            )
        Fmat = effective_torsion_constant_matrix(m_vals, spec.F_alpha)
        D = np.diag(x.astype(float)).astype(complex)
        kinetic = D @ Fmat @ D
        diag_kin = np.diag(kinetic)
        diag = (
            E_rot
            + np.real_if_close(diag_kin)
            + float(spec.F4) * (x**4)
            + float(spec.F6) * (x**6)
            + float(spec.c_mk) * (m_vals * k)
            + float(spec.c_k2) * (k * k)
        )
    else:
        diag = (
            E_rot
            + float(spec.F) * (x**2)
            + float(spec.F4) * (x**4)
            + float(spec.F6) * (x**6)
            + float(spec.c_mk) * (m_vals * k)
            + float(spec.c_k2) * (k * k)
        )
    H = np.diag(diag.astype(float)).astype(complex)
    if spec.F_alpha is not None:
        H += kinetic - np.diag(np.diag(kinetic))
        if float(spec.F_alpha.f0) <= 0.0:
            warnings.append("F_alpha.f0 <= 0 is unusual; verify effective torsion constant model.")
        harmonics = list(spec.F_alpha.fcos.values()) + list(spec.F_alpha.fsin.values())
        if harmonics and abs(float(spec.F_alpha.f0)) > 0.0:
            max_rel = max(abs(float(v)) for v in harmonics) / abs(float(spec.F_alpha.f0))
            if max_rel > 1.0:
                warnings.append(
                    "F_alpha harmonic amplitude exceeds |f0|; kinetic model may be strongly nonuniform."
                )
        alpha_grid = np.linspace(-np.pi, np.pi, 181)
        F_grid = evaluate_effective_torsion_constant_on_grid(spec.F_alpha, alpha_grid)
        if np.min(F_grid) <= 0.0:
            warnings.append("F(alpha) is non-positive on part of grid; kinetic model may be unphysical.")
        if np.max(np.abs(F_grid)) > 50.0 * max(abs(float(spec.F_alpha.f0)), 1e-12):
            warnings.append("F(alpha) range is very large relative to f0; verify Fourier coefficients.")

    V = fourier_potential_matrix(m_vals, spec.potential)
    H += V
    H = 0.5 * (H + H.conj().T)
    return H, m_vals, warnings


def _canonical_symmetry_mode(symmetry_mode: Optional[str]) -> Optional[str]:
    if symmetry_mode is None:
        return None
    mode = str(symmetry_mode).strip().lower()
    if mode in {"", "none", "off"}:
        return None
    if mode in {"c3", "3fold", "threefold"}:
        return "c3"
    raise ValueError(f"Unsupported symmetry_mode={symmetry_mode!r}. Supported: None, 'c3'.")


def _warn_if_symmetry_mismatch(
    potential: TorsionFourierPotential,
    symmetry_mode: Optional[str],
    warnings: list[str],
) -> None:
    if symmetry_mode != "c3":
        return
    tol = 1e-12
    bad_harmonics: list[int] = []
    for k, amp in potential.vcos.items():
        kk = int(k)
        if kk > 0 and abs(float(amp)) > tol and kk % 3 != 0:
            bad_harmonics.append(kk)
    for k, amp in potential.vsin.items():
        kk = int(k)
        if kk > 0 and abs(float(amp)) > tol and kk % 3 != 0:
            bad_harmonics.append(kk)
    if bad_harmonics:
        uniq = sorted(set(bad_harmonics))
        warnings.append(
            "symmetry_mode='c3' requested but potential contains non-3-fold harmonics "
            f"{uniq}; A/E block purity may be reduced."
        )


def _c3_residue_indices(m_values: np.ndarray) -> dict[int, np.ndarray]:
    m = np.asarray(m_values, dtype=int).ravel()
    return {r: np.where(np.mod(m, 3) == r)[0] for r in (0, 1, 2)}


def periodic_wavefunction_diagnostics(
    eigenvector: np.ndarray,
    m_values: np.ndarray,
    rotor_fold: int = 3,
    alpha_grid: Optional[np.ndarray] = None,
) -> dict:
    """
    Periodic-grid and Fourier-space symmetry diagnostics for one wavefunction.
    """
    c = np.asarray(eigenvector, dtype=complex).ravel()
    m = np.asarray(m_values, dtype=int).ravel()
    if c.size != m.size:
        raise ValueError("eigenvector and m_values size mismatch.")
    fold = int(rotor_fold)
    if fold <= 1:
        raise ValueError("rotor_fold must be >= 2.")

    norm = float(np.sum(np.abs(c) ** 2))
    if norm <= 0.0:
        raise ValueError("eigenvector norm must be positive.")
    c = c / np.sqrt(norm)

    residues = {}
    for r in range(fold):
        mask = np.mod(m, fold) == r
        residues[r] = float(np.sum(np.abs(c[mask]) ** 2))
    dominant_res = max(residues, key=residues.get)
    purity = float(residues[dominant_res])

    # Translation alpha -> alpha + 2pi/fold in Fourier basis.
    trans_phase = np.exp(1j * 2.0 * np.pi * (m / float(fold)))
    translation_expectation = complex(np.vdot(c, trans_phase * c))

    if alpha_grid is None:
        alpha = np.linspace(-np.pi, np.pi, 361, endpoint=False)
    else:
        alpha = np.asarray(alpha_grid, dtype=float).ravel()
    phase = np.exp(1j * np.outer(alpha, m))
    psi = phase @ c
    pref = phase.conj() @ c  # psi(-alpha)
    p = np.abs(psi) ** 2
    pref_p = np.abs(pref) ** 2
    density_mirror_l1 = float(np.mean(np.abs(p - pref_p)))
    overlap = complex(np.vdot(psi, pref) / max(np.vdot(psi, psi).real, 1e-15))

    parity_label = "mixed"
    if abs(abs(overlap) - 1.0) < 1e-3 and abs(overlap.imag) < 1e-3:
        parity_label = "even" if overlap.real >= 0 else "odd"

    out = {
        "rotor_fold": fold,
        "translation_expectation": translation_expectation,
        "residue_weights": residues,
        "dominant_residue": int(dominant_res),
        "symmetry_purity": purity,
        "density_mirror_l1": density_mirror_l1,
        "parity_overlap": overlap,
        "parity_label": parity_label,
    }
    if fold == 3:
        sub = {0: "A", 1: "E1", 2: "E2"}[dominant_res]
        out["symmetry_label"] = "A" if dominant_res == 0 else "E"
        out["symmetry_sublabel"] = sub
    return out


def solve_ram_lite_levels(
    spec: TorsionHamiltonianSpec,
    J: int = 0,
    K: int = 0,
    n_levels: Optional[int] = None,
    symmetry_mode: Optional[str] = None,
    return_blocks: bool = False,
    label_levels: bool = False,
) -> dict:
    """
    Diagonalize RAM-lite Hamiltonian and return energies/eigenvectors.
    """
    mode = _canonical_symmetry_mode(symmetry_mode)
    H, m_vals, warnings = build_ram_lite_hamiltonian(spec, J=J, K=K)
    _warn_if_symmetry_mismatch(spec.potential, mode, warnings)
    e, U = np.linalg.eigh(H)
    e = np.real_if_close(e).astype(float)

    labels: list[str] = []
    sublabels: list[str] = []
    purities: list[float] = []
    diags: list[dict] = []
    if label_levels or mode == "c3":
        fold = 3 if mode == "c3" else 2
        for i in range(U.shape[1]):
            d = periodic_wavefunction_diagnostics(U[:, i], m_vals, rotor_fold=fold)
            diags.append(d)
            if mode == "c3":
                labels.append(str(d.get("symmetry_label", "unknown")))
                sublabels.append(str(d.get("symmetry_sublabel", "unknown")))
            else:
                labels.append(str(d.get("parity_label", "mixed")))
                sublabels.append(str(d.get("dominant_residue")))
            purities.append(float(d.get("symmetry_purity", 0.0)))

    if n_levels is not None:
        n = max(1, int(n_levels))
        e = e[:n]
        U = U[:, :n]
        if labels:
            labels = labels[:n]
            sublabels = sublabels[:n]
            purities = purities[:n]
            diags = diags[:n]

    block_data = None
    if return_blocks and mode == "c3":
        residues = _c3_residue_indices(m_vals)
        block_data = {}
        for r, idx in residues.items():
            if idx.size == 0:
                continue
            Hb = H[np.ix_(idx, idx)]
            eb, Ub = np.linalg.eigh(Hb)
            eb = np.real_if_close(eb).astype(float)
            key = {0: "A", 1: "E1", 2: "E2"}[r]
            block_data[key] = {
                "residue": int(r),
                "m_values": m_vals[idx],
                "indices": idx,
                "energies_cm-1": eb,
                "eigenvectors_block": Ub,
            }

    out = {
        "energies_cm-1": e,
        "eigenvectors": U,
        "m_values": m_vals,
        "hamiltonian": H,
        "warnings": warnings
        + [
            "RAM-lite Phase-1 approximation: not a full torsion-rotation Hamiltonian fit."
        ],
    }
    if labels:
        out["symmetry_labels"] = np.asarray(labels, dtype=object)
        out["symmetry_sublabels"] = np.asarray(sublabels, dtype=object)
        out["symmetry_purity"] = np.asarray(purities, dtype=float)
        out["state_diagnostics"] = diags
    if block_data is not None:
        out["symmetry_blocks"] = block_data
    return out


def torsion_probability_density(eigenvector: np.ndarray, alpha_grid: np.ndarray, m_values: np.ndarray) -> np.ndarray:
    """
    Compute |psi(alpha)|^2 for one eigenvector in Fourier basis.
    """
    c = np.asarray(eigenvector, dtype=complex).ravel()
    a = np.asarray(alpha_grid, dtype=float).ravel()
    m = np.asarray(m_values, dtype=int).ravel()
    if c.size != m.size:
        raise ValueError("eigenvector and m_values size mismatch.")
    phase = np.exp(1j * np.outer(a, m))
    psi = phase @ c
    p = np.abs(psi) ** 2
    # normalize on discrete grid
    s = float(np.sum(p))
    if s > 0:
        p /= s
    return p


def motion_average_constants_on_grid(
    B_grid_abc: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Average rotational constants [A,B,C] over a probability vector on grid points.
    """
    B = np.asarray(B_grid_abc, dtype=float)
    p = np.asarray(probabilities, dtype=float).ravel()
    if B.ndim != 2 or B.shape[1] != 3:
        raise ValueError("B_grid_abc must have shape (G, 3).")
    if p.size != B.shape[0]:
        raise ValueError("probabilities length must match grid length.")
    if np.any(p < 0.0):
        raise ValueError("probabilities must be non-negative.")
    if float(np.sum(p)) <= 0.0:
        raise ValueError("probabilities cannot sum to zero.")
    p = p / np.sum(p)
    return np.sum(p[:, None] * B, axis=0)


def torsion_objective_from_levels(
    predicted_rows: list[dict],
    target_rows: list[dict],
) -> dict:
    """
    Compare predicted torsion levels to user targets and return RMS residual.

    Matching key: (J, K, level_index)
    Residual unit: cm^-1
    """
    out = assign_levels_by_keys(predicted_rows, target_rows)
    if not out["rows"]:
        out["warnings"] = out["warnings"] + ["No matched torsion target levels."]
    return out


def _alias_key_name(name: str) -> str:
    s = str(name).strip()
    lower = s.lower()
    if lower in {"index", "level_index"}:
        return "index"
    if lower in {"label", "symmetry_label"}:
        return "label"
    if lower in {"sublabel", "symmetry_sublabel"}:
        return "sublabel"
    if lower == "j":
        return "J"
    if lower == "k":
        return "K"
    return s


def _get_key_value(row: dict, key_name: str):
    key = _alias_key_name(key_name)
    if key == "J":
        return None if "J" not in row else int(row["J"])
    if key == "K":
        return None if "K" not in row else int(row["K"])
    if key == "index":
        if "index" in row:
            return int(row["index"])
        if "level_index" in row:
            return int(row["level_index"])
        return None
    if key == "label":
        if "label" in row:
            return str(row["label"])
        if "symmetry_label" in row:
            return str(row["symmetry_label"])
        return None
    if key == "sublabel":
        if "sublabel" in row:
            return str(row["sublabel"])
        if "symmetry_sublabel" in row:
            return str(row["symmetry_sublabel"])
        return None
    return row.get(key_name)


def _resolve_key_priority(key_priority: Optional[list[tuple[str, ...]]]) -> list[tuple[str, ...]]:
    if key_priority is None:
        return [
            ("J", "K", "label", "sublabel", "index"),
            ("J", "K", "label", "index"),
            ("J", "K", "index"),
        ]
    out: list[tuple[str, ...]] = []
    for keys in key_priority:
        kk = tuple(_alias_key_name(k) for k in keys)
        if kk:
            out.append(kk)
    if not out:
        raise ValueError("key_priority must contain at least one non-empty key tuple.")
    return out


def assign_levels_by_keys(
    predicted_rows: list[dict],
    target_rows: list[dict],
    key_priority: Optional[list[tuple[str, ...]]] = None,
) -> dict:
    """
    Match predicted/target torsion rows using a prioritized list of key tuples.

    Returns:
      - rows/matches: matched rows with observed/predicted/residual energies
      - unmatched_predicted / unmatched_target: row indices not matched
      - warnings: diagnostics including ambiguous key collisions
      - rms_cm-1: RMS residual over matched rows (inf if no rows matched)
    """
    key_specs = _resolve_key_priority(key_priority)
    unmatched_pred = set(range(len(predicted_rows)))
    unmatched_targ = set(range(len(target_rows)))
    matches: list[dict] = []
    warnings: list[str] = []

    for keys in key_specs:
        pred_bins: dict[tuple, list[int]] = {}
        targ_bins: dict[tuple, list[int]] = {}

        for i in sorted(unmatched_pred):
            key = tuple(_get_key_value(predicted_rows[i], k) for k in keys)
            if any(v is None for v in key):
                continue
            pred_bins.setdefault(key, []).append(i)
        for j in sorted(unmatched_targ):
            key = tuple(_get_key_value(target_rows[j], k) for k in keys)
            if any(v is None for v in key):
                continue
            targ_bins.setdefault(key, []).append(j)

        for key in sorted(set(pred_bins.keys()) & set(targ_bins.keys())):
            pred_ids = pred_bins[key]
            targ_ids = targ_bins[key]
            n = min(len(pred_ids), len(targ_ids))
            if len(pred_ids) > 1 or len(targ_ids) > 1:
                warnings.append(
                    f"Ambiguous collision for keys={keys}, key={key}: "
                    f"{len(pred_ids)} predicted vs {len(targ_ids)} target; pairing by row order."
                )
            for idx in range(n):
                pi = pred_ids[idx]
                ti = targ_ids[idx]
                prow = predicted_rows[pi]
                trow = target_rows[ti]
                if "energy_cm-1" not in prow or "energy_cm-1" not in trow:
                    warnings.append(
                        f"Missing energy_cm-1 for matched pair predicted_index={pi}, target_index={ti}; skipped."
                    )
                    unmatched_pred.discard(pi)
                    unmatched_targ.discard(ti)
                    continue
                pred = float(prow["energy_cm-1"])
                obs = float(trow["energy_cm-1"])
                row = {
                    "J": _get_key_value(trow, "J"),
                    "K": _get_key_value(trow, "K"),
                    "level_index": _get_key_value(trow, "index"),
                    "observed_cm-1": obs,
                    "predicted_cm-1": pred,
                    "residual_cm-1": obs - pred,
                    "matched_keys": keys,
                    "predicted_row_index": pi,
                    "target_row_index": ti,
                }
                matches.append(row)
                unmatched_pred.discard(pi)
                unmatched_targ.discard(ti)

    for ti in sorted(unmatched_targ):
        trow = target_rows[ti]
        key = (
            _get_key_value(trow, "J"),
            _get_key_value(trow, "K"),
            _get_key_value(trow, "index"),
        )
        warnings.append(f"Missing predicted level for target row index={ti}, key={key}")

    if not matches:
        return {
            "rows": [],
            "matches": [],
            "rms_cm-1": float("inf"),
            "warnings": warnings,
            "unmatched_predicted": sorted(unmatched_pred),
            "unmatched_target": sorted(unmatched_targ),
            "key_priority_used": key_specs,
        }
    res = np.asarray([r["residual_cm-1"] for r in matches], dtype=float)
    rms = float(np.sqrt(np.mean(res ** 2)))
    return {
        "rows": matches,
        "matches": matches,
        "rms_cm-1": rms,
        "warnings": warnings,
        "unmatched_predicted": sorted(unmatched_pred),
        "unmatched_target": sorted(unmatched_targ),
        "key_priority_used": key_specs,
    }


def auto_assign_levels_by_proximity(
    predicted_rows: list[dict],
    observed_energies_cm1: list[float],
    *,
    max_delta_cm1: float = float("inf"),
    symmetry_filter: Optional[str] = None,
    method: str = "global",
    ambiguity_tol_cm1: float = 0.05,
) -> dict:
    """
    Assign observed energies to predicted levels by energy proximity.

    Uses a global minimum-cost assignment by default, falling back to the
    historical greedy nearest-neighbour method for large candidate sets.
    Optionally restricted to levels matching a symmetry label.

    Parameters
    ----------
    predicted_rows : list of dicts with 'energy_cm-1' and optional
                     'J', 'K', 'level_index', 'symmetry_label' keys
    observed_energies_cm1 : observed energies to assign
    max_delta_cm1 : maximum allowed |predicted - observed| for a match;
                    unmatched entries are flagged in the output
    symmetry_filter : if given, only consider predicted rows whose
                      'symmetry_label' equals this string (e.g. 'A' or 'E')
    method : 'global' (default), 'greedy', or 'auto'
    ambiguity_tol_cm1 : warn when another candidate is this close to the
                        selected candidate's absolute residual

    Returns
    -------
    dict with:
      assignments : list of dicts — one per observed energy, each with:
          observed_cm-1, predicted_cm-1, delta_cm-1, predicted_row_index,
          matched (bool), symmetry_label (if available)
      rms_cm-1 : float — RMS over matched assignments (inf if none)
      n_matched : int
      n_unmatched : int
      warnings : list[str]
    """
    obs = np.asarray([float(e) for e in observed_energies_cm1], dtype=float).ravel()
    warnings: list[str] = []

    # Filter candidate predicted rows by symmetry if requested.
    candidate_indices = [
        i for i, r in enumerate(predicted_rows)
        if (symmetry_filter is None or
            str(r.get("symmetry_label", "")).upper() == str(symmetry_filter).upper())
        and "energy_cm-1" in r
    ]
    if not candidate_indices:
        return {
            "assignments": [],
            "rms_cm-1": float("inf"),
            "n_matched": 0,
            "n_unmatched": int(obs.size),
            "method_used": str(method or "global").strip().lower(),
            "warnings": ["No predicted rows available for proximity assignment."],
        }

    pred_e = np.asarray(
        [float(predicted_rows[i]["energy_cm-1"]) for i in candidate_indices], dtype=float
    )

    method_l = str(method or "global").strip().lower()
    if method_l not in {"global", "greedy", "auto"}:
        warnings.append(f"Unknown auto assignment method '{method}'; using global.")
        method_l = "global"
    if method_l == "auto":
        method_l = "global" if len(candidate_indices) <= 18 else "greedy"
    if method_l == "global" and len(candidate_indices) > 18:
        warnings.append(
            "Global proximity assignment skipped because more than 18 candidate levels "
            "were available; using greedy assignment."
        )
        method_l = "greedy"

    obs_order = np.argsort(obs, kind="stable")
    assignments_by_obs_idx: dict[int, dict] = {}

    matched_pairs: list[tuple[int, int]] = []
    if method_l == "global":
        n_obs = len(obs_order)
        n_pred = len(candidate_indices)
        inf = float("inf")
        dp: dict[tuple[int, int], tuple[float, list[tuple[int, int]]]] = {
            (0, 0): (0.0, [])
        }
        for pos in range(n_obs):
            next_dp: dict[tuple[int, int], tuple[float, list[tuple[int, int]]]] = {}
            obs_idx = int(obs_order[pos])
            obs_val = float(obs[obs_idx])
            for (_old_pos, mask), (cost, pairs) in dp.items():
                # Leave this observation unmatched with a finite penalty so the
                # assignment can still complete when max_delta filters all rows.
                penalty = float(max_delta_cm1)
                if not np.isfinite(penalty):
                    penalty = max(float(np.max(np.abs(pred_e - obs_val))) * 2.0, 1.0)
                else:
                    penalty = penalty * 1.000001
                un_key = (pos + 1, mask)
                un_cost = cost + penalty ** 2
                if un_cost < next_dp.get(un_key, (inf, []))[0]:
                    next_dp[un_key] = (un_cost, pairs)
                for cand in range(n_pred):
                    if mask & (1 << cand):
                        continue
                    delta = abs(float(pred_e[cand]) - obs_val)
                    if delta > float(max_delta_cm1):
                        continue
                    key = (pos + 1, mask | (1 << cand))
                    new_cost = cost + delta ** 2
                    if new_cost < next_dp.get(key, (inf, []))[0]:
                        next_dp[key] = (new_cost, pairs + [(obs_idx, cand)])
            dp = next_dp
        if dp:
            _best_key, (_best_cost, matched_pairs) = min(dp.items(), key=lambda kv: kv[1][0])
    else:
        available = set(range(len(candidate_indices)))
        for obs_idx in obs_order:
            obs_val = float(obs[obs_idx])
            if not available:
                break
            avail_list = sorted(available)
            deltas = np.abs(pred_e[avail_list] - obs_val)
            best_pos = int(np.argmin(deltas))
            best_cand = avail_list[best_pos]
            best_delta = float(deltas[best_pos])
            if best_delta <= float(max_delta_cm1):
                available.discard(best_cand)
                matched_pairs.append((int(obs_idx), int(best_cand)))

    matched_by_obs = {obs_idx: cand for obs_idx, cand in matched_pairs}
    for obs_idx in range(obs.size):
        obs_val = float(obs[obs_idx])
        if int(obs_idx) not in matched_by_obs:
            nearest_delta = float(np.min(np.abs(pred_e - obs_val))) if pred_e.size else float("inf")
            warnings.append(
                f"Observed energy {obs_val:.4f} cm^-1 has no match within "
                f"{max_delta_cm1:.2f} cm^-1 (nearest delta={nearest_delta:.4f})."
            )
            assignments_by_obs_idx[int(obs_idx)] = {
                "observed_cm-1": obs_val,
                "predicted_cm-1": float("nan"),
                "delta_cm-1": float("inf"),
                "predicted_row_index": -1,
                "matched": False,
            }
            continue
        best_cand = int(matched_by_obs[int(obs_idx)])
        best_delta = abs(float(pred_e[best_cand]) - obs_val)
        pred_row_global = candidate_indices[best_cand]
        pred_row = predicted_rows[pred_row_global]

        entry: dict = {
            "observed_cm-1": obs_val,
            "predicted_cm-1": float(pred_e[best_cand]),
            "delta_cm-1": best_delta,
            "predicted_row_index": pred_row_global,
            "matched": True,
            "assignment_method": method_l,
        }
        for k in ("J", "K", "level_index", "symmetry_label", "symmetry_sublabel"):
            if k in pred_row:
                entry[k] = pred_row[k]
        all_deltas = np.sort(np.abs(pred_e - obs_val))
        if all_deltas.size > 1 and float(all_deltas[1] - all_deltas[0]) <= float(ambiguity_tol_cm1):
            warnings.append(
                f"Observed energy {obs_val:.4f} cm^-1 has ambiguous proximity assignment "
                f"(nearest deltas differ by {float(all_deltas[1] - all_deltas[0]):.4f} cm^-1)."
            )
        assignments_by_obs_idx[int(obs_idx)] = entry

    # Reassemble in original observation order.
    assignments = [assignments_by_obs_idx.get(i, {
        "observed_cm-1": float(obs[i]),
        "predicted_cm-1": float("nan"),
        "delta_cm-1": float("inf"),
        "predicted_row_index": -1,
        "matched": False,
    }) for i in range(obs.size)]

    matched_deltas = [r["delta_cm-1"] for r in assignments if r["matched"]]
    rms = float(np.sqrt(np.mean(np.asarray(matched_deltas, dtype=float) ** 2))) if matched_deltas else float("inf")
    return {
        "assignments": assignments,
        "rms_cm-1": rms,
        "n_matched": int(len(matched_deltas)),
        "n_unmatched": int(obs.size - len(matched_deltas)),
        "method_used": method_l,
        "warnings": warnings,
    }
