"""
Kraitchman substitution-structure (rs) analysis and inertial-defect diagnostics.

Single isotopic substitution changes the principal moments of inertia in a way
that depends only on the substituted atom's position in the *parent* principal
axis system.  Kraitchman's equations (Gordy & Cook, *Microwave Molecular
Spectra*, 3rd ed., Ch. 13) invert that relationship, giving |a|, |b|, |c| of
the substituted atom directly from the two sets of rotational constants — no
geometry model required.  This is the field-standard cross-check for any
fitted structure and is what programs like KRA and STRFIT report.

All moments are in amu·Å², constants in MHz, coordinates in Å.

Conventions
-----------
* Planar moments: P_a = Σ m_i a_i² = ( -I_a + I_b + I_c ) / 2, and cyclic.
* Reduced mass of substitution: μ = M·Δm / (M + Δm).
* Costain rule: σ(|z|) ≈ K / |z| with K = 0.0015 Å² accounts for zero-point
  vibration effects, and dominates propagated measurement error for typical
  microwave data.

Small negative squared coordinates (atom near a principal axis) are clamped to
zero and flagged rather than raised, matching standard practice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# h / (8π² · amu · Å²) → MHz  (same constant as backend.spectral)
_INERTIA_TO_MHZ = 505379.0084353526

# Costain constant for rs coordinate uncertainty [Å²]
COSTAIN_K = 0.0015


def moments_from_constants(constants_mhz) -> np.ndarray:
    """Principal moments of inertia (amu·Å²) from rotational constants A ≥ B ≥ C in MHz."""
    c = np.asarray(constants_mhz, dtype=float)
    if np.any(c <= 0):
        raise ValueError("Rotational constants must be positive.")
    return _INERTIA_TO_MHZ / c


def planar_moments(constants_mhz) -> np.ndarray:
    """
    Planar moments (P_a, P_b, P_c) in amu·Å² from constants (A, B, C) in MHz.

    P_a = Σ m_i a_i² = (-I_a + I_b + I_c)/2, and cyclic permutations.
    """
    I = moments_from_constants(constants_mhz)
    Ia, Ib, Ic = I
    return np.array([
        0.5 * (-Ia + Ib + Ic),
        0.5 * (Ia - Ib + Ic),
        0.5 * (Ia + Ib - Ic),
    ])


def inertial_defect(constants_mhz) -> float:
    """
    Inertial defect Δ = I_c − I_a − I_b in amu·Å².

    Δ ≈ 0 (typically +0.05…+0.5 from zero-point effects) for a rigid planar
    molecule; large or negative values indicate non-planarity or
    large-amplitude motion contaminating the ground-state constants.
    """
    Ia, Ib, Ic = moments_from_constants(constants_mhz)
    return float(Ic - Ia - Ib)


@dataclass
class KraitchmanResult:
    """rs coordinates of one substituted atom in the parent principal axis system."""

    atom_index: int | None
    delta_mass: float
    reduced_mass: float
    coords_abs: np.ndarray            # |a|, |b|, |c| in Å (0.0 where imaginary)
    coords_sq: np.ndarray             # signed squared coordinates a², b², c² [Å²]
    sigma_costain: np.ndarray         # Costain uncertainties per coordinate [Å]
    imaginary_axes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def kraitchman_single_substitution(
    parent_constants_mhz,
    substituted_constants_mhz,
    parent_total_mass_amu: float,
    delta_mass_amu: float,
    atom_index: int | None = None,
) -> KraitchmanResult:
    """
    Solve Kraitchman's equations for a single isotopic substitution in a
    (possibly asymmetric-top) molecule.

    Parameters
    ----------
    parent_constants_mhz, substituted_constants_mhz : (A, B, C) in MHz
    parent_total_mass_amu : total mass M of the parent species
    delta_mass_amu : mass change Δm of the substituted atom (may be negative)
    atom_index : optional atom label carried through to the result

    Returns
    -------
    KraitchmanResult with |a|, |b|, |c| in the parent principal axis system.
    """
    M = float(parent_total_mass_amu)
    dm = float(delta_mass_amu)
    if M <= 0:
        raise ValueError("parent_total_mass_amu must be positive.")
    if dm == 0:
        raise ValueError("delta_mass_amu must be nonzero for a substitution.")
    mu = M * dm / (M + dm)

    P = planar_moments(parent_constants_mhz)
    Ps = planar_moments(substituted_constants_mhz)

    warnings: list[str] = []
    imaginary: list[str] = []
    coords_sq = np.zeros(3)
    axes = ("a", "b", "c")

    # Exact rank-one-update identity: the substituted planar-moment tensor in
    # parent axes is P + μ·xxᵀ, so its eigenvalues P'_k determine
    #   x_i² = Π_k (P'_k − P_i) / ( μ · Π_{k≠i} (P_k − P_i) ),
    # equivalent to Kraitchman's equations (Gordy & Cook, Ch. 13).
    for i in range(3):
        num = float(np.prod(Ps - P[i]))
        denom = mu
        for k in range(3):
            if k == i:
                continue
            gap = P[k] - P[i]
            if abs(gap) < 1e-9:
                warnings.append(
                    f"Planar moments P_{axes[i]} and P_{axes[k]} nearly equal; "
                    f"{axes[i]}-coordinate is ill-determined (symmetric-top degeneracy)."
                )
                gap = 1e-9 if gap >= 0 else -1e-9
            denom *= gap
        coords_sq[i] = num / denom

    coords_abs = np.zeros(3)
    sigma = np.zeros(3)
    for i in range(3):
        if coords_sq[i] < 0:
            imaginary.append(axes[i])
            if coords_sq[i] < -0.01:
                warnings.append(
                    f"{axes[i]}² = {coords_sq[i]:.5f} Å² is significantly negative; "
                    "check constants or vibrational contamination."
                )
            coords_abs[i] = 0.0
            sigma[i] = float("nan")
        else:
            coords_abs[i] = float(np.sqrt(coords_sq[i]))
            # Costain rule; diverges for atoms on/near an axis, cap at 1 Å.
            sigma[i] = min(COSTAIN_K / coords_abs[i], 1.0) if coords_abs[i] > 0 else 1.0

    return KraitchmanResult(
        atom_index=atom_index,
        delta_mass=dm,
        reduced_mass=mu,
        coords_abs=coords_abs,
        coords_sq=coords_sq,
        sigma_costain=sigma,
        imaginary_axes=imaginary,
        warnings=warnings,
    )


def _find_single_substitution(parent_masses, iso_masses, tol=1e-6):
    """Return (atom_index, delta_mass) if exactly one mass differs, else None."""
    p = np.asarray(parent_masses, dtype=float)
    s = np.asarray(iso_masses, dtype=float)
    if p.shape != s.shape:
        return None
    diff = np.where(np.abs(s - p) > tol)[0]
    if diff.size != 1:
        return None
    idx = int(diff[0])
    return idx, float(s[idx] - p[idx])


def kraitchman_analysis(isotopologues: list[dict], parent_index: int = 0) -> dict:
    """
    Run Kraitchman analysis on every singly-substituted isotopologue.

    Parameters
    ----------
    isotopologues : list of dicts with at least
        'masses' (per-atom amu) and 'obs_constants' (A, B, C in MHz).
        Ground-state (B0) constants are the standard input for rs structures.
        Optional 'name' is carried into the rows.
    parent_index : which entry is the parent species (default: first).

    Returns
    -------
    dict with:
      rows      : list of dicts (one per substituted species) with atom_index,
                  |a|, |b|, |c|, Costain sigmas, and flags
      defects   : list of dicts with per-species inertial defects
      warnings  : list of strings
      skipped   : names of isotopologues that are not single substitutions
    """
    if not isotopologues:
        return {"rows": [], "defects": [], "warnings": ["No isotopologues provided."], "skipped": []}
    parent = isotopologues[parent_index]
    p_masses = np.asarray(parent["masses"], dtype=float)
    p_const = np.asarray(parent["obs_constants"], dtype=float)
    M = float(p_masses.sum())

    rows: list[dict] = []
    defects: list[dict] = []
    warnings: list[str] = []
    skipped: list[str] = []

    for i, iso in enumerate(isotopologues):
        name = str(iso.get("name", f"iso_{i}"))
        const = np.asarray(iso["obs_constants"], dtype=float)
        if const.size == 3 and np.all(const > 0):
            defects.append({"name": name, "inertial_defect_amuA2": inertial_defect(const)})
        if i == parent_index:
            continue
        if const.size != 3 or np.any(const <= 0):
            skipped.append(name)
            warnings.append(f"{name}: needs all three positive constants for Kraitchman analysis.")
            continue
        sub = _find_single_substitution(p_masses, iso["masses"])
        if sub is None:
            skipped.append(name)
            continue
        idx, dm = sub
        res = kraitchman_single_substitution(p_const, const, M, dm, atom_index=idx)
        warnings.extend(f"{name}: {w}" for w in res.warnings)
        rows.append({
            "name": name,
            "atom_index": idx,
            "delta_mass_amu": dm,
            "abs_a_angstrom": float(res.coords_abs[0]),
            "abs_b_angstrom": float(res.coords_abs[1]),
            "abs_c_angstrom": float(res.coords_abs[2]),
            "sigma_a_angstrom": float(res.sigma_costain[0]),
            "sigma_b_angstrom": float(res.sigma_costain[1]),
            "sigma_c_angstrom": float(res.sigma_costain[2]),
            "imaginary_axes": ",".join(res.imaginary_axes),
        })

    return {"rows": rows, "defects": defects, "warnings": warnings, "skipped": skipped}


def principal_axis_coords(coords, masses) -> np.ndarray:
    """
    Rotate/translate coordinates into the principal axis system (a, b, c),
    with axes ordered so that I_a ≤ I_b ≤ I_c.
    """
    c = np.asarray(coords, dtype=float)
    m = np.asarray(masses, dtype=float)
    cm = m @ c / m.sum()
    r = c - cm
    r2 = np.einsum("ij,ij->i", r, r)
    I = np.einsum("i,jk->jk", m * r2, np.eye(3)) - np.einsum("i,ij,ik->jk", m, r, r)
    evals, evecs = np.linalg.eigh(I)  # ascending: I_a, I_b, I_c
    if np.linalg.det(evecs) < 0:
        evecs[:, 2] *= -1.0
    return r @ evecs


def compare_rs_to_geometry(rows: list[dict], coords, masses) -> list[dict]:
    """
    Compare Kraitchman |a|,|b|,|c| values against a fitted geometry.

    The fitted coordinates are rotated into their own principal axis system and
    the absolute values compared per substituted atom.

    Returns rows augmented with fit_|a|,|b|,|c| and per-axis deviations.
    """
    pac = np.abs(principal_axis_coords(coords, masses))
    out = []
    for row in rows:
        idx = int(row["atom_index"])
        if idx < 0 or idx >= len(pac):
            continue
        r = dict(row)
        for ax, col in zip(range(3), ("a", "b", "c")):
            r[f"fit_abs_{col}_angstrom"] = float(pac[idx, ax])
            r[f"delta_{col}_angstrom"] = float(pac[idx, ax] - row[f"abs_{col}_angstrom"])
        out.append(r)
    return out
