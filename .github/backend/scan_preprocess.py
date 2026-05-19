"""
Torsion scan preprocessing utilities.

Preprocessing pipeline for 1D torsion potential scans before Fourier fitting:
  1. sort_scan            — sort by ascending torsion angle
  2. deduplicate_endpoint — remove duplicate periodic endpoint
  3. extend_by_symmetry   — replicate n-fold symmetric scan to fill full period
  4. preprocess_scan      — combined pipeline with configurable steps

All functions operate on (phi_rad, energies_cm1) pairs and return new arrays
without modifying the inputs. The combined pipeline returns an info dict
describing which steps were applied and how many points resulted.

Typical usage for a 3-fold symmetric rotor (e.g. methanol CH3):
    phi_pp, e_pp, info = preprocess_scan(
        phi_rad, energies_cm1,
        symmetry_number=3,
        do_deduplicate=True,
        do_extend_by_symmetry=True,
    )
"""

from __future__ import annotations

import numpy as np


def sort_scan(
    phi_rad: np.ndarray,
    energies_cm1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort scan points by ascending torsion angle.

    Returns new sorted arrays (no mutation of inputs).
    """
    phi = np.asarray(phi_rad, dtype=float).ravel()
    e = np.asarray(energies_cm1, dtype=float).ravel()
    if phi.size != e.size:
        raise ValueError(
            f"phi_rad ({phi.size}) and energies_cm1 ({e.size}) must have the same length."
        )
    idx = np.argsort(phi, kind="stable")
    return phi[idx], e[idx]


def deduplicate_endpoint(
    phi_rad: np.ndarray,
    energies_cm1: np.ndarray,
    *,
    period_rad: float = 2.0 * np.pi,
    tol_rad: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Remove the duplicate periodic endpoint if present.

    The last point is considered a duplicate when::

        |phi[-1] - phi[0] - period_rad| < tol_rad

    Such points arise when quantum chemistry codes include both phi=0 and
    phi=period as separate scan points.  Keeping the duplicate biases v0
    in the Fourier fit.

    Parameters
    ----------
    phi_rad : torsion angles in radians (should be sorted ascending)
    energies_cm1 : potential energies in cm^-1
    period_rad : expected period of the torsion coordinate
    tol_rad : tolerance for detecting the duplicate endpoint (radians)

    Returns
    -------
    (phi_out, energies_out, removed)
    removed is True when a duplicate was found and stripped.
    """
    phi = np.asarray(phi_rad, dtype=float).ravel()
    e = np.asarray(energies_cm1, dtype=float).ravel()
    if phi.size != e.size:
        raise ValueError(
            f"phi_rad ({phi.size}) and energies_cm1 ({e.size}) must have the same length."
        )
    if phi.size < 2:
        return phi.copy(), e.copy(), False

    residual = abs(float(phi[-1] - phi[0]) - float(period_rad))
    if residual < float(tol_rad):
        return phi[:-1].copy(), e[:-1].copy(), True
    return phi.copy(), e.copy(), False


def extend_by_symmetry(
    phi_rad: np.ndarray,
    energies_cm1: np.ndarray,
    symmetry_number: int,
    *,
    period_rad: float = 2.0 * np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend a partial torsion scan to full period using n-fold rotational symmetry.

    For a rotor with fold-symmetry ``symmetry_number`` n, the torsional
    potential repeats with period ``period_rad / n``.  This function
    replicates the supplied scan points into all n copies shifted by
    ``k * (period_rad / n)`` for k = 0, ..., n-1, then sorts the result.

    Duplicate angle values (within 1e-9 rad) are removed after replication.

    Parameters
    ----------
    phi_rad : torsion angles covering approximately one symmetry segment
              (~period_rad / symmetry_number)
    energies_cm1 : potential energies at each angle
    symmetry_number : fold symmetry of the rotor (1 = no extension)
    period_rad : full torsion period (default 2*pi)

    Returns
    -------
    (phi_extended, energies_extended) — sorted, near-duplicate-free
    """
    n = int(symmetry_number)
    if n < 1:
        raise ValueError("symmetry_number must be >= 1.")
    phi = np.asarray(phi_rad, dtype=float).ravel()
    e = np.asarray(energies_cm1, dtype=float).ravel()
    if phi.size != e.size:
        raise ValueError(
            f"phi_rad ({phi.size}) and energies_cm1 ({e.size}) must have the same length."
        )
    if n == 1:
        return phi.copy(), e.copy()

    period = float(period_rad)
    step = period / n
    phi_parts = [phi]
    e_parts = [e]
    for k in range(1, n):
        phi_parts.append(phi + k * step)
        e_parts.append(e.copy())

    phi_all = np.concatenate(phi_parts)
    e_all = np.concatenate(e_parts)

    idx = np.argsort(phi_all, kind="stable")
    phi_sorted = phi_all[idx]
    e_sorted = e_all[idx]

    # Remove near-duplicate angles (within 1e-9 rad)
    if phi_sorted.size > 1:
        diffs = np.diff(phi_sorted)
        keep = np.concatenate([[True], diffs > 1e-9])
        phi_sorted = phi_sorted[keep]
        e_sorted = e_sorted[keep]

    return phi_sorted, e_sorted


def preprocess_scan(
    phi_rad: np.ndarray,
    energies_cm1: np.ndarray,
    *,
    symmetry_number: int = 1,
    period_rad: float = 2.0 * np.pi,
    do_sort: bool = True,
    do_deduplicate: bool = True,
    do_extend_by_symmetry: bool = False,
    endpoint_tol_rad: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Combined preprocessing pipeline for torsion scan data.

    Steps applied in order (each optional via keyword flags):

    1. **sort** — sort by ascending torsion angle (``do_sort``)
    2. **deduplicate** — remove duplicate periodic endpoint (``do_deduplicate``)
    3. **extend_by_symmetry** — replicate n-fold symmetric scan to fill
       the full period (``do_extend_by_symmetry``)

    Parameters
    ----------
    phi_rad : torsion angles in radians
    energies_cm1 : potential energies in cm^-1
    symmetry_number : rotor fold symmetry (used in extension step)
    period_rad : full torsion period (default 2*pi)
    do_sort : sort by ascending angle
    do_deduplicate : remove duplicate periodic endpoint
    do_extend_by_symmetry : replicate scan using n-fold symmetry
    endpoint_tol_rad : tolerance for duplicate endpoint detection (rad)

    Returns
    -------
    (phi_out, energies_out, info)

    info keys:
      sorted : bool
      deduplicated : bool — True if a duplicate endpoint was removed
      extended_by_symmetry : bool
      symmetry_number : int — value used for extension
      n_points_in : int — input point count
      n_points_out : int — output point count
      warnings : list[str]
    """
    phi = np.asarray(phi_rad, dtype=float).ravel()
    e = np.asarray(energies_cm1, dtype=float).ravel()
    if phi.size != e.size:
        raise ValueError(
            f"phi_rad ({phi.size}) and energies_cm1 ({e.size}) must have the same length."
        )

    info: dict = {
        "sorted": False,
        "deduplicated": False,
        "extended_by_symmetry": False,
        "symmetry_number": int(symmetry_number),
        "n_points_in": int(phi.size),
        "n_points_out": int(phi.size),
        "warnings": [],
    }

    if do_sort:
        phi, e = sort_scan(phi, e)
        info["sorted"] = True

    if do_deduplicate:
        phi, e, removed = deduplicate_endpoint(
            phi, e, period_rad=period_rad, tol_rad=endpoint_tol_rad
        )
        info["deduplicated"] = removed
        if removed:
            info["warnings"].append(
                "Duplicate periodic endpoint detected and removed before fitting."
            )

    if do_extend_by_symmetry and int(symmetry_number) > 1:
        phi, e = extend_by_symmetry(phi, e, symmetry_number, period_rad=period_rad)
        info["extended_by_symmetry"] = True

    info["n_points_out"] = int(phi.size)
    return phi, e, info
