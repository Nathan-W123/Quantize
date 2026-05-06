"""
Identifiability analysis for internal-coordinate structure refinement.

Labels each bond/angle/dihedral as:
    constrained     — dominated by spectral data; SVD participation score high
    partial         — moderate spectral sensitivity; mix of data and prior
    weak            — low spectral sensitivity; prior plays a significant role
    prior-dominated — essentially no spectral sensitivity; value set by prior
    unidentifiable  — zero or near-zero sensitivity and no informative prior

The key diagnostic is the participation score:
    p_j = sum_i (v_{ij}^2 * s_i^2) / sum_i s_i^2
where v_{ij} is the j-th component of the i-th right singular vector and s_i is the
i-th singular value of the weighted internal Jacobian W^{1/2} Jq.

High p_j → coordinate j is well-determined by spectral data.
Low p_j  → coordinate j lies mostly in the null space.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# Threshold defaults for labelling
_THRESH_CONSTRAINED = 0.5      # p_j > this → constrained
_THRESH_PARTIAL = 0.2          # p_j in (partial, constrained) → partial
_THRESH_WEAK = 0.05            # p_j in (weak, partial) → weak
# below weak → prior-dominated (if prior exists) or unidentifiable


def participation_scores(Jq: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute SVD participation scores for each internal coordinate.

    Parameters
    ----------
    Jq : (m, n_q)  Internal-coordinate Jacobian (already weighted by observation sigma).

    Returns
    -------
    scores : (n_q,)  Participation score per coordinate in [0, 1].
    sv     : (min(m,n_q),)  Singular values.
    rank   : int    Effective rank (sv > 1e-3 * sv_max).
    """
    Jq = np.asarray(Jq, dtype=float)
    U, sv, Vt = np.linalg.svd(Jq, full_matrices=False)
    s2 = sv ** 2
    total = max(float(s2.sum()), 1e-12)
    # score_j = sum_i (Vt[i,j]^2 * s_i^2) / total
    scores = (Vt.T ** 2) @ s2 / total

    rank = int(np.sum(sv > max(1e-3 * sv[0], 1e-12))) if sv[0] > 0 else 0
    return scores, sv, rank


def label_identifiability(
    scores: np.ndarray,
    sigma_prior: Optional[np.ndarray] = None,
    thresh_constrained: float = _THRESH_CONSTRAINED,
    thresh_partial: float = _THRESH_PARTIAL,
    thresh_weak: float = _THRESH_WEAK,
) -> list[str]:
    """
    Assign identifiability labels to each internal coordinate.

    Parameters
    ----------
    scores       : (n_q,)  Participation scores from participation_scores().
    sigma_prior  : (n_q,) or None  Prior sigmas (Å, rad).  None means no informative prior.
    thresh_*     : float   Label boundaries (see module docstring).

    Returns
    -------
    labels : list[str]  One of "constrained", "partial", "weak",
                        "prior-dominated", "unidentifiable".
    """
    labels = []
    for j, p in enumerate(scores):
        if p >= thresh_constrained:
            labels.append("constrained")
        elif p >= thresh_partial:
            labels.append("partial")
        elif p >= thresh_weak:
            labels.append("weak")
        else:
            has_prior = (
                sigma_prior is not None
                and j < len(sigma_prior)
                and np.isfinite(sigma_prior[j])
                and sigma_prior[j] < 1e6
            )
            labels.append("prior-dominated" if has_prior else "unidentifiable")
    return labels


def identifiability_table(
    coord_set,
    Jq: np.ndarray,
    sigma_prior: Optional[np.ndarray] = None,
    thresh_constrained: float = _THRESH_CONSTRAINED,
    thresh_partial: float = _THRESH_PARTIAL,
    thresh_weak: float = _THRESH_WEAK,
) -> list[dict]:
    """
    Build a human-readable identifiability table.

    Parameters
    ----------
    coord_set  : InternalCoordinateSet
    Jq         : (m, n_q)  Internal Jacobian (already weighted).
    sigma_prior : (n_q,) or None
    thresh_*   : float

    Returns
    -------
    rows : list of dict with keys:
        name, score, rank_contribution, label
    """
    scores, sv, rank = participation_scores(Jq)
    labels = label_identifiability(scores, sigma_prior, thresh_constrained, thresh_partial, thresh_weak)
    active = coord_set.active_coords()

    rows = []
    for ic, p, lbl in zip(active, scores, labels):
        rows.append({
            "name": ic.name,
            "score": float(p),
            "label": lbl,
            "sv_rank": rank,
        })
    return rows, sv, rank


def print_identifiability_table(rows: list[dict], sv: np.ndarray, rank: int) -> None:
    """Pretty-print the identifiability table."""
    print(f"\nIdentifiability analysis  (SVD rank = {rank}, "
          f"top sv = {sv[0]:.3e}, bottom sv = {sv[-1]:.3e})")
    print(f"{'Coordinate':<28}  {'Score':>8}  {'Label'}")
    print("-" * 55)
    for r in rows:
        bar = "#" * int(r["score"] * 20)
        print(f"{r['name']:<28}  {r['score']:>8.4f}  {r['label']}  |{bar:<20}|")
