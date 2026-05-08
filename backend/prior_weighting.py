from __future__ import annotations

import numpy as np


def identifiability_scores_from_jacobian(
    j_spectral_q: np.ndarray,
    sv_rel_threshold: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute coordinate visibility scores in [0, 1] from spectral J(q).

    score_j = sum_k V_{jk}^2 over spectroscopically visible singular vectors k.
    """
    j = np.asarray(j_spectral_q, dtype=float)
    if j.ndim != 2 or j.shape[1] == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float), 0

    _, sv, vt = np.linalg.svd(j, full_matrices=False)
    if sv.size == 0 or sv[0] <= 0.0:
        return np.zeros(j.shape[1], dtype=float), sv, 0

    rel = max(float(sv_rel_threshold), 1e-12)
    rank = int(np.sum(sv > max(rel * sv[0], 1e-12)))
    rank = max(rank, 0)
    v = vt.T
    if rank == 0:
        return np.zeros(v.shape[0], dtype=float), sv, 0
    scores = np.sum(v[:, :rank] ** 2, axis=1)
    scores = np.clip(scores, 0.0, 1.0)
    return scores, sv, rank


def adaptive_sigma_from_identifiability(
    sigma_base: np.ndarray,
    scores: np.ndarray,
    gamma: float = 2.0,
    min_sigma_scale: float = 0.5,
    max_sigma_scale: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    sigma_eff = sigma_base * (1 + gamma * score), clipped to scale bounds.
    """
    sb = np.asarray(sigma_base, dtype=float)
    sc = np.asarray(scores, dtype=float)
    if sb.shape != sc.shape:
        raise ValueError(f"sigma_base and scores shape mismatch: {sb.shape} vs {sc.shape}")

    g = max(float(gamma), 0.0)
    smin = max(float(min_sigma_scale), 1e-6)
    smax = max(float(max_sigma_scale), smin)
    raw_scale = 1.0 + g * np.clip(sc, 0.0, 1.0)
    scale = np.clip(raw_scale, smin, smax)
    sigma_eff = np.maximum(sb * scale, 1e-12)
    return sigma_eff, scale

