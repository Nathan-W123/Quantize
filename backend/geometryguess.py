"""
Initial geometry guess utilities.

These helpers provide deterministic, template-based starting geometries so
driver scripts do not need hardcoded coordinate arrays.
"""

import numpy as np

_COV_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39, "Si": 1.11, "B": 0.84,
}


def _bond_length_guess(e1, e2):
    r1 = _COV_RADII.get(e1, 0.80)
    r2 = _COV_RADII.get(e2, 0.80)
    return 1.10 * (r1 + r2)


def _unit(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def _relax_geometry(
    coords,
    bonds,
    targets,
    n_steps=200,
    dt=0.01,
    k_bond=5.0,
    k_rep=0.01,
    rep_cut=1.2,
):
    """Simple deterministic spring/repulsion relaxation."""
    coords = coords.copy()
    n = len(coords)
    bonded = {tuple(sorted((i, j))) for i, j in bonds}
    for _ in range(n_steps):
        f = np.zeros_like(coords)
        for i, j in bonds:
            rij = coords[j] - coords[i]
            d = np.linalg.norm(rij)
            if d < 1e-9:
                continue
            u = rij / d
            t = targets[(min(i, j), max(i, j))]
            fb = k_bond * (d - t)
            f[i] += fb * u
            f[j] -= fb * u
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in bonded:
                    continue
                rij = coords[j] - coords[i]
                d = np.linalg.norm(rij)
                if 1e-9 < d < rep_cut:
                    fr = k_rep * (rep_cut - d) / d
                    f[i] -= fr * rij
                    f[j] += fr * rij
        coords -= dt * f
        coords -= coords.mean(axis=0, keepdims=True)
    return coords


def guess_bent_triatomic(
    central_elem,
    terminal1_elem,
    terminal2_elem,
    r1=1.0,
    r2=1.0,
    angle_deg=109.5,
    planar_tilt_deg=0.0,
):
    """
    Build a bent triatomic guess with atom order [central, terminal1, terminal2].
    """
    _ = (central_elem, terminal1_elem, terminal2_elem)  # semantic placeholders
    theta = np.radians(float(angle_deg))
    tilt = np.radians(float(planar_tilt_deg))
    x1 = float(r1) * np.sin(theta / 2.0)
    y1 = float(r1) * np.cos(theta / 2.0)
    x2 = -float(r2) * np.sin(theta / 2.0)
    y2 = float(r2) * np.cos(theta / 2.0)
    z = np.tan(tilt) * max(abs(x1), abs(x2), 1e-12)
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [x1, y1, z],
            [x2, y2, -z],
        ],
        dtype=float,
    )


def guess_linear_triatomic(
    left_elem,
    center_elem,
    right_elem,
    r_left_center=1.2,
    r_center_right=1.6,
    bend_deg=0.0,
):
    """
    Build a near-linear triatomic guess with atom order [left, center, right].
    """
    _ = (left_elem, center_elem, right_elem)  # semantic placeholders
    bend = np.radians(float(bend_deg))
    return np.array(
        [
            [-float(r_left_center), 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [float(r_center_right) * np.cos(bend), float(r_center_right) * np.sin(bend), 0.0],
        ],
        dtype=float,
    )


def guess_geometry(elems, bonds, relax_kwargs=None):
    """
    General initial-guess builder from element list + bond graph.

    Parameters
    ----------
    elems : list[str]
        Element symbols in atom order.
    bonds : list[tuple[int, int]]
        Undirected bonded atom pairs (0-based).
    relax_kwargs : dict, optional
        Keyword arguments forwarded to `_relax_geometry` (e.g. ``n_steps``,
        ``dt``, ``k_bond``, ``k_rep``, ``rep_cut``).
    """
    n = len(elems)
    coords = np.full((n, 3), np.nan, dtype=float)
    nbrs = [[] for _ in range(n)]
    for i, j in bonds:
        nbrs[i].append(j)
        nbrs[j].append(i)
    targets = {(min(i, j), max(i, j)): _bond_length_guess(elems[i], elems[j]) for i, j in bonds}
    candidate_dirs = [
        np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]),
        _unit(np.array([1.0, 1.0, 1.0])), _unit(np.array([-1.0, 1.0, 1.0])),
        _unit(np.array([1.0, -1.0, 1.0])), _unit(np.array([1.0, 1.0, -1.0])),
    ]

    visited = set()
    component_shift = 0
    for root in range(n):
        if root in visited:
            continue
        coords[root] = np.array([4.0 * component_shift, 0.0, 0.0])
        component_shift += 1
        queue = [root]
        parent = {root: None}
        visited.add(root)
        while queue:
            i = queue.pop(0)
            p = parent[i]
            back_dir = None
            if p is not None:
                back_dir = _unit(coords[i] - coords[p])
            placed_dirs = []
            for k in nbrs[i]:
                if not np.isnan(coords[k, 0]):
                    placed_dirs.append(_unit(coords[k] - coords[i]))
            for j in nbrs[i]:
                if not np.isnan(coords[j, 0]):
                    continue
                t = targets[(min(i, j), max(i, j))]
                best = None
                best_score = -1e18
                for d in candidate_dirs:
                    score = 0.0
                    if back_dir is not None:
                        score -= 2.0 * abs(np.dot(d, back_dir))
                    for pd in placed_dirs:
                        score -= abs(np.dot(d, pd))
                    trial = coords[i] + t * d
                    for k in range(n):
                        if np.isnan(coords[k, 0]) or k == i:
                            continue
                        score += 0.05 * np.linalg.norm(trial - coords[k])
                    if score > best_score:
                        best_score = score
                        best = d
                coords[j] = coords[i] + t * best
                placed_dirs.append(best)
                visited.add(j)
                parent[j] = i
                queue.append(j)

    # Fill isolated atoms safely.
    for i in range(n):
        if np.isnan(coords[i, 0]):
            coords[i] = np.array([4.0 * (i + 1), 0.0, 0.0])

    if bonds:
        rk = {} if relax_kwargs is None else dict(relax_kwargs)
        coords = _relax_geometry(coords, bonds, targets, **rk)
    return coords
