"""
Initial geometry guess utilities.

These helpers provide deterministic, template-based starting geometries so
driver scripts do not need hardcoded coordinate arrays.

Use :func:`guess_geometry_molecular_input` for **PubChem 3D** (name, CID, SMILES)
or :func:`guess_geometry` when you already know ``elems`` + ``bonds``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

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


def guess_staggered_methanol(
    r_co=1.428,
    r_ch=1.094,
    r_oh=0.963,
    angle_coh_deg=108.7,
    center=True,
):
    """
    Staggered CH₃OH geometry (C_s), atom order **C, O, H, H, H, Ho** with Ho the
    hydroxyl hydrogen bonded to O.

    The methyl takes three vertices of a regular tetrahedron around C (fourth
    vertex is the C–O bond); the OH bond is placed in a plane defined by the C→O
    axis and a perpendicular so **∠COH ≈ angle_coh_deg** (gas-phase ~108.7°).

    Bond lengths default near equilibrium gas-phase / microwave-derived r₀ values
    (order Å): CO ~1.43, CH ~1.09, OH ~0.96.

    This avoids ``guess_geometry(...)``, whose greedy placement often yields
    distorted methyl angles unsuitable as an optimization seed.

    Parameters
    ----------
    center : bool
        If True (default), subtract the centroid so the cluster sits near the origin.
    """
    # Four tetrahedral directions from C (alternating vertices); assign O to one.
    v = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    )
    v /= np.linalg.norm(v[0])

    coords = np.zeros((6, 3), dtype=float)
    coords[1] = float(r_co) * v[0]
    for i in range(3):
        coords[2 + i] = float(r_ch) * v[1 + i]

    o = coords[1]
    u_oc = _unit(coords[0] - o)
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(u_oc, ref)) > 0.85:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    e_perp = _unit(np.cross(u_oc, ref))
    g = np.radians(float(angle_coh_deg))
    u_oh = np.cos(g) * u_oc + np.sin(g) * e_perp
    u_oh = _unit(u_oh)
    coords[5] = o + float(r_oh) * u_oh

    if center:
        coords -= coords.mean(axis=0, keepdims=True)
    return coords


def guess_planar_formaldehyde(r_co=1.205, r_ch=1.111, angle_hch_deg=116.13):
    """
    Planar H₂CO (C₂ᵥ) guess; atom order **O, C, H, H** matching experimental fits.

    Orientation: C at origin, O on +z; hydrogens symmetric in the yz plane (∠HCH ≈ angle_hch_deg).
    Defaults align with NIST CCCBDB experimental geometry for formaldehyde (Cas No. 50-00-0).
    """
    half = 0.5 * np.radians(float(angle_hch_deg))
    o = np.array([0.0, 0.0, float(r_co)], dtype=float)
    c = np.array([0.0, 0.0, 0.0], dtype=float)
    y = float(r_ch) * np.sin(half)
    z_h = -float(r_ch) * np.cos(half)
    h1 = np.array([0.0, y, z_h], dtype=float)
    h2 = np.array([0.0, -y, z_h], dtype=float)
    return np.array([o, c, h1, h2], dtype=float)


def guess_planar_benzene(r_cc=1.3971, r_ch=1.0804):
    """
    Planar D6h benzene in the xy plane (standard orientation: σh = xy).

    Carbons sit on a regular hexagon at circumradius R = r_cc (Å); hydrogens lie
    outward along the same radial directions at distance (r_cc + r_ch) from the
    origin (COM at origin for equal masses).

    Atom order: C0…C5 clock-wise, then H0…H5 with Hi bonded to Ci.

    Default bond lengths match equilibrium values inferred by In Heo et al., RSC Adv.
    2022, 12, 21406–21416 (doi:10.1039/D2RA03431J) from rotational Raman data.
    """
    coords = np.zeros((12, 3))
    for i in range(6):
        th = i * (np.pi / 3.0)
        ux, uy = np.cos(th), np.sin(th)
        rc = r_cc
        coords[i] = np.array([rc * ux, rc * uy, 0.0])
        coords[6 + i] = np.array([(rc + r_ch) * ux, (rc + r_ch) * uy, 0.0])
    return coords


def guess_geometry_molecular_input(
    identifier: Optional[str] = None,
    *,
    elems: Optional[Sequence[str]] = None,
    bonds: Optional[Sequence[Tuple[int, int]]] = None,
    pubchem_timeout: float = 60.0,
    center: bool = True,
    pubchem_prefer: str = "auto",
) -> tuple[np.ndarray, list[str]]:
    """
    Unified geometry guess for arbitrary input.

    **1. Explicit connectivity** — supply ``elems`` and ``bonds`` (same contract as
    :func:`guess_geometry`). Good offline when you have a bond graph.

    **2. PubChem 3-D** — supply ``identifier`` only: compound **name**, numeric **CID**, or
    **SMILES**. Requires internet; uses MMFF94-relaxed PubChem conformers
    (``record_type=3d``). For ``prefer="auto"``: all-digit strings resolve as CID;
    otherwise SMILES is tried first, then name.

    If both ``elems``/``bonds`` and ``identifier`` are given, the explicit graph wins.

    Parameters
    ----------
    center
        If True (default), subtract the centroid so the molecule sits near the origin.
    pubchem_prefer
        ``auto`` | ``cid`` | ``smiles`` | ``name`` — passed to PubChem resolution.

    Returns
    -------
    coords : ndarray, shape (N, 3)
        Cartesian coordinates in Å.
    elems : list of str
        Element symbols in the same order as coordinate rows.
    """
    if elems is not None and bonds is not None:
        e_list = [str(x).strip() for x in elems]
        b_list = [(int(i), int(j)) for i, j in bonds]
        coords = guess_geometry(e_list, b_list)
        if center:
            coords = coords - coords.mean(axis=0, keepdims=True)
        return coords, e_list

    if identifier is None or not str(identifier).strip():
        raise ValueError("Provide (elems, bonds) or a non-empty identifier for PubChem.")

    from backend.pubchem_geometry import coords_elems_from_pubchem

    coords, e_list = coords_elems_from_pubchem(
        str(identifier).strip(),
        timeout=pubchem_timeout,
        prefer=pubchem_prefer,
    )
    if center:
        coords = coords.copy()
        coords -= coords.mean(axis=0, keepdims=True)
    return coords, e_list
