"""
Point-group symmetry constraints for Cartesian geometry optimisation.

Usage
-----
    from symmetry import PointGroupSymmetry

    sym = PointGroupSymmetry("C2v", ["O", "H", "H"], initial_coords)
    print(sym.summary())

    # inside the optimisation step:
    dx_sym = sym.project_step(dx)        # remove symmetry-breaking part of step
    coords  = sym.symmetrize(coords)     # snap coordinates to symmetric subspace

Standard orientation expected (auto_orient=True will try to fix it automatically):
  Cnv        : principal axis (Cn) along +z; one off-axis atom in xz half-plane
  Cnh / Dnh  : same + horizontal mirror = xy plane
  Cs         : mirror plane = xy  (molecule lies in or near xy)
  Cinf_v     : molecular axis along +z
  Dinf_h     : molecular axis along +z, inversion centre at origin
"""

import numpy as np
from typing import List, Optional, Tuple


# ── 3×3 symmetry-operation generators ────────────────────────────────────────

def _Cn_z(n: int, k: int = 1) -> np.ndarray:
    """Proper rotation by 2πk/n about z."""
    t = 2 * np.pi * k / n
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _sigma_v(phi: float) -> np.ndarray:
    """Vertical mirror plane containing z; normal to the plane is at phi+π/2 from x."""
    c2, s2 = np.cos(2 * phi), np.sin(2 * phi)
    return np.array([[c2, s2, 0.], [s2, -c2, 0.], [0., 0., 1.]])


def _C2_perp(phi: float) -> np.ndarray:
    """C2 rotation about the in-plane axis at angle phi from x."""
    n = np.array([np.cos(phi), np.sin(phi), 0.])
    return 2 * np.outer(n, n) - np.eye(3)


def _Sn_z(n: int, k: int = 1) -> np.ndarray:
    """Improper rotation: Cn(z)^k followed by σh."""
    return _Cn_z(n, k) @ _SH


# ── module-level operation constants ─────────────────────────────────────────

_E   = np.eye(3)
_SH  = np.diag([1., 1., -1.])    # σh  (reflection in xy)
_SXZ = np.diag([1., -1., 1.])    # σv  (reflection in xz)
_SYZ = np.diag([-1., 1., 1.])    # σv' (reflection in yz)
_INV = np.diag([-1., -1., -1.])  # inversion i
_C2Z = _Cn_z(2)
_C2X = np.diag([1., -1., -1.])
_C2Y = np.diag([-1., 1., -1.])
_PI  = np.pi

# ── operation tables (standard orientation: principal axis along z) ───────────

_GROUP_OPS: dict = {
    # ── non-axial ──────────────────────────────────────────────────────────────
    "C1":  [_E],
    "Ci":  [_E, _INV],
    "Cs":  [_E, _SH],
    # ── pure rotation (Cn) ────────────────────────────────────────────────────
    "C2":  [_E, _C2Z],
    "C3":  [_E, _Cn_z(3, 1), _Cn_z(3, 2)],
    "C4":  [_E, _Cn_z(4, 1), _C2Z, _Cn_z(4, 3)],
    "C5":  [_E] + [_Cn_z(5, k) for k in range(1, 5)],
    "C6":  [_E] + [_Cn_z(6, k) for k in range(1, 6)],
    # ── pyramidal / conical (Cnv) ─────────────────────────────────────────────
    "C2v": [_E, _C2Z, _SXZ, _SYZ],
    "C3v": [_E, _Cn_z(3, 1), _Cn_z(3, 2),
             _sigma_v(0),           _sigma_v(2*_PI/3),  _sigma_v(4*_PI/3)],
    "C4v": [_E, _Cn_z(4, 1), _C2Z, _Cn_z(4, 3),
             _SXZ, _SYZ,
             _sigma_v(_PI/4),       _sigma_v(3*_PI/4)],
    "C5v": ([_E] + [_Cn_z(5, k) for k in range(1, 5)] +
             [_sigma_v(k * _PI/5)   for k in range(5)]),
    "C6v": ([_E] + [_Cn_z(6, k) for k in range(1, 6)] +
             [_sigma_v(k * _PI/6)   for k in range(6)]),
    # ── rotoreflection (Cnh) ──────────────────────────────────────────────────
    "C2h": [_E, _C2Z, _INV, _SH],
    "C3h": [_E, _Cn_z(3, 1), _Cn_z(3, 2), _SH,
             _Sn_z(3, 1),    _Sn_z(3, 2)],
    # ── dihedral (Dnh) ────────────────────────────────────────────────────────
    "D2h": [_E, _C2Z, _C2X, _C2Y, _INV, _SH, _SXZ, _SYZ],
    "D3h": [_E, _Cn_z(3, 1), _Cn_z(3, 2),
             _C2_perp(0),         _C2_perp(2*_PI/3),  _C2_perp(4*_PI/3),
             _SH,
             _Sn_z(3, 1),         _Sn_z(3, 2),
             _sigma_v(0),          _sigma_v(2*_PI/3),  _sigma_v(4*_PI/3)],
    "D4h": [_E, _Cn_z(4, 1), _C2Z, _Cn_z(4, 3),
             _C2_perp(0),     _C2_perp(_PI/2),
             _C2_perp(_PI/4), _C2_perp(3*_PI/4),
             _INV, _SH,
             _Sn_z(4, 1),    _Sn_z(4, 3),
             _SXZ, _SYZ,
             _sigma_v(_PI/4), _sigma_v(3*_PI/4)],
    "D6h": ([_E] + [_Cn_z(6, k) for k in range(1, 6)] +
             [_C2_perp(k * _PI/3)        for k in range(3)] +
             [_C2_perp(_PI/6 + k*_PI/3)  for k in range(3)] +
             [_INV, _SH,
              _Sn_z(6, 1), _Sn_z(6, 5),
              _Sn_z(3, 1), _Sn_z(3, 2)] +
             [_sigma_v(k * _PI/3)        for k in range(3)] +
             [_sigma_v(_PI/6 + k*_PI/3)  for k in range(3)]),
    # ── linear (handled separately) ───────────────────────────────────────────
    "Cinf_v": None,
    "Dinf_h": None,
}

# Groups whose principal axis is *perpendicular* to the molecular plane:
# PCA assigns the axis of minimum spread to z.
_PLANAR_PG = {"Cs", "C2h", "C3h", "D2h", "D3h", "D4h", "D6h"}

_ALIASES = {
    "C2V": "C2v", "C3V": "C3v", "C4V": "C4v", "C5V": "C5v", "C6V": "C6v",
    "C2H": "C2h", "C3H": "C3h",
    "D2H": "D2h", "D3H": "D3h", "D4H": "D4h", "D6H": "D6h",
    "CINF_V":  "Cinf_v", "CINFV":  "Cinf_v", "C*V":  "Cinf_v",
    "DINF_H":  "Dinf_h", "DINFH":  "Dinf_h", "D*H":  "Dinf_h",
}


def _normalize_pg(pg: str) -> str:
    return _ALIASES.get(pg.strip().upper(), pg.strip())


# ── orientation ──────────────────────────────────────────────────────────────

def _topological_axis(elements: List[str], c: np.ndarray) -> Optional[np.ndarray]:
    """
    Estimate the principal symmetry axis from element counts.

    Strategy: the minority-count element(s) are usually the apex/unique atoms
    (e.g. O in H₂O, N in NH₃, S in SO₂).  The axis points from the centroid
    of the majority atoms toward the centroid of the minority atoms.

    Returns a unit vector, or None if the result is degenerate or ambiguous.
    """
    from collections import Counter
    counts = Counter(elements)
    if len(counts) < 2:
        return None
    min_cnt = min(counts.values())
    max_cnt = max(counts.values())
    if min_cnt == max_cnt:
        return None   # all elements appear equally often — ambiguous

    mask = np.array([counts[e] == min_cnt for e in elements])
    c_apex = c[mask].mean(axis=0)
    c_base = c[~mask].mean(axis=0)
    axis = c_apex - c_base
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return None   # apex is at the centroid of the base (e.g. B in BF₃)
    return axis / norm


def _orient_to_standard(
    elements: List[str],
    coords: np.ndarray,
    pg: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Centre the molecule and rotate so the principal symmetry axis → z.

    Returns
    -------
    coords_std : (N, 3)   coordinates in the standard frame
    R          : (3, 3)   rotation  (coords_std = (coords − centroid) @ R.T)

    Axis-selection strategy
    -----------------------
    - Planar groups (Cs, C2h, C3h, Dnh): smallest PCA eigenvalue → z
      (normal to the molecular plane).
    - All others: try topological axis first (minority-element centroid →
      majority-element centroid; works for C2v, C3v, …), then fall back to
      the largest PCA eigenvalue.
    """
    centroid = coords.mean(axis=0)
    c = coords - centroid
    _, evecs = np.linalg.eigh(c.T @ c)   # eigenvalues ascending

    if pg in _PLANAR_PG:
        # Normal to molecular plane = axis of minimum spread
        z_vec = evecs[:, 0]
    else:
        # Topology-based: works for C2v (O in H₂O), C3v (N in NH₃), etc.
        z_vec = _topological_axis(elements, c)
        if z_vec is None:
            # Fallback: largest PCA eigenvalue (works for linear-like molecules)
            z_vec = evecs[:, 2]

    # Build right-handed orthonormal frame with z_vec → z
    ref = np.array([1., 0., 0.]) if abs(z_vec[0]) < 0.9 else np.array([0., 1., 0.])
    x_vec = ref - np.dot(ref, z_vec) * z_vec
    x_vec /= np.linalg.norm(x_vec)
    y_vec = np.cross(z_vec, x_vec)
    R = np.vstack([x_vec, y_vec, z_vec])
    if np.linalg.det(R) < 0:
        R[0] = -R[0]

    c_rot = c @ R.T

    # Rotate about z so the first significantly off-axis atom lies in xz (y → 0)
    rho = np.hypot(c_rot[:, 0], c_rot[:, 1])
    if rho.max() > 1e-6:
        i_ref = int(np.argmax(rho))
        phi = np.arctan2(c_rot[i_ref, 1], c_rot[i_ref, 0])
        cp, sp = np.cos(-phi), np.sin(-phi)
        Rz = np.array([[cp, -sp, 0.], [sp, cp, 0.], [0., 0., 1.]])
        c_rot = c_rot @ Rz.T
        R = Rz @ R

    return c_rot, R


# ── permutation detection ─────────────────────────────────────────────────────

def _find_permutations(
    ops: List[np.ndarray],
    coords: np.ndarray,
    elements: List[str],
    tol: float,
) -> List[List[int]]:
    """
    For each 3×3 operation, find which atom index maps to which.
    Uses greedy nearest-neighbour matching restricted to same-element pairs.
    """
    n = len(elements)
    perms: List[List[int]] = []
    for R in ops:
        transformed = coords @ R.T
        perm = [-1] * n
        used: set = set()
        candidates = []
        for i in range(n):
            for j in range(n):
                if elements[i] == elements[j]:
                    d = float(np.linalg.norm(transformed[i] - coords[j]))
                    candidates.append((d, i, j))
        candidates.sort()
        for d, i, j in candidates:
            if perm[i] == -1 and j not in used and d < tol:
                perm[i] = j
                used.add(j)
        if -1 in perm:
            bad = [i for i in range(n) if perm[i] == -1]
            raise ValueError(
                f"Atom permutation detection failed under operation\nR =\n{np.round(R, 4)}\n"
                f"Unmatched atom indices: {bad}  (tol = {tol:.3f} Å).\n"
                f"Tip: increase tol, or verify the geometry is in standard orientation "
                f"(principal axis along z, one off-axis atom in xz plane)."
            )
        perms.append(perm)
    return perms


# ── 3N × 3N superoperator ────────────────────────────────────────────────────

def _superop(R: np.ndarray, perm: List[int]) -> np.ndarray:
    """Build the 3N×3N matrix for symmetry operation (R, perm)."""
    n = len(perm)
    S = np.zeros((3 * n, 3 * n))
    for i, j in enumerate(perm):
        # atom i is rotated by R and placed at atom j's position
        S[3*j:3*j+3, 3*i:3*i+3] = R
    return S


# ── equivalent-atom groups ────────────────────────────────────────────────────

def _equiv_groups(n: int, perms: List[List[int]]) -> List[List[int]]:
    """Union-find: collect atoms connected by any symmetry permutation."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for perm in perms:
        for i, j in enumerate(perm):
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def infer_point_group(elements: List[str], coords, linear_tol_deg: float = 2.0) -> str:
    """
    Lightweight point-group inference for common molecular topologies.

    This intentionally favors robust defaults over exhaustive classification.
    Returns a supported label usable by PointGroupSymmetry.
    """
    elems = list(elements)
    xyz = np.asarray(coords, dtype=float)
    n = len(elems)
    if n < 2:
        return "C1"

    if n == 2:
        return "Dinf_h" if elems[0] == elems[1] else "Cinf_v"

    if n == 3:
        # Treat index 1 as central for triatomic driver conventions.
        v1 = xyz[0] - xyz[1]
        v2 = xyz[2] - xyz[1]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 > 1e-12 and n2 > 1e-12:
            ang = float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))
            if abs(180.0 - ang) <= float(linear_tol_deg):
                return "Dinf_h" if elems[0] == elems[2] else "Cinf_v"
        return "C2v" if elems[0] == elems[2] else "Cs"

    # Simple tetra-like pyramidal detector (e.g., NH3)
    unique = {}
    for e in elems:
        unique[e] = unique.get(e, 0) + 1
    if sorted(unique.values()) == [1, 3]:
        return "C3v"

    return "C1"


def create_symmetry(
    point_group: Optional[str],
    elements: List[str],
    coords,
    tol: float = 0.5,
    auto_orient: bool = True,
) -> "PointGroupSymmetry":
    """
    Create a PointGroupSymmetry instance from explicit or inferred group.

    point_group:
      - None, "", or "auto" => infer a supported point-group label.
      - otherwise            => use the provided group label directly.
    """
    pg = point_group
    if pg is None or str(pg).strip() == "" or str(pg).strip().lower() == "auto":
        pg = infer_point_group(elements, coords)
    return PointGroupSymmetry(str(pg), elements, coords, tol=tol, auto_orient=auto_orient)


# ── main class ────────────────────────────────────────────────────────────────

class PointGroupSymmetry:
    """
    Symmetry projection operator for a molecular point group.

    Builds a 3N×3N projection matrix P such that P @ step removes the
    symmetry-breaking component of any Cartesian displacement.  Applying P
    at every optimisation step constrains the geometry to the symmetric
    subspace, reducing effective degrees of freedom.

    Parameters
    ----------
    point_group : str
        Point group label.  Case-insensitive; common aliases accepted.
        Supported: C1, Ci, Cs, C2–C6, C2v–C6v, C2h, C3h,
                   D2h, D3h, D4h, D6h, Cinf_v, Dinf_h.
    elements : list[str]
        Element symbols, length N.
    coords : array-like (N, 3)
        Approximate Cartesian coordinates (Å) used once to detect atom
        permutations.  Need not be exactly symmetric.
    tol : float, optional
        Atom-matching tolerance (Å) for permutation detection.  Default 0.5 Å.
    auto_orient : bool, optional
        If True (default) the molecule is auto-rotated to standard orientation
        (principal axis → z) before permutation detection.  Set False if you
        have pre-oriented the coordinates manually.

    Notes
    -----
    Standard orientation (auto_orient=True attempts to impose this):
      - Cnv, Cn   : main axis along +z; one off-axis atom in xz half-plane
      - Cnh, Dnh  : same + σh = xy plane  (planar groups: small-spread axis → z)
      - Cs        : mirror plane = xy
      - Cinf_v    : molecular axis along +z
      - Dinf_h    : same + inversion centre at origin

    Auto-orient uses PCA and works reliably for the most common cases
    (C2v, C3v, D3h, Cinf_v, Dinf_h, Cs).  For C2h, D2h with non-obvious
    orientations, pass pre-oriented coordinates and set auto_orient=False.
    """

    def __init__(
        self,
        point_group: str,
        elements: List[str],
        coords,
        tol: float = 0.5,
        auto_orient: bool = True,
    ):
        self.point_group: str = _normalize_pg(point_group)
        self.elements: List[str] = list(elements)
        self.n_atoms: int = len(elements)
        self.tol: float = float(tol)
        self._projection: Optional[np.ndarray] = None
        self._equiv: Optional[List[List[int]]] = None
        self._setup(np.asarray(coords, dtype=float), auto_orient=auto_orient)

    # ── internal setup ────────────────────────────────────────────────────────

    def _setup(self, coords: np.ndarray, auto_orient: bool) -> None:
        pg = self.point_group
        if pg == "Cinf_v":
            self._setup_linear(coords, apolar=False)
            return
        if pg == "Dinf_h":
            self._setup_linear(coords, apolar=True)
            return

        raw_ops = _GROUP_OPS.get(pg)
        if raw_ops is None:
            supported = sorted(k for k, v in _GROUP_OPS.items() if v is not None)
            raise ValueError(
                f"Unsupported point group: {pg!r}.  "
                f"Supported: {supported + ['Cinf_v', 'Dinf_h']}"
            )

        if auto_orient:
            coords_std, R_orient = _orient_to_standard(self.elements, coords, pg)
        else:
            coords_std = coords - coords.mean(axis=0)
            R_orient = np.eye(3)

        perms = _find_permutations(raw_ops, coords_std, self.elements, self.tol)

        # Projection in standard frame, then rotate to lab frame
        P_std = sum(_superop(R_op, perm) for R_op, perm in zip(raw_ops, perms))
        P_std /= len(raw_ops)

        # P_lab = T^T P_std T  where T = kron(I_N, R_orient)  (block-diagonal rotation)
        T = np.kron(np.eye(self.n_atoms), R_orient)
        self._projection = T.T @ P_std @ T
        self._equiv = _equiv_groups(self.n_atoms, perms)

    def _setup_linear(self, coords: np.ndarray, apolar: bool) -> None:
        """C∞v or D∞h: project all atoms onto the molecular axis (z after orient)."""
        n, N = self.n_atoms, 3 * self.n_atoms

        centroid = coords.mean(axis=0)
        c = coords - centroid
        _, evecs = np.linalg.eigh(c.T @ c)
        z_vec = evecs[:, 2]   # largest spread = molecular axis
        ref = np.array([1., 0., 0.]) if abs(z_vec[0]) < 0.9 else np.array([0., 1., 0.])
        x_vec = ref - np.dot(ref, z_vec) * z_vec
        x_vec /= np.linalg.norm(x_vec)
        y_vec = np.cross(z_vec, x_vec)
        R = np.vstack([x_vec, y_vec, z_vec])
        if np.linalg.det(R) < 0:
            R[0] = -R[0]
        coords_std = c @ R.T

        P_std = np.zeros((N, N))

        if apolar:
            inv_perm = self._detect_inversion_pairs(coords_std)
            if inv_perm is not None:
                for i in range(n):
                    j = inv_perm[i]
                    if i != j:
                        # enforce z_i = −z_j  via antisymmetric average
                        P_std[3*i+2, 3*i+2] =  0.5
                        P_std[3*i+2, 3*j+2] = -0.5
                    # central atom (i==j): z row stays zero → locked at 0
                self._equiv = self._build_dinf_equiv(n, inv_perm)
            else:
                # fallback: treat as C∞v
                for i in range(n):
                    P_std[3*i+2, 3*i+2] = 1.0
                self._equiv = [[i] for i in range(n)]
        else:
            # C∞v: each atom keeps only its z component; x and y are zeroed
            for i in range(n):
                P_std[3*i+2, 3*i+2] = 1.0
            self._equiv = [[i] for i in range(n)]

        T = np.kron(np.eye(n), R)
        self._projection = T.T @ P_std @ T

    def _detect_inversion_pairs(self, coords_std: np.ndarray) -> Optional[List[int]]:
        """Return inversion permutation for D∞h, or None if pairs cannot be found."""
        n = self.n_atoms
        inv_perm = [-1] * n
        for i in range(n):
            for j in range(n):
                if self.elements[i] == self.elements[j]:
                    if np.linalg.norm(-coords_std[i] - coords_std[j]) < self.tol:
                        inv_perm[i] = j
                        break
        return None if -1 in inv_perm else inv_perm

    @staticmethod
    def _build_dinf_equiv(n: int, inv_perm: List[int]) -> List[List[int]]:
        seen: set = set()
        groups: List[List[int]] = []
        for i in range(n):
            if i in seen:
                continue
            j = inv_perm[i]
            if i == j:
                groups.append([i])
            else:
                groups.append(sorted([i, j]))
                seen.update([i, j])
        return groups

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def projection(self) -> np.ndarray:
        """3N × 3N symmetry projection matrix (lab frame)."""
        return self._projection

    @property
    def equiv_groups(self) -> List[List[int]]:
        """Groups of symmetry-equivalent atom indices."""
        return self._equiv

    def project_step(self, step) -> np.ndarray:
        """
        Remove the symmetry-breaking component of a (3N,) Cartesian step.

        Returns the projected step as a (3N,) array.
        """
        return self._projection @ np.asarray(step, dtype=float).ravel()

    def symmetrize(self, coords) -> np.ndarray:
        """
        Return the nearest point in the symmetric subspace.

        Parameters
        ----------
        coords : (N, 3) or (3N,) array

        Returns
        -------
        Array of the same shape as input.
        """
        arr = np.asarray(coords, dtype=float)
        return (self._projection @ arr.ravel()).reshape(arr.shape)

    def summary(self) -> str:
        """Short human-readable description of the symmetry constraints."""
        lines = [f"Point group: {self.point_group}"]
        for g in (self._equiv or []):
            if len(g) > 1:
                lbl = ", ".join(f"{self.elements[i]}({i+1})" for i in g)
                lines.append(f"  Equivalent atoms: {lbl}")
        n_dof_free = int(np.round(np.trace(self._projection))) if self._projection is not None else 3 * self.n_atoms
        lines.append(f"  Symmetric degrees of freedom: {n_dof_free} / {3 * self.n_atoms}")
        return "\n".join(lines)

    @staticmethod
    def supported_groups() -> List[str]:
        """Return all supported point group labels."""
        return sorted(k for k, v in _GROUP_OPS.items() if v is not None) + ["Cinf_v", "Dinf_h"]
