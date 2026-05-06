"""
Quantum engine: Wilson B-matrix and ORCA gradient/Hessian parser.

Primitive coordinate derivative code adapted from geomeTRIC
(Lee-Ping Wang et al., BSD-3 licence, https://github.com/leeping/geomeTRIC).
ORCA .engrad / .hess parsing written from scratch (geomeTRIC has no ORCA engine).
"""

import numpy as np
import re
from collections import defaultdict
from scipy import constants

# ── Unit conversions ──────────────────────────────────────────────────────────
BOHR_TO_ANG = constants.physical_constants["Bohr radius"][0] * 1e10   # ≈ 0.529177
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG                                       # ≈ 1.889726

# ── Element data (geomeTRIC / Cordero et al. Dalton Trans. 2008) ─────────────
_ELEMENTS = [
    "None", "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
]

_COV_RADII = [  # Å, index matches atomic number − 1
    0.31, 0.28,
    1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,
    0.00, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,
    2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.61, 1.52, 1.50,
    1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16,
    2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42,
    1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40,
    2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98,
    1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87,
    1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36,
    1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50,
]


def _cov_radius(elem):
    return _COV_RADII[_ELEMENTS.index(elem) - 1]


# ── Primitive derivatives (adapted from geomeTRIC internal.py) ────────────────

def _cross(a, b):
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ])


def _bond_deriv(xyz, m, n):
    """∂r_{mn}/∂x — shape (N, 3)."""
    d = np.zeros_like(xyz)
    u = (xyz[m] - xyz[n]) / np.linalg.norm(xyz[m] - xyz[n])
    d[m] = u
    d[n] = -u
    return d


def _angle_deriv(xyz, m, o, n):
    """∂θ_{mon}/∂x, vertex at o — shape (N, 3)."""
    d = np.zeros_like(xyz)
    u_p = xyz[m] - xyz[o]; u_n = np.linalg.norm(u_p); u = u_p / u_n
    v_p = xyz[n] - xyz[o]; v_n = np.linalg.norm(v_p); v = v_p / v_n
    V1 = np.array([ 1., -1.,  1.]) / np.sqrt(3)
    V2 = np.array([-1.,  1.,  1.]) / np.sqrt(3)
    if np.linalg.norm(u + v) < 1e-10 or np.linalg.norm(u - v) < 1e-10:
        w_p = _cross(u, V2) if (np.linalg.norm(u + V1) >= 1e-10 and
                                 np.linalg.norm(u - V2) >= 1e-10) else _cross(u, V1)
    else:
        w_p = _cross(u, v)
    w = w_p / np.linalg.norm(w_p)
    t1 = _cross(u, w) / u_n
    t2 = _cross(w, v) / v_n
    d[m] = t1; d[n] = t2; d[o] = -(t1 + t2)
    return d


def _dihedral_deriv(xyz, m, o, p, n):
    """∂φ_{mopn}/∂x — shape (N, 3)."""
    d = np.zeros_like(xyz)
    u_p = xyz[m] - xyz[o]; u_n = np.linalg.norm(u_p); u = u_p / u_n
    w_p = xyz[p] - xyz[o]; w_n = np.linalg.norm(w_p); w = w_p / w_n
    v_p = xyz[n] - xyz[p]; v_n = np.linalg.norm(v_p); v = v_p / v_n
    su2 = 1.0 - np.dot(u, w) ** 2
    sv2 = 1.0 - np.dot(v, w) ** 2
    if su2 < 1e-6 or sv2 < 1e-6:
        return d
    t1 = _cross(u, w) / (u_n * su2)
    t2 = _cross(v, w) / (v_n * sv2)
    t3 = _cross(u, w) * np.dot(u, w) / (w_n * su2)
    t4 = _cross(v, w) * np.dot(v, w) / (w_n * sv2)
    d[m] = t1; d[n] = -t2
    d[o] = -t1 + t3 - t4; d[p] = t2 - t3 + t4
    return d


# ── Connectivity ──────────────────────────────────────────────────────────────

def _detect_bonds(coords, elems, fac=1.2):
    """Bonds from covalent-radii criterion: dist < fac*(R_i + R_j)."""
    radii = np.array([_cov_radius(e) for e in elems])
    bonds = []
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            if np.linalg.norm(coords[i] - coords[j]) < fac * (radii[i] + radii[j]):
                bonds.append((i, j))
    return bonds


def _detect_angles(bonds):
    """Angle triples (i, vertex, k) from bond list."""
    nbrs = defaultdict(set)
    for i, j in bonds:
        nbrs[i].add(j); nbrs[j].add(i)
    angles = []
    for j, neighbors in nbrs.items():
        nb = sorted(neighbors)
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                angles.append((nb[a], j, nb[b]))
    return angles


def _detect_dihedrals(bonds):
    """Dihedral quadruples (i, j, k, l) from bond list."""
    nbrs = defaultdict(set)
    for a, b in bonds:
        nbrs[a].add(b); nbrs[b].add(a)
    seen, dihedrals = set(), []
    for j, k in bonds:
        for i in nbrs[j]:
            if i == k:
                continue
            for l in nbrs[k]:
                if l == j or l == i:
                    continue
                fwd, rev = (i, j, k, l), (l, k, j, i)
                key = min(fwd, rev)
                if key not in seen:
                    seen.add(key)
                    dihedrals.append(fwd)
    return dihedrals


# ── Wilson B-matrix ───────────────────────────────────────────────────────────

def wilson_B(coords, elems):
    """
    Wilson B-matrix (n_internals × 3N).

    Rows: bonds [Å], valence angles [rad], proper dihedrals [rad].
    coords : (N, 3) array in Angstroms.
    elems  : list of element symbols.
    Returns (B, labels) — B shape (n_int, 3N), labels list of str.
    """
    coords = np.asarray(coords, dtype=float)
    bonds = _detect_bonds(coords, elems)
    angles = _detect_angles(bonds)
    dihedrals = _detect_dihedrals(bonds)

    rows, labels = [], []
    for i, j in bonds:
        rows.append(_bond_deriv(coords, i, j).ravel())
        labels.append(f"bond {i+1}-{j+1}")
    for i, j, k in angles:
        rows.append(_angle_deriv(coords, i, j, k).ravel())
        labels.append(f"angle {i+1}-{j+1}-{k+1}")
    for i, j, k, l in dihedrals:
        rows.append(_dihedral_deriv(coords, i, j, k, l).ravel())
        labels.append(f"dihedral {i+1}-{j+1}-{k+1}-{l+1}")

    return np.array(rows), labels


# ── ORCA file parsers ─────────────────────────────────────────────────────────

def parse_engrad(path):
    """
    Parse an ORCA .engrad file.

    Returns
    -------
    energy   : float   Total energy in Hartree.
    gradient : ndarray Shape (3N,), in Hartree/Bohr.
    """
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    it = iter(lines)
    n_atoms = int(next(it))
    energy = float(next(it))
    gradient = np.array([float(next(it)) for _ in range(3 * n_atoms)])
    return energy, gradient


def parse_hess(path):
    """
    Parse an ORCA .hess file.

    Returns
    -------
    hessian : ndarray   Shape (3N, 3N), in Hartree/Bohr².
    """
    with open(path) as f:
        content = f.read()

    block_start = content.index("$hessian")
    lines = [l for l in content[block_start:].split("\n")[1:] if l.strip()]

    dim = int(lines[0])
    H = np.zeros((dim, dim))
    pos = 1

    while pos < len(lines):
        # Column-index header: all tokens are non-negative integers
        header = lines[pos].split()
        try:
            col_indices = [int(c) for c in header]
        except ValueError:
            break
        pos += 1

        for _ in range(dim):
            parts = lines[pos].split()
            row = int(parts[0])
            for k, v in enumerate(parts[1:]):
                H[row, col_indices[k]] = float(v)
            pos += 1

        if col_indices[-1] == dim - 1:
            break

    return H


def parse_orca_rovib(path):
    """Parse an ORCA output for vibration-rotation alpha constants and metadata.

    Returns
    -------
    ParsedRovibResult
        ``alpha_abc`` is a length-3 array (A, B, C) in MHz with NaN for
        missing components.  ``parse_status`` is one of ``ok``, ``partial``,
        or ``parse_failed``.  Warnings include detected resonances, imaginary
        modes, low-frequency modes (< 50 cm^-1), and a marker if the run did
        not appear to invoke VPT2.
    """
    from backend.correction_models import ParsedRovibResult  # local import to avoid cycles

    alpha = np.full(3, np.nan, dtype=float)
    frequencies: list[float] = []
    warnings: list[str] = []
    label_to_idx = {"A": 0, "B": 1, "C": 2}

    pat_labeled = re.compile(
        r"(?i)\balpha\(?\s*([ABC])\s*\)?\s*[:=]?\s*([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)"
    )
    pat_row = re.compile(
        r"^\s*([ABC])\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)"
    )
    pat_triplet = re.compile(
        r"(?i)\balpha[^0-9A-Za-z+-]*"
        r"([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s+"
        r"([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s+"
        r"([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)"
    )
    # Generic frequency table line: "<idx>: <freq cm**-1> ..."
    pat_freq = re.compile(
        r"^\s*\d+\s*:\s*([-+]?\d+(?:\.\d+)?)\s*(?:cm\*\*-1|cm-1|cm\^-1)?",
        re.IGNORECASE,
    )

    in_alpha_table = False
    saw_vpt2_marker = False
    in_freq_block = False

    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                low = line.lower()
                if "vpt2" in low or "second order" in low or "second-order" in low:
                    saw_vpt2_marker = True
                if "vibrational-rotational" in low or "vibration-rotation" in low or "alpha constants" in low:
                    in_alpha_table = True
                if in_alpha_table and not line.strip():
                    in_alpha_table = False

                # Detect VPT2 warnings / resonance markers.
                if "warning" in low and ("vpt2" in low or "resonance" in low or "fermi" in low):
                    warnings.append(line.strip())
                elif "fermi" in low and "resonance" in low:
                    warnings.append(line.strip())
                elif "darling-dennison" in low or "coriolis resonance" in low:
                    warnings.append(line.strip())

                # Detect frequency tables (harmonic or fundamental).
                if "vibrational frequencies" in low or "fundamental frequencies" in low:
                    in_freq_block = True
                    continue
                if in_freq_block:
                    if not line.strip():
                        in_freq_block = False
                    else:
                        mfreq = pat_freq.match(line)
                        if mfreq:
                            try:
                                frequencies.append(float(mfreq.group(1)))
                            except ValueError:
                                pass

                m = pat_labeled.search(line)
                if m:
                    idx = label_to_idx[m.group(1).upper()]
                    alpha[idx] = float(m.group(2))
                    continue
                m3 = pat_triplet.search(line)
                if m3 and ("scaling param" not in low):
                    alpha[0] = float(m3.group(1))
                    alpha[1] = float(m3.group(2))
                    alpha[2] = float(m3.group(3))
                    continue
                m2 = pat_row.match(line)
                if m2 and "alpha" in low:
                    idx = label_to_idx[m2.group(1).upper()]
                    alpha[idx] = float(m2.group(2))
                    continue
                if in_alpha_table:
                    m4 = pat_row.match(line)
                    if m4:
                        key = m4.group(1).upper()
                        if key in label_to_idx:
                            idx = label_to_idx[key]
                            if not np.isfinite(alpha[idx]):
                                alpha[idx] = float(m4.group(2))
    except OSError as e:
        warnings.append(f"could not read file: {e}")

    freq_arr = np.asarray(frequencies, dtype=float) if frequencies else None

    if freq_arr is not None and freq_arr.size:
        imag = freq_arr[freq_arr < 0.0]
        if imag.size:
            warnings.append(
                f"detected {imag.size} imaginary frequency mode(s) (min "
                f"{float(np.min(imag)):.2f} cm^-1)"
            )
        low_modes = freq_arr[(freq_arr > 0.0) & (freq_arr < 50.0)]
        if low_modes.size:
            warnings.append(
                f"detected {low_modes.size} low-frequency mode(s) below 50 cm^-1"
            )

    if not saw_vpt2_marker:
        warnings.append("no VPT2/second-order marker found in output")

    finite_count = int(np.sum(np.isfinite(alpha)))
    if finite_count == 0:
        parse_status = "parse_failed"
    elif finite_count < 3:
        parse_status = "partial"
    else:
        parse_status = "ok"

    return ParsedRovibResult(
        alpha_abc=alpha,
        frequencies=freq_arr,
        warnings=warnings,
        source_files=[str(path)],
        parse_status=parse_status,
        units="MHz",
    )


def parse_orca_rovib_alpha(path):
    """Backward-compatible wrapper returning just the alpha(A,B,C) vector."""
    return parse_orca_rovib(path).alpha_abc


# ── QuantumEngine ─────────────────────────────────────────────────────────────

class QuantumEngine:
    """
    Gradient, Hessian, and Wilson B-matrix from ORCA output files.

    Parameters
    ----------
    engrad_path : str        Path to ORCA .engrad file.
    hess_path   : str        Path to ORCA .hess file.
    elems       : list[str]  Element symbols in atom order.
    """

    def __init__(self, engrad_path, hess_path, elems):
        self.elems = list(elems)
        self.energy, self._gradient_bohr = parse_engrad(engrad_path)
        self._hessian_bohr = parse_hess(hess_path)

    @property
    def gradient(self):
        """Gradient in Hartree/Å, shape (3N,)."""
        return self._gradient_bohr * ANG_TO_BOHR

    @property
    def hessian(self):
        """Hessian in Hartree/Å², shape (3N, 3N)."""
        return self._hessian_bohr * ANG_TO_BOHR ** 2

    def wilson_B(self, coords):
        """
        Wilson B-matrix at the given geometry.

        coords : (N, 3) array in Angstroms.
        Returns (B [n_int × 3N], labels [list of str]).
        """
        return wilson_B(np.asarray(coords, dtype=float), self.elems)
