"""
Single-point ORCA **geometry optimization** using a lightweight method/basis.

Intended only for **cheap initial guesses** before expensive hybrid inversion. Not
spectroscopic ``rₑ``: a fast surrogate minimum on the HF/DFTB-composite PES you choose.

Requires a working ORCA install (same discovery rules as ``MolecularOptimizer``).
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from backend.quantize import _find_orca

# Default balances speed vs reasonable organics-first behavior; override with QUANTIZE_CHEAP_OPT_KEYWORD.
_DEFAULT_OPT_KEYWORD = "HF-3c Opt TightSCF"


def _opt_keyword_line() -> str:
    raw = os.environ.get("QUANTIZE_CHEAP_OPT_KEYWORD", "").strip()
    return raw if raw else _DEFAULT_OPT_KEYWORD


def _write_opt_input(
    path: Path,
    *,
    coords: np.ndarray,
    elems: Sequence[str],
    charge: int,
    multiplicity: int,
    bang_line: str,
) -> None:
    coords = np.asarray(coords, dtype=float)
    elems = list(elems)
    if coords.shape != (len(elems), 3):
        raise ValueError(f"coords shape {coords.shape} incompatible with len(elems)={len(elems)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"! {bang_line}",
        "%pal",
        "  nprocs 1",
        "end",
        f"* xyz {int(charge)} {int(multiplicity)}",
    ]
    for elem, row in zip(elems, coords):
        lines.append(f"  {str(elem):2s}  {row[0]:16.10f}  {row[1]:16.10f}  {row[2]:16.10f}")
    lines.append("*")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _exec_orca(orca_exe: str, workdir: Path, inp_rel: str) -> None:
    env = os.environ.copy()
    orca_dir = os.path.dirname(os.path.abspath(orca_exe))
    if orca_dir not in env.get("PATH", ""):
        env["PATH"] = orca_dir + os.pathsep + env.get("PATH", "")
    workdir = workdir.resolve()
    result = subprocess.run(
        [orca_exe, inp_rel],
        capture_output=True,
        text=True,
        cwd=str(workdir),
        env=env,
    )
    out_path = workdir / inp_rel.replace(".inp", ".out")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.stdout or "", encoding="utf-8", errors="ignore")
    err_path = workdir / inp_rel.replace(".inp", ".err")
    err_path.write_text(result.stderr or "", encoding="utf-8", errors="ignore")
    if result.returncode != 0:
        tail = (result.stderr or "")[-3000:]
        raise RuntimeError(
            f"ORCA cheap-opt failed (exit {result.returncode}). Last stderr chars:\n{tail}"
        )


# Indexed table: ``  1  O   x y z`` (some ORCA versions).
_RE_CART_INDEXED = re.compile(
    r"^\s*\d+\s+([A-Za-z][a-z]?)\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)"
    r"\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s*$"
)
# ORCA 6+ final block: ``  O   x y z`` (no leading atom index).
_RE_CART_SYMBOL_FIRST = re.compile(
    r"^\s*([A-Z][a-z]?)\s+([-+]?\d+(?:\.\d+)?(?:[EeDd][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?(?:[EeDd][-+]?\d+)?)\s+([-+]?\d+(?:\.\d+)?(?:[EeDd][-+]?\d+)?)\s*$"
)


def parse_xyz_trajectory_last(path: Path) -> tuple[list[str], np.ndarray]:
    """Return elems, coords (N×3 Å) using the **last** frame of a multi-structure ``.xyz`` file."""
    text = path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    if not text:
        raise ValueError(f"Empty xyz: {path}")
    frames = []
    i = 0
    while i < len(text):
        line = text[i].strip()
        if line == "":
            i += 1
            continue
        try:
            n_line = line.split()[0]
            nat = int(n_line)
        except (ValueError, IndexError):
            i += 1
            continue
        i += 1
        if i >= len(text):
            break
        i += 1  # skip title/comment line
        elems_chunk: list[str] = []
        rows: list[list[float]] = []
        for _ in range(nat):
            if i >= len(text):
                raise ValueError(f"Truncated XYZ frame near line {i} in {path}")
            parts = text[i].split()
            if len(parts) < 4:
                raise ValueError(f"Bad XYZ atom line {i}: {text[i]!r}")
            sym = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            elems_chunk.append(sym[0].upper() + sym[1:].lower() if len(sym) > 1 else sym.upper())
            rows.append([x, y, z])
            i += 1
        frames.append((elems_chunk, np.asarray(rows, dtype=float)))

    if not frames:
        raise ValueError(f"No XYZ frames parsed from {path}")
    els, crs = frames[-1]
    return els, crs


def _parse_last_cartesian_block_from_orca_out(out_text: str) -> tuple[list[str], np.ndarray] | None:
    """Fallback: grab the final ``CARTESIAN COORDINATES (ANGSTROEM)`` table from ORCA stdout."""
    lines = out_text.splitlines()
    last_start = None
    for i, ln in enumerate(lines):
        if "CARTESIAN COORDINATES (ANGSTR" in ln.upper():
            last_start = i
    if last_start is None:
        return None
    elems: list[str] = []
    rows: list[list[float]] = []
    for ln in lines[last_start + 1 :]:
        st = ln.strip()
        if not st or set(st) <= {"-", "="}:
            if rows:
                break
            continue
        m = _RE_CART_INDEXED.match(ln) or _RE_CART_SYMBOL_FIRST.match(ln)
        if m:
            sym = m.group(1)
            x, y, z = float(m.group(2)), float(m.group(3)), float(m.group(4))
            if len(sym) == 1:
                sym_u = sym.upper()
            else:
                sym_u = sym[0].upper() + sym[1].lower()
            elems.append(sym_u)
            rows.append([x, y, z])
        elif rows and ("CARTESIAN COORDINATES" in ln.upper() or "INTERNAL COORDINATES" in ln.upper()):
            break
    if len(rows) >= 1:
        return elems, np.asarray(rows, dtype=float)
    return None


def _normalize_elem(sym: str) -> str:
    s = sym.strip()
    if len(s) == 1:
        return s.upper()
    return s[0].upper() + s[1:].lower()


def _try_read_orca_xyz_file(path: Path, elems_in: list[str]) -> np.ndarray | None:
    """Parse ORCA ``.xyz`` / trajectory; return coords if element count/order matches ``elems_in``."""
    if not path.is_file():
        return None
    try:
        sym_out, crd = parse_xyz_trajectory_last(path)
    except (ValueError, OSError):
        return None
    if len(sym_out) != len(elems_in):
        return None
    for so, inp in zip(sym_out, elems_in):
        if _normalize_elem(so) != _normalize_elem(str(inp)):
            return None
    return crd


def _resolve_optimized_coords(work: Path, stem: str, elems_in: list[str]) -> np.ndarray:
    """
    Read optimized Cartesians from ORCA output files (version-dependent names).

    Tries ``{stem}_opt.xyz``, ``{stem}.xyz`` (ORCA 6+), ``*_opt.xyz``, ``{stem}_trj.xyz``,
    then parses the last ``CARTESIAN COORDINATES (ANGSTROEM)`` block from ``{stem}.out``.
    """
    stem = stem.strip().replace(".inp", "")
    inp_name = f"{stem}.inp"
    candidates: list[Path] = [
        work / f"{stem}_opt.xyz",
        work / f"{stem}.xyz",
    ]
    for p in sorted(glob.glob(str(work / "*_opt.xyz"))):
        candidates.append(Path(p))
    trj = work / f"{stem}_trj.xyz"
    if trj.is_file():
        candidates.append(trj)

    seen: set[str] = set()
    for cand in candidates:
        key = str(cand.resolve())
        if key in seen or not cand.is_file():
            continue
        seen.add(key)
        crd = _try_read_orca_xyz_file(cand, elems_in)
        if crd is not None:
            return crd

    outp = work / inp_name.replace(".inp", ".out")
    if not outp.is_file():
        raise RuntimeError(f"No ORCA output at {outp} and no readable .xyz under {work}")
    parsed = _parse_last_cartesian_block_from_orca_out(outp.read_text(encoding="utf-8", errors="ignore"))
    if parsed is None or len(parsed[0]) != len(elems_in):
        raise RuntimeError(
            f"Could not parse optimized geometry from {outp}. "
            f"Expected {len(elems_in)} atoms; check ORCA completed successfully."
        )
    sym_out, crd = parsed
    for so, inp in zip(sym_out, elems_in):
        if _normalize_elem(so) != _normalize_elem(str(inp)):
            raise RuntimeError(f"Element mismatch parsing {outp}: got {sym_out} vs {elems_in}")
    return crd


def minimize_geometry_cheap_orca(
    coords,
    elems: Sequence[str],
    *,
    workdir: os.PathLike[str] | str,
    charge: int = 0,
    multiplicity: int = 1,
    orca_executable: str | None = None,
    stem: str = "cheap_guess_orca",
    opt_bang_line: str | None = None,
    center: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Run one ORCA ``Opt`` job and return the optimized Cartesians (Å).

    Parameters
    ----------
    coords
        (N, 3) Å — starting guess.
    elems
        Element symbols aligned with coords.
    workdir
        Fresh directory recommended (isolates xyz/out/err).
    orca_executable
        None → same resolution as ``MolecularOptimizer``.
    opt_bang_line
        ORCA preamble after ``"!"`` (excluding the ``!``), e.g. ``"HF-3c Opt TightSCF"``.
    center
        Subtract centroid before returning.

    Raises
    ------
    RuntimeError
        ORCA missing, non-zero exit, or geometry could not be read back.
    """
    bang = opt_bang_line if opt_bang_line is not None else _opt_keyword_line()
    work = Path(workdir)
    exe = _find_orca(orca_executable)
    stem = stem.strip().replace(".inp", "")
    inp_name = f"{stem}.inp"
    inp_path = work / inp_name
    _write_opt_input(
        inp_path,
        coords=np.asarray(coords, dtype=float),
        elems=elems,
        charge=charge,
        multiplicity=multiplicity,
        bang_line=bang,
    )
    _exec_orca(exe, work, inp_name)

    elems_in = [str(e).strip() for e in elems]
    out_coords = _resolve_optimized_coords(work, stem, elems_in)

    if center:
        out_coords = out_coords.copy()
        out_coords -= out_coords.mean(axis=0, keepdims=True)
    return out_coords, elems_in
