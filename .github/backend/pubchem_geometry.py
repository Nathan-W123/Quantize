"""
Fetch 3-D Cartesian coordinates from PubChem PUG REST (MMFF94-relaxed conformers).

Uses ``record_type=3d`` SDF (V2000). Requires network access.

PubChem API reference: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from typing import Tuple

import numpy as np

USER_AGENT = "Research/geometry-guess (PubChem PUG REST; numpy seed geometries)"

PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"


def _fetch_text(url: str, timeout: float) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _normalize_element_symbol(raw: str) -> str:
    s = raw.strip()
    if not s:
        return "X"
    if len(s) == 1:
        return s.upper()
    return s[0].upper() + s[1:].lower()


def parse_sdf_v2000_first_mol(sdf_text: str) -> Tuple[list[str], np.ndarray]:
    """
    Extract atom symbols and coordinates from the first V2000 molecule in an SD file.

    Coordinates are read from the CTAB atom block (Å).
    """
    block = sdf_text.split("$$$$")[0]
    lines = block.splitlines()
    counts_idx = None
    for i, ln in enumerate(lines):
        if "V2000" in ln:
            counts_idx = i
            break
        if "V3000" in ln:
            raise ValueError(
                "SDF uses V3000 CTAB; only V2000 is implemented. Try another PubChem record."
            )

    if counts_idx is None:
        raise ValueError("No V2000 counts line found in SDF.")

    counts_line = lines[counts_idx]
    try:
        n_atoms = int(counts_line[0:3])
    except ValueError as e:
        raise ValueError(f"Could not parse atom count from CTAB: {counts_line!r}") from e

    elems: list[str] = []
    xyz_rows: list[list[float]] = []
    for k in range(n_atoms):
        ln = lines[counts_idx + 1 + k]
        if len(ln) < 34:
            raise ValueError(f"Atom line too short (line {counts_idx + 2 + k}): {ln!r}")
        x = float(ln[0:10])
        y = float(ln[10:20])
        z = float(ln[20:30])
        sym_field = ln[31:34]
        sym = sym_field.split()[0] if sym_field.split() else ""
        elems.append(_normalize_element_symbol(sym))
        xyz_rows.append([x, y, z])

    coords = np.asarray(xyz_rows, dtype=float)
    return elems, coords


def _url_cid(cid: str) -> str:
    return f"{PUG_BASE}/cid/{urllib.parse.quote(cid, safe='')}/record/SDF/?record_type=3d"


def _url_smiles(smiles: str) -> str:
    enc = urllib.parse.quote(smiles, safe="")
    return f"{PUG_BASE}/smiles/{enc}/record/SDF/?record_type=3d"


def _url_name(name: str) -> str:
    enc = urllib.parse.quote(name, safe="")
    return f"{PUG_BASE}/name/{enc}/record/SDF/?record_type=3d"


def coords_elems_from_pubchem(
    identifier: str,
    *,
    timeout: float = 60.0,
    prefer: str = "auto",
) -> Tuple[np.ndarray, list[str]]:
    """
    Fetch a 3-D conformer from PubChem.

    Parameters
    ----------
    identifier
        PubChem compound identifier: numeric **CID**, **SMILES**, or **common/IUPAC name**.
    prefer
        ``"auto"`` — if ``identifier`` is all digits, use CID only; otherwise try SMILES,
        then name. ``"cid"`` | ``"smiles"`` | ``"name"`` — force that lookup mode.

    Returns
    -------
    coords
        (N, 3) array in Å.
    elems
        Element symbols aligned with ``coords``.
    """
    ident = identifier.strip()
    if not ident:
        raise ValueError("PubChem identifier must be non-empty.")

    pref = prefer.strip().lower()
    if pref not in {"auto", "cid", "smiles", "name"}:
        raise ValueError("prefer must be one of: auto, cid, smiles, name")

    urls: list[tuple[str, str]] = []
    if pref == "cid":
        urls.append(("cid", _url_cid(ident)))
    elif pref == "smiles":
        urls.append(("smiles", _url_smiles(ident)))
    elif pref == "name":
        urls.append(("name", _url_name(ident)))
    else:
        if ident.isdigit():
            urls.append(("cid", _url_cid(ident)))
        else:
            urls.append(("smiles", _url_smiles(ident)))
            urls.append(("name", _url_name(ident)))

    last_exc: Exception | None = None
    for label, url in urls:
        try:
            sdf = _fetch_text(url, timeout)
            elems, coords = parse_sdf_v2000_first_mol(sdf)
            return coords, elems
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code in (404, 400):
                continue
            raise
        except urllib.error.URLError as e:
            last_exc = e
            continue

    modes = ", ".join(l for l, _ in urls)
    raise RuntimeError(
        f"PubChem lookup failed for {ident!r} (tried: {modes}). "
        "Check spelling/SMILES or network."
    ) from last_exc
