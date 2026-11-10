"""On-disk cache for Hessians, keyed by level of theory and geometry.

WHY. A correlated correction level has no analytic Hessian, so each one costs
6N gradient evaluations, and the ``normal_mode`` cubic scheme needs 2*n_vib of
them -- for formyl fluoride at MP2/cc-pVTZ that is around 288 MP2 gradients,
about an hour. The benchmark cached them in a plain dict inside the
per-molecule loop and wrote its results file only after a whole molecule
finished, so anything that interrupted the run threw away every Hessian it had
computed. That happened twice in one session: once to a container restart and
once to a process that died mid-molecule leaving no traceback. Both times the
hour was simply lost.

Caching to disk makes the expensive part resumable. It serves accuracy and
cost at once rather than trading them: the measurement actually completes, and
no Hessian is ever computed twice.

WHAT MAKES A CACHE HIT, and why each part is in the key. A Hessian is a
property of the electronic structure method, the basis, the elements and the
geometry -- and of nothing else. It does NOT depend on the masses, which is why
one cached Hessian serves every isotopologue, and that is most of the saving.

So the key is a hash of (method, basis, elements, coordinates rounded to 1e-9
Ang). Dropping any of those would return a confidently wrong matrix:

  * method or basis -- the usual silent catastrophe. An HF Hessian handed back
    for an MP2 request looks entirely plausible and is simply a different
    force field, which is the exact mistake an earlier version of the
    benchmark made by evaluating one level's Hessian at another level's
    geometry, and which it then reported as "the correction level does not
    matter".
  * elements -- two molecules can have the same atom count and geometry shape.
  * coordinates -- obviously, and rounded to match the in-memory cache the
    benchmark already used, so the two agree on what counts as the same point.

VERIFIED ON READ, NOT TRUSTED. The stored file carries the method, basis,
elements and coordinates it was computed for, and a load checks all of them
against the request before returning anything. A hash collision, a corrupted
file, a half-written file from a killed process, or a cache directory copied
between machines therefore produces a MISS and a recomputation rather than a
wrong Hessian. The cost of being wrong here is a whole benchmark conclusion;
the cost of a spurious miss is one Hessian.

WRITES ARE ATOMIC. Written to a temporary file in the same directory and then
renamed, because the process this exists to protect against is one that dies at
an arbitrary moment. A rename within a directory is atomic on POSIX, so a
reader never sees a partial file.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

#: Geometry rounding, in Angstrom, for the cache key. Matches the rounding the
#: benchmark's in-memory caches use so the two agree on identity.
_COORD_DECIMALS = 9


def _normalise_level(method: str, basis: str) -> tuple[str, str]:
    return str(method).strip().lower(), str(basis).strip().lower()


def cache_key(method: str, basis: str, elements, coords_ang) -> str:
    """Hex digest identifying one Hessian computation."""
    m, b = _normalise_level(method, basis)
    elems = [str(e).strip().capitalize() for e in elements]
    coords = np.asarray(coords_ang, dtype=float).round(_COORD_DECIMALS)

    h = hashlib.sha256()
    h.update(m.encode())
    h.update(b"\x00")
    h.update(b.encode())
    h.update(b"\x00")
    h.update("|".join(elems).encode())
    h.update(b"\x00")
    h.update(np.ascontiguousarray(coords, dtype="<f8").tobytes())
    return h.hexdigest()


class HessianCache:
    """Disk-backed Hessian store. ``enabled=False`` makes every call a miss.

    Wrap a backend's Hessian call with :meth:`wrap` and the result is cached
    transparently::

        cache = HessianCache(root, method="mp2", basis="cc-pvtz",
                             elements=mol.elems)
        hessian_fn = cache.wrap(lambda c: backend.run_hessian(c).hessian_bohr)
    """

    def __init__(self, root, method: str, basis: str, elements,
                 enabled: bool = True):
        self.root = Path(root)
        self.method, self.basis = _normalise_level(method, basis)
        self.elements = [str(e).strip().capitalize() for e in elements]
        self.enabled = bool(enabled)
        self.hits = 0
        self.misses = 0
        self.rejected = 0

    def path_for(self, coords_ang) -> Path:
        key = cache_key(self.method, self.basis, self.elements, coords_ang)
        return self.root / f"{self.method}_{self.basis}" / f"{key}.npz"

    def load(self, coords_ang):
        """The cached Hessian, or None. Verifies what it loads before trusting it."""
        if not self.enabled:
            return None
        path = self.path_for(coords_ang)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                stored_method = str(z["method"].item())
                stored_basis = str(z["basis"].item())
                stored_elems = [str(x) for x in z["elements"].tolist()]
                stored_coords = np.asarray(z["coords"], dtype=float)
                hessian = np.asarray(z["hessian"], dtype=float)
        except Exception:
            # Corrupt, truncated, or written by an incompatible numpy. A miss
            # costs one Hessian; trusting it could cost a conclusion.
            self.rejected += 1
            return None

        want = np.asarray(coords_ang, dtype=float).round(_COORD_DECIMALS)
        n = want.shape[0]
        if (stored_method != self.method
                or stored_basis != self.basis
                or stored_elems != self.elements
                or stored_coords.shape != want.shape
                or not np.allclose(stored_coords, want, rtol=0.0,
                                   atol=10.0 ** -_COORD_DECIMALS)
                or hessian.shape != (3 * n, 3 * n)):
            self.rejected += 1
            return None

        self.hits += 1
        return hessian

    def store(self, coords_ang, hessian) -> None:
        """Write a Hessian atomically, so a killed process leaves no half file."""
        if not self.enabled:
            return
        path = self.path_for(coords_ang)
        path.parent.mkdir(parents=True, exist_ok=True)
        coords = np.asarray(coords_ang, dtype=float).round(_COORD_DECIMALS)
        payload = {
            "method": np.array(self.method),
            "basis": np.array(self.basis),
            "elements": np.array(self.elements),
            "coords": coords,
            "hessian": np.asarray(hessian, dtype=float),
        }
        # Write through an open HANDLE, not a path. np.savez appends ".npz"
        # to a path that lacks it, so passing a mkstemp name like "x.npz.tmp"
        # makes it write "x.npz.tmp.npz" while the empty mkstemp file stays
        # behind -- and renaming that empty file over the target produced a
        # cache that stored nothing and missed on every read. Handed a file
        # object, savez writes exactly there and appends nothing.
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npz.tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez(fh, **payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def wrap(self, compute):
        """``compute(coords) -> hessian``, with the disk cache in front of it."""
        def cached(coords_ang):
            got = self.load(coords_ang)
            if got is not None:
                return got
            self.misses += 1
            hessian = compute(coords_ang)
            self.store(coords_ang, hessian)
            return hessian
        return cached

    def summary(self) -> dict:
        return {"hits": int(self.hits), "misses": int(self.misses),
                "rejected": int(self.rejected), "enabled": bool(self.enabled),
                "level": f"{self.method}/{self.basis}"}
