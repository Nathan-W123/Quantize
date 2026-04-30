"""
Psi4 backend for quantum prior (energy/gradient/Hessian).

This module mirrors the unit conventions used by QuantumEngine in quantum.py:
  - stored gradient/Hessian are in Hartree/Bohr and Hartree/Bohr^2
  - exposed properties return Hartree/Angstrom and Hartree/Angstrom^2
"""

import os
import numpy as np

from quantum import ANG_TO_BOHR


class Psi4State:
    """Container with QuantumEngine-compatible properties."""

    def __init__(self, energy, gradient_bohr, hessian_bohr):
        self.energy = float(energy)
        self._gradient_bohr = np.asarray(gradient_bohr, dtype=float).ravel()
        self._hessian_bohr = np.asarray(hessian_bohr, dtype=float)

    @property
    def gradient(self):
        return self._gradient_bohr * ANG_TO_BOHR

    @property
    def hessian(self):
        return self._hessian_bohr * ANG_TO_BOHR ** 2


class Psi4Engine:
    """
    Compute energy/gradient/Hessian directly with Psi4.
    """

    def __init__(
        self,
        elems,
        method="B3LYP",
        basis="cc-pVDZ",
        charge=0,
        multiplicity=1,
        memory="2 GB",
        num_threads=1,
        output_file=None,
    ):
        self.elems = list(elems)
        self.method = str(method).strip()
        self.basis = str(basis).strip()
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.memory = memory
        self.num_threads = int(num_threads)
        self.output_file = output_file
        self._psi4 = None
        self._cache_hessian = {}
        self._cache_gradient = {}
        self._cache_decimals = 8
        self._configure_psi4()

    def _coords_key(self, coords_ang):
        arr = np.asarray(coords_ang, dtype=float).ravel()
        return tuple(np.round(arr, self._cache_decimals))

    def _configure_psi4(self):
        try:
            import psi4  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise RuntimeError(
                "Psi4 backend requested but psi4 is not installed. "
                "Install with conda/pip and retry."
            ) from e
        self._psi4 = psi4
        psi4.set_memory(self.memory)
        psi4.set_num_threads(self.num_threads)
        out = self.output_file or f"psi4_{os.getpid()}.out"
        psi4.core.set_output_file(out, False)

    def _molecule(self, coords_ang):
        lines = [f"{self.charge} {self.multiplicity}"]
        for elem, (x, y, z) in zip(self.elems, coords_ang):
            lines.append(f"{elem} {x:.12f} {y:.12f} {z:.12f}")
        lines += ["units angstrom", "no_com", "no_reorient"]
        text = "\n".join(lines)
        return self._psi4.geometry(text)

    def _method_spec(self):
        return f"{self.method}/{self.basis}" if self.basis else self.method

    def run_hessian(self, coords_ang):
        key = (self._method_spec(), self._coords_key(coords_ang))
        if key in self._cache_hessian:
            return self._cache_hessian[key]
        mol = self._molecule(np.asarray(coords_ang, dtype=float))
        spec = self._method_spec()
        e, wfn = self._psi4.energy(spec, molecule=mol, return_wfn=True)
        g = self._psi4.gradient(spec, molecule=mol, ref_wfn=wfn)
        h = self._psi4.hessian(spec, molecule=mol, ref_wfn=wfn)
        g_bohr = np.array(g.np, dtype=float).ravel()
        h_bohr = np.array(h.np, dtype=float)
        state = Psi4State(e, g_bohr, h_bohr)
        self._cache_hessian[key] = state
        self._cache_gradient[key] = (float(e), g_bohr.copy())
        return state

    def run_gradient(self, coords_ang):
        key = (self._method_spec(), self._coords_key(coords_ang))
        if key in self._cache_gradient:
            return self._cache_gradient[key]
        mol = self._molecule(np.asarray(coords_ang, dtype=float))
        spec = self._method_spec()
        e, wfn = self._psi4.energy(spec, molecule=mol, return_wfn=True)
        g = self._psi4.gradient(spec, molecule=mol, ref_wfn=wfn)
        g_bohr = np.array(g.np, dtype=float).ravel()
        out = (float(e), g_bohr)
        self._cache_gradient[key] = (out[0], out[1].copy())
        return out
