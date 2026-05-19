"""Compatibility shim. Use backend.torsion.torsion_hamiltonian_2d."""

from backend.torsion import torsion_hamiltonian_2d as _m

globals().update({k: getattr(_m, k) for k in dir(_m) if not k.startswith('__')})
__all__ = [k for k in dir(_m) if not k.startswith('__')]
