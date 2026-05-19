"""Compatibility shim. Use backend.torsion.torsion_lam_integration."""

from backend.torsion import torsion_lam_integration as _m

globals().update({k: getattr(_m, k) for k in dir(_m) if not k.startswith('__')})
__all__ = [k for k in dir(_m) if not k.startswith('__')]
