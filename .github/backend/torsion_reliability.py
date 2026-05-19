"""Compatibility shim. Use backend.torsion.torsion_reliability."""

from backend.torsion import torsion_reliability as _m

globals().update({k: getattr(_m, k) for k in dir(_m) if not k.startswith('__')})
__all__ = [k for k in dir(_m) if not k.startswith('__')]
