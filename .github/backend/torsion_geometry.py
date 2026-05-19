"""Compatibility shim. Use backend.torsion.torsion_geometry."""

from backend.torsion import torsion_geometry as _m

globals().update({k: getattr(_m, k) for k in dir(_m) if not k.startswith('__')})
__all__ = [k for k in dir(_m) if not k.startswith('__')]
