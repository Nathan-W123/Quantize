"""Compatibility shim. Use backend.torsion.torsion_plots."""

from backend.torsion import torsion_plots as _m

globals().update({k: getattr(_m, k) for k in dir(_m) if not k.startswith('__')})
__all__ = [k for k in dir(_m) if not k.startswith('__')]
