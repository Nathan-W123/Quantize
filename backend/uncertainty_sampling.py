"""Compatibility shim. Use backend.uncertainty.sampling."""

from backend.uncertainty import sampling as _m

globals().update({k: getattr(_m, k) for k in dir(_m) if not k.startswith('__')})
__all__ = [k for k in dir(_m) if not k.startswith('__')]
