"""
Backend registry: maps string names to QuantumBackend subclasses.

Usage
-----
Backends self-register when their module is imported::

    from backend.registry import register_backend

    @register_backend
    class MyBackend(QuantumBackend):
        name = "mybackend"
        ...

Consumers look up backends by name::

    from backend.registry import get_backend, list_backends
    cls = get_backend("orca")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.base_backend import QuantumBackend

_REGISTRY: dict[str, type[QuantumBackend]] = {}


def register_backend(cls: type) -> type:
    """Class decorator that registers a QuantumBackend subclass by its ``name``."""
    key = str(cls.name).strip().lower()
    _REGISTRY[key] = cls
    return cls


def get_backend(name: str) -> type[QuantumBackend]:
    """Return the backend class for *name*, raising ValueError if unknown."""
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown quantum_backend '{name}'. "
            f"Available backends: {list_backends()}"
        )
    return _REGISTRY[key]


def list_backends() -> list[str]:
    """Return sorted list of registered backend names."""
    return sorted(_REGISTRY)
