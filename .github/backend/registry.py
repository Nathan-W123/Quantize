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


def fork_hang_warning(backend_name: str, n_workers: int,
                      backend_threads: int | None = None) -> str | None:
    """What to say before forking ``n_workers`` processes from this one, or None.

    A child forked from a process whose quantum backend has started threads
    inherits their locks and can hang for ever: a water bootstrap (pyscf
    HF/6-31G, OMP_NUM_THREADS=4, 50 samples) hung 3 of 3 times (T-42). One
    worker runs in-process and cannot; several fork, so this says so when the
    backend runs in this process and OMP_NUM_THREADS is not 1. It warns only:
    a forkserver pool would avoid the hang but costs ~0.5 s a run and needs a
    __main__ guard in calling scripts (decisions D-103).
    """
    import os
    if int(n_workers) <= 1:
        return None
    try:
        cls = get_backend(backend_name)
    except ValueError:
        return None
    if not getattr(cls, "runs_in_process", True):
        return None
    omp = os.environ.get("OMP_NUM_THREADS", "").strip()
    # A backend may set its own thread count (Psi4: psi4_num_threads), which
    # OMP_NUM_THREADS=1 does not limit.
    own = int(backend_threads) if backend_threads is not None else 1
    if omp == "1" and own <= 1:
        return None
    threads = (f"OMP_NUM_THREADS={omp or 'unset'}" if own <= 1
               else f"{own} threads of its own")
    return (f"{int(n_workers)} workers will be forked from this process, and the "
            f"'{backend_name}' backend computes in it with {threads}: a worker "
            f"forked from a threaded process can hang for ever (a water bootstrap "
            f"hung 3 of 3 times at four threads). Set OMP_NUM_THREADS=1 and the "
            f"backend's threads to 1, use one worker, or use ORCA.")

