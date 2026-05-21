# Backend package for optimization engine modules.
#
# Importing the backend modules below triggers their @register_backend decorators
# so that get_backend("orca") / get_backend("psi4") work anywhere in the codebase.
# To add a new backend: create backend/<name>_backend.py and add one import line here.
from backend.orca import orca_backend as _orca_backend  # noqa: F401
from backend.psi4 import psi4_backend as _psi4_backend  # noqa: F401
from backend.registry import get_backend, list_backends  # noqa: F401
