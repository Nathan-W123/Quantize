"""The rigid-mode projector must remove the modes the Hessian is actually flat along.

`_rigid_mode_projector` removes translations and rotations from the quantum
gradient and Hessian before they are used as the prior. Which vectors those are
depends on whether the Hessian is mass-weighted, and this one is not: the
optimiser works in plain Cartesian Angstrom with plain Hartree/Ang^2
derivatives.

The projector used to build sqrt(m)-weighted modes -- the null vectors of a
*mass-weighted* Hessian -- and apply them to the plain one. That removes a
subspace which is not the rigid one, deleting real energy-changing directions
while leaving the rigid contamination it exists to remove. These tests measure
that directly against an analytic Hessian, so the distinction is settled by
numbers rather than by argument.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import dev.analytic_water_backend  # noqa: F401,E402  (registers analytic_water)
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.registry import get_backend  # noqa: E402

MASSES = np.array([15.9949146196, 1.00782503207, 1.00782503207])


def _water_coords(r=0.95785, deg=104.508):
    th = np.radians(deg)
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
        [-r * np.sin(th / 2), r * np.cos(th / 2), 0.0],
    ])


def _hessian(coords):
    h = get_backend("analytic_water")(elems=["O", "H", "H"]).run_hessian(coords)
    H = np.asarray(h.hessian_bohr, dtype=float)
    return 0.5 * (H + H.T)


def _rigid_vectors(coords, masses, weighted):
    """Translation/rotation generators, optionally sqrt(m)-weighted."""
    n = coords.shape[0]
    com = (masses[:, None] * coords).sum(axis=0) / masses.sum()
    rel = coords - com
    w = np.sqrt(masses)[:, None] if weighted else np.ones((n, 1))
    out = []
    for k in range(3):
        v = np.zeros((n, 3))
        v[:, k] = 1.0
        out.append((f"translation {'xyz'[k]}", (v * w).ravel()))
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        v = np.cross(np.tile(e, (n, 1)), rel)
        if np.linalg.norm(v) > 1e-12:
            out.append((f"rotation {'xyz'[k]}", (v * w).ravel()))
    return out


@pytest.mark.parametrize("name_idx", range(3))
def test_unweighted_translations_are_null_vectors_of_the_cartesian_hessian(name_idx):
    """Displacing every atom equally cannot change the energy."""
    coords = _water_coords()
    H = _hessian(coords)
    name, v = _rigid_vectors(coords, MASSES, weighted=False)[name_idx]
    v = v / np.linalg.norm(v)
    assert np.linalg.norm(H @ v) < 1e-6, f"{name} is not flat: {np.linalg.norm(H @ v):.3e}"


def test_sqrt_mass_weighted_translations_are_not_null_vectors():
    """The old projector's modes. Moving heavy and light atoms by different
    amounts stretches bonds, so of course the energy changes -- these belong to
    a mass-weighted Hessian, not this one."""
    coords = _water_coords()
    H = _hessian(coords)
    residuals = []
    for name, v in _rigid_vectors(coords, MASSES, weighted=True)[:3]:
        v = v / np.linalg.norm(v)
        residuals.append(float(np.linalg.norm(H @ v)))
    assert max(residuals) > 1e-2, (
        f"expected sqrt(m)-weighted translations to be far from null, got {residuals}"
    )


def test_projector_annihilates_the_rigid_subspace_it_claims_to_remove():
    """P v = 0 for every genuine rigid mode, and P leaves vibrations alone."""
    coords = _water_coords()
    opt = MolecularOptimizer(
        elems=["O", "H", "H"], coords=coords,
        isotopologues=[{
            "name": "H2-16O", "masses": MASSES.tolist(),
            "obs_constants": [835840.29, 435351.72, 278138.70],
            "sigma_constants": [0.2, 0.2, 0.2],
        }],
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=1,
        project_rigid_modes=True,
    )
    P = opt._rigid_mode_projector(coords, MASSES)
    for name, v in _rigid_vectors(coords, MASSES, weighted=False):
        v = v / np.linalg.norm(v)
        assert np.linalg.norm(P @ v) < 1e-8, f"{name} survives the projector"
    # A symmetric stretch is not rigid and must be preserved.
    stretch = np.zeros((3, 3))
    stretch[1] = coords[1] - coords[0]
    stretch[2] = coords[2] - coords[0]
    s = stretch.ravel() / np.linalg.norm(stretch)
    assert np.linalg.norm(P @ s) > 0.5, "projector is eating a vibrational mode"


def test_projected_hessian_keeps_the_vibrational_spectrum():
    """Projection must remove the rigid block and nothing else.

    Water has 3N-6 = 3 vibrations; after projection exactly three eigenvalues
    should remain non-zero, and they must match the vibrational eigenvalues of
    the unprojected Hessian.
    """
    coords = _water_coords()
    H = _hessian(coords)
    opt = MolecularOptimizer(
        elems=["O", "H", "H"], coords=coords,
        isotopologues=[{
            "name": "H2-16O", "masses": MASSES.tolist(),
            "obs_constants": [835840.29, 435351.72, 278138.70],
            "sigma_constants": [0.2, 0.2, 0.2],
        }],
        quantum_backend="analytic_water", use_autoconfig=False, max_iter=1,
        project_rigid_modes=True,
    )
    P = opt._rigid_mode_projector(coords, MASSES)
    ev_proj = np.sort(np.linalg.eigvalsh(P @ H @ P))
    scale = float(np.max(np.abs(ev_proj)))
    n_nonzero = int((np.abs(ev_proj) > 1e-6 * scale).sum())
    assert n_nonzero == 3, f"expected 3 vibrations, got {n_nonzero}"

    ev_raw = np.sort(np.linalg.eigvalsh(H))
    assert np.allclose(ev_proj[-3:], ev_raw[-3:], rtol=1e-3), (
        "projection changed the vibrational eigenvalues it should have left alone"
    )


def test_linear_molecule_keeps_five_rigid_modes_not_six():
    """Rotation about a linear molecule's own axis moves no atom.

    The rank has to come from singular values: a reduced QR returns six
    unit-norm columns regardless, so a column-norm test would project out an
    arbitrary sixth direction -- a real vibration.
    """
    coords = np.array([[0.0, 0.0, -1.06], [0.0, 0.0, 0.0], [0.0, 0.0, 1.16]])
    masses = np.array([1.00782503207, 12.0, 14.0030740048])
    opt = MolecularOptimizer(
        elems=["H", "C", "N"], coords=coords,
        isotopologues=[{
            "name": "HCN", "masses": masses.tolist(),
            "obs_constants": [44315.98], "component_indices": [1],
            "sigma_constants": [1.0],
        }],
        quantum_backend=None, spectral_only=True, use_autoconfig=False, max_iter=1,
        project_rigid_modes=True,
    )
    P = opt._rigid_mode_projector(coords, masses)
    # trace of a projector is the dimension of the space it keeps
    kept = int(round(float(np.trace(P))))
    assert kept == 9 - 5, f"expected 5 rigid modes removed from 9, kept {kept}"
