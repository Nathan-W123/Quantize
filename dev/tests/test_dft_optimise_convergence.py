"""A DFT geometry optimisation must run on its energy's own gradient, and say
when it did not converge.

pyscf's RKS gradient leaves out the grid-weight response by default, so it is
not the derivative of the energy L-BFGS-B is minimising. The line search then
stops on its energy tolerance wherever the energy stops falling, with the
gradient still well above gtol: the cached B3LYP/6-31G(d) theory geometries
were left at |g| up to 3.2e-4 Ha/A against a gtol of 2e-5, bonds up to
0.24 mA from the energy's minimum (fluoroethane, formyl fluoride, vinyl
fluoride), and optimise() returned them without a word (decisions D-57). With
the grid response the same twelve converge (regenerated at the default
gtol: <= 2.0e-5).

These tests replace the electronic structure with a quadratic surface, so
they run in milliseconds; what they pin is what optimise() asks for and what
it reports.
"""

import numpy as np
import pytest

import dev.pyscf_backend  # noqa: F401  (registers "pyscf_hf")
from backend.registry import get_backend

ELEMS = ["O", "H", "H"]
X_MIN = np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
K = 0.6          # Ha/bohr^2 per coordinate, about a stretch
A2B = 1.8897259886


def _quadratic(offset_bohr=0.0, calls=None):
    """E = K/2 |x - x_min|^2 (bohr), and a gradient that is its derivative
    plus a constant offset -- the size of the missing grid term."""
    def energy_gradient(coords_ang, **kw):
        if calls is not None:
            calls.append(kw)
        d = (np.asarray(coords_ang, dtype=float) - X_MIN).ravel() * A2B
        return 0.5 * K * float(d @ d), K * d + offset_bohr
    return energy_gradient


@pytest.mark.parametrize("method, wants", [("b3lyp", True), ("hf", False)])
def test_optimise_takes_the_dft_gradient_with_the_grid_response(monkeypatch, method, wants):
    backend = get_backend("pyscf_hf")(elems=ELEMS, method=method, basis="sto-3g")
    calls = []
    monkeypatch.setattr(backend, "_energy_gradient", _quadratic(calls=calls))
    backend.optimise(X_MIN + 0.02)
    assert calls, "the optimiser evaluated nothing"
    assert all(bool(kw.get("grid_response", False)) is wants for kw in calls)


def test_optimise_reports_a_run_that_stopped_above_gtol(monkeypatch, capsys):
    backend = get_backend("pyscf_hf")(elems=ELEMS, method="b3lyp", basis="sto-3g")
    # A gradient 2e-4 Ha/bohr off its energy's derivative: the line search
    # stalls where the energy stops falling. Measured: L-BFGS-B ends on an
    # abnormal line search after 3 iterations, |g| 3.4e-4 Ha/A, 17x the
    # default gtol (the real vinyl fluoride run: ftol stop, 2.0e-4, 10x).
    monkeypatch.setattr(backend, "_energy_gradient", _quadratic(offset_bohr=2e-4))
    backend.optimise(X_MIN + 0.02)
    info = backend.last_optimise
    assert info["converged"] is False
    assert info["g_max_ha_per_ang"] > 10 * info["gtol"]
    assert "did not converge" in capsys.readouterr().out


def test_optimise_records_a_converged_run(monkeypatch, capsys):
    backend = get_backend("pyscf_hf")(elems=ELEMS, method="b3lyp", basis="sto-3g")
    monkeypatch.setattr(backend, "_energy_gradient", _quadratic())
    relaxed = backend.optimise(X_MIN + 0.02)
    info = backend.last_optimise
    # Measured |g| 2.7e-10 Ha/A, 1.3e-10 A from the minimum.
    assert info["converged"] is True and info["g_max_ha_per_ang"] <= info["gtol"]
    assert np.abs(relaxed - X_MIN).max() < 1e-6
    assert "did not converge" not in capsys.readouterr().out
