"""The gradient a DFT fit takes is the derivative of its energy (D-164).

pyscf's RKS gradient leaves out the grid-weight response by default, so it is
not the energy's derivative; run_gradient and run_hessian now ask for it. On
displaced water at B3LYP/sto-3g the gradient with the response matches a
central finite difference of the energy to 3.7e-7 Ha/bohr, against 4.8e-6
without it (6-31G: 4.2e-7 against 3.0e-6). Real pyscf calls, about 7 s.
"""
import numpy as np
import pytest
from backend.registry import get_backend

import dev.pyscf_backend  # noqa: F401  (registers "pyscf_hf")

pytest.importorskip("pyscf")

A2B = 1.8897259886
X = (np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
     + np.array([[0.01, -0.02, 0.03], [0.02, 0.01, -0.01], [-0.01, 0.02, 0.0]]))


@pytest.fixture(scope="module")
def water_b3lyp():
    be = get_backend("pyscf_hf")(elems=["O", "H", "H"], method="b3lyp", basis="sto-3g")
    h = 1e-3
    fd = []
    for i in range(X.size):
        xp, xm = X.ravel().copy(), X.ravel().copy()
        xp[i] += h
        xm[i] -= h
        ep = be.run_gradient(xp.reshape(X.shape)).energy
        em = be.run_gradient(xm.reshape(X.shape)).energy
        fd.append((ep - em) / (2.0 * h) / A2B)
    return be, np.asarray(fd)


def test_run_gradient_is_the_energy_derivative(water_b3lyp):
    be, fd = water_b3lyp
    g = be.run_gradient(X).gradient_bohr
    assert np.max(np.abs(g - fd)) < 1.5e-6


def test_run_hessian_returns_the_same_gradient(water_b3lyp):
    be, fd = water_b3lyp
    g = be.run_hessian(X).gradient_bohr
    assert np.max(np.abs(g - fd)) < 1.5e-6
