"""Bias detection has a floor, and it is far above the biases that matter.

Every residual-based detector in this engine is blind to a correction bias that
is common-mode across a molecule's isotopologues, because the fit absorbs it
into the geometry and the residuals stay small -- measured, reduced chi-square
never leaves [0.25, 4] anywhere on the reference set.

The one opening is that a correction is not common-mode: Delta_K depends on the
isotopologue through its masses and the geometry does not. Fitting a scale on
each component's correction alongside the geometry exploits that, and on water
it does recover the true bias from the measurements alone.

These tests pin the part that decides whether any of it is usable: the size of
the bias such a fit could distinguish from none. It is computed from masses, a
geometry and the correction sigmas -- no measured constant enters -- so it is an
experimental-design number, available before a species is ever made.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.spectral.centrifugal_distortion import rotational_constants_mhz
from dev.monofluoro_references import _bent_xy

_M_H, _M_D = 1.00782503, 2.01410178
_M_O16, _M_O18 = 15.99491462, 17.99915961


def _limit(iso_masses, delta_of, sigma_of, r=0.9578, theta=104.48):
    """2-sigma threshold on |s - 1| per component; see the script of the same name."""
    b0 = {}
    for label, masses in iso_masses:
        ref = rotational_constants_mhz(_bent_xy(r, theta),
                                       np.asarray(masses, dtype=float))
        b0[label] = ref - delta_of(label)

    def residual(p):
        rr, tt, sc = p[0], p[1], p[2:5]
        out = []
        for label, masses in iso_masses:
            calc = rotational_constants_mhz(_bent_xy(rr, tt),
                                            np.asarray(masses, dtype=float))
            out.extend(((calc - (b0[label] + sc * delta_of(label)))
                        / sigma_of(label)).tolist())
        return np.asarray(out)

    p0 = np.array([r, theta, 1.0, 1.0, 1.0])
    base = residual(p0)
    jac = np.zeros((base.size, 5))
    for i in range(5):
        q = p0.copy()
        q[i] += 1e-5
        jac[:, i] = (residual(q) - base) / 1e-5
    cov = np.linalg.inv(jac.T @ jac)
    return 2.0 * np.sqrt(np.clip(np.diag(cov)[2:5], 0.0, None))


#: Representative water corrections and their sigmas at B3LYP/6-31G(d), in MHz,
#: as the engine computes them. Fixed here so the test needs no quantum
#: chemistry; the limit depends on their scale, not on their being exact.
_WATER = {
    "H2-16O": (np.array([-7865.0, 2918.9, 7151.4]),
               np.array([9523.5, 3473.8, 1701.3])),
    "D2-16O": (np.array([-3151.5, 1085.6, 2685.0]),
               np.array([3921.2, 1240.6, 638.0])),
    # The two 18-O species are not measured; their corrections are scaled from
    # the measured ones, which is enough for a design calculation that depends
    # on the scale of sigma rather than on its exact value.
    "HD-16O": (np.array([-5200.0, 1900.0, 4600.0]),
               np.array([6300.0, 2300.0, 1120.0])),
    "H2-18O": (np.array([-7800.0, 2600.0, 6700.0]),
               np.array([9400.0, 3100.0, 1600.0])),
}
_MASSES = {
    "H2-16O": [_M_O16, _M_H, _M_H],
    "D2-16O": [_M_O16, _M_D, _M_D],
    "HD-16O": [_M_O16, _M_H, _M_D],
    "H2-18O": [_M_O18, _M_H, _M_H],
}


def _water_set(labels):
    return ([(l, _MASSES[l]) for l in labels],
            lambda l: _WATER[l][0], lambda l: _WATER[l][1])


def test_the_limit_is_finite_and_the_jacobian_is_not_singular():
    """A singular normal matrix reports every limit as infinite, which reads
    exactly like 'undetectable' and hides a broken calculation. It happened:
    building the notional observation from the fitted geometry made it cancel
    out of the residual, leaving no geometry columns at all."""
    lim = _limit(*_water_set(["H2-16O", "D2-16O"]))
    assert np.all(np.isfinite(lim))
    assert np.all(lim > 0.0)


def test_water_cannot_detect_its_own_measured_bias():
    """The finding this whole exercise produced.

    Water's A correction was measured 87% wrong against its published
    equilibrium structure. Two isotopologues put the detection floor around ten
    times that, so no residual-based test on that data could ever have flagged
    it -- which is why nothing did, for months.
    """
    lim = _limit(*_water_set(["H2-16O", "D2-16O"]))
    assert lim[0] > 5.0, f"A limit {lim[0]:.1f} unexpectedly tight"
    assert lim[0] > 5.0 * 0.87


def test_more_isotopologues_help_only_by_about_a_factor_of_two():
    """And then saturate, which is what makes this a dead end rather than a cost.

    The correction's isotopic signature is nearly parallel to the geometry's, so
    extra species add little independent information.
    """
    two = _limit(*_water_set(["H2-16O", "D2-16O"]))
    four = _limit(*_water_set(["H2-16O", "D2-16O", "HD-16O", "H2-18O"]))
    assert np.all(four < two)
    assert np.all(four > 0.35 * two), (
        "a bigger gain than measured -- recheck the design calculation"
    )


def test_a_tighter_correction_sigma_tightens_the_limit_proportionally():
    """The limit is set by how uncertain the correction already is.

    Useful because it says what would have to improve: a correction known ten
    times better moves the floor ten times, and water's A would still sit near
    100%. That is the quantitative reason to prevent bias rather than detect it.
    """
    isos, delta_of, sigma_of = _water_set(["H2-16O", "D2-16O"])
    loose = _limit(isos, delta_of, sigma_of)
    tight = _limit(isos, delta_of, lambda l: 0.1 * sigma_of(l))
    assert np.allclose(tight, 0.1 * loose, rtol=0.05)


@pytest.mark.parametrize("comp,idx", [("A", 0), ("B", 1), ("C", 2)])
def test_c_is_the_best_determined_component(comp, idx):
    """C's floor is the lowest, and it is also the component the corrected
    engine gets right to 3%. The ordering is not a coincidence: both follow
    from C having the least cancellation between its correction terms."""
    lim = _limit(*_water_set(["H2-16O", "D2-16O"]))
    assert lim[2] <= lim[idx] + 1e-9
