"""Converting a torsional scan into the correction chain's replacement.

The torsion package could already solve a hindered rotor and average the
rotational constants over its wavefunction; harmonic_alpha could already accept
a computed torsional contribution in place of the perturbative one. Nothing
converted between them, so the averaging could not reach the correction.

The conversion is two characters wide and both of them matter:

    alpha_K = -2 (<B_K> - B_K(ref))

A dropped factor of two halves the correction. A dropped minus sign pushes the
fitted structure the wrong way by twice its own size, and neither shows up as
an error -- just a plausible number. So these check the arithmetic against a
scan whose answer is known in advance rather than against itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.spectral.harmonic_alpha import TorsionReplacement
from backend.spectral.spectral import _rotational_constants
from backend.torsion.torsion_average import TorsionGridPoint, TorsionScan
from backend.torsion.torsion_hamiltonian import (
    TorsionFourierPotential,
    TorsionHamiltonianSpec,
)
from backend.torsion.torsion_lam_integration import torsion_replacement_from_scan

ELEMENTS = ["O", "H", "H"]
MASSES = np.array([15.9949146196, 1.00782503207, 1.00782503207])


def _water(r=0.9578, theta_deg=104.5):
    half = np.radians(theta_deg) / 2.0
    return np.array([
        [0.0, 0.0, 0.0],
        [r * np.sin(half), r * np.cos(half), 0.0],
        [-r * np.sin(half), r * np.cos(half), 0.0],
    ])


def _spec():
    """A hindered rotor with a real barrier, so the weights are not uniform."""
    return TorsionHamiltonianSpec(
        F=5.0, n_basis=8,
        potential=TorsionFourierPotential(vcos={3: -150.0}),
    )


def _flat_scan(constants, n=12):
    """A scan whose every grid point carries the SAME rotational constants.

    Any weighted average of identical values is that value, whatever the
    torsional wavefunction turns out to be. That makes the expected answer
    exact and independent of the Hamiltonian, which is what lets the next test
    check the conversion rather than the averaging.
    """
    geom = _water()
    return TorsionScan(
        name="flat", atoms=None, angle_unit="degrees",
        grid_points=[
            TorsionGridPoint(phi=360.0 * i / n, geometry=geom,
                             rotational_constants=np.asarray(constants, dtype=float))
            for i in range(n)
        ],
    )


def test_a_scan_that_averages_to_the_reference_gives_no_correction():
    """The null case. If <B> == B_ref the torsional contribution is zero."""
    ref = _water()
    exact = np.asarray(_rotational_constants(ref, MASSES), dtype=float)
    replacement, _ = torsion_replacement_from_scan(
        ELEMENTS, _flat_scan(exact), _spec(), ref, masses=MASSES)
    for k in ("A", "B", "C"):
        assert replacement.alpha_mhz[k] == pytest.approx(0.0, abs=1e-6)


def test_the_conversion_is_minus_two_times_the_shift():
    """Sign and factor, against an answer computed by hand.

    Every grid point carries B_ref minus a known offset, so <B> - B_ref is that
    offset exactly and alpha must be -2 times it.
    """
    ref = _water()
    exact = np.asarray(_rotational_constants(ref, MASSES), dtype=float)
    offset = np.array([-500.0, -40.0, -25.0])      # averaging lowers B
    replacement, _ = torsion_replacement_from_scan(
        ELEMENTS, _flat_scan(exact + offset), _spec(), ref, masses=MASSES)

    for i, k in enumerate(("A", "B", "C")):
        assert replacement.alpha_mhz[k] == pytest.approx(-2.0 * offset[i], rel=1e-9)


def test_lowering_b_gives_positive_alpha():
    """The physical sanity check the algebra has to agree with.

    Averaging over a floppy coordinate samples geometries spread away from the
    minimum, raising the moments of inertia and lowering B. Watson's
    B_0 = B_e - (1/2) sum alpha then requires positive alpha, because the
    vibrationally averaged constant is the smaller one. This is the assertion
    that fails if the minus sign is dropped.
    """
    ref = _water()
    exact = np.asarray(_rotational_constants(ref, MASSES), dtype=float)
    replacement, _ = torsion_replacement_from_scan(
        ELEMENTS, _flat_scan(exact * 0.99), _spec(), ref, masses=MASSES)
    for k in ("A", "B", "C"):
        assert replacement.alpha_mhz[k] > 0.0, (
            f"{k}: lowering B gave a negative alpha -- the sign is inverted"
        )


def test_uncertainty_carries_the_same_factor_of_two():
    """sigma is an uncertainty on <B>, so it scales exactly as alpha does."""
    ref = _water()
    exact = np.asarray(_rotational_constants(ref, MASSES), dtype=float)
    scan = _flat_scan(exact)
    for gp in scan.grid_points:
        gp.sigma_abc = np.array([10.0, 2.0, 1.0])

    replacement, result = torsion_replacement_from_scan(
        ELEMENTS, scan, _spec(), ref, masses=MASSES)
    sigma_avg = np.asarray(result["sigma_averaged"], dtype=float).ravel()[:3]
    for i, k in enumerate(("A", "B", "C")):
        assert replacement.sigma_mhz[k] == pytest.approx(2.0 * abs(sigma_avg[i]), rel=1e-9)
        assert replacement.sigma_mhz[k] >= 0.0


def test_it_returns_something_the_correction_chain_accepts():
    """End of the chain: the result must be a usable TorsionReplacement."""
    ref = _water()
    exact = np.asarray(_rotational_constants(ref, MASSES), dtype=float)
    replacement, result = torsion_replacement_from_scan(
        ELEMENTS, _flat_scan(exact * 0.995), _spec(), ref, masses=MASSES,
        mode_indices=(0,))
    assert isinstance(replacement, TorsionReplacement)
    assert replacement.modes_for([120.0, 1600.0, 3800.0]) == [0]
    # the averaging result comes back too, so a caller can report provenance
    assert "weights" in result and "torsional_energies_cm1" in result
    assert np.isclose(float(np.sum(result["weights"])), 1.0, atol=1e-9)


def test_the_whole_chain_runs_from_scan_to_correction():
    """Scan -> averaging -> replacement -> correction table, in one go.

    The two halves are tested separately (here, and in test_torsion_bridge),
    but the point of this work was that nothing joined them. This is the join:
    a torsional scan must be able to change the correction a fit actually sees,
    with no hand-computed quantity in between.
    """
    import dev.analytic_water_backend  # noqa: F401  (registers analytic_water)
    from backend.registry import get_backend
    from backend.spectral.harmonic_alpha import build_correction_table_from_hessian

    ref = _water(r=0.95785, theta_deg=104.508)
    exact = np.asarray(_rotational_constants(ref, MASSES), dtype=float)
    # A scan whose average sits 1% below the reference: a floppy coordinate
    # raising the moments of inertia, which is the physical case.
    replacement, _ = torsion_replacement_from_scan(
        ELEMENTS, _flat_scan(exact * 0.99), _spec(), ref, masses=MASSES,
        mode_indices=(0,))

    hess = get_backend("analytic_water")(elems=ELEMENTS).run_hessian(ref).hessian_bohr
    isos = [{"name": "H2-16O", "masses": MASSES.tolist(),
             "component_indices": [0, 1, 2]}]

    base, _ = build_correction_table_from_hessian(hess, ref, isos)
    with_torsion, _ = build_correction_table_from_hessian(
        hess, ref, isos, torsion_replacement=replacement)

    def alphas(table):
        entry = table["H2-16O"]
        return {k: float(v["alpha_sum_mhz"]) for k, v in entry.items()
                if isinstance(v, dict) and "alpha_sum_mhz" in v}

    a_base, a_tors = alphas(base), alphas(with_torsion)
    assert a_tors != a_base, "the scan never reached the correction table"
    # The replacement's own contribution is +2% of each reference constant
    # (-2 x -1%), large enough that it must dominate the mode it replaced.
    assert a_tors["A"] > a_base["A"], "the torsional contribution went in with the wrong sign"
