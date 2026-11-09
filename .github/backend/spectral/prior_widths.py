"""How far the quantum surface is trusted, by level of theory.

The joint objective weights the quantum prior against the spectroscopic data
through one number: sigma_x, the displacement over which the electronic surface
is believed. It is not a free knob. It is an estimate of the level of theory's
own geometry error, which is what makes alpha_q interpretable rather than
arbitrary -- trust the surface over the distance it is actually right about.

That number depends on the level and on nothing else, and this repository has
measured it. The values below are the all-bond RMS error of each level against
published reference structures over the benchmark set.

THE PROBLEM THIS MODULE FIXES. Those measurements lived in
scripts/mixed_estimation_baseline.py, a benchmark script, while the engine
itself shipped one flat constant for every level. Every benchmark passed
sigma_x explicitly and got the calibrated value; every CLI and browser run took
the flat 0.020. At B3LYP/cc-pVTZ that is a prior five times looser than the
measurement supports, so the fit leaned on the data in directions where theory
was the better witness. The shipped engine and the benchmarked engine were not
the same engine.

WHAT A WRONG sigma_x COSTS, measured. Too loose and the quantum term stops
holding the directions the spectrum cannot see. Too tight is worse: water's
B3LYP/6-31G(d) row went 1.36 -> 4.13 mA when sigma_x claimed 7 mA for a method
that was 10.89 mA out on O-H, because the fit was instructed to trust a bad
prior. Both failures are the same mistake -- asserting a precision the surface
does not have -- so an uncalibrated level falls back to the loose default
rather than to an optimistic guess.

Per-level, not per-molecule. ``bond_offsets.prior_sigma_for_molecule`` computes
a better width bond by bond, but it needs offsets measured across a calibration
set of molecules with known reference structures, which a production run of one
unknown molecule does not have. It stays a benchmark tool. This is the part
that can ship.
"""

from __future__ import annotations

#: All-bond RMS geometry error by (method, basis), in Angstrom, measured
#: against published reference structures over the benchmark set. Keys are
#: lowercase. Extend this when a level is benchmarked, not when one is guessed
#: at: an entry here is a claim that the number was measured.
PRIOR_SIGMA_BY_LEVEL: dict[tuple[str, str], float] = {
    ("hf", "6-31g"): 0.020,
    ("b3lyp", "6-31g(d)"): 0.007,
    ("b3lyp", "cc-pvtz"): 0.004,
}

#: Used when the level is not in the table. Deliberately the loosest calibrated
#: value rather than a mean or an extrapolation: an unmeasured level has earned
#: no confidence, and the failure mode of an over-tight prior is worse than
#: that of a loose one.
DEFAULT_PRIOR_SIGMA_ANG = 0.020


def normalise_level(method, basis) -> tuple[str, str]:
    """(method, basis) as the table keys them: lowercase, stripped."""
    return (str(method or "").strip().lower(), str(basis or "").strip().lower())


def level_for_backend(quantum_backend, orca_method=None, orca_basis=None,
                      psi4_method=None, psi4_basis=None) -> tuple[str, str]:
    """Which method/basis pair a backend will actually run at.

    MolecularOptimizer carries two pairs and picks between them by backend
    name: psi4 reads its own, and everything else -- ORCA, PySCF and any
    registered third-party backend -- reads the orca_* pair, which is why the
    benchmark scripts set orca_method while running PySCF.
    """
    if str(quantum_backend or "").strip().lower() == "psi4":
        return normalise_level(psi4_method, psi4_basis)
    return normalise_level(orca_method, orca_basis)


def prior_sigma_for_level(method, basis, default: float = DEFAULT_PRIOR_SIGMA_ANG):
    """Calibrated prior width for a level, and whether it was calibrated.

    Returns ``(sigma_ang, calibrated)``. The second value exists so callers can
    say which they used: silently substituting a default for a measurement is
    how the shipped engine drifted from the benchmarked one in the first place.
    """
    key = normalise_level(method, basis)
    if key in PRIOR_SIGMA_BY_LEVEL:
        return float(PRIOR_SIGMA_BY_LEVEL[key]), True
    return float(default), False
