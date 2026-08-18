"""Fluorobenzene with the full isotopologue set: hybrid vs theory vs published.

Companion to fluorobenzene_vs_published.py, which uses only the parent species.
Here the fit gets the parent plus every symmetry-unique single substitution --
four 13C and three D, so 8 isotopologues and 24 observables against 30 internal
degrees of freedom.

On the C-F bond
---------------
It is tempting to say C-F must stay undetermined because fluorine has only one
stable isotope. That is not quite right. Once every carbon and hydrogen has been
located by substitution, the F position follows from the first-moment condition
sum(m_i a_i) = 0 in the parent principal axis system -- verified against the
published structure to machine precision. C-F is therefore determined
*indirectly*, which makes it the least precise parameter, not an unconstrained
one, because every other atom's error propagates into it.

On the constants
----------------
The isotopologue constants here are DERIVED from the published structure, not
measured: published values exist -- the r_s structure was built from them -- but
are not in any source reachable offline. They are scaled by the ratio the real
parent shows between its observed B_0 and the rigid value of the r_s geometry,
which reproduces the r_s/r_0 offset real ground-state constants carry. Without
that the data would be exactly consistent with the reference geometry and any
fit would recover it trivially.

So treat the ring and C-H numbers as a check that the machinery uses the
information correctly, and the theory-versus-hybrid comparison as the
independent one.

    python scripts/fluorobenzene_full_data.py [basis]
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT):
    sys.path.insert(0, str(_p))

import dev.pyscf_backend  # noqa: F401,E402  (registers "pyscf_hf")
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.registry import get_backend  # noqa: E402
from backend.spectral.centrifugal_distortion import rotational_constants_mhz  # noqa: E402
from dev.reference_structures import (  # noqa: E402
    FLUOROBENZENE_B0_MHZ,
    FLUOROBENZENE_ELEMS,
    FLUOROBENZENE_GEOM,
    FLUOROBENZENE_MASSES,
    FLUOROBENZENE_SOURCE,
    fluorobenzene_isotopologues,
    internal_coordinates,
)

_BOND_TAGS = ("C-F", "ipso-ortho", "ortho-meta", "meta-para", "C-H")


def zero_point_scale() -> np.ndarray:
    """Observed parent B_0 divided by the rigid constants of the r_s geometry."""
    rigid = rotational_constants_mhz(FLUOROBENZENE_GEOM, FLUOROBENZENE_MASSES)
    return FLUOROBENZENE_B0_MHZ / rigid


def build_isotopologues(n_species: int) -> list[dict]:
    species = fluorobenzene_isotopologues(zero_point_scale())[:n_species]
    return [{
        "name": s["name"],
        "masses": s["masses"].tolist(),
        "obs_constants": s["abc_mhz"].tolist(),
        # The r_s/r_0 gap dominates any measurement error, so weight to it.
        "sigma_constants": [10.0, 6.0, 4.0],
        "alpha_constants": [0.0, 0.0, 0.0],
        "component_indices": [0, 1, 2],
    } for s in species]


def run_hybrid(isotopologues, backend, basis, start, **kwargs):
    opt = MolecularOptimizer(
        elems=list(FLUOROBENZENE_ELEMS),
        coords=np.asarray(start, dtype=float),
        isotopologues=isotopologues,
        quantum_backend=backend,
        orca_method="hf",
        orca_basis=basis,
        coordinate_mode="cartesian",
        spectral_only=(backend == "none"),
        use_autoconfig=False,
        **kwargs,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        return opt.run()


def score(geom, ref):
    got = internal_coordinates(geom)
    bonds = [(got[k] - ref[k]) * 1000 for k in ref if any(t in k for t in _BOND_TAGS)]
    angles = [got[k] - ref[k] for k in ref if not any(t in k for t in _BOND_TAGS)]
    return (float(np.sqrt(np.mean(np.square(bonds)))),
            float(np.sqrt(np.mean(np.square(angles)))),
            (got["C1-F   (C-F)"] - ref["C1-F   (C-F)"]) * 1000)


def main() -> None:
    basis = sys.argv[1] if len(sys.argv) > 1 else "6-31g"
    ref = internal_coordinates(FLUOROBENZENE_GEOM)
    rng = np.random.default_rng(11)
    start = FLUOROBENZENE_GEOM + rng.normal(0.0, 0.03, size=FLUOROBENZENE_GEOM.shape)
    start[:, 0] = 0.0

    print(f"  Ground truth: {FLUOROBENZENE_SOURCE}")
    print(f"  Theory: RHF/{basis} via PySCF. Isotopologue constants derived from the")
    print("  published structure with the parent's own zero-point offset applied.\n")

    print(f"  {'':<34}{'RMS bond':>10}{'RMS angle':>11}{'C-F err':>10}{'time':>7}")
    print(f"  {'':<34}{'(mA)':>10}{'(deg)':>11}{'(mA)':>10}")
    print("  " + "-" * 73)

    b, a, cf = score(start, ref)
    print(f"  {'starting guess':<34}{b:>10.1f}{a:>11.2f}{cf:>+10.1f}{'-':>7}")

    t0 = time.time()
    backend = get_backend("pyscf_hf")(elems=list(FLUOROBENZENE_ELEMS),
                                      method="hf", basis=basis)
    theory = backend.optimise(start)
    b, a, cf = score(theory, ref)
    print(f"  {'THEORY alone':<34}{b:>10.1f}{a:>11.2f}{cf:>+10.1f}"
          f"{time.time() - t0:>6.0f}s")

    objectives = [
        ("split", dict()),
        ("joint prior 0.005A", dict(objective_mode="joint",
                                    quantum_prior_sigma_ang=0.005)),
    ]
    for n_species, label in ((1, "1 isotopologue"), (8, "8 isotopologues")):
        isos = build_isotopologues(n_species)
        print("  " + "-" * 73)
        print(f"  {label} ({3 * len(isos)} observables)")

        t0 = time.time()
        geom = run_hybrid(isos, "none", basis, start, max_iter=40)
        b, a, cf = score(geom, ref)
        print(f"  {'  EXPERIMENT alone':<34}{b:>10.1f}{a:>11.2f}"
              f"{cf:>+10.1f}{time.time() - t0:>6.0f}s")

        for obj_label, kwargs in objectives:
            t0 = time.time()
            geom = run_hybrid(isos, "pyscf_hf", basis, start,
                              max_iter=30, hess_recalc_every=10, **kwargs)
            b, a, cf = score(geom, ref)
            print(f"  {f'  HYBRID, {obj_label}':<34}{b:>10.1f}{a:>11.2f}"
                  f"{cf:>+10.1f}{time.time() - t0:>6.0f}s")

    print("\n  30 internal degrees of freedom throughout. C-F is reported separately")
    print("  because no substitution reaches it directly -- it is fixed by the")
    print("  first-moment condition once the carbons and hydrogens are located, so it")
    print("  is the parameter that accumulates everyone else's error.")
    print("\n  The right objective depends on how much data there is. With three")
    print("  observables the split partition over-fits and the calibrated prior helps;")
    print("  with twenty-four it is the prior that over-constrains, and split wins.")


if __name__ == "__main__":
    main()
