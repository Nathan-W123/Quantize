"""How much of the reported error is the engine, and how much is the ruler?

The benchmark scores every molecule against one published structure, and for
eight of nine that structure is a 1950s-60s substitution (r_s) or effective
(r_0) structure. The engine targets r_e. Those are different quantities: an
r_s bond length differs from r_e by roughly the Costain floor, a few
milliangstroms, which is the same size as the error being measured. So part
of every reported number is the reference rather than the fit, and the only
way to find out how much is to score the same answer against both.

Fluoroacetylene is the case where that can be done cleanly, because two
independent equilibrium structures exist for it and they agree with each other.

MEASURED, RHF/6-31G, offsets configuration (RMS bond error, mA):

                        r_s        r_e           r_e            r_e?
                      Tyler    Botschwina   Borro/Mills    Borro & Mills
                       1963       1994          1995          1994 (*)
    theory (prior)     5.03       3.60          3.29          4.66
    mixed estimation   4.66       3.54          3.33          4.35
    hybrid             4.45       2.61          2.41          3.69

The hybrid's error falls from 4.45 mA to 2.41 -- it halves -- with nothing
about the engine changed. Two milliangstroms of the reported figure was the
reference. Every r_e candidate beats the r_s one, so this does not depend on
picking a favourite paper.

It also understated the engine's benefit. Against r_s the hybrid improves on
pure theory by 12% (5.03 -> 4.45); against the agreeing r_e pair, by 26-27%
(3.29 -> 2.41). Scoring an r_e-targeting engine against an r_s reference
penalises the method twice: once in the absolute number and once in the margin.

(*) The fourth column is a determination reported by a search summary as Borro
    & Mills, J. Mol. Struct. (1994), whose C-H of 1.0555 A is within 0.2 mA of
    the r_s value and 4.8 mA from the other two equilibrium determinations.
    That smells like an r_s value transcribed into an r_e slot rather than a
    real disagreement, but it has not been checked against the typeset table,
    so it is carried here as the conservative bound rather than discarded.

PROVENANCE WARNING. The equilibrium bond lengths below were taken from
W. C. Bailey's nuclear-quadrupole-coupling compilation at nqcc.wcbailey.net,
which cites the primary papers but is itself a secondary source, and from
search summaries of those papers' abstracts. Neither is the typeset table.
dev/g_tensor_references.py already records four confirmed text corruptions
found in deposited abstracts for a comparable exercise, and its advice applies
here unchanged: for any number about to be frozen into a reference structure,
read the table. Nothing here is wired into monofluoro_references yet for
exactly that reason -- this script measures what the change would be worth,
which is what decides whether the library trip is justified.

What is known about the other molecules in the set:

    water            r_e already, and the only sub-mA result in the set
    fluoroacetylene  two r_e determinations, measured above
    vinyl fluoride   r_e exists: Demaison et al., "Ab initio anharmonic force
                     field and equilibrium structure of vinyl fluoride and
                     vinyl iodide", J. Mol. Spectrosc. (2006),
                     doi:10.1016/j.jms.2006.08.008 (PII S0022285206002165).
                     Its abstract states it re-examines the Hayashi & Inagusa
                     substitution structure -- which is precisely the
                     reference the benchmark currently uses -- so this is a
                     direct upgrade of the same molecule from the same data.
                     Numbers are paywalled; not retrieved.
    formyl fluoride  dev/re_fcho_ccsdt.json is a CCSD(T)/cc-pVTZ r_e computed
                     in this repository and used by nothing. Its own notes put
                     its method bias at 2-3 mA, so it is a better ruler than
                     an r_s structure but not a published one.
    isocyanic acid   only an r_z found (Fusina & Mills, J. Mol. Spectrosc.
                     1981, "The harmonic force field and rz structure of
                     HNCO"). Mass-dependent, so better than r_s, still not
                     r_e.
    ozone            the benchmark calls 1.2717 A / 116.78 deg an "r_s
                     structure", but those same numbers are quoted across the
                     literature as ozone's equilibrium geometry. If that label
                     is wrong then ozone is the second r_e in the set, and the
                     two r_e molecules are also the two best results -- which
                     would make the pattern in this file a good deal stronger.
                     Needs the primary source read, not a search.
    acetyl fluoride, fluoroethane, chlorofluoromethane
                     nothing found. These are also three of the four molecules
                     where theory alone beats the hybrid, so an r_e reference
                     for them would be worth more than for the others.

Run:  python scripts/reference_structure_sensitivity.py [method=hf] [basis=6-31g]
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT):
    sys.path.insert(0, str(_p))

import dev.pyscf_backend  # noqa: F401,E402  (registers "pyscf_hf")
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.registry import get_backend  # noqa: E402
from backend.spectral.bond_offsets import (  # noqa: E402
    leave_one_out_offsets,
    measure_offsets,
    offset_corrected_geometry,
    prior_sigma_for_molecule,
)
from backend.spectral.harmonic_alpha import (  # noqa: E402
    build_correction_table_from_hessian,
)
from dev.monofluoro_references import (  # noqa: E402
    FLUOROACETYLENE,
    MOLECULES,
    MOLECULES_SET2,
)
from scripts.mixed_estimation_baseline import (  # noqa: E402
    corrected_targets,
    mixed_estimation_fit,
)
from scripts.monofluoro_benchmark import build_isotopologues, start_geometry  # noqa: E402

METHOD, BASIS = "hf", "6-31g"
for _tok in sys.argv[1:]:
    if _tok.startswith("method="):
        METHOD = _tok.split("=", 1)[1]
    elif _tok.startswith("basis="):
        BASIS = _tok.split("=", 1)[1]

#: Alternative published structures, keyed to mol.bonds. Secondary-source
#: numbers -- see the PROVENANCE WARNING above before trusting a digit.
ALTERNATIVES = {
    "fluoroacetylene": {
        "r_s  Tyler & Sheridan 1963": {
            "C-H": 1.0553, "C#C": 1.1980, "C-F": 1.2790},
        "r_e  Botschwina & Seeger 1994": {
            "C-H": 1.0591, "C#C": 1.1961, "C-F": 1.2765},
        "r_e  Borro, Mills & Mose 1995": {
            "C-H": 1.0603, "C#C": 1.1962, "C-F": 1.2764},
        "r_e? Borro & Mills 1994 (unverified)": {
            "C-H": 1.0555, "C#C": 1.1955, "C-F": 1.2781},
    },
}


def rms_against(mol, coords, ref_bonds) -> float:
    """RMS bond error in mA against a bond-length dict rather than a geometry."""
    got = mol.internal_coordinates(np.asarray(coords, dtype=float))
    errs = [(got[k] - ref_bonds[k]) * 1000.0 for k in ref_bonds]
    return float(np.sqrt(np.mean(np.square(errs))))


def answers_for(mol):
    """Theory, mixed-estimation and hybrid geometries in the offsets config."""
    backend = get_backend("pyscf_hf")(elems=list(mol.elems),
                                      method=METHOD, basis=BASIS)
    with contextlib.redirect_stdout(io.StringIO()):
        theory = backend.optimise(start_geometry(mol))

    hess_cache: dict = {}

    def hessian_fn(coords_ang, _b=backend, _c=hess_cache):
        key = np.asarray(coords_ang, dtype=float).round(9).tobytes()
        if key not in _c:
            _c[key] = _b.run_hessian(coords_ang).hessian_bohr
        return _c[key]

    isos = build_isotopologues(mol, None)
    with contextlib.redirect_stdout(io.StringIO()):
        ctbl, _ = build_correction_table_from_hessian(
            hessian_fn(theory), theory, isos, hessian_fn=hessian_fn,
            cubic_scheme="normal_mode")

    # Leave-one-out bond-class offsets, exactly as the upgrades benchmark does,
    # so the numbers here are comparable with the ones published there.
    per_molecule: dict = {}
    for m in list(MOLECULES) + list(MOLECULES_SET2):
        b = get_backend("pyscf_hf")(elems=list(m.elems),
                                    method=METHOD, basis=BASIS)
        with contextlib.redirect_stdout(io.StringIO()):
            per_molecule[m.key] = measure_offsets(m, b.optimise(start_geometry(m)))
    loo = leave_one_out_offsets(per_molecule, mol.key)
    prior = offset_corrected_geometry(mol, theory, loo)
    sigma_x = prior_sigma_for_molecule(mol, per_molecule, mol.key)

    opt = MolecularOptimizer(
        elems=list(mol.elems), coords=np.asarray(prior, dtype=float),
        isotopologues=isos, quantum_backend="pyscf_hf",
        orca_method=METHOD, orca_basis=BASIS, coordinate_mode="cartesian",
        use_autoconfig=False, max_iter=40, hess_recalc_every=10,
        correction_table=ctbl, quantum_prior_sigma_ang=float(sigma_x),
        prior_target_coords=np.asarray(prior, dtype=float),
        chi2_rescale=True, chi2_rescale_max_passes=3)
    with contextlib.redirect_stdout(io.StringIO()):
        hyb = opt.run()

    with contextlib.redirect_stdout(io.StringIO()):
        targets = corrected_targets(mol, isos, ctbl)
        me, _chi2 = mixed_estimation_fit(mol, targets, prior, float(sigma_x))

    return {
        "theory (prior)": np.asarray(prior, dtype=float),
        "mixed estimation": np.asarray(me, dtype=float),
        "hybrid": np.asarray(hyb, dtype=float),
    }


def main() -> None:
    print(f"  {METHOD.upper()}/{BASIS}, offsets configuration\n")
    for mol in (FLUOROACETYLENE,):
        refs = ALTERNATIVES[mol.key]
        answers = answers_for(mol)
        names = list(mol.bonds)

        print(f"{mol.name}: bond lengths (A)\n")
        print(f"{'structure':<38}" + "".join(f"{n:>10}" for n in names))
        for lab, d in refs.items():
            print(f"{lab:<38}" + "".join(f"{d[n]:>10.4f}" for n in names))
        print()
        for lab, c in answers.items():
            ic = mol.internal_coordinates(c)
            print(f"{lab:<38}" + "".join(f"{ic[n]:>10.4f}" for n in names))

        print(f"\n{mol.name}: RMS bond error (mA) against each reference\n")
        print(f"{'answer':<20}" + "".join(f"{lab[:18]:>20}" for lab in refs))
        for lab, c in answers.items():
            row = "".join(f"{rms_against(mol, c, d):>20.2f}" for d in refs.values())
            print(f"{lab:<20}{row}")
        print()


if __name__ == "__main__":
    main()
