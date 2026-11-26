"""Validation set: molecules no calibration in this repository has seen.

The nine benchmark molecules (dev.monofluoro_references) are the ones every
default was tuned on, and the three calibration-only references feed the
bond-class offsets. A score on them is partly a score of the tuning. These
molecules are held out of both: nothing is fitted to them, they are not in the
offsets pool, and the benchmark scores them in a results file of its own
(``scripts/accuracy_upgrades_benchmark.py scored=validation``), so the existing
table cannot change by adding them (decisions D-154).

Every molecule has a semi-experimental equilibrium structure (r_e^SE) with a
stated uncertainty, so the score measures the engine against the quantity it
targets rather than an r_s or r_0 yardstick, plus the ground-state A0, B0, C0
of several isotopologues with their stated uncertainties.

    formaldehyde      H2CO   r_e^SE, Penocchio et al. 2016             6 species
    thioformaldehyde  H2CS   r_e^SE, Muller et al. 2019                8 species
    ketene            H2CCO  r_e^SE, Smith et al. 2023                16 species
    sulfur_dioxide    SO2    r_e, Flaud & Lafferty 1993 via Demaison   3 species
    pyrimidine        C4H4N2 r_e^SE, Heim et al. 2020                 16 species
    thiophene         C4H4S  r_e^SE, Esselman et al. 2025;            24 species
                             constants Orr et al. 2021

The numbers live in dev/validation_molecules/<name>.py, transcribed from
downloaded sources, with <name>.numbers.json beside each giving the table, row,
column and exact source line of every number. Independent re-reads against
their own downloads found no transcription discrepancy; two found that the
shared M_O18 is 1.4e-6 u above the current NIST value, which is kept (D-156).

Conventions decided here, not in the sources
--------------------------------------------
* The reference bar (RE_REFERENCE_MAX_UNCERTAINTY_MA, 0.5 mA) is read at
  1-sigma, as the existing references quote it. A source that quotes 2-sigma is
  halved: ketene's 0.69 mA becomes 0.345, pyrimidine's 0.45 becomes 0.225,
  thiophene's 0.3 becomes 0.15 (D-155). The entry modules keep the
  figure as printed; the conversion is in ONE_SIGMA_UNCERTAINTY_MA below.
* Sulfur dioxide is scored on its three species with fitted, stated-sigma
  constants (32S, 33S, 34S16O2). The other 16 ground-state species in its
  source are Kivelson-Wilson determinable constants printed without
  uncertainties; they stay in the entry module but are not scored (D-155).
  Formaldehyde's HDCO and D2CO also lack a printed sigma, but they are modern
  S-reduction fits of the same kind as the other species, so they are kept and
  floored at their quoted precision like any constant without a stated sigma.
* Thiophene's 2025 study prints its constants without uncertainties, so its
  entry scores the same group's 2021 S-reduced constants, which print them,
  for the 24 species that study's structure used (D-158).
"""

from __future__ import annotations

import numpy as np

from dev.monofluoro_references import Isotopologue, ReferenceMolecule
from dev.validation_molecules import (
    formaldehyde,
    ketene,
    pyrimidine,
    sulfur_dioxide,
    thioformaldehyde,
    thiophene,
)

#: The entry modules, in the order the set is reported.
ENTRY_MODULES = (formaldehyde, thioformaldehyde, ketene, sulfur_dioxide, pyrimidine,
                 thiophene)

#: The raw entries, with everything the ReferenceMolecule leaves out: each
#: species' source and note, and a second equilibrium structure where the
#: sources give one (formaldehyde's purely experimental r_e, ketene's 1995 one,
#: pyrimidine's two other r_e^SE columns, thiophene's 2021 structure and its
#: 2025 constants, SPECIES in its module).
ENTRIES = {m.ENTRY["key"]: m.ENTRY for m in ENTRY_MODULES}

#: 1-sigma bond uncertainty (mA) where the source quotes another level (D-155).
ONE_SIGMA_UNCERTAINTY_MA = {
    # Smith et al. 2023 Table IV, r(C-C) 1.31218(69) A, 2-sigma as the
    # abstract states.
    "ketene": 0.345,
    # Heim et al. 2020 Table IV, r(C4-C5) 1.38675(45) A, 2-sigma (footnote c,
    # abstract, Fig. 5 caption). The cross-ring C2...C5 parameter, 0.52 mA at
    # 2-sigma, is not a bond and not scored.
    "pyrimidine": 0.225,
    # Esselman et al. 2025 Table IV, r(C3-C4) 1.4222(3) A (a dependent
    # parameter; the largest independent one is r(C2-C3), 0.2), 2-sigma
    # (footnote c).
    "thiophene": 0.15,
}

#: Species scored, by label, where not every transcribed species is (D-155).
SCORED_SPECIES = {
    "sulfur_dioxide": ("32S16O2", "33S16O2", "34S16O2"),
}


def _isotopologue(sp: dict) -> Isotopologue:
    sigma = sp.get("measured_sigma_mhz")
    return Isotopologue(
        label=str(sp["label"]),
        subs={int(i): float(m) for i, m in sp["subs"].items()},
        abc_mhz=tuple(float(v) for v in sp["abc_mhz"]),
        decimals=tuple(int(d) for d in sp["decimals"]),
        measured_sigma_mhz=None if sigma is None else tuple(float(s) for s in sigma),
    )


def reference_molecule(entry: dict) -> ReferenceMolecule:
    """The ReferenceMolecule for one entry, with this module's conventions."""
    key = entry["key"]
    species = list(entry["species"])
    keep = SCORED_SPECIES.get(key)
    if keep is not None:
        by_label = {sp["label"]: sp for sp in species}
        missing = [lbl for lbl in keep if lbl not in by_label]
        if missing:
            raise KeyError(f"{key}: scored species not in the entry: {missing}")
        species = [by_label[lbl] for lbl in keep]
    if species[0]["subs"]:
        raise ValueError(f"{key}: the first species must be the parent")
    return ReferenceMolecule(
        key=key,
        name=entry["name"],
        formula=entry["formula"],
        elems=list(entry["elems"]),
        geometry=np.asarray(entry["geometry"], dtype=float),
        masses=np.asarray(entry["masses"], dtype=float),
        species=[_isotopologue(sp) for sp in species],
        structure_source=entry["structure_source"],
        constants_source=entry["constants_source"],
        structure_type=entry["structure_type"],
        geometry_re_uncertainty_ma=ONE_SIGMA_UNCERTAINTY_MA.get(
            key, entry["geometry_re_uncertainty_ma"]),
        bonds={k: [tuple(p) for p in v] for k, v in entry["bonds"].items()},
        angles={k: [tuple(t) for t in v] for k, v in entry["angles"].items()},
    )


FORMALDEHYDE = reference_molecule(formaldehyde.ENTRY)
THIOFORMALDEHYDE = reference_molecule(thioformaldehyde.ENTRY)
KETENE = reference_molecule(ketene.ENTRY)
SULFUR_DIOXIDE = reference_molecule(sulfur_dioxide.ENTRY)
PYRIMIDINE = reference_molecule(pyrimidine.ENTRY)
THIOPHENE = reference_molecule(thiophene.ENTRY)

#: The validation set, in report order. Never pooled into the offsets.
VALIDATION_SET = [FORMALDEHYDE, THIOFORMALDEHYDE, KETENE, SULFUR_DIOXIDE, PYRIMIDINE,
                  THIOPHENE]
