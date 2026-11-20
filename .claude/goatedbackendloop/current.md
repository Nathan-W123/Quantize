# Current task
Task: T-07
Status: verifying
Request: calibration molecules for the uncovered bond classes (backlog N16's
root fix), literature first, no engine change (D-34, auto-picked at Q-02).

## Brief
- Outcome: candidate molecules, with sources, that would give C-Cl, N-H and
  C-N (and possibly O-H, O-O) a second member in the offsets pool.
- What a calibration entry needs (bond_offsets.measure_offsets): a reference
  geometry (mol.geometry) and a theory geometry per level; no constants or
  fits. Adding one also adds its C-H etc. to the shared pool, so every
  molecule's offsets move and the offsets rows need rerunning (gate).
- Known limit: the existing pool measures against each molecule's primary
  reference, mostly r_s / effective (+/-5 mA), not r_e (N3).

## Graph
- N1 · plan · design choices Q-03 (entry type, classes, reference rule) · after: none · done · handoff: D-35 (auto-picked)
- N2 · explore · literature search for candidates per class · after: none · done · handoff: C-Cl, C-N, O-O/O-H found; N-H none meeting the rule
- N3 · gate · Q-04: add CH3Cl, HCN, H2O2 and measure offsets? · after: N1, N2 · done · handoff: D-36 (auto-picked)
- N4 · build · calibration-only entries + theory geometries + pool · after: N3 · done · handoff: refs + calibration=extended + offsets_pool; 10 tests, 3 mutations caught
- N5 · verify · offsets shifts per molecule; offsets fits for CH2ClF, HNCO, water, ozone vs r_e · after: N4 · running · handoff: scratchpad calib/calib_{hf,b3}.log
- N6 · gate · rerun the full offsets rows and README tables · after: N5 · pending · handoff: -

## Progress
- Candidates meeting the rule (r_e / r_e^SE, stated sd <= 0.5 mA):
  - C-Cl: CH3Cl r_e^SE, Melosso et al., Molecules 30, 1604 (2025),
    doi 10.3390/molecules30071604, Table 5 (CCSD(T) vib. corrections, 14
    isotopologues, 95% CI): C-Cl 1.777716(10), C-H 1.083484(17) A,
    Cl-C-H 108.3723(18) deg. Agrees with ref. 61 (Pietropolli Charmet et al.,
    JQSRT 305, 108624, 2023: 1.777725(11), 1.083450(17), 108.3740(19)) and
    CCSD(T)/CBS composite 1.77792 / 1.08339 / 108.384. Read from the open
    PDF (extracted locally; scratchpad dl_ch3cl/).
  - C-N: HCN r_e, Carter, Mills & Handy, J. Chem. Phys. (1992), "The
    equilibrium structure of HCN": C-H 1.06501(8), C-N 1.15324(2) A (8
    isotopologues, variational B0 - Be). From the abstract (search result);
    full text not read. C-N class would pool C#N with HNCO's N=C (bond
    order ignored by design, bond_class).
  - O-O, O-H: H2O2 r_e^SE, Baraban, Changala & Stanton, J. Mol. Spectrosc.
    343, 92-95 (2018): O-O 1.4523(2), O-H 0.9618(2) A, HOO 99.78(4),
    HOOH 113.5(2) deg -- as tabulated in Herberth, Giesen & Yamada, JMS 362,
    37 (2019) (arXiv 2105.09186, Table 5), whose own fit agrees: 1.4518(6),
    0.962(1), 100.0(1), 113.7(2). Baraban's paper itself not read.
- Built (D-36): dev/monofluoro_references.py METHYL_CHLORIDE,
  HYDROGEN_CYANIDE, HYDROGEN_PEROXIDE, CALIBRATION_ONLY (species=[],
  structure_type r_e); scripts/accuracy_upgrades_benchmark.py
  calibration=extended (rows to *_calext.json) and offsets_pool(theory, keys).
  dev/tests/test_calibration_references.py: 10 tests; breaking the CH3Cl
  builder, the H2O2 dihedral, or offsets_pool's key filter each fails one.
- N-H: nothing readable meets the rule. NH3 r_e 1.0110 A / 106.70 deg
  (Demaison, Margules & Boggs, Chem. Phys. 260, 65-81, 2000, "The
  equilibrium N-H bond length"), no uncertainty on the secondary page
  (NQCC); paper paywalled. Formamide (Demaison et al. 2007, "reBO") and
  methanimine (Margules et al., SE) likewise without a readable sd.
- Not tried: HNCS r_e^SE (McGuire et al. 2016, values only in a figure /
  SI).


## Observations

## Pending question

## Handoff
