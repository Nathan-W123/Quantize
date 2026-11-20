# Current task
Task: T-07
Status: clarifying
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
- N1 · plan · design choices Q-03 (entry type, classes, reference rule) · after: none · running · handoff: pending answer
- N2 · explore · literature search for candidates per class · after: none · done · handoff: C-Cl, C-N, O-O/O-H found; N-H none meeting the rule
- N3 · gate · report candidates; approval before adding molecules / rerunning offsets rows · after: N1, N2 · pending · handoff: -

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
- N-H: nothing readable meets the rule. NH3 r_e 1.0110 A / 106.70 deg
  (Demaison, Margules & Boggs, Chem. Phys. 260, 65-81, 2000, "The
  equilibrium N-H bond length"), no uncertainty on the secondary page
  (NQCC); paper paywalled. Formamide (Demaison et al. 2007, "reBO") and
  methanimine (Margules et al., SE) likewise without a readable sd.
- Not tried: HNCS r_e^SE (McGuire et al. 2016, values only in a figure /
  SI).


## Observations

## Pending question
- Q-03 · Q1 entry type: 1 calibration-only (Recommended), 2 full benchmark
  molecule; Q2 classes: 1 C-Cl, N-H, C-N + O-H, O-O (Recommended), 2 only the
  three; Q3 references: 1 r_e / r_e^SE with stated sd <= 0.5 mA
  (Recommended), 2 also r_s / effective · deadline 01:51 UTC 2026-11-20 ·
  wake-up trig_014JWXTTmSnEhnzQ71J3SxJz

## Handoff
