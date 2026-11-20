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
- N5 · verify · offsets shifts per molecule; offsets fits for CH2ClF, HNCO, water, ozone vs r_e · after: N4 · done · handoff: CH2ClF big gain both levels; HNCO bonds better, angles worse; ozone worse at B3LYP (O-O bond order); others' sigma_x wider
- N6 · gate · Q-05: rerun offsets config for all 9 with calibration=extended? · after: N5 · running · handoff: pending answer

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
- Measurement (scratchpad calib/calib.py; offsets configuration as the
  benchmark runs it; default rows reproduce the archive to 4 decimals):
  theory offsets of the new entries, RHF / B3LYP (mA): CH3Cl C-Cl +85.6 /
  +25.4, C-H -9.9 / +6.8; HCN C-H -12.3 / +5.2, C-N -9.1 / +3.8; H2O2 O-O
  +10.0 / +3.3, O-H -7.4 / +12.0 (water's own O-H error: -8.1 / +10.9;
  ozone's O-O: -22.1 / -8.7, opposite sign to H2O2's: single vs 1.5-order).
  B3LYP fits, default -> extended, bond RMS vs r_e (HNCO vs SE100):
  CH2ClF 11.22 -> 6.00 (C-Cl +14.1 -> +7.4, C-F -13.3 -> -7.2); HNCO 2.03
  -> 1.74 (angle 0.68 -> 0.78); water 0.69 -> 0.46 (angle 0.13 -> 0.03);
  ozone 0.37 -> 1.02 (angle 0.060 -> 0.027). Other five: sigma_x 3.9-4.8 ->
  4.2-4.8 mA, C-H offsets move <= 0.3 mA.
  RHF first pass invalid for H2O2 (see review): CH2ClF 17.64 -> 2.94, HNCO
  20.01 -> 7.28 (angle 1.49 -> 2.79), water 0.65 -> 0.02, ozone 2.86 ->
  6.22; other five sigma_x 5.1-6.3 -> 7.0-7.6.
  RHF rerun with the guard (calib_hf2.log): H2O2 left out (0.274 A);
  CH2ClF 17.64 -> 2.91 (C-Cl +23.2 -> +1.0, C-F -18.5 -> +2.9, angle 0.67
  -> 0.89); HNCO 20.01 -> 7.14 (N-H +33 -> +8), angle 1.49 -> 2.78 deg;
  water 0.65 -> 0.64; ozone 2.86 -> 2.86; other five sigma_x 5.1-6.3 ->
  6.1-6.9 (not refitted).
- Independent review (fresh agent, brief + criteria + diff): default path
  and never-scored PASS. Found: HF/6-31G optimises H2O2 trans-planar
  (179.8 deg; B3LYP 117.25, r_e 113.5) -- confirmed; per-results-file lock
  let default + extended runs race on the shared theory cache; --table
  ignored the flag; archived hybrid_sigma_diagnostic.py pooled the whole
  cache; tests too weak (8 mutants survived). All fixed in 565d69e:
  conformer guard (aligned RMSD <= 0.10 A; ordinary 0.004-0.034, HF H2O2
  0.274), per-level lock, calext table, key-filtered archived script,
  stronger tests incl. main() end to end -- 8 / 8 mutations caught.
  Suite before the fixes: 1606 passed, 22 skipped.
- N-H: nothing readable meets the rule. NH3 r_e 1.0110 A / 106.70 deg
  (Demaison, Margules & Boggs, Chem. Phys. 260, 65-81, 2000, "The
  equilibrium N-H bond length"), no uncertainty on the secondary page
  (NQCC); paper paywalled. Formamide (Demaison et al. 2007, "reBO") and
  methanimine (Margules et al., SE) likewise without a readable sd.
- Not tried: HNCS r_e^SE (McGuire et al. 2016, values only in a figure /
  SI).


## Observations
- 2026-11-20: the independent verifier found the H2O2 conformer flip by
  running one HF optimisation; my own run had the trans geometry on disk
  and I never looked at its shape, only its bond lengths. Check a
  calibration geometry's shape (dihedrals), not just the bonds measured.
- 2026-11-20: 8 of my 10 first tests let real breaks through (verifier's
  mutants); L-15 again -- the main() wiring was untested.

## Pending question
- Q-05 · gate: 1 rerun only the offsets configuration, all 9 molecules,
  both levels, calibration=extended, into the _calext files; compare with
  the archived rows; no default / README change (Recommended); 2 same but
  without H2O2's O-O (bond-order mismatch with ozone; chosen after seeing
  results); 3 stop, keep it opt-in and recorded · deadline 02:45 UTC
  2026-11-20 · wake-up trig_01LNuWmimjxhg2jNJcYcKu88

## Handoff
