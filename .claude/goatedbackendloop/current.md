# Current task
Task: T-75
Status: building
Request: Find more validation molecules (published equilibrium structures + isotopologue rotational constants) for the benchmark (D-152 item 2).
## Brief
- Outcome: a ranked, source-checked list of candidate molecules the benchmark can score against, each with an equilibrium-structure source (r_e or r_e^SE, its stated uncertainty, open or not) and an isotopologue-constant source (count, open or not, e.g. CDMS/JPL catalog or paper SI).
- Criteria (from dev/monofluoro_references.py): semirigid, <= 12 atoms, a dipole (microwave constants), >= 2 isotopologues with published B0 and sigmas; reference uncertainty <= 0.5 mA on bonds (RE_REFERENCE_MAX_UNCERTAINTY_MA), unless Q-106 says otherwise; numbers read off the source, never guessed.
- Approach: three parallel web-search nodes split by family; merge and spot-check sources myself; gate: the user picks which to wire in (transcription is the next task).
- Already searched (N3, module notes): SE100 (no sd), Piccardo 2015, Gambi 2019, NQCC, CCCBDB, Demaison 2007 review (HAL); nothing for HCOF, fluoroethane, acetyl fluoride; HNCO SE100 only.
## Graph
- N1 · plan · inventory + criteria · after: none · done · handoff: 4 scored r_e (water, ozone, VF, CH2ClF) + 3 calibration-only (CH3Cl, HCN, H2O2); bar 0.5 mA
- N2 · explore · search: fluorine/halogen small molecules · after: N1 · running
- N3 · explore · search: classic small polyatomics with dipoles · after: N1 · running
- N4 · explore · search: larger semirigid organics (<= 12 atoms) · after: N1 · running
- N5 · merge · rank, dedupe, spot-check two sources per tier · after: N2, N3, N4 · pending
- N6 · gate · user picks which to wire in · after: N5 · pending
## Progress
## Observations
## Pending question
- Q-105/Q-106 (asked 00:18 UTC, deadline 00:33): scope (recommended: any semirigid <= 12 atoms with a dipole) and reference bar (recommended: keep 0.5 mA for headline scoring, list <= 1 mA as a second tier). Wake-up trig_01Va7sMh5HXd4G2we7kyLrTA.
## Handoff
