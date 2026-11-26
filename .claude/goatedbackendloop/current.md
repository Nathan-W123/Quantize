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
- N2 · explore · search: fluorine/halogen small molecules · after: N1 · done · handoff: t75/n2/REPORT.md: little with open sd + open B0; best CH2=CFCl (tier 2)
- N3 · explore · search: classic small polyatomics with dipoles · after: N1 · done · handoff: t75/n3/REPORT.md: H2CS, ketene, H2CO, SO2, HNCO (B,C only), HNO, H2CCS; key open source Penocchio et al. Can J Chem 2016
- N4 · explore · search: larger semirigid organics (<= 12 atoms) · after: N1 · done · handoff: t75/n4/REPORT.md: pyrimidine, thiophene, thiazole, ketene, pyridazine, pyridine, oxirane, acrolein, formic acid, triazole; McMahon/Woods 2-sigma papers + Piccardo 2015 SI
- N5 · merge · rank, dedupe, spot-check two sources per tier · after: N2, N3, N4 · done · handoff: tier 1 H2CO, H2CS, ketene, SO2, thiophene, pyrimidine, thiazole, pyridine; tier 2 pyridazine, triazole, oxirane, acrolein, formic acid, CH2=CFCl, HOF, HNO, HNCO(B,C); linear N2O, OCS, HC3N
- N6 · gate · user picks which to wire in (Q-107) and how (Q-108) · after: N5 · pending
- N7 · build · draft entries (scratch only) H2CO + H2CS · after: N5 · running
- N8 · build · draft entries (scratch only) ketene + SO2 · after: N5 · running
- N9 · verify · independent re-read of every transcribed number against its source · after: N7, N8 · pending
- N10 · build · wire the verified entries in (as Q-108 says) + tests · after: N6, N9 · pending
- N11 · verify · run the new molecules' benchmark rows · after: N10 · pending
## Progress
## Observations
- Spot-checks (N5, own downloads in t75/dl/): pyrimidine Table IV r_e^SE(CCSD(T)) C2-N1 1.33333(38) A (2-sigma) matches N4; H2CS Table 7 r_e^SE(CCSD(T)) CS 160.909(1), CH 108.531(2) pm matches N3. Same table: r_e^SE with B3LYP corrections gives CS 160.975, 0.66 mA longer -> least-squares sigmas understate the real (correction-model) uncertainty; rank by method spread too.
## Pending question
- Q-107 (which: recommended six tier-1) / Q-108 (how: recommended separate validation set), asked 00:44 UTC, deadline 00:59, wake-up see send_later.
## Handoff
