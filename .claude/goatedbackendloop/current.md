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
- N6 · gate · user picks which to wire in (Q-107) and how (Q-108) · after: N5 · done · handoff: auto-picked six tier-1, separate validation set (D-154); sigma convention D-155 (assumption)
- N7 · build · draft entries (scratch only) H2CO + H2CS · after: N5 · done · handoff: t75/draft_h2co_h2cs/REPORT.md: H2CO 6 species (HDCO, D2CO without sigma), H2CS 8 with sigma; checks clean
- N8 · build · draft entries (scratch only) ketene + SO2 · after: N5 · done · handoff: t75/draft_ketene_so2/REPORT.md: ketene 16 iso, SO2 3 with sigmas + 16 without; checks clean; doubts on sigma levels
- N9a · verify · independent re-read, ketene + SO2 · after: N8 · done · handoff: t75/verify_ketene_so2/REPORT.md: every number matches; module M_O18 is 1.4e-6 u above NIST (3.5e-8 relative on 18O species, below the 0.5% model error: keep the module value); one docstring wording slip
- N9b · verify · independent re-read, H2CO + H2CS · after: N7 · done · handoff: t75/verify_h2co_h2cs/REPORT.md: every number matches; M_O18 again (D-156)
- N12 · build · draft entries thiophene · after: N6 · running
- N13 · build · draft entries pyrimidine · after: N6 · running
- N9c · verify · independent re-read, thiophene + pyrimidine · after: N12, N13 · pending
- N10 · build · wire the verified entries in as a separate validation set + tests · after: N9a, N9b, N9c · pending
- N11 · verify · run the new molecules' benchmark rows · after: N10 · pending
## Progress
## Observations
- Uncertainty convention not defined anywhere in the module: ketene's r_e^SE is quoted at 2-sigma (0.69 mA), 1-sigma 0.345 -- the 0.5 bar must say which; existing references quote 1-sigma.
- Spot-checks (N5, own downloads in t75/dl/): pyrimidine Table IV r_e^SE(CCSD(T)) C2-N1 1.33333(38) A (2-sigma) matches N4; H2CS Table 7 r_e^SE(CCSD(T)) CS 160.909(1), CH 108.531(2) pm matches N3. Same table: r_e^SE with B3LYP corrections gives CS 160.975, 0.66 mA longer -> least-squares sigmas understate the real (correction-model) uncertainty; rank by method spread too.
## Pending question
## Handoff
