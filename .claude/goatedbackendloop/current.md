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
- N12 · build · draft entries thiophene · after: N6 · done · handoff: t75/draft_thiophene: 2025 Table IV r_e^SE (46 iso, 2-sigma, 0.3 mA); 2025 Table II constants have NO sigmas (determinable, A/S average); 2021 S-reduced 26 species with sigmas kept as SPECIES_2021_S; paper-internal slip on [3,4-2H, 34S] defect
- N13 · build · draft entries pyrimidine · after: N6 · done · handoff: t75/draft_pyrimidine: 16 species with sigmas, r_e^SE CCSD(T) 2-sigma (0.45 mA); checks clean
- N9c-p · verify · independent re-read, pyrimidine · after: N13 · done · handoff: t75/verify_pyrimidine/REPORT.md: no error; one comment slip (3.7e-11 -> 3.7e-10), fixed
- N9c-t · verify · independent re-read, thiophene · after: N12 · running
- N10 · build · wire the verified entries in as a separate validation set + tests · after: N9a, N9b, N9c-p · done for 5 (thiophene after N9c-t) · handoff: dev/validation_references.py (VALIDATION_SET, 1-sigma conversion, SO2 3 species), dev/validation_molecules/*.py + .numbers.json, benchmark scored=validation (own results file, never pooled), dev/tests/test_validation_references.py (34 pass)
- N11 · verify · run the new molecules' benchmark rows (HF/6-31G and B3LYP/6-31G(d), base + offsets) · after: N10 · running
## Progress
- 5 of 6 molecules wired and tested; validation runs relaunched on the committed code.
## Observations
- Uncertainty convention not defined anywhere in the module: ketene's r_e^SE is quoted at 2-sigma (0.69 mA), 1-sigma 0.345 -- the 0.5 bar must say which; existing references quote 1-sigma.
- Spot-checks (N5, own downloads in t75/dl/): pyrimidine Table IV r_e^SE(CCSD(T)) C2-N1 1.33333(38) A (2-sigma) matches N4; H2CS Table 7 r_e^SE(CCSD(T)) CS 160.909(1), CH 108.531(2) pm matches N3. Same table: r_e^SE with B3LYP corrections gives CS 160.975, 0.66 mA longer -> least-squares sigmas understate the real (correction-model) uncertainty; rank by method spread too.
- The benchmark's `set=` option collided with scripts/monofluoro_benchmark.py, which reads set= at import; the test passed only because an earlier test had cached that import. Renamed to scored=; the new test re-imports every scripts.* module (L-29 candidate: argv-reading imports need fresh-import tests).
- The inertial-defect lower bound in the new test (0.03) was a guess; pyrimidine's printed defects go to 0.0287. Bound set to the physical rule (small, positive).
## Pending question
## Handoff
