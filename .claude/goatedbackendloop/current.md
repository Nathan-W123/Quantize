# Current task
Status: done
Request: N13 -- regenerate the README / benchmark results table at the
current code (RHF/6-31G and B3LYP/6-31G(d), all configurations).

## Brief
- Outcome: the README table and the benchmark docstring's table measured at
  one code vintage (5cfc5d4 or later), replacing September's numbers.
- Approach: scripts/accuracy_upgrades_benchmark.py as it stands, every
  same-level configuration (base, offsets, elec, lam, all, bob, offsets+bob,
  defect, offsets+defect, track, offsets+track, track_direct,
  track_quadratic, torsion, offsets+torsion; cd and offsets+cd at RHF only,
  to re-check they are still no-ops), into fresh results files: the old
  JSON and the theory-geometry cache (Nov 10, before ca3d8b2 changed the
  optimiser) are renamed aside, so no old-vintage row can be reused. corr
  is a second level of theory and stays out of a per-level table.
- Components: theory geometries -> correction table (Hessian disk cache,
  resumable) -> per configuration ME and hybrid fits -> --table.
- Tradeoff: "best configuration per leg" takes each molecule's minimum over
  ~17 configurations, which flatters all three methods and grows with the
  number of configurations; the base configuration alone (one setting for
  everything, no selection) is also reported so the comparison is clean.
- Success: both tables (best-of and base-only) regenerated from JSON rows of
  one vintage (warn_on_mixed_vintages silent), numbers copied into README
  and docstring with commit, threads and date; claims in the surrounding
  prose re-checked against the new numbers.
- Runs: RHF and B3LYP concurrently, OMP_NUM_THREADS=2 each (4 cores),
  exit codes captured (L-P03); B3LYP scope revisited once RHF timings show
  what track/torsion cost.

## Progress
- 17:22Z both levels started from a frozen detached worktree at 5cfc5d4
  (scratchpad n13/wt): the first start stamped its rows "5cfc5d4-dirty"
  because this records file is tracked and was modified, and a commit
  between pass 1 and pass 2 would have split the vintage. Logs and status:
  scratchpad n13/{hf,b3lyp}.pass{1,2}.log, *.status.
- 17:27Z RHF stopped on `elec`: No module named 'pyscf.prop'; pip install
  of pyscf-properties blocked by the permission check; user chose to leave
  elec and all out (D-29). geomeTRIC is missing too but nothing here uses
  it (torsion scans use SLSQP). 17:28Z both levels restarted without them;
  vinyl fluoride's 9 RHF rows (same vintage) kept.

- Side request (user, mid-N13): water at CCSD/cc-pVDZ -- theory alone, the
  joint hybrid on H2O + D2O ("full") and on H2O alone ("partial", user's
  choice), sigma_x 0.020 (default for an uncalibrated level), 0.010, 0.005.
  Script and log: scratchpad wccsd/water_ccsd.{py,log,json}. Theory alone
  r 0.96435 A (+6.58 mA vs r_e 0.95777), angle 102.21 (-2.27 deg).
  DONE (water_ccsd2.json; correction table 7 CCSD Hessians, 6 min; fits
  1-2 s, plateau at 11-12 it, restart from the answer moves 0.000 mA):
  full sx 0.020: -0.58 +- 1.71 mA, +0.164 +- 0.31 deg; partial (H2O):
  -0.67 +- 2.07, +0.199 +- 0.37. sx 0.010: full -0.38 / +0.13, partial
  -0.16 / +0.11; sx 0.005: full +0.30 / +0.01, partial +1.32 / -0.20.
  chi2/obs 0.05-0.8: widths are mostly the corrections' sigma. A first
  run labelled the angle sd (radians) as degrees and cached Hessians under
  Python's per-process hash(); both fixed in water_ccsd2.py.
  Data only (sx 1.0, water_ccsd_dataonly.json): full -0.64 / +0.176,
  partial -0.86 / +0.230 -- the residual error is the cc-pVDZ corrections'
  (the prior at the default width adds 0.06 / 0.19 mA); water has 6 (3)
  constants for 2 parameters, so it cannot show the prior's main value.
  Corrections at MP2/cc-pVTZ (own minimum, 7 Hessians, 3.7 min), prior and
  fit still CCSD/cc-pVDZ (water_corr.json): full -0.03 / +0.136, partial
  -0.02 / +0.163, data only -0.10 / +0.149 and -0.21 / +0.200. The bond's
  0.6 mA was the corrections' (now under the reference's 0.1 mA); the
  angle's +0.14-0.16 deg is not (0.4 sd). Corrections at CCSD/cc-pVTZ
  (33 min): full -0.08 / +0.102, partial -0.09 / +0.122, data only -0.15 /
  +0.115 and -0.28 / +0.159: the angle improves a little (MP2 0.136), the
  bond is indistinguishable (both under the reference's 0.1 mA).
- 18:51Z B3LYP pass 1: 1 of 9 molecules (~50 min each at 2 threads with
  the cores shared): several hours yet; RHF pass 2: 3 of 9.
- 17:59Z RHF pass 1 done (30 min; acetyl fluoride 11 min, fluoroethane
  8.6); pass 2 (track, torsion) started. B3LYP theory geometries done ~18:00.

- Partial-data test (user chose step 2), RHF/6-31G, parent isotopologue
  only, benchmark base recipe, caches from the frozen worktree
  (scratchpad partial/partial_data.py, partial.json). Full-data rows equal
  the benchmark's base rows to 4 decimals (all 9). Mean all-bond RMS (mA):
  theory 21.47; full hybrid 11.48, full ME 17.59; parent hybrid 13.76,
  parent ME 16.16, parent data-only (sx 1.0) 13.89. Parent hybrid beats
  theory 8/9 (not isocyanic acid, 26.84 -> 30.08) and parent ME 7/9 (not
  ozone, isocyanic acid); vs data-only 3 better, 2 tied, 4 worse: at HF the
  directions one isotopologue cannot see stay at theory with or without the
  prior (the start is the prior centre), so the prior's weight adds ~0.1 mA.
  Angles: water 7.06 -> 0.19, isocyanic 21.6 -> 1.28 deg (ME 19.6).

- User chose to repeat the partial-data test at B3LYP/6-31G(d):
  scratchpad partial/after_b3lyp.sh (detached) waits for the benchmark's
  "end pass 1 b3lyp", then runs partial_data.py with PD_METHOD=b3lyp
  (sigma_x 0.007) on the cached Hessians -> partial_b3lyp.json.

- 19:03Z RHF done (pass 2 64 min): best-of-15 theory 15.29 / ME 12.90 /
  hybrid 7.74 mA; base only 21.47 / 17.59 / 11.48 (hybrid beats theory 9/9,
  ME 5/9). Archived to reports/accuracy_upgrades (a7473f3).
- 19:04Z B3LYP restarted with THREADS=4 (cores freed; vinyl and acetyl
  fluoride rows and the Hessian cache kept); the wrapper was stopped before
  python so no "end pass 1" line fired the queued partial-data test early.

- 19:26Z container restart killed every process (scratchpad kept): B3LYP
  had 2/9 molecules (vinyl, acetyl fluoride; archived to reports/ as a
  partial file so a rerun skips them); restarted with THREADS=4, the
  partial-data chain re-armed.

- 20:15Z B3LYP pass 1 done (48 min for 7 molecules at 4 threads); pass
  2 (track, torsion) running; pass-1 rows archived (3b3278c).
- Partial-data test at B3LYP/6-31G(d) (sx 0.007, partial_b3lyp.json; full
  rows equal the benchmark's base rows): mean theory 9.19; all data hybrid
  4.74, ME 5.59; one isotopologue hybrid 4.90, ME 5.82, data-only 5.10.
  Hybrid beats theory 8/9, ME 6/9, data-only 6/9 (RHF: 3/9). The prior's
  weight starts to pay at the better level (0.2 mA mean), and one
  isotopologue with the hybrid comes within 0.16 mA of all of them; both
  differences are below the r_s references' resolution (N3).

- 03:25Z container restart had stopped B3LYP pass 2 at ~21:34Z (vinyl
  fluoride and acetyl fluoride's tracking rows kept, archived 5a244f6);
  pass 2 restarted on 4 threads.
- User chose the correction-level option (step 1): 2aa8755
  (correction_method / correction_basis, D-30). Engine path on water
  reproduces the scratch result (-0.033 mA / +0.1364 deg full, -0.026 /
  +0.1637 H2O only); default path bit-identical; clean suite 1630 passed.

- 02:20Z (11-19) another container restart; B3LYP pass 2 restarted.
- User chose step 4 (r_e references): chlorofluoromethane r_e^SE (Vogt,
  Demaison & Rudolph 2014, via Bailey's NQCC table; checked: B_e reproduce
  B0 + B3LYP corrections to 0.06%) wired, 221ef62; scored vs r_e: B3LYP
  hybrid 9.24 mA (effective 10.85), RHF 10.80 (13.38). Others: leads only
  (backlog N3). Clean suite 1631 passed.

- User chose "Read SE100's data": SI Table S1 has HNCO and CH2=CHF r_e^SE
  but no uncertainties (unwired, values in the module comment). Both
  reproduce B0 + B3LYP corrections like Vogt's control (HNCO 0.042%; vinyl
  fluoride B, C 0.035-0.057%, H5 cis to F as in the module). It exposed
  bad deuterated A0 values in vinyl fluoride's data (784, 788: negative
  inertial defects, 2-3% misses; compilation re-read, no transcription
  slip): backlog N15, not changed (benchmark inputs, user's call).

- User chose "Measure the A-row pull": base config, both levels, the
  benchmark's own fit path (all-rows runs reproduce the archived 5cfc5d4
  rows to the digit). Dropping A of 784/788: RHF hybrid 18.34 -> 13.63 mA
  vs ref (14.86 -> 10.98 vs SE100); B3LYP 6.20 -> 5.42 (4.62 -> 4.45).
  Dropping all D-species A: data-only lands 0.9-1.0 mA from SE100 at both
  levels. Robust losses don't fix it. Per-species defect: the parent's
  correction works (ratio 0.01); the round-number-A species drive the
  ratio. Corrected two misattributed claims (rovib_corrections
  _apply_defect_bias_floor docstring; README monofluoro paragraph).

- User chose "Down-weight those A rows": measured_sigma_mhz field, values
  910/160/1350/280/70 MHz from the inertial defects (D-31). All 22 rows,
  RHF hybrid 13.90 mA vs ref (11.05 vs SE100), B3LYP 5.48 (4.35): same as
  dropping 784/788. Data-only 17.9 (was 25.6). Tests: 7 new, 5 mutations
  caught; two pinned values updated with reasons; suite 1638 passed.
  Next: rerun vinyl fluoride rows (both levels, all configs) at the head
  commit once B3LYP pass 2 ends (n13/vf_rerun.sh), merge, then N13 tables.

- 06:25Z vinyl fluoride rerun done (72e3a7c, 28 rows): hybrid better in
  28/28 configs (RHF base 18.34 -> 13.90, B3LYP 6.20 -> 5.48); ME better
  except RHF defect (13.67 -> 14.69) and offsets+defect (6.41 -> 6.44);
  theory identical everywhere. Merged into reports/accuracy_upgrades
  (n13/merge_vf.py; index marks the mix). Pass 2 per molecule: 5 molecules
  banked by 06:20Z, CH2ClF running.

- 17:25Z container restart (pass 2 died in fluoroethane); relaunched
  fluoroethane + acetyl fluoride on 4 threads (MOLS= in chain_p2.sh).
- User chose "Check prior vs data balance". Brief: for water and ozone
  (independent r_e), CH2ClF (r_e^SE) and HNCO / vinyl fluoride (SE100,
  caveat), fit theory, ME, hybrid at sigma_x x1/x2/x5/x20, x1 without chi2
  rescale, data only; score vs r_e; record rank, chi2/nu, rescale events;
  x1 must reproduce the archived base row. Diagnostic only -- any change
  goes back to the user. Running: scratchpad balance/balance.py (1 thread).

- Prior-vs-data balance done (scratchpad balance/*.json; every hybrid x1
  reproduced its archived base row). Bond RMS mA vs yardstick, default
  hybrid / best looser hybrid / data only:
    independent r_e: water RHF 0.56 / 0.66 / 0.69, B3LYP 0.05 / 0.63 / 0.87;
    ozone RHF 3.13 / 2.75 / 2.76, B3LYP 0.06 / 0.34 / 0.43. Angles best at
    the default for both at both levels.
    SE-type: CH2ClF (Vogt, rank 5/9) RHF 10.80 / 8.68 / 8.56, B3LYP
    9.24 / 7.51 / 3.48; HNCO (SE100) RHF 22.36 / 7.28 / 7.06, B3LYP
    2.27 / 1.83 / 1.76; vinyl fluoride (SE100, stated sigmas) RHF 11.05 /
    9.48 / 17.10, B3LYP 4.35 / 4.39 / 17.10.
  Verdict: no evidence the default prior holds back fully determined fits --
  on the independent r_e it is the best option by 10-17x at B3LYP. The
  CH2ClF gap is not prior width (20x looser: C-F still -9.5 mA): both fits
  reproduce the data (chi2/nu 0.0009 vs 0.0000); the hybrid's joint
  objective picks a structure that moves C-F -11 mA, data-only's minimum-
  norm step puts it all in C-Cl (-0.8). chi2 rescale: floored by design for
  corrected targets, so it refits 3x doing nothing (only vinyl fluoride
  moves). No change made.

- User chose "Examine the split" (N16). CH2ClF B3LYP: under the x1
  hybrid's own Phi the hybrid answer is the optimum (Phi 13.57; data-only
  point 16.69; x20 14.05) and a restart from the data-only answer returns
  to it exactly -> not the solver; the chooser is the objective's quantum
  prior 1/2 w d^T H d (prior covariance ~ H^-1): it splits a correction the
  data cannot attribute by bond stiffness (C-Cl -28 + C-F -10 cheaper than
  C-Cl -39), which fails when the theory's error is concentrated in one
  bond (B3LYP C-Cl +38, C-F ~0). Offsets cannot help: C-Cl is a one-member
  class (leave-one-out leaves it), and bond_offsets.prior_sigma_for_molecule
  computes per-bond widths then collapses them to one RMS. Linear metric
  solve diverged (rigid modes not projected) -- not needed. Others (B3LYP,
  vs effective refs): hybrid vs data-only differ 4-6 mA bond RMS in formyl
  fluoride, fluoroethane, acetyl fluoride (HNCO 0.85); fluoroethane and
  acetyl fluoride show the same fix-one-bond-push-a-neighbour pattern
  (C-F -5 -> -14; C-C -2 -> -15) but data-only does it too there, and the
  references are old effective structures, so right/wrong is unknown.

- User chose "Find more references". Found: Piccardo et al. 2015 JPCA
  (open accepted manuscript, cris.unibo.it 11585/523965) Table 1 gives
  vinyl fluoride r_e^SE (CCSD(T) corrections) WITH fit sd (0.1-0.2 mA) --
  the source of SE100's numbers -- and its SI tabulates Hayashi & Inagusa
  1989's constants for 10 isotopologues. Checked: with those constants plus
  B3LYP corrections the r_e reproduces all 24 constants of our 8 species to
  0.034-0.086% (rms 0.058%); every inertial defect +0.084..+0.104. Wired
  vinyl fluoride's r_e (0.2 mA, rule met; tests + 3 mutations; suite 1639
  passed). Our deuterated A0 (NBS 1968) are off by 118-1350 MHz from the
  1989 values; D-31's defect-sized sigmas were 60-100% of the true errors.
  Nothing new for HCOF, fluoroethane, acetyl fluoride, HNCO (SE100 only,
  unstated) or HCCF (0.8 mA, too coarse). Pending user: replace the
  constants (N15).

- User approved replacing the constants: Hayashi & Inagusa 1989 for the 8
  species (24 constants, stated uncertainties as measured sigma), D-32
  supersedes D-31; tests rewritten (test_vinyl_fluoride_constants.py, 9
  tests, 2 mutations caught), default-sigma test restored, pinned residual
  0.93 -> 1.00; suite 1641 passed. Next: vinyl fluoride rerun at this
  commit (n13/wt_vf2), merge, then N13 tables.

- N17 done: run() stops the chi2 loop when a downward scale changes no
  sigma (rtol 1e-12) and says the floor binds. Measured first: the no-op
  refits cost 0.1-0.4 s per fit at RHF, no new Hessians, and moved results
  by <2e-7 A -- so the gain is an honest log, not speed. Answers unchanged
  (4 molecules bit-identical, ozone 2e-8 A). 2 tests (premise asserted
  after a fixture first widened instead of tightening), mutations caught;
  suite 1643 passed.

- User asked for a bloat audit (mid-task). Tools: vulture 2.16, pyflakes,
  an AST import graph from cli.py/runner (package __init__s counted), an
  AST duplicate-body scan, a kwargs-never-passed scan; every 'referenced
  nowhere' candidate re-checked with git grep. Findings (scratchpad
  audit/): (1) dead, zero behaviour: 79 unused imports, ~15 dead locals /
  write-only attributes (incl. quantize.py hessian_fn x2, _orca_call_count,
  _mass_dependent_c), 7 unreferenced defs (RunDiagnostics module,
  print_identifiability_table, symmetry.equiv_groups/supported_groups,
  VALID_BACKENDS, 2 unused script backends); (2) 11 investigation scripts
  (~3.1k lines) referenced by no doc, code or test; (3) 13 backend modules
  (~3.2k lines) unreachable from the CLI/runner -- torsion extras 1.4k,
  model_selection, prior_sensitivity, bayes_tune, geometric_bridge,
  rovib_tables, plus benchmark infra (hessian_cache, bond_offsets,
  electronic_g) -- and 47 functions only tests/scripts call; (4) 13 of
  143 optimizer kwargs never passed anywhere (guardrails active by
  default, so not dead). Copy-paste negligible (77 lines). Not dead:
  load_orca (named in ORCA error messages), get_run_settings.

- 19:20Z vinyl fluoride rerun at 676eb0d done and archived (28 rows, now
  also scored vs Piccardo r_e). vs r_e, base / offsets: RHF hybrid 11.39 /
  2.84 mA (theory 17.98 / 4.30, ME 10.70 / 3.12); B3LYP hybrid 3.36 / 1.03
  (theory 5.86 / 0.90, ME 3.87 / 2.27). defect == base now (consistent
  data). vs the effective reference everything sits ~5.2 mA.
- B3LYP pass 2 chain stopped at the 2 h background limit (it already had
  the maximum timeout, so not restarted). Banked: fluoroethane's 4 tracking
  rows. Missing: torsion + offsets+torsion for fluoroethane and acetyl
  fluoride (4 of 117 rows); a single B3LYP torsion config has outlived a
  3 h run before. Asked the user how to proceed.

- Cleanups done (user: "do 1-3"): efabab3 dead code; a2aafb9 ten scripts
  archived; 988bad5 CSV correction tables wired (real bug: validated, then
  failed in the engine) and five unused module groups deleted (-2265 lines
  incl. tests). Suite 1596 passed.
- N13 DONE (user: publish without the 4 B3LYP torsion rows): README Results
  and the benchmark docstring regenerated from reports/accuracy_upgrades --
  best-of RHF 15.29 / 12.92 / 7.76, B3LYP 8.12 / 4.98 / 4.20; base 21.47 /
  17.35 / 11.03 and 9.19 / 5.42 / 4.63; plus an r_e table for water, ozone,
  vinyl fluoride and CH2ClF (hybrid beats theory on all four at both levels,
  best of three on all four at B3LYP). Stale 1.36 -> 4.13 water claims
  corrected in prior_widths, holdout_validation, test_prior_widths.

## Observations
- 2026-11-19: I estimated N17's saving ("2 wasted refits") without
  measuring; measured, it was ~0.2 s and no QC calls. Measure cost before
  pitching an efficiency fix.
- 2026-11-19: a rescale test passed for the wrong reason (the fixture
  widened, the test assumed tightening); asserting the premise in the test
  caught it (L-15 again).
- 2026-11-19: classifying bonds vs angles by counting '-' in their names
  garbled C=O / O=C-F rows; caught by eye (a 451 'mA' angle). Use the
  molecule's own bonds / angles keys, never the label's spelling.
- 2026-11-19: the hypothesis I offered (prior holds back determined fits)
  came from an SE-type yardstick; the independent r_e reversed it. Rank the
  yardsticks by independence before proposing an engine change.
- 2026-11-19: reproducing the archived base rows to the digit before
  reporting the pull (L-P04) is what lets the drop-row numbers be read as
  the benchmark's own, not a script's.
- 2026-11-19: the SE100 consistency check, run per constant rather than as
  one RMS, separated a reference that is fine (all B, C at 0.04%) from bad
  input data (five A rows). An aggregate 0.67% would have read as "SE100
  disagrees". Break a failed consistency check down by row before judging
  the reference.
- 2026-11-19: I wrote a verification claim into a comment ("swapping the
  labels costs 0.3%") from a different computation; caught on re-read
  (the label swap is 5.8%). Tie each number in a comment to the run that
  produced it.
- 2026-11-19: a secondary table (NQCC) was the only readable source for
  the CH2FCl r_e; checking its numbers against the repo's own B0 and the
  engine's corrections (0.06%, inside the corrections' sigma) is what made
  it safe to wire in. Use a data-consistency check whenever the primary is
  not readable.
- 2026-11-17: code_vintage() reads `git status --porcelain`, so editing the
  loop records (tracked) while a benchmark starts stamps it dirty. Caught
  from the log header within a minute; runs moved to a frozen worktree.

- 2026-11-17: the internal-coordinate posterior gives angle widths in
  radians with no unit field; my script printed them as degrees (0.005 for
  0.29). Check a quantity's units against a known value before reporting.

## Handoff
Two background chains (scratchpad n13/), both resumable (finished rows are
skipped; each invocation banks its rows when it ends):
- chain_p2.sh: B3LYP pass 2 in worktree n13/wt (5cfc5d4), one molecule per
  invocation, cheapest first, acetyl fluoride (torsion, >3 h) last. Status in
  n13/b3lyp.status.
- chain_vf.sh: vinyl fluoride rerun in worktree n13/wt_vf (72e3a7c, D-31),
  hf then b3lyp, each split in two. Status in n13/vf.status.
After a container restart: check `pgrep -af accuracy_upgrades`; delete the
stale *.lock in that worktree's output/; relaunch the chain with THREADS=2
(both) or THREADS=4 (one alone). If a worktree is gone: `git worktree add
--detach <path> <sha>`, copy theory_geometries_*.json and hessian_cache from
the other worktree (or reports/accuracy_upgrades/*.json for rows).
When both are done: archive wt rows (scripts/archive_benchmark_results.py),
then replace the vinyl_fluoride entry in reports/accuracy_upgrades/*.json
with wt_vf's (vintage 72e3a7c; only D-31 differs from 5cfc5d4 for it), then
`--table results=...` (best-of and configs=base), README + docstring, noting
the vintage mix, elec/all excluded (D-29), CH2ClF r_e scoring.
