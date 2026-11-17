# Current task
Status: building
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

## Observations
- 2026-11-17: code_vintage() reads `git status --porcelain`, so editing the
  loop records (tracked) while a benchmark starts stamps it dirty. Caught
  from the log header within a minute; runs moved to a frozen worktree.

- 2026-11-17: the internal-coordinate posterior gives angle widths in
  radians with no unit field; my script printed them as degrees (0.005 for
  0.29). Check a quantity's units against a known value before reporting.

## Handoff
