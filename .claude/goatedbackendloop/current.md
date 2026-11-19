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

## Observations
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
