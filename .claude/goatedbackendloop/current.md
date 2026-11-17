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

## Observations
- 2026-11-17: code_vintage() reads `git status --porcelain`, so editing the
  loop records (tracked) while a benchmark starts stamps it dirty. Caught
  from the log header within a minute; runs moved to a frozen worktree.

## Handoff
