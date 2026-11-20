# Current task
Task: T-12
Status: building
Request: count how many base / offsets benchmark rows sit near the chi2
rescale switch (N19; D-46, auto-picked at Q-14); measurement only.

## Brief
- Outcome: for each of the 36 rows (9 molecules x base/offsets x HF/B3LYP),
  the first-pass reduced chi2, the effective dof, and how many passes ran;
  a count of rows within ~25% of a threshold (4.0 above, 0.25 below) where
  the gate is open (dof >= CHI2_RESCALE_MIN_DOF = 1). A tightening pass
  stops at each row's model-error floor, so a low-chi2 row that already
  sits on the floor is not fragile -- the passes / fit change shows it.
- How: t12/chi2run.py runs the benchmark's own main() from a fresh
  worktree (HEAD 04ef980, engine = c8904da) with the five earlier Hessian
  caches merged (no overwrite), patching only MolecularOptimizer.run /
  _optimise_once / reduced_chi_square to log, and hybrid_fit to tag each
  call with its molecule and score. The run's rows must reproduce the
  archived ones; that is the check the instrumentation changed nothing.
- Then: for rows near a threshold, the size of the jump (a sigma sweep as
  in T-11) is a follow-up, not part of this count.
- Success: 36 logged rows matching the archive, the count, and a fresh-agent
  check.

## Graph
- N1 · build · chi2run.py, both levels in the background · after: none · running · handoff: -
- N2 · verify · fresh agent: rows reproduce the archive, counts recomputed · after: N1 · pending · handoff: -
- N3 · gate · Q-15: what to do with the count · after: N2 · pending · handoff: -

## Progress
- 14:20 UTC: worktree t12/wt at 04ef980; 4367 cached Hessians merged from
  n13/wt, n13/wt_vf, n13/wt_vf2, t07/wt, t07/wt2; runs started (logs
  t12/chi2_{hf,b3lyp}.log, data chi2_{hf,b3lyp}.json).

## Observations

## Pending question

## Handoff
