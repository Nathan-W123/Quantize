# Current task
Task: T-30
Status: building
Request: (D-75, auto-picked at Q-42) N25: the chi-square rescale's high-side widening multiplies the stated sigma while the weights use max(stated, 0.1% of the constant, ...); measure on real fits how often and how much, then propose.

## Brief
- Outcome: a measured answer to "does any real fit widen on unchanged
  weights", and a proposal with its cost.
- Reach (checked): config / CLI cannot enable the rescale (chi2_rescale is not
  an optimizer key); Python API and four scripts do. The benchmark never
  widens (T-15, T-29). Left: holdout_validation (README quotes it),
  uncertainty_calibration, mixed_estimation_baseline.
- How: run those three at HEAD 0147704 instrumented (t30/instr.py: engine's
  own chi2_rescale_checks plus the weighting sigmas before each fit and the
  geometry after it), one worktree each with t29's caches; HF and B3LYP.
- Success: per script, the number of widening refits, how many ran on
  unchanged weights, and the geometry each moved; then a proposal.

## Graph
- N1 · diagnose · reach of the rescale · after: none · done · handoff: -
- N2 · measure · instrumented runs (hh, hb, uh, ub, mh) · after: N1 · running (mh rerun) · handoff: -
- N3 · gate · proposal (Q-43) · after: N2 · pending · handoff: -

## Progress
- 2026-11-22 · first batch (4 parallel, then mh) hit the 2 h background limit.
  hh, uh, ub finished (exit 0); hb stopped at 30 of 35 molecules (log written
  per run, so the 30 stand); mh never started, rerun alone at 4 threads.
- Counts (first check of each fit, engine's own chi2_rescale_checks):
  hh 35 fits: 35 below 1/4 (max chi2 0.085), 0 above 4; 35 tighten refits,
  all weights changed, then "at floor"; geometry move max 2.5e-4 Å.
  hb 30 fits: 30 below 1/4 (max 0.063), 0 above 4; move max 3.2e-4 Å.
  uh 6 fits: 4 below, 2 in band, 0 above; move max 5.7e-6 Å.
  ub 6 fits: 3 below, 3 in band (max 1.65), 0 above; move max 1.3e-10 Å.
  So far 0 of 77 real fits widen; the unchanged-weight case is only the
  synthetic 5 MHz water test. Benchmark: 0 of 362 (T-15, T-29).

## Observations

## Pending question
