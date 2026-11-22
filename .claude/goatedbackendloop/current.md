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
- N2 · measure · instrumented runs (hh, hb, uh, ub, mh) · after: N1 · running · handoff: -
- N3 · gate · proposal (Q-43) · after: N2 · pending · handoff: -

## Progress

## Observations

## Pending question
