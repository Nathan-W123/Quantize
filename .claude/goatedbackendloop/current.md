# Current task
Task: T-29
Status: building
Request: (D-73, auto-picked at Q-40) N19 (a): measure how often fits sit near the chi-square rescale switch (re-weights when a pass's reduced chi2 passes 4.0 or falls below 0.25), then propose; no behaviour change without a proposal.

## Brief
- Outcome: a measured answer to "does the switch's position decide any
  current result", and a proposal (keep / smooth / remove) with its cost.
- Facts so far: engine default chi2_rescale=False; the CLI never enables it;
  four scripts do (accuracy_upgrades_benchmark, holdout_validation,
  mixed_estimation_baseline, uncertainty_calibration), threshold 4, 3 passes.
- How: reuse T-15's band scan (b29cec3, 248 rows, 362 fits) for the shape;
  rerun at HEAD, instrumented (t13/chi2run2.py, worktree t29/wt), the fits
  nearest the switch: HF HNCO offsets (largest chi2 1.85), B3LYP ozone
  offsets (0.226), B3LYP water base (0.217); check their rows match the
  archive (same vintage).
- Success: current chi2 for those fits and the distance to each switch
  point; a proposal.

## Graph
- N1 · diagnose · T-15 logs: 362 fits; none above 4 on the first pass (max 1.85, HNCO offsets family); 340 below 1/4 (fires, tightening): 202 < 0.05, 126 in 0.05-0.2, 12 in 0.2-0.25; 22 in [1/3, 3] · after: none · done · handoff: -
- N2 · measure · targeted rerun at HEAD (vintage moved: ba67cfe sum rule, 4614cd0 B3LYP geometries) · after: N1 · running · handoff: -
- N3 · gate · proposal (Q-41) · after: N2 · pending · handoff: -

## Progress

## Observations

## Pending question
