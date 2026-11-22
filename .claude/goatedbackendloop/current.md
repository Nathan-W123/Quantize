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
- N2 · measure · targeted rerun at HEAD 03e7d03, fresh caches (t29/wt): HF HNCO offsets chi2 1.8529, 1 pass, 28.5574 mA (archive 28.5553); B3LYP water base 0.2175 (fires; archive match 2e-6 mA), ozone offsets 0.2267 (fires; 0.03324 vs archive 0.03302), water offsets 0.0486, ozone base 0.0407 · after: N1 · done · handoff: -
- N2b · measure · rescale off (NORESCALE=1, fresh results file) vs on, B3LYP: ozone offsets +9.7e-6 mA, ozone base +1.4e-6, water base -8.6e-10, water offsets -4.9e-9 -- the switch at 1/4 moves these rows < 1e-5 mA (the tightened sigmas sit at their model-error floors; chi2 identical before and after the refit) · after: N2 · done · handoff: -
- N3 · gate · proposal (Q-41) · after: N2b · asked · handoff: -

## Progress

## Observations
- A second run in the same worktree resumed from its results file and fitted nothing (the runner's docstring says "fresh results file"); moved it aside. Check that a rerun did work before reading its log.

## Pending question
- Q-41 (asked 00:59 UTC, deadline 01:14): N19 proposal. 1 keep the switch; benchmark docstring to the 03e7d03 margins; the engine keeps each rescale check (chi2, dof, fired) and the benchmark stores them per row, so a later rerun shows near-switch fits without instrumentation; no number changes (recommended) · 2 docstring only, close N19 (a) · 3 make the rescale continuous (flagged: moves every firing row, full rerun, no measured benefit; not recommended) · 4 done for now.
