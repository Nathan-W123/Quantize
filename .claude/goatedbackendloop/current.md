# Current task
Task: T-30
Status: verifying
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
- N2 · measure · instrumented runs (hh, hb, uh, ub, mh) · after: N1 · done · handoff: -
- N3 · gate · proposal (Q-43) · after: N2 · done (D-76, auto-picked 1) · handoff: -
- N4 · build · skip a refit on unchanged weights, keep scaling · after: N3 · done · handoff: -
- N5 · verify · old-vs-new sweep, tests + mutations, CI selection · after: N4 · done · handoff: -
- N6 · verify · independent verifier (different mechanism) · after: N4 · pending · handoff: -

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
- mh 9 fits (exit 0): 8 below 1/4, 1 in band (max 1.03), 0 above 4; 8 tighten
  refits, all weights changed; move max 9.3e-6 Å.
  Total: 0 of 86 script fits and 0 of 362 benchmark fits widen.

- N4: run() compares _weighting_sigmas() before and after scale_sigma; equal
  (exactly) and stated changed -> "weights unchanged", continue. Guarded off
  with autoconfig or a mass-dependent correction (both read stated sigma).
- N5: t30/cmp.py, 17 analytic-water cases (0.05 MHz to 1e5 MHz, 1-5 passes)
  old tree (0147704) vs new: coords, sigmas, measurement sigmas, chi2
  sequence, final chi2, geometry_uncertainty widths bit-identical; fits
  4->1 (<= 2 MHz), 4->2 (5), 4->3 (20), 6->4 (5 MHz, 5 passes).
  Guards measured: without the mass-dependent guard the answer moves
  5.2e-6 A (refits are real there); without the autoconfig guard 0 A on this
  case (kept: its controls read the median stated sigma).
  Tests: 3 new (one parametrised x2); mutations M1 no skip, M2 stop instead of
  continue, M3/M4 each guard dropped, M5 weights read after the scale: each
  fails a test. CI selection 1640 passed, 15 skipped.

## Observations
- 2026-11-22 · the brief's option 1 said "stops"; working out the water case
  showed a stop changes answers (pass 3 crosses a floor). Checking the promise
  ("no result change") against the mechanism before coding caught it.

## Pending question
