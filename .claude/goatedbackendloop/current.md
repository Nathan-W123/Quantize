# Current task
Task: T-33
Status: building
Request: (D-82, auto-picked at Q-48) Diagnose why fits with distortion rows do not converge, then propose.

## Brief
- Outcome: the cause of T-32's non-convergence (as shipped 5 of 9 distortion-
  row fits still moving at 200 iterations; 11-12 without rows), with
  evidence, and a proposal.
- Hypotheses: (a) the rows' prediction uses the quantum Hessian, which is
  refreshed every hess_recalc_every (10) iterations, so the objective moves
  under the fit; (b) the rows' Jacobian (_cd_jacobian) disagrees with finite
  differences of the rows; (c) a curvature the Gauss-Newton step misses
  (the rows' second derivative) makes steps overshoot and oscillate.
- How: formyl fluoride, floor, 1.0 x prediction (the fastest unconverged
  case, ~20 s): full history; rerun with no Hessian refresh; FD check of
  the rows' Jacobian at the stopping point.
- Success: one hypothesis shown by an experiment that turns convergence on
  or off; then a proposal.

## Graph
- N1 · measure · full history of the failing fit · after: none · done · handoff: -
- N2 · measure · no Hessian refresh · after: N1 · done: not the cause · handoff: -
- N3 · measure · rows' Jacobian vs FD · after: N1 · done: not the cause · handoff: -
- N3b · measure · trust radius: autoconfig on / growth prototype, 9 cases + no-row · after: N3 · done · handoff: -
- N3c · build · correct T-32's README note and comment (my own wrong figures) · after: N3b · done · handoff: -
- N3d · verify · independent fact-check of the corrected note · after: N3c · done: 1 wrong, conclusion overstated · handoff: -
- N3e · build · note rewritten: no fit shown at an optimum; 0.2-29 mA · after: N3d · done · handoff: -
- N4 · gate · proposal (Q-49) · after: N3e · pending · handoff: -

## Progress
- N1 (t33/h10.json): formyl fluoride, floor, x1.0, 200 iterations: rejections
  at iterations 9-16 halve the trust radius to 1.5625e-3 A, after which every
  step is accepted (step ratio 0.62-0.88, almost all in the null space) and
  wrms falls ~2e-4 per step: a crawl, not an oscillation.
- N2 (hinf.json): hess_recalc_every 1e5: same history (wrms to 1e-5).
- N3 (jac.py): along a random direction the prediction is smooth (residual
  from a quadratic over +-2 mA 1e-11-1e-7 of its range); J.d agrees with
  central differences at h 1e-6..1e-2 to 4e-4 relative or better.
- Cause: SubspaceOptimizer.adapt_lambda halves the trust radius on every
  rejection and never enlarges it; with use_autoconfig False (every script,
  T-32's fits) nothing else does either, while autoconfig (on by default in
  the API and CLI) resets it to its target outside a rejection streak.
  Same fit with autoconfig on: plateau at 18 iterations; with a prototype
  that doubles the radius after an accepted step (capped at the start
  value): plateau at 84. All three answers agree within 0.4 mA / 0.02 deg.
- N3b (t33/fits, compare.py): the 9 shipped cases + no-row fits, autoconfig on
  and growth prototype, max_iter 200. Autoconfig on: no-row 12 iterations;
  with rows 7 of 9 stop (14-167), 2 hit 200 (FF and VF at x0.7); rows 30% off
  move bonds 1.1-3.5 mA, angles <= 0.15 deg from the no-row fit (shipped,
  autoconfig off: up to 42 mA / 2.2 deg). Same data, autoconfig on vs off:
  44.4 and 37.0 mA apart (FF x0.7, CH2ClF x1.3), 1.8-11.2 elsewhere. The growth
  prototype still hits 200 in 4 of 9 and ends 2-39 mA from its no-row fit.
  So T-32's "up to 42 mA" was a property of autoconfig-off fits that had not
  converged, not of the rows' optimum. Which answer is the optimum in the
  disagreeing cases is not settled (the rows' prediction uses the Hessian of
  each fit's last quantum point, so each fit's objective differs too).
- N3c: README note and the comment at the floor rewritten on both settings'
  numbers (each recomputed by script).

- N3d (fc33/check.py): figures reproduce except floor removal 0.2-29 (not
  31: 30.6 only with CH2ClF's two C-H taken separately). 8 of the 11 "stopped"
  row fits hit the 40-flat-iteration plateau fallback while still taking
  full-radius steps; only 3 of 18 ended with a step under 1e-3 A. The note
  led with autoconfig's small moves as if they were the effect; the growth
  prototype puts CH2ClF x1.3 at 38.9 mA, near the autoconfig-off answer.
- N3e: README note says no mismatched fit is shown at an optimum and how to
  tell (step_norm); comment 0.2-29 mA.

## Observations
- 2026-11-22 · correcting my own prose twice in one task: each rewrite
  replaced one unsupported framing with another (first "rows move 42 mA",
  then "with the default only 1-3.5 mA"). The data only support "where a
  run stops depends on the settings". Say what is unknown before what is.

## Pending question
