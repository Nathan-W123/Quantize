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
- N3d · verify · independent fact-check of the corrected note · after: N3c · pending · handoff: -
- N4 · gate · proposal (Q-49) · after: N2, N3 · pending · handoff: -

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

## Observations

## Pending question
