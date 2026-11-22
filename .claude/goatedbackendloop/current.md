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
- N3b · measure · trust radius: autoconfig on / growth prototype, 9 cases + no-row · after: N3 · running · handoff: -
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

## Observations

## Pending question
