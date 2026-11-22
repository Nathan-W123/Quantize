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
- N1 · measure · full history of the failing fit · after: none · pending · handoff: -
- N2 · measure · no Hessian refresh · after: N1 · pending · handoff: -
- N3 · measure · rows' Jacobian vs FD · after: N1 · pending · handoff: -
- N4 · gate · proposal (Q-49) · after: N2, N3 · pending · handoff: -

## Progress

## Observations

## Pending question
