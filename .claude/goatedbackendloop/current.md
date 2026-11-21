# Current task
Task: T-16
Status: done
Request: fix the failing test_prior_target check (backlog N20; D-53, auto-picked at Q-20).

## Brief
- Outcome: test_a_target_prior_keeps_the_target_s_curvature passes on the CI
  interpreter (3.12) and asserts what it claims -- a target-centred prior
  keeps the target's curvature: a Hessian refresh recomputes the same
  Hessian at the target and the answer stays put to solver roundoff.
- Approach: understand the change first (L-09: exact-zero assertion on a
  re-converged geometry), then bound by a measured scale; test-only change.
- Success: passes on both numpy builds; the old defect (Hessian at the
  answer) fails each new assertion; CI selection green on 3.12.

## Graph
- N1 · diagnose · why moved_ang is 7e-11 A on 3.12 · after: none · done · handoff: t16/probe.py
- N2 · build · measured bounds + docstring · after: N1 · done · handoff: -
- N3 · verify · break check (L-15) + CI selection on 3.12 · after: N2 · done · handoff: t16/break.py, t16/full_suite.log

## Progress
- Cause: numpy, not code, Python or scipy. Same 3.12 venv, scipy fixed:
  numpy 2.4.6 -> moved 0.0, numpy 2.5.5 -> 6.97e-11 A (t16/venv_np246,
  both swaps). Mechanism (probe.py iteration tables): the first fit stops on
  a plateau with a Gauss-Newton step left of 3.9e-14 A (2.4.6) or 2.09e-10 A
  (2.5.5); the refresh re-converges on the identical target Hessian, and on
  2.5.5 iteration 10 accepts that roundoff step. T-14's suite ran on 3.11
  (numpy 2.4.6), T-15's on the 3.12 venv -- hence "passed at 19:33".
- Scales: restart / refresh move 5e-11 / 0 mA (2.4.6), 7.0e-8 / 7.0e-8 mA
  (2.5.5); the old defect moves the refresh 4.56e-5 mA and leaves the
  Hessian 7.7e-3 A from the target.
- Fix (test only): assert age_after_ang < 1e-12 (the refreshed Hessian is
  the target's) and moved < 1e-6 mA, the restart's bound (14x above noise,
  46x below the defect); docstring says why neither move is exactly zero.
- Verified: test file 17/17 on 3.11 + numpy 2.4.6, 3.12 + 2.5.5; the
  defect fails both new assertions on both builds (break.py, A and B); CI
  selection on 3.12: 1612 passed, 15 skipped, exit 0.

## Observations
- 2026-11-21 · L-P02 again: "passed in T-14's suite" meant a different interpreter and numpy build; running the test on both interpreters found the cause in one step after a day of "cause unknown".

## Pending question
Q-21 (asked 03:03 UTC, deadline 03:19 UTC; send_later trig_015D6mF4zciSw9GLxzujdxc8): what's next after T-16?
1. Measure the acoustic-sum-rule repair of the B3LYP Hessian, backlog N19 (b)(ii) -- prototype + rerun, propose before any engine change (Recommended; auto-pick)
2. More r_e^SE references for the benchmark yardstick, backlog N3
3. Offsets configs take alpha at the stationary geometry, backlog N19 (b)(i) -- measure first
4. Done for now

## Handoff
