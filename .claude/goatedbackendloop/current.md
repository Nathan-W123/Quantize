# Current task
Task: T-23
Status: building
Request: (D-64, auto-picked at Q-31) the multistart ranks its starts on a stale energy (backlog N24): measure, then propose a no-cost fix before making it.

## Brief
- Outcome: per start, stale vs model vs true energy at the answer, which
  start each ranks first, and how much the choice moves the answer.
- How: t23/rank.py wraps MolecularOptimizer.run inside `cli run` (the
  verifier's configs; HF/6-31G, BALANCED 3 starts, internal coordinates).
- Tradeoff: the model energy is free but approximate; the true energy costs
  one SCF per start.
- Success: the table; a proposal (or not).

## Graph
- N1 · diagnose · 4 cases (HCOF, CH2ClF x warm/cold) · after: none · done (t23/*.json) · handoff: -
- N1b · diagnose · 8 more (VF, AF, FE x warm/cold; HCOF, CH2ClF cold repeat) · after: N1 · running (t23/rank_all.log) · handoff: -
- N2 · gate · propose (or not) · after: N1b · todo · handoff: -

## Progress
- Code: multistart._run_single_start records energy = history[-1]['energy'],
  the quantum energy at the start's last quantum point; select_best_result
  takes the lowest within a 2x spectral gate (here every start passes).
  The fit's objective is not comparable across starts (each start's alpha_q
  comes from its own Hessian: HCOF warm 1930.6 / 1662.2 / 2031.2).
- N1: stale - true at the answer up to 6.4e-4 Ha; model - true 1.7e-7 to
  2.8e-5, one start 4.3e-4 (HCOF cold start 2, 4.9 mA from its last quantum
  point). Ranking: HCOF warm stale picks 1, model and true 3; CH2ClF warm
  stale and model pick 1, true 2 (1.3e-6 Ha apart, model error 1.4e-5);
  both cold cases all agree. The starts' answers lie 0.15-0.77 mA apart
  (max atom), bond RMS within 0.22 mA -- they differ mainly because each
  stops short on its own start's Hessian (N23).

## Observations

## Pending question

## Handoff
