# Current task
Task: T-35
Status: building
Request: (D-85, auto-picked at Q-51) Why distortion-row steps beyond ~2 mA fail the ratio test; then propose.

## Brief
- Outcome: which part of the objective the Gauss-Newton model gets wrong on
  the refused steps (rotational rows, distortion rows, quantum term), how
  large the miss is against the predicted gain, and a proposal.
- How: wrap MolecularOptimizer._joint_step_ratio in the formyl fluoride x1.0
  and x0.7 distortion fits (HF/6-31G, autoconfig off, shipped trust rule and
  T-34's growth rule): per step |dp|, rho, and per block actual vs predicted
  decrease (rows split by position: the distortion rows are stacked last);
  the nonlinearity |r1 - (r0 - J dp)| / |J dp| per block.
- Success: the block that makes rho fail, with numbers; then a proposal.

## Graph
- N1 · measure · per-block step accounting · after: none · done · handoff: -
- N2 · gate · proposal (Q-52) · after: N1 · asked · handoff: -

## Progress
- N1 (t35/acct.py, ff_{ship,grow}_{1.0,0.7}.json): formyl fluoride, 200
  iterations each, every step's ratio test split by block (3 distortion rows
  stacked last). On refused steps the ROTATIONAL rows miss: GN predicts a
  decrease (median +1.1e-2 ship x1.0, +1.7e-1 ship x0.7) and they rise;
  their nonlinearity |r1 - (r0 - J dp)| / |J dp| has median 1.0-3.1 (max
  5.4-8.9); on accepted crawl steps (1.56e-3 A) it is ~0.2 already. The
  distortion rows stay near-linear (median 0.2-1.1) and are 1e2-1e3 smaller
  (pred 3e-6..1e-2). Quantum term: exact here (no target), cancels in rho.
  T-33's history: the steps are almost all null-space (dx_range 4e-6, dx_null
  1.56e-3 A). So: the distortion rows pull along directions the rotational
  constants do not see to first order; there their second derivative, which
  Gauss-Newton drops, outweighs the linear change beyond ~2 mA, and every
  longer step is refused.

## Observations

## Pending question
Q-52 · asked 2026-11-22T08:55Z · deadline 09:10Z · auto-pick 1
1. (Recommended) Flag and document, no solver change: record the cause in
   backlog N8 and the README's distortion note (the rotational rows'
   curvature in the directions the distortion rows push, which the
   Gauss-Newton model drops); a solver fix for this opt-in, data-gated
   feature is your call.
2. Prototype a second-order term for the rotational rows along the step
   (their curvature in the null directions) and measure on the distortion
   cases and the benchmark's fits without rows; then propose.
3. N7: measure how much split-mode widths miss the step's range-null
   coupling, then propose.
4. Done for now.
