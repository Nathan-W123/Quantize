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
- N1 · measure · per-block step accounting · after: none · pending · handoff: -
- N2 · gate · proposal (Q-52) · after: N1 · pending · handoff: -

## Progress

## Observations

## Pending question
