# Current task
Task: T-34
Status: building
Request: (D-83, auto-picked at Q-49) N26: measure what a growing trust radius does to the scripts' fits without distortion rows; measurement only.

## Brief
- Outcome: per benchmark molecule, iterations, stop reason and answer
  (internal coordinates) for the shipped trust rule vs the standard growth
  rule; whether any answer moves, and by how much against the reported
  widths.
- Prototype: wrap _joint_step_ratio to keep rho and the realised step norm;
  wrap SubspaceOptimizer.adapt_lambda: after an accepted step with rho > 0.75
  and |step| >= 0.99 x radius, double both radii, capped at their values at
  the start of the fit. Shrink on rejection unchanged.
- Fits: (a) the T-32 plain joint setup (HF/6-31G, autoconfig off, max_iter 40
  as the scripts, and 200), the nine reference molecules, no distortion rows;
  (b) if (a) moves anything, a benchmark-script row through the real script.
- Success: a table of iterations and answer shifts; then a proposal.

## Graph
- N1 · build · growth prototype (monkeypatch) · after: none · pending · handoff: -
- N2 · measure · nine molecules, shipped vs growth, max_iter 40 and 200 · after: N1 · pending · handoff: -
- N3 · gate · proposal (Q-50) · after: N2 · pending · handoff: -

## Progress

## Observations

## Pending question
