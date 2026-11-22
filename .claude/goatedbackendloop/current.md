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
- N1 · build · growth prototype (monkeypatch) · after: none · done · handoff: -
- N2 · measure · nine molecules, shipped vs growth, max_iter 40 and 200 · after: N1 · done · handoff: -
- N2b · measure · the rule on the distortion crawl and fluoroacetylene · after: N2 · done · handoff: -
- N3 · gate · proposal (Q-50) · after: N2b · asked · handoff: -

## Progress
- N2 (t34/fit.py, fits/, compare.py): 9 molecules x {ship, grow} x {40, 200},
  plain joint HF/6-31G, autoconfig off, no distortion rows. Every pair
  bit-identical (0.000 mA); the rule's condition (rho > 0.75 with the step at
  the radius) fired 0-6 times per fit, always before any rejection, when the
  radius was already at its cap: grown 0 everywhere. Iterations 11-20, except
  fluoroacetylene (max_iter at 40, plateau at 57; crawling at ~95% of its
  radius, 6.25e-3 A, under the 0.99 test; with 0.9: still 57, grown 0).
- N2b: formyl fluoride with distortion rows (x1.0): shipped 200 iterations,
  6 rejected; with the rule grown 25 times, 30 rejected, still 200: each
  doubled step to 3.1e-3 A is refused and the radius saw-tooths 1.6-3.1e-3.
  The T-33 prototype (double after every accepted step) stopped at 84. So
  the crawl is the step model failing beyond ~2 mA, not the bookkeeping.

## Observations

## Pending question
Q-50 · asked 2026-11-22T08:13Z · deadline 08:28Z · auto-pick 1
1. (Recommended) Close N26 as measured, no change: the standard growth rule
   never fires on the scripts' fits (answers bit-identical on nine
   molecules) and does not fix the distortion crawl (doubled steps are
   refused); record the narrowed cause in backlog N8.
2. Diagnose why steps beyond ~2 mA fail on distortion-row fits: actual
   objective change against the Gauss-Newton model and one with the rows'
   second derivatives, along refused steps; then propose.
3. N3: look again for published r_e^SE references (sd <= 0.5 mA, D-35).
4. Done for now.
