# Current task
Status: clarifying
Request: T-56 (D-127): measure why the water dimer in joint internal mode (quantum on) ends off its posterior's mode
## Brief
- Outcome: a measured cause for the dimer's gap to its mode (0.47 widths in the T-49 verifier, 0.29 in N34), then a proposal. No code change before the measurement.
- Hypothesis to test, not assumed (L-27): without dihedrals the dimer's internal set has 10 coordinates for 12 degrees of freedom, so the step cannot move two directions the posterior weighs.
- Success: the gap split into the part the internal step can reach and the part it cannot, on the dimer at HEAD; the same split with dihedrals auto.
## Progress
- N1 · measure (t56/dimer_reach.py, dimer_dirs.py) · C1 starts (8/15): 10 coords reach 10/12, gap 0.068-4.0 widths lies along the two H-bond torsions (within reached 5e-5-0.041; seed 11 0.32); Cs starts (7/15): reach 8/8, <=0.023; auto: 12/12, 0.003-0.080; no report note below 2 widths
## Observations
- 2026-11-23 · my first split of the gap (M-norm shares, then widths of each part of the gradient) did not add up (162%; 14.5 + 13.4 against 1.1): the parts are not M-orthogonal. The distance to the mode within the reached directions is the measure that holds.
- 2026-11-23 · the dimer's start noise decides its point group (Cs in 7 of 15), and with it whether the torsions exist for the fit: one seed would have hidden or exaggerated the effect
## Pending question
- Q-84 (asked 03:09 UTC, deadline 03:24 UTC): 1 (Recommended) report note only (internal mode without dihedrals: say how many directions bonds and angles reach, that the rest cannot move, and that use_dihedrals auto adds them; no fit changes); 2 make use_dihedrals auto the default (user's call, D-100: moves answers, costs iterations; never auto-picked); 3 seed 23's second minimum; 4 Done for now. Deadline takes 1.
## Handoff
