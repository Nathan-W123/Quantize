# Current task
Status: done
Request: T-56 (D-127, D-128): measure why the water dimer in joint internal mode ends off its mode; then report a set that misses directions
## Brief
- Outcome: a measured cause for the dimer's gap to its mode (0.47 widths in the T-49 verifier, 0.29 in N34), then a proposal. No code change before the measurement.
- Hypothesis to test, not assumed (L-27): without dihedrals the dimer's internal set has 10 coordinates for 12 degrees of freedom, so the step cannot move two directions the posterior weighs.
- Success: the gap split into the part the internal step can reach and the part it cannot, on the dimer at HEAD; the same split with dihedrals auto.
## Progress
- N1 · measure (t56/dimer_reach.py, dimer_dirs.py) · C1 starts (8/15): 10 coords reach 10/12, gap 0.068-4.0 widths lies along the two H-bond torsions (within reached 5e-5-0.041; seed 11 0.32); Cs starts (7/15): reach 8/8, <=0.023; auto: 12/12, 0.003-0.080; no report note below 2 widths
- N2 · report note (D-128, auto-picked Q-84): _resolve_ic_dihedrals records a short set for dihedrals off (and auto when dihedrals reach no more); start-of-run line; ic_reach_warnings in internal_coordinate_posterior and the Laplace summary (runner report and CLI read those); README sentence; 7 tests, 9/9 mutations caught (t56/mutate.py); CI 1752 passed
- N3 · independent check (L-10; t56v/REPORT.md): no fit changes (bit-identical); findings fixed: remedy names use_dihedrals: true (multistart), short sets with dihedrals reported, neutral wording, start-geometry tense, saddle and no-covariance paths, constructor cannot raise, docstring 'only' claim (acetone, fluorobenzene). 17 tests in the section, 17/17 mutations caught; CI 1759 passed; fits re-checked bit-identical vs 20ec153
## Observations
- 2026-11-23 · the verifier caught a false 'only' claim: my reach survey imported 4 of the reference module's lists (L-18 revised)
- 2026-11-23 · the verifier caught a remedy that is false under multistart (auto becomes the first start's True/False): advice in a message has to hold on every entry path, not just the direct API (L-25)
- 2026-11-23 · fact-check (L-18) caught my docstring's 'the missed directions keep the start's values': the dimer's torsions moved 8-168 deg from the start (t56/note_check.py)
- 2026-11-23 · my first split of the gap (M-norm shares, then widths of each part of the gradient) did not add up (162%; 14.5 + 13.4 against 1.1): the parts are not M-orthogonal. The distance to the mode within the reached directions is the measure that holds.
- 2026-11-23 · the dimer's start noise decides its point group (Cs in 7 of 15), and with it whether the torsions exist for the fit: one seed would have hidden or exaggerated the effect
## Pending question
- none (Q-84 auto-picked option 1 at 03:26 UTC)
## Handoff
