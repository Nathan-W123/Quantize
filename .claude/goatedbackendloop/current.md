# Current task
Task: T-26
Status: clarifying
Request: (D-69, auto-picked at Q-36) have the multistart say when its top starts are near-tied but far apart (N24); thresholds from T-23 data; propose before building.

## Brief
- Outcome: the run reports which other starts end within the energies'
  resolution of the selected one but far from it, so a user sees that the
  answer is one of several the fit does not decide between.
- How (proposed, Q-37): multistart._run_single_start records the fit's own
  model energy at the answer (quantum_prior_energy, free);
  multistart.near_tied_starts(results, best, tie 5e-5 Ha, distance 2 mA,
  Kabsch-aligned max atom) lists the others; run_generic stores it on the
  best result, prints it, and the report (md + html) lists it under "Best
  start"; "Final energy" relabelled as the last quantum point's. Ranking
  unchanged.
- Thresholds (t23 data, 20 start pairs): |model - true| at an answer median
  5.6e-6, 90th percentile 2.5e-5, max 4.3e-4 Ha -> tie 5e-5 (2x the 90th);
  distance 2 mA leaves out the 0.15-1.2 mA pairs. Flags 5 of 10 runs (VF
  cold 8.7 mA; AF warm 20.6; AF cold 18.9 / 9.5; FE warm 11.6; FE cold 12.0
  / 18.3), none of HCOF / CH2ClF (pairs <= 1.2 mA apart or > 5e-5 apart).
- Success: a unit test on synthetic results (flags tie+far only); the report
  line; one real CLI run showing it; CI selection.

## Graph
- N1 · diagnose · thresholds from t23 · after: none · done · handoff: -
- N2 · gate · exact behaviour (Q-37) · after: N1 · asked · handoff: -
- N3 · build · helper, recording, report, test · after: N2 · todo · handoff: -
- N4 · verify · fresh agent: a real CLI run shows it; test fails on old code · after: N3 · todo · handoff: -

## Progress

## Observations

## Pending question
- Q-37 (asked 22:41 UTC, deadline 22:57): the N24 behaviour. 1 as proposed (recommended) · 2 record in the bundle only, no report line or print · 3 as proposed with the thresholds in config · 4 done for now.

## Handoff
