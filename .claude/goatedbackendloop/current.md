# Current task
Task: T-26
Status: verifying
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
- N2 · gate · exact behaviour (Q-37) · after: N1 · done: as proposed (D-70, auto-picked) · handoff: -
- N3 · build · helper, recording, report, test · after: N2 · done (6 tests; payload carries it for rebuilds; CI selection 1631 passed) · handoff: -
- N4 · verify · fresh agent: a real CLI run shows it; quiet cases stay quiet · after: N3 · running · handoff: -

## Progress
- Real `cli run` (acetyl fluoride, RHF minimum start, HF/6-31G): printed and
  in report.md / report.html: "near-tied starts: start 2 (+1.9e-05 Eh, 20.6
  mA)". The first run's payload dropped it (report_payload.json keeps five
  fields of best) -- added; after a rerun, `cli report` rebuilds it.

## Observations

## Pending question

## Handoff
