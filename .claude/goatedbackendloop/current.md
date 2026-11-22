# Current task
Task: T-31
Status: building
Request: (D-78, auto-picked at Q-44) N24: measure how far the reported answer depends on which near-tied multistart start wins, then propose.

## Brief
- Outcome: per CLI run, whether the selected start's reported 1-sigma bond
  and angle widths (geometry_uncertainty) cover the distance to its
  near-tied rivals; then a proposal with its cost.
- Known (T-23, D-65, D-70): near-tied starts (model energy within 5e-5 Ha)
  sit 8.7-20.6 mA apart (component RMS) on the methyl-torsion molecules and
  VF cold; bond RMS against references moves <= 1.2 mA between them.
- How: rerun T-23's ten configs (5 molecules x warm/cold, HF/6-31G, BALANCED,
  3 starts) at HEAD via `cli run`, run() wrapped to record per start coords,
  model energy, stop reason and geometry_uncertainty(); output root in t31.
  Compare per bond / angle |q_i - q_j| with sigma_i and sigma_j.
- Success: per run the near-tied pairs (D-70 rule), max |dq| / sigma over
  bonds and angles, and which coordinates carry it.

## Graph
- N1 · measure · rerun 10 configs instrumented · after: none · pending · handoff: -
- N2 · analyse · |dq| / sigma per near-tied pair · after: N1 · pending · handoff: -
- N3 · gate · proposal (Q-45) · after: N2 · pending · handoff: -

## Progress

## Observations

## Pending question
