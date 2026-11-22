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
- N1 · measure · rerun 10 configs instrumented · after: none · done · handoff: -
- N2 · analyse · |dq| / sigma per near-tied pair · after: N1 · done · handoff: -
- N3 · gate · proposal (Q-45) · after: N2 · asked · handoff: -

## Progress
- N1: 10 `cli run`s at 62beb3f (t31/logs), rc 0, 3 starts each; the runs'
  own near_tied_starts listed 7 pairs in 5 runs (8.6-20.6 mA largest atom,
  same set as T-26: acetyl fluoride x2, fluoroethane x2, VF cold).
- N2 (t31/analyse.py, analyse.out): per near-tied pair, the largest
  |q_other - q_selected| / sigma_selected over the reported bonds and
  angles (the posterior has no dihedrals): bonds max 0.29 (median 0.11),
  angles max 0.21 (median 0.09); none above 1. Largest: fluoroethane warm
  bond 2-8, 2.93 mA against sigma 10.2 mA; angle 0.35 deg against 1.83 deg.
  Reported sigmas: bonds 9.2-16.7 mA, angles 1.2-2.1 deg. So the 9-21 mA
  atom spread is along torsions (H positions); the quoted bonds and angles
  move by under a third of their own width. The one start above 1 sigma
  (formyl fluoride cold, start 2: bond 1-2 1.07) is 6.9e-4 Ha higher and
  correctly not selected or listed.

## Observations

## Pending question
Q-45 · asked 2026-11-22T05:38Z · deadline 05:53Z · auto-pick 1
1. (Recommended) Put the measure in the near-tie note: per listed start, the
   largest |dq| / sigma over the reported bonds and angles, from the two
   starts' own posteriors (no extra compute); answers and ranking unchanged.
2. Docstring only: T-31's numbers beside near_tied_starts; close N24.
3. Close N24 in the records only.
4. Done for now.
