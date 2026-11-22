# Current task
Task: T-31
Status: done
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
- N3 · gate · proposal (Q-45) · after: N2 · done (D-79, auto-picked 1) · handoff: -
- N4 · build · shift in sigma on each near-tied start + note · after: N3 · done · handoff: -
- N5 · verify · tests + mutations, replay of t31 data, CI selection · after: N4 · done · handoff: -
- N6 · verify · independent verifier · after: N4 · done: holds, 2 wording fixes · handoff: -
- N7 · build · fallback note wording, docstring median · after: N6 · done · handoff: -

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

- N4: multistart._reported_shift_in_sigma; near_tied_starts carries
  max_shift_in_sigma / max_shift_coordinate; near_tie_note prints them.
- N5: 3 tests; mutations no dihedral wrap, other start's sigma, smallest
  instead of largest, note without the shift, zero width not skipped: each
  fails a test. A guard for a missing posterior survived (equivalent: the
  except already returns None) and was removed. Replaying the t31 logs through
  the helper gives analyse.py's figures (max 0.287, fluoroethane warm bond
  2-8). CI selection 1639 passed, 15 skipped.

- N6: verifier (t31v/): `cli run` old vs new, OMP 1, fluoroethane warm and
  acetyl fluoride warm: of 23 output files only the near-tie line (stdout,
  md, html) and two JSON keys differ (plus timestamps / config path);
  selected start, coords, widths identical. Own bond / angle / dihedral code
  gives 0.2870 (bond 2-8) and 0.1293 (bond 2-7), as reported. Found: the
  note's explanation sentence printed with no figure; "median 0.11" is the
  bonds-only median (per pair 0.129).
- N7: explanation only when a listed start has a figure (test + mutation M7
  fails it); docstring median 0.13. CI selection 1639 passed, 15 skipped.

## Observations
- 2026-11-22 · replaying the measurement data through the shipped helper
  was a cheap check that it computes what the analysis measured.
- 2026-11-22 · I quoted a median without saying over what; the verifier
  caught it. State what a summary statistic is taken over.

## Pending question
Q-46 · asked 2026-11-22T05:49Z · deadline 06:04Z · auto-pick 1
1. (Recommended) N8 (b): measure whether the 0.01 MHz distortion sigma floor
   swamps heavy molecules' kHz-size DJ, then propose; no behaviour change
   without a proposal.
2. N3: look again for published r_e^SE references (sd <= 0.5 mA, D-35) for
   HNCO, fluoroacetylene, HCOF; nothing guessed.
3. N7: measure how much split-mode widths miss the step's range-null
   coupling, then propose.
4. Done for now.
