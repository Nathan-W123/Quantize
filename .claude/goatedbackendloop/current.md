# Current task
Task: T-32
Status: building
Request: (D-80, auto-picked at Q-46) N8 (b): measure whether the 0.01 MHz distortion sigma floor swamps heavy molecules' kHz-size DJ, then propose.

## Brief
- Outcome: per reference molecule and key (DJ, DJK, DK), the predicted
  constant, the 40% model sigma, whether the 0.01 MHz floor replaces it, and
  the weight lost (floor / model)^2; then a proposal with its cost.
- Code: SpectralEngine._cd_rows: sigma = max(hypot(cd_sigma, model), 0.01),
  model = cd_model_sigma or _default_cd_sigma = max(0.4 |observed|, 0.01).
  Units MHz. Water's DJ is ~0.04 MHz, so 40% clears the floor there.
- Data gate: no reference molecule has observed distortion constants; the
  predicted constants (compute_cd_constants at the reference geometry, HF/
  6-31G Hessian, the reduction the engine would pick) stand in.
- Success: the count of floored rows and their weight factors; whether any
  geometry effect is worth measuring next.

## Graph
- N1 · measure · predicted CD constants vs the floor · after: none · done · handoff: -
- N1b · measure · geometry with / without the floor, synthetic CD data · after: N1 · done (unconverged) · handoff: -
- N1c · measure · same at max_iter 200 (FF, VF, CH2ClF), history kept · after: N1b · done · handoff: -
- N2 · gate · proposal (Q-47) · after: N1c · asked · handoff: -

## Progress
- N1 (t32/cdfloor.py, cd.json, floor_table.txt): harmonic constants at the
  reference geometry, HF/6-31G, reduction auto, parent isotopologue. Water
  DJ 28.9 MHz (obs 37.59). Heavy molecules' DJ 0.0029-0.0083 MHz, so
  0.4 |DJ| = 1.2-3.3 kHz against the 10 kHz floor. 13 of 27 rows floored
  (every DJ except water's, plus HNCO DJK, acetyl fluoride DJK and DK, all
  of fluoroacetylene -- linear, its DJK / DK are not meaningful); their
  weight cut 1.9x to 75x below what the 40% model error implies.
  The floor dates from 5f1842a (2026-11-13), no stated reason; a zero guard.
- N1b: fits with synthetic cd_observed = 0.7 / 1.0 / 1.3 x prediction,
  cd_weight 1, floor vs 1e-9 floor (t32/wt_floor, wt_nofloor), plus no rows.
  6 molecules x (none + 3 scales x 2 trees) = 42 fits (t32/fits, compare.py).
  With rows, 34 of 36 fits stopped on max_iter 40 (without rows all plateau
  in 11-20 iterations; fluoroacetylene, linear, is erratic both ways). Their
  geometries sit 0.5-47 mA from the no-row fit (bond max) and floor vs no
  floor 0.4-29 mA apart -- but unconverged, so not answers. Rerunning at 200.
- N1c (t32/fits200, compare200.py): at max_iter 200, 12 of 18 fits with rows
  plateau after 59-199 iterations (no rows: 11-13); 6 still hit 200. The
  answers agree with the 40-iteration ones within a few mA. Against the
  no-row answer (bond max): x1.0 rows 0.5-3.3 mA; x0.7 / x1.3 (a mismatch
  inside the stated 40% model error) 1.1-45 mA and up to 2.4 deg -- several
  times the reported bond widths (9-17 mA). Floor vs no floor 0.2-29 mA,
  bond RMS against the reference better without the floor in 3 of 9 cases,
  worse in 5: no consistent direction.
  So the floor is the smaller issue: the rows themselves, as weighted, slow
  the fit and pull the structure by more than its width at a model-sized
  mismatch (opt-in: fit_cd_constants and cd_weight > 0, config reachable).

## Observations

## Pending question
Q-47 · asked 2026-11-22T06:42Z · deadline 06:57Z · auto-pick 1
1. (Recommended) Flag, no behaviour change: keep the floor; write T-32's
   numbers where fit_cd_constants / cd_weight are documented (README note,
   docstrings) and in backlog N8.
2. Measure where the rows' pull goes (which coordinates, prior-held
   directions) and what model sigma would keep a 30% mismatch inside the
   reported widths; then propose (any change is accuracy-side: your call).
3. Remove the floor (sigma = the stated 40% model error): changes opt-in
   heavy-molecule distortion fits by 0.2-29 mA, no consistent accuracy
   direction. Accuracy change: not auto-picked.
4. Done for now.
