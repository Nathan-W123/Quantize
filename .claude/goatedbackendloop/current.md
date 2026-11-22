# Current task
Task: T-36
Status: done
Request: (D-88, auto-picked at Q-53) Fix the orientation defect in the distortion prediction.

## Brief
- Outcome: the harmonic distortion prediction is invariant under a rigid
  rotation of the trial geometry; distortion-row fits stop turning the
  molecule.
- How: set_hessian_for_cd(hess, coords=None) keeps the Hessian's geometry;
  _predict_cd_for_iso aligns the trial geometry onto it before
  compute_cd_constants. MolecularOptimizer passes _hessian_ref_coords at its
  three hand-offs. No geometry stated: old behaviour.
- Checks: rotation invariance (1e-2, 1e-1, 1 rad); the rows' Jacobian along a
  rigid rotation ~0; at the Hessian's own geometry the prediction unchanged;
  the optimizer passes the geometry; mutations; CI selection; rerun T-32's
  cases (formyl fluoride, vinyl fluoride, chlorofluoromethane; none, 0.7, 1.0,
  1.3; autoconfig off and on): iterations, stop reason, moves; README note.
- Trade-off: none expected (correctness); one Kabsch per prediction, cheap.

## Graph
- N1 · build · frame-consistent prediction · after: none · done · handoff: -
- N2 · verify · tests + mutations + CI selection · after: N1 · done · handoff: -
- N3 · measure · T-32 cases rerun on the fix · after: N1 · done · handoff: -
- N4 · verify · independent verifier · after: N2, N3, N5 · done: holds, 1 test gap · handoff: -
- N6 · build · target-prior test, alignment comment, backlog N27 · after: N4 · done · handoff: -
- N5 · build · README note on the fixed numbers · after: N3 · done · handoff: -

## Progress
- N1: SpectralEngine.set_hessian_for_cd(hess, coords=None) and cd_frame_coords
  (unweighted Kabsch onto the Hessian's geometry); _predict_cd_for_iso
  predicts at cd_frame_coords(coords); MolecularOptimizer passes
  _hessian_ref_coords at the three hand-offs, and the harmonic report table
  is built at the aligned geometry too. Formyl fluoride: rotations 0.01-1 rad
  plus a shift change the prediction <= 9e-10 relative (was up to 5.5e-2);
  the rows' Jacobian along rotations <= 1.2e-7 of its largest element.
- N2: dev/tests/test_cd_frame.py, 7 tests (premises asserted: without the
  frame a 0.05 rad turn moves the prediction > 1e-4, the rotation pull at a
  turned geometry > 1e-3). Mutations: trial-frame prediction; geometry
  dropped in set_hessian_for_cd; each of the three hand-offs without it;
  report table in the trial frame -- each fails a test (first try: the
  harmonic and gradient hand-offs and the Jacobian test were not covered;
  fixed). CI selection 1646 passed, 15 skipped.
- N3 (t36/fits): 18 fits with rows (3 molecules x 3 scales x autoconfig off
  / on) stop on the plateau after 11-12 iterations, last accepted step
  <= 6.1e-4 A (on: <= 1.7e-5); moves from the no-row fit: x0.7 / x1.3 bonds
  0.08-3.87 mA, angles <= 0.135 deg; x1.0 <= 0.08 mA. Autoconfig off vs on:
  FF <= 0.013 mA, VF 0.42, CH2ClF 1.69 -- the same as without rows.
  Floor vs no floor (wt_nofloor, autoconfig off): <= 0.076 mA, except FF x0.7
  2.895 mA (bond RMS 12.59 -> 13.69 without the floor).
- N5: README note rewritten on the fixed numbers; comment at the floor.
- N4 (t36v): frame holds -- 0.01-3.1 rad turns move the prediction <= 7.8e-10,
  rotating the Hessian into the trial frame instead agrees to 3.9e-10; Eckart
  vs unweighted alignment: 0.3-2 deg apart, constants 0.1-0.5% (<= 0.022 row
  sigmas), bonds <= 0.062 mA on the 9 autoconfig-off fits. Target-prior path
  holds (rotated Hessian vs fresh 1.3e-6). README numbers reproduce and the
  fits are converged (distance_to_mode <= 3.0e-3 widths; 11 more iterations
  move bonds <= 0.0087 mA). Gap: no test of the target path. Pre-existing:
  linear molecules' prediction is nonsense (N27).
- N6: test_with_a_target_prior_... (Cartesian mode -- my first try ran in
  internal mode where the target prior is off and passed for the wrong
  reason; now asserts the premise); mutation of the refresh hand-off to the
  fit's geometry fails it. Comment: why unweighted, with the measured Eckart
  difference. CI selection 1647 passed, 15 skipped.

## Observations
- 2026-11-22 · the target-prior test first passed for the wrong reason (the
  fixture's internal mode switches the target prior off); asserting the
  premise caught it -- L-15 again.

## Pending question
Q-54 · asked 2026-11-22T10:00Z · deadline 10:15Z · auto-pick 1
1. (Recommended) N27: linear molecules' distortion prediction (fluoroacetylene
   DJ ~1e8 MHz at bent geometries): measure what compute_cd_constants does
   for a linear molecule and what its rows should be (a linear molecule has
   only D), then propose.
2. N7: measure how much split-mode widths miss the step's range-null
   coupling, then propose.
3. N3: look again for published r_e^SE references (sd <= 0.5 mA, D-35).
4. Done for now.
