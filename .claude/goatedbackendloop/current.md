# Current task
Task: T-36
Status: building
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
- N4 · verify · independent verifier · after: N2, N3, N5 · pending · handoff: -
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

## Observations

## Pending question
