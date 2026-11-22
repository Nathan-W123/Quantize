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
- N1 · build · frame-consistent prediction · after: none · pending · handoff: -
- N2 · verify · tests + mutations + CI selection · after: N1 · pending · handoff: -
- N3 · measure · T-32 cases rerun on the fix · after: N1 · pending · handoff: -
- N4 · verify · independent verifier · after: N2, N3 · pending · handoff: -
- N5 · build · README note on the fixed numbers · after: N3, N4 · pending · handoff: -

## Progress

## Observations

## Pending question
