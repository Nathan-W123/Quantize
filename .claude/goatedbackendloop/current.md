# Current task
Task: T-37
Status: building
Request: (D-89, auto-picked at Q-54) N27: what the distortion prediction and rows do for a linear molecule, and what they should be; then propose.

## Brief
- Outcome: for fluoroacetylene (and HCN, the calibration-only linear
  molecule), the predicted DJ / DJK / DK at the linear reference and at
  geometries bent by a few degrees; whether the rows are built; what a user
  with a linear molecule's D gets; a proposal.
- Physics: a linear molecule has one quartic constant, D (= DJ), and its
  bending modes are degenerate; DJK and DK do not exist.
- How: compute_cd_constants with the HF/6-31G Hessian at the reference;
  is_linear branches in centrifugal_distortion.py; the rows' keys
  (CD_FIT_KEYS) for a linear isotopologue.

## Graph
- N1 · measure · linear-molecule predictions and code paths · after: none · done · handoff: -
- N2 · gate · proposal (Q-55) · after: N1 · asked · handoff: -

## Progress
- N1 (t37/linear.py): HF/6-31G Hessian at the linear reference. At the linear
  geometry compute_cd_constants gives DJ = -tau'_bb/4 = the linear D
  (fluoroacetylene 2.887e-3 MHz, HCN 7.383e-2 MHz), with placeholder DJK =
  -2D and DK = D (no K for a linear molecule). Bent 0.5 / 1 / 3 deg (the H
  end): DJ 4e11-8e12 MHz, DJK and DK alike -- A turns finite but enormous
  (rotational_constants_mhz keeps I_a > 1e-10 amu A^2) and the asymmetric-top
  reduction runs on it. The rows' finite-difference Jacobian (step ~1e-4 x
  |x|) bends the molecule at every column, so it is garbage even at a linear
  fit. Nothing refuses DJK / DK rows for a linear molecule.

## Observations

## Pending question
Q-55 · asked 2026-11-22T10:19Z · deadline 10:34Z · auto-pick 1
1. (Recommended) Linear branch: when the Hessian's geometry is linear,
   predict only D = -(tau'_bb + tau'_cc)/8 from the two perpendicular axes
   (smooth through small bends; the rigid-mode count taken from the
   Hessian's geometry), refuse DJK / DK rows with a warning, and test D at
   linear and bent geometries and through a fit. Opt-in distortion fits of
   linear molecules only; a correctness fix.
2. Refuse distortion rows for linear molecules altogether (warn, no rows).
3. Document as a known limitation.
4. Done for now.
