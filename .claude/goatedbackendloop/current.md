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
- N1 · measure · linear-molecule predictions and code paths · after: none · pending · handoff: -
- N2 · gate · proposal (Q-55) · after: N1 · pending · handoff: -

## Progress

## Observations

## Pending question
