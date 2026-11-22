# Current task
Task: T-37
Status: verifying
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
- N2 · gate · proposal (Q-55) · after: N1 · done (D-90, auto-picked 1) · handoff: -
- N3 · build · linear branch + D-only rows · after: N2 · done · handoff: -
- N4 · verify · tests + mutations, fixed-Hessian identity, FA fits, CI · after: N3 · done · handoff: -
- N5 · verify · independent verifier · after: N4 · pending · handoff: -

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

- N3: centrifugal_distortion.CD_LINEAR_REL_TOL = 1e-3 and _linear_cd_constants
  (D = -(tau'_bb + tau'_cc)/8, five rigid modes, NaN for the rest);
  SpectralEngine._cd_keys keeps DJ alone for a "linear" prediction (note once)
  in both _cd_rows and _cd_jacobian; README sentence.
- N4: t37/linear.py rerun: FA and HCN at 0, 0.5, 1, 3 deg give D 2.887e-3 /
  7.383e-2 MHz (3 deg: 2.888e-3 / 7.392e-2), DJK and DK NaN. Old and new
  compute_cd_constants on one Hessian per molecule: bit-identical for all
  eight non-linear molecules, FA differs only in DJK/DK/delta (now NaN).
  dev/tests/test_cd_linear.py, 4 tests on an embedded HF/STO-3G HCN Hessian
  (a pyscf fixture left OpenMP threads that made later forking tests warn:
  37 -> 123 warnings; embedded, back to 37). Mutations: tolerance back to
  1e-6, the a axis included, DJK/DK rows kept, Jacobian ignoring the linear
  keys -- each fails; six rigid modes survives (equivalent: the extra mode is
  a bend with dB/dQ = 0 at linear, and the 50 cm-1 cut removes near-zero
  modes either way). FA fits with D rows (t37/fits): finite, D sigma at the
  0.01 MHz floor; but FA crawls with or without rows (N28). CI selection
  1651 passed, 15 skipped; all 139 distortion tests incl. slow pass.

## Observations
- 2026-11-22 · a test fixture that runs pyscf at module scope changed other
  tests' warnings (forks of a multi-threaded process); embedding the
  Hessian removed it. Tests should not start quantum chemistry when data
  will do.

## Pending question
