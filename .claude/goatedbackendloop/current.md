# Current task
Task: T-37
Status: done
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
- N5 · verify · independent verifier · after: N4 · done: REFUTED (crash at the threshold) · handoff: -
- N6 · build · pin the branch once per isotopologue (D-91) · after: N5 · done · handoff: -
- N7 · verify · re-run the verifier's reproducers on the fix · after: N6 · done: crash fixed; 2 refuted (HNCO, diatomic) · handoff: -
- N8 · build · pin rule B-only + off-axis 0.4 A; diatomics; table pinned (D-92) · after: N7 · done · handoff: -
- N9 · verify · verifier's pin4 / jitter3 / diatomic on the fix · after: N8 · done: holds; notes suppressed · handoff: -
- N10 · build · notes printed when rows are configured; margin documented · after: N9 · done · handoff: -

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

- N5 (t37v): physics holds (HCN D 73.3165 kHz by code, closed form and full HF
  PES under centrifugal load; experiment 87.24(6) kHz, NIST). Refuted: the
  per-geometry 1e-3 switch -- just past it the asymmetric-top rows (DJ, DJK,
  DK) met linear-branch FD steps (NaN), SVD failed, a CLI HCN+DCN fit aborted
  (parent completes); HNCO at its HF minimum is 3.1e-3, not 13x away; long
  polyyne symmetric tops sit at 8e-4. Non-linear bit-identical holds.
- N6: _cd_reduction pins "linear" (no A observed, first geometry < 2e-2);
  stated cd_reduction: linear accepted (engine and CLI validator);
  compute_cd_constants switches only for "auto" or "linear"; report warning
  for dropped DJK/DK; harmonic printout gives D. The verifier's nanjac.py
  now gives a (1, 9) finite J and SVD ok at the threshold; its hcn_lin.yaml
  completes (report: both isotopologues "linear (auto ...)", the warning, D
  rows at +0.41 / +0.40 sigma). 3 more tests; mutations (no pin, A observed
  ignored, any reduction switching by geometry, stated linear not
  normalised) each fail. CI selection 1654 passed; 213 distortion + CLI tests
  incl. slow pass.

- N7 (t37v pin3.py, diatomic.py, jitter2.py): crash fixed (no NaN in any of
  three CLI fits; jittered starts max ratio 6.3e-3, all pinned linear);
  refuted: HNCO pinned linear (observes B, C), B-only CH3C#CCN-type tops too;
  a pinned diatomic raised "Unknown reduction 'LINEAR'"; the harmonic table
  still decided per geometry (printed DJ 1e8 MHz for jittered starts).
- N8: off-axis distance: jittered HCN / FA starts <= 0.212 A (20000 each);
  bent reference molecules >= 0.444 A (ozone), HNCO 0.696, CH3Cl 1.028.
  Rule: B alone observed and off-axis < 0.4 A. Diatomic test against
  4B^3/omega^2 (analytic Hessian, rel 1e-3). Mutations: components ignored,
  no off-axis test, diatomics excluded -- each fails. hcn_lin CLI rerun:
  exit 0, harmonic printout D 72.3-73.8 kHz for every start. CI selection
  1656 passed; 215 distortion + CLI tests incl. slow pass.

- N9 (t37v pin4.py, jitter3.py, diatomic.py, run3_*): every reference
  molecule pinned correctly (fluoroacetylene linear, HNCO S at reference and
  HF minimum); B-only methyl tops (propyne .. CH3(C#C)7CN) 1.02 A off-axis,
  S; jittered starts <= 0.21 A (HCN), 0.16 (FA), 0.14 (HC7N); CO D 0.16324
  MHz = 4B^3/omega^2 to 6 digits; three CLI fits complete, no NaN. Refuted:
  the early pin (harmonic table) suppressed the [cd-note] lines (gated on
  rows active). Margin: the 0.4 A test alone would pin HNCO's HF minimum
  (0.33 A); the B-alone test keeps it out.
- N10: notes gated on rows configured (cd_weight > 0 and fit_cd_constants);
  test + mutation; the margin written at CD_LINEAR_PIN_OFF_AXIS_ANG. CI
  selection 1657 passed; 216 distortion + CLI tests incl. slow pass; hcn_lin
  CLI exit 0 with the linear note printed.
- 2026-11-22 · I introduced a crash with a per-geometry branch inside a
  finite-difference loop; my own tests sat away from the boundary. A branch
  a derivative can cross must be decided once outside it -- test at the
  boundary itself.
- 2026-11-22 · a test fixture that runs pyscf at module scope changed other
  tests' warnings (forks of a multi-threaded process); embedding the
  Hessian removed it. Tests should not start quantum chemistry when data
  will do.

## Pending question
Q-56 (asked 11:37 UTC, deadline 11:52 UTC): what's next after T-37?
1. N28: diagnose fluoroacetylene's crawl (does the solver treat a linear
   molecule's 5 rigid modes as 6?), then propose a fix (Recommended)
2. N7: measure the split estimator's widths against its spread, then propose
3. N3: search again for semi-experimental references (HCOF, fluoroethane,
   acetyl fluoride)
4. Done for now
