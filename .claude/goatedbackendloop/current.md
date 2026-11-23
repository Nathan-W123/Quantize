# Current task
Status: verifying
Request: T-71 (D-145): exact curvature where Gauss-Newton is flagged, in the step and in the report
## Brief
- Outcome: the joint step and the reported widths/distance/gate use the exact curvature along flagged soft directions.
- Approach: one correction builder from gauss_newton_discrepancy rows; _posterior_blocks(exact_curvature=True) adds it to the spectral block (the prior part untouched, so the error-model target stays right); the check reads Gauss-Newton blocks; the step gets it through SubspaceOptimizer.step(curvature_correction=...), q-space image with the reach rule in internal mode; the Cartesian fallback step too.
- Trade-off: -5% to +22% per fit where it acts (T-69), answers move on three benchmark molecules by <= 0.2 mA bond RMS (internal mode).
- Success: dimer reproduces T-68's prototype numbers; benchmark rows bit-identical where it does not act; tests for the step, the widths, the gate and the wording; CI selection green.
## Progress
- N1 · step (SVD._joint_step/step curvature_correction; quantize._step_curvature_correction, main loop + Cartesian fallback) and report (_posterior_blocks/_mode_decomposition/distance_to_mode/laplace exact_curvature; _exact_curvature_correction; warning wording) done; option exact_curvature_where_flagged (default True)
- N2 · L-28 check: dimer seed 4 plain and refreshed answers bit-identical to the T-68 prototype runs, both modes
- N3 · tests: test_exact_curvature.py (16), wording test updated, two step spies pass **kw; CI selection 1788 passed, 13 skipped
- N4 · dimer, 8 starts x 2 modes: kept 16/16 everywhere; widths 6.6-9.6 deg (Gauss-Newton 13.5-331); one exact-below-Gauss-Newton direction (internal seed 4) -> report rule made two-way (D-145 refined); README note added
- N5 · running: HF base+offsets with the correction on/off (t71/on, t71/off)
## Observations
- The 227 -> ~1.5 deg figure I gave the user was T-58's old answer; at the corrected, refreshed answer seed 3's widest angle goes 14.2 -> 6.65 deg. Correct it in the wrap-up.
## Pending question
## Handoff
