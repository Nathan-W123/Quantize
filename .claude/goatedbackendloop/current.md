# Current task
Status: building
Request: T-38 (N28, D-93): diagnose why fluoroacetylene's plain joint fit crawls, then propose a fix
## Brief
- Outcome: a measured cause for the crawl (HF/6-31G, autoconfig off, cartesian,
  T-34 fit.py; 57 iterations, last steps ~6e-3 A at ~95% of the radius) and a
  proposed fix, measured before it is written.
- Approach: split the last steps into rigid vs internal motion (L-23); check
  how the solver counts rigid modes for a linear molecule (5, not 6); look at
  the ratio test along the step (rho, predicted vs actual).
- Success: the cause is named with numbers that a different mechanism would
  not produce; the fix (if any) is proposed, not implemented, until agreed.
## Progress
- N1 · diagnose · step decomposition + rigid-mode count · done (t38 diag.py, terms.py)
- N2 · prototype · Cartesian steps on J P_rigid, 9 molecules off/on · done (t38 fits/, compare.py)
- N3 · propose · Q-57 · asked
- N1: the rigid-mode count is right (6 kept at 179.7 deg, s_min 7e-4 > 1e-8;
  5 only when exactly linear). From iteration ~10 the steps are 95%
  translation + rotation (5.64e-3 of 5.95e-3 A translation; internal part
  5e-6 A per step). Cause: once B ~ C (gap/scale 4.6e-7 < 1e-4) the B
  Jacobian falls back to central finite differences, whose per-atom steps
  break translation/rotation invariance at 1.5e-5 of |J| (analytic: 0);
  J^T r then has a rigid part (2.6e-3 translation, 1.0e-3 rotation; alpha g's
  is 1e-15), and the rigid directions of the joint matrix carry only the
  1e-8 spd floor and lambda_damp (decayed to 1e-8), so the unclipped step is
  3.75 A, 3.55 of it translation; clipped to the 6.25e-3 A radius the
  internal motion is ~1.6e-3 of what it should be. The comment at SVD.py
  ~316 ("both J^T r and g are orthogonal to them") is false on the
  finite-difference path. Internal mode is immune (J B+ excludes rigid
  motion); the CLI default is internal, the benchmark Cartesian.
- N2 (t38 compare.py; HF/6-31G, autoconfig off, cartesian, 200 iterations):
  fluoroacetylene 57 -> 19 iterations, last step 5.9e-3 -> 2.4e-8 A, 17 -> 7 s;
  C-H 1.05194 -> 1.04939, C#C 1.17863 -> 1.17917, C-F 1.29783 -> 1.29689 A,
  bond RMS 15.719 -> 15.376 mA (the crawl had stopped up to 2.6 mA from the
  fit's own optimum). Second mechanism: the analytic Jacobian (fallback off)
  gives the same 19 iterations and 15.376. The other 8 molecules never hit
  the fallback (0 finite-difference calls, leak <= 1.3e-15): same iteration
  counts, geometries within 1e-6 mA and 2e-9 deg. With D rows (scale 1.0,
  40 iterations): off still crawling at the cap (last step 2.8e-3 A), on
  plateau at 19 (1.7e-8 A), bond RMS 15.676 -> 15.368.
- 2026-11-22 · I wrote a guessed question time again (Q-57: 12:41 written at
  12:02), the third time (Q-50, Q-55). Read `date -u` first, then write.
- Side note, not chased: the converged angle is 179.796 deg, not 180.

## Pending question
## Handoff
Q-57 (asked 12:03 UTC, deadline 12:18 UTC): fix fluoroacetylene's crawl?
1. Implement: Cartesian steps use the Jacobian with its rigid part removed
   (J P_rigid at the current geometry; exact for the true Jacobian), in the
   joint and split steps and the ratio test; correct the SVD.py comment;
   tests with premises; independent verifier (Recommended)
2. As 1, and also measure lowering the analytic-Jacobian fallback threshold
   (1e-4) so near-linear molecules keep the analytic derivative
3. Leave the code; note in the README that Cartesian fits of near-linear
   molecules crawl
4. Done for now
