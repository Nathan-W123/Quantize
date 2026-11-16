# Current task
Status: verifying (round 9: physics fixed, CLI check running)
Request: N6 -- is the benchmark recipe (scripts/accuracy_upgrades_benchmark.py
hybrid_fit: Cartesian, max_iter 40, hess_recalc_every 10, chi2 rescale x3)
restart-stationary, and does the published table depend on where it stopped?

Earlier tasks (the 7-item verification, steps 1-8, rounds 1-8): their
progress log is in this file's git history up to d5ce548; decisions D-01..D-24
and lessons L-01..L-12 carry what they settled.

## Brief
Finding: the benchmark fits were converged (same-object restart <= 0.0003
mA, <= 0.21 widths from their mode); "not restart-stationary" came from
restarts that recomputed the Hessian at the answer. With prior_target_coords
the prior is a Gaussian at the target whose precision is H, so a Hessian
taken elsewhere moves the prior. Fix (D-25): such a prior keeps the
target's Hessian (computed once there, rotated into the frame after).
Success: restarts and the refresh stationary; benchmark answers unchanged.

## Progress
- scratchpad n6/n6.py on 9 molecules (worktree at d5ce548): fit1, same-object
  restart (Hessian kept), fresh restart (Hessian at x1) twice, and the N1
  refresh then a fresh restart.
- Before (d5ce548): fresh restart moves isocyanic acid 97.2 mA (then 38.3;
  refresh unsettled after 3 passes, bond 23.98 -> 40.13), CFM 3.98, VF
  0.69, FF 0.54, rest 0.03-0.30; same-object restarts <= 0.0003 mA.
- DONE 078633f: target prior keeps the target's curvature. After: fresh
  restarts <= 0.0011 mA, refresh <= 0.0009; fit1 scores agree to 1e-8 mA
  (README table stands); Hessians 1/fit as before. Tests: rotation exact
  (analytic water, FD-limited 7e-8), target-prior restart < 1e-6 mA
  (8e-10; was 4.6e-5), refresh moves 0 -- both fail on the old code.
- Clean-checkout suite at 078633f: 1565 passed, 23 skipped.
- Round 9 check (user: "Check recent fixes"), physics agent: nothing
  refuted; rotated target Hessian exact under PySCF (1e-7 rel, linear
  included); benchmark restarts 97 -> 0 mA reproduced. Fixed: 84b609c
  wording (null cut binds only with an explicit null radius; small radius
  still unsafe on isocyanic acid split, 100.3 MHz; split + target changes,
  14.60 -> 15.52 mA), 5e40ada strict JSON (NaN -> null in all five files),
  9a0c9ce detectable bounds = 50%-power points (48% / 39% reproduced, seed
  20261117), no tail caveat for a zero-spread cloud. CLI agent pending.

## Observations
- The step-7 restart changed two things at once (a fresh optimizer is also a
  fresh Hessian), and the backlog item read the result as a stopping-rule
  defect. Splitting the restart into same-object / fresh / refresh took one
  9-molecule run (~4 min) and named the real cause. Separate the variables a
  "restart" changes before diagnosing it.

## Handoff
