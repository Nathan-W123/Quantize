# Current task
Status: clarifying (measuring)
Request: N6 -- is the benchmark recipe (scripts/accuracy_upgrades_benchmark.py
hybrid_fit: Cartesian, max_iter 40, hess_recalc_every 10, chi2 rescale x3)
restart-stationary, and does the published table depend on where it stopped?

Earlier tasks (the 7-item verification, steps 1-8, rounds 1-8): their
progress log is in this file's git history up to d5ce548; decisions D-01..D-24
and lessons L-01..L-12 carry what they settled.

## Brief
(pending measurement) Step 7 found 4 of 9 molecules not restart-stationary
under the recipe (isocyanic acid 97 mA / 13.8 deg). Every fit stopped on the
plateau rule at iteration 11-13 with one Hessian (at the start). A restart
was a fresh optimizer, so it also had a fresh Hessian: premature stop and a
stale quantum model are confounded. Measuring both separately at HEAD.

## Progress
- scratchpad n6/n6.py on 9 molecules (worktree at d5ce548): fit1, same-object
  restart (Hessian kept), fresh restart (Hessian at x1) twice, and the N1
  refresh then a fresh restart.

## Observations

## Handoff
