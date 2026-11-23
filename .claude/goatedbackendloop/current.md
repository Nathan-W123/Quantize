# Current task
Status: clarifying
Request: T-62 (D-136): scratch prototype of the joint step with the exact curvature on Gauss-Newton-flagged directions; measure, do not commit code
## Brief
- Outcome: evidence for page item 9: how the dimer fit and its refits and the nine benchmark molecules (Cartesian, joint HF/6-31G) change, and at what cost. No repository code change.
- Method: class-level patch of SubspaceOptimizer._joint_step; each optimizer carries a back-reference to its MolecularOptimizer (deepcopy keeps it consistent for refits); add (c_exact - c_GN) v v^T for flagged directions (stable, >50% apart, exact > GN, same dimension).
- Scope limit: Cartesian mode only (internal mode's step is in q-space).
## Progress
- N1 · dimer (t62/dimer_*.txt): 32 -> 16 it; refreshed exact 0.054 -> 0.0099; bootstrap 8/16 -> 15/16
- N2 · nine molecules (t62/bench_*.txt): acts on 3; 8 unchanged to 0.004 mA; formyl fluoride 12.47 -> 14.14 mA, the old answer a stopping point (t62/formyl.txt); cost 2-12%
- N3 · page item 9 v6
## Observations
- 2026-11-23 · formyl fluoride's 1.67 mA move looked like harm until measured on the real objective: the old answer was the stopping point (L-27: measure before calling a change good or bad)
## Pending question
- Q-93 (asked 06:50 UTC, deadline 07:05 UTC): 1 (Recommended) extend the scratch trial to internal mode, the default (flagged Cartesian directions carried into q-space), on the dimer and the nine molecules; scratchpad only; 2 report on the 13 never-passed settings; 3 adopt the step fix in the repository (user's call: changes answers; never auto-picked); 4 Done for now. Deadline takes 1.
## Handoff
