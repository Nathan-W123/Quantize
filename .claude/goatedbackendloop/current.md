# Current task
Status: building
Request: T-62 (D-136): scratch prototype of the joint step with the exact curvature on Gauss-Newton-flagged directions; measure, do not commit code
## Brief
- Outcome: evidence for page item 9: how the dimer fit and its refits and the nine benchmark molecules (Cartesian, joint HF/6-31G) change, and at what cost. No repository code change.
- Method: class-level patch of SubspaceOptimizer._joint_step; each optimizer carries a back-reference to its MolecularOptimizer (deepcopy keeps it consistent for refits); add (c_exact - c_GN) v v^T for flagged directions (stable, >50% apart, exact > GN, same dimension).
- Scope limit: Cartesian mode only (internal mode's step is in q-space).
## Progress
- N1 · running: t62/proto.py (class-level patch), t62/dimer.py and t62/bench.py, off and on, four processes in parallel (equal load)
## Observations
## Pending question
## Handoff
