# Current task
Status: clarifying
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
- N1 · diagnose · step decomposition + rigid-mode count · pending
## Observations
## Pending question
## Handoff
