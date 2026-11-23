# Current task
Status: building
Request: T-69 (D-143): the trial fix's real cost, every-step check against once per quantum Hessian; nothing in the repository
## Brief
- Variants: off (code as it stands), every step (t65v/ideal), once per Hessian (cached rows, reused until the Hessian changes).
- Sequential, 3 repeats, wall and CPU time; answers of the cached variant against every-step.
## Progress
- N1 · running (t69/timing.py, sequential off -> every -> cached, 3 repeats): nine molecules internal defaults + dimer seed 4 internal/Cartesian with a 16-refit bootstrap
## Observations
## Pending question
## Handoff
