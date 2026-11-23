# Current task
Status: building
Request: T-64 (D-138): internal-mode prototype corrects only directions the q-space step reaches; rerun; nothing in the repository
## Brief
- Rule: correct a flagged direction v in q-space only if v^T B+ B v > 0.5 (B+ the step's damped pseudo-inverse).
- Compare with T-63's "off" runs (same code as it stands); real-objective check for any that move.
## Progress
- N1 · running: t64/proto.py (reach rule), t64/dimer.py and t64/bench.py, on only (off = T-63)
## Observations
## Pending question
## Handoff
