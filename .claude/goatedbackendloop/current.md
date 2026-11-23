# Current task
Status: building
Request: T-61 (D-135): why half the water-dimer error-model refits stop 0.18-1.39 widths short after the polish passes
## Brief
- Outcome: a measured reason, then a proposal. No code change before the measurement.
- Record per refit: trust radius at the end, termination, last steps (norm, accepted), polish passes, the direction carrying the distance (GN vs exact curvature), and whether more passes would reach the mode.
## Progress
- N1 · running (t61/stopshort.py): seed-3 refreshed fit, 16 error-model refits (seed 7, as T-60), per-pass trace; then the same draws with 20 polish passes
## Observations
## Pending question
## Handoff
