# Current task
Status: done
Request: T-74 (D-151): profile the per-step exact-curvature check (scratch only)
## Brief
- Profiled seven HF/6-31G fits with timers on the check and its parts.
## Progress
- N1 · check = 0.3-2.2% of a benchmark molecule's fit, 4% of the dimer's; FD residuals ~2/3 of it; Jacobian reuse would save <1%: not proposed.
- N2 · corrected T-71's attribution (README, docstring, page v16, D-145 notes, L-27 outcome).
## Observations
- L-27 broken in T-71 (named the check as the cause of a wall gap without profiling); caught by measuring in T-74.
## Pending question
- Q-104 (asked 20:10 UTC): every option is the user's call (page decisions, table rebuild, a cost change), so no auto-pick and no reminder; wait.
## Handoff
