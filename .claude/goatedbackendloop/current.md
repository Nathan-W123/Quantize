# Current task
Status: done
Request: T-50 (D-118): narrow independent check of af66799
## Brief
- Outcome: af66799's claims held / refuted with measurements; defects fixed.
## Progress
- N1 · verify · fresh subagent · done (hand-back saved to scratchpad t50v/REPORT.md)
- N2 · fixes · one cut rule for both null spaces; record carried over the last step; off-mode note names the coupling when null spaces agree; docs · done (D-119)
- N3 · checks · 24 tests, 11/11 mutations, CI 1740 passed; benchmark re-measured: note on 9 of 18 fits
## Observations
- 2026-11-23 · a comparison between two subspaces cut by two different rules read 1.0 for identical spaces: compare like with like, by one rule
- 2026-11-23 · "skip when the record is a step behind" hid real cases; carrying the record over the step (exact for the quadratic model) kept them
## Pending question
- Q-77 (asked 00:03 UTC, deadline 00:18 UTC): what next? 1 (Recommended) F3: embed the two 3-atom geometries so the multireference tests run in a fresh clone (they always skip now); 2 a narrow check of c4eb2b6; 3 N34 dimer in joint mode; 4 Done for now. Deadline takes 1.
## Handoff
