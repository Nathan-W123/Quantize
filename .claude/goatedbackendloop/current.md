# Current task
Status: clarifying
Request: T-63 (D-137): the scratch step prototype in internal mode (the default), dimer and nine molecules; nothing in the repository
## Brief
- Outcome: evidence for page item 9 on the default path: answers, iterations, distances, run time, off against on.
- Method: the flagged Cartesian direction v with added curvature D maps to the q-space step through the step's own Cartesian back-map dx = B+ dq: add D (B+^T v)(B+^T v)^T to the q-space system.
## Progress
- N1 · dimer: refreshed exact 0.024 -> 0.0006; bootstrap 12 -> 14/16
- N2 · nine: six unchanged; formyl to the restart value; fluoroacetylene and acetyl fluoride worse on the real objective (t63/compare.txt)
- N3 · page item 9 v7
## Observations
- 2026-11-23 · the Gauss-Newton distance said 'worse' for fluoroacetylene and acetyl fluoride; the real objective and restarts confirmed it this time (fluoroacetylene) and qualified it (acetyl fluoride: both are stopping points)
## Pending question
- Q-94 (asked 07:12 UTC, deadline 07:27 UTC): 1 (Recommended) refine the scratch prototype for internal mode: correct only flagged directions the q-space step reaches (the reach rule of _resolve_ic_dihedrals), rerun the internal comparison; scratchpad only; 2 report on the 13 never-passed settings; 3 adopt the step fix for Cartesian mode only (user's call: changes answers; never auto-picked); 4 Done for now. Deadline takes 1.
## Handoff
