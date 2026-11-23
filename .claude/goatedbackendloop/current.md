# Current task
Status: clarifying
Request: T-64 (D-138): internal-mode prototype corrects only directions the q-space step reaches; rerun; nothing in the repository
## Brief
- Rule: correct a flagged direction v in q-space only if v^T B+ B v > 0.5 (B+ the step's damped pseudo-inverse).
- Compare with T-63's "off" runs (same code as it stands); real-objective check for any that move.
## Progress
- N1 · refined (t64/bench_on.txt, compare.txt): fluoroacetylene harm gone; dimer, formyl gains kept; six unchanged; acetyl fluoride (no dihedrals) +0.008 between stopping points
- N2 · acetyl fluoride with dihedrals auto (t64/af_auto.txt): unchanged (0.29 mA, Phi within 0.0003)
- N3 · page item 9 v8
## Observations
## Pending question
- Q-95 (asked 07:33 UTC, deadline 07:48 UTC): 1 (Recommended) an independent check (L-10) of the prototype evidence on page item 9 (T-62 to T-64 scripts and figures) before the user decides; no repository change; 2 report on the 13 never-passed settings; 3 adopt the refined step fix in the repository (user's call: changes answers; never auto-picked); 4 Done for now. Deadline takes 1.
## Handoff
