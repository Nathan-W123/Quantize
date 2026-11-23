# Current task
Status: clarifying
Request: T-61 (D-135): why half the water-dimer error-model refits stop 0.18-1.39 widths short after the polish passes
## Brief
- Outcome: a measured reason, then a proposal. No code change before the measurement.
- Record per refit: trust radius at the end, termination, last steps (norm, accepted), polish passes, the direction carrying the distance (GN vs exact curvature), and whether more passes would reach the mode.
## Progress
- N1 · trace (t61/stopshort.txt): refused refits end on plateau at trust 0.78 mA; exact distance along the stiffest direction; 20 passes: slow progress, two reach their exact mode and are still refused (GN 0.13, 0.61)
- N2 · cause (t61/steps.txt): proposed steps 97.6-100% along the GN-softest direction (torsion), refused down to 0.8-3.1 mA; stiff directions barely move; page item 9 v5, backlog N31
## Observations
- 2026-11-23 · L-27 held twice: I named the step-direction mechanism only after measuring it (98-100% along the torsion)
## Pending question
- Q-92 (asked 06:28 UTC, deadline 06:43 UTC): 1 (Recommended) prototype in the scratchpad only, no repository change: give the joint step the exact curvature along the directions the Gauss-Newton check flags, and measure the dimer's refits (kept, distances, iterations) and the nine benchmark fits' answers and run time, as evidence for page item 9; 2 report on the 13 never-passed settings; 3 Done for now. Deadline takes 1.
## Handoff
