# Current task
Status: building
Request: T-73 (D-150): scratch measurement, ratio test on the corrected model
## Brief
- Patch a scratch copy: stash the step's K, subtract 1/2 dp^T K dp from the ratio test's predicted reduction. Run the HF base+offsets benchmark rows and the dimer (8 starts x 2 modes). Compare with t71/on (current) and t71/off, and log p at isocyanic acid's answers.
## Progress
- N1 · HF base+offsets with the patch: all 18 rows identical to today's code to 4 decimals, isocyanic acid 24.2325 / 29.0037 (not back to 23.98 / 28.56). Probe: the patch fired on both of HNCO's corrected steps (ratio_k 2), both accepted either way, so acceptance is not what steers it: the step direction is. Hypothesis refuted.
- N2 · dimer 8 x 2 with the patch: running (t73/dimer)
## Observations
## Pending question
## Handoff
