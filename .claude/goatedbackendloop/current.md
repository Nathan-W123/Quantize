# Current task
Status: done
Request: T-69 (D-143): the trial fix's real cost, every-step check against once per quantum Hessian; nothing in the repository
## Brief
- Variants: off (code as it stands), every step (t65v/ideal), once per Hessian (cached rows, reused until the Hessian changes).
- Sequential, 3 repeats, wall and CPU time; answers of the cached variant against every-step.
## Progress
- N1 · cost (t69/summary.txt): every-step check +17-22% where it acts, bootstraps +24-45%, else within 10-20% run-to-run noise; once-per-Hessian not viable (dimer 7 mA, Cartesian bootstrap 0/16)
- N2 · page item 9 v11
## Observations
- 2026-11-23 · identical fits vary 10-20% in wall time between sequential runs here: cost claims under that need repeats in one process, interleaved (L-11)
- 2026-11-23 · a driver imported a module by a fixed name and the cached variant's file had another; the run failed silently into a log (check each variant's output exists before reading)
## Pending question
- Q-100 (asked 09:41 UTC, deadline 09:56 UTC): 1 (Recommended) an independent fact-check (L-10, L-18) of page item 9's rows added since T-65 (seven more starts, cost) against t68/ and t69/ outputs; no repository change; 2 adopt the step fix (user's call: changes answers; never auto-picked); 3 remove null_convergence_requires_energy (user's call); 4 Done for now. Deadline takes 1.
## Handoff
