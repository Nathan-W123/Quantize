# Current task
Status: done
Request: T-66 (D-140): report on the 13 MolecularOptimizer settings nothing outside quantize.py passes
## Brief
- For each: what it does (read the code), every entry path that could set it (API keyword, config runner, CLI, scripts, tests), whether a test exercises a non-default value, and a recommendation (keep / expose / candidate for removal). Report only.
## Progress
- N1 · report (t66/REPORT.md): keep 11, null_convergence_requires_energy a removal candidate, guardrails on in the runner but untested (0 refusals on the nine fits); backlog N35
## Observations
## Pending question
- Q-97 (asked 08:27 UTC, deadline 08:42 UTC): 1 (Recommended) N35: a test of the geometry guardrails (a bond-crushing step refused, damping and trust radius adjusted; off by default stays off), test-only, with mutation checks; 2 the corrected trial fix on more dimer starts (scratchpad only); 3 remove null_convergence_requires_energy (user's call: an API change); 4 Done for now. Deadline takes 1.
## Handoff
