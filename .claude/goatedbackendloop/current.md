# Current task
Status: done
Request: T-60 (D-134): does the bootstrap's 0.25-width gate refuse water-dimer refits that sit at their exact-Hessian mode?
## Brief
- Outcome: per refit, the Gauss-Newton distance the gate reads and the exact-Hessian distance; kept counts; what the summary says. No code change before the measurement.
## Progress
- N1 · gate (t60/gate.txt): 8/16 kept for both; refused refits 0.25-2.71 GN widths and 0.18-1.39 exact-corrected (all recorded ones): they stopped short; the gate is right
- N2 · correction (t60/base_dirs.txt): seed-3 base distance 0.054 exact-corrected (0.003 was the torsion component); docstring and page v4 corrected
## Observations
- 2026-11-23 · L-27 held: I set out to test 'the gate refuses refits at their exact mode' and measured it false
- 2026-11-23 · fact-check by measurement caught my T-58 '0.003' (one direction's component quoted as a maximum) (L-18)
- 2026-11-23 · keying a recorder on id(obj) lost entries (ids are reused after garbage collection)
## Pending question
- Q-91 (asked 06:08 UTC, deadline 06:23 UTC): 1 (Recommended) measure why half the water-dimer refits stop 0.18-1.39 widths short even after the polish passes (trust radius, plateau rule, which direction; no code change before); 2 report on the 13 never-passed settings (nothing deleted); 3 Done for now. Deadline takes 1.
## Handoff
