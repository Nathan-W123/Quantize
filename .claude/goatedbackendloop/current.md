# Current task
Status: done
Request: T-71/T-72 (D-145 to D-149): exact curvature where Gauss-Newton is flagged; Hessian refresh in quantize run; use_dihedrals auto default; multistart dihedrals over every start; null_convergence_requires_energy removed. Results-table rebuild waits (user).
## Brief
- Done and verified; see decisions D-145 to D-149 and their measured/reviewed notes.
## Progress
- Implementation reproduces the prototype bit for bit; dimer 8 starts x 2 modes: kept 14-16 of 16, widest angle 6.6-9.6 deg (GN 13.5-331); cost <= 5% per fit; RHF benchmark rows: only isocyanic acid moves (+0.26/+0.45 mA, a near-tied stationary point).
- Independent review (t72v/REPORT.md): 7 findings, 6 fixed, 1 to backlog (N39); 9 mutations caught; CI 1806 passed.
## Observations
- The 227 -> ~1.5 deg figure I gave the user was from an old answer; corrected (14.2 -> 6.65 deg at seed 3).
- My "waiting on you" list missed 8 page items; corrected in chat.
- L-25 missed the bootstrap target's assumption; the review caught it.
## Pending question
- Q-102: what's next (asked in chat; no auto-pick: options are the user's calls or need their go-ahead on cost).
## Handoff
