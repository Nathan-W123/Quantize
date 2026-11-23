# Current task
Status: done
Request: T-58 (D-131): why a refreshed fit can stand 0.5 widths from its mode while the refresh reports settled
## Brief
- Outcome: a measured reason (dimer seed 3, Cartesian, joint HF/6-31G) and a proposal. No code change before the measurement.
- Questions: what "settled" checks; how the fit terminated and what its last steps were; how far the Newton step to the mode is in A; whether continuing (same Hessian) reaches the mode; how common it is across the T-57 fits.
## Progress
- N1 · how it stopped (t58/settled.py): plateau, trust radius 0.78 mA; the 0.505 lies along one direction whose GN width is 11.4 A (Newton step 1.36 A rms); continuing 0.17 mA moved it to 0.148
- N2 · the direction (t58/soft.py): the H-bond torsion; raw HF Hessian negative there at all 4 answers (-0.0007..-0.0017), floored; GN curvature = data's only (0.0077 seed 3; 1.5-3.0 others); exact 188-252 (omitted residual term)
- N3 · the report (t58/report.py): widths angle 2-1-5/3-1-5 227 deg (seed 3), 17.6 (seed 23); exact-curvature ~1.3-1.6 deg; warning says 'order-of-magnitude' for 181x and 11x; 'settled' = Hessian age, not the mode; D-04 revisit condition met
- N4 · wording (D-132, auto-picked Q-88): the doubtful warning gives the exact Hessian's width factor and direction; settled docstring; tests; 5/6 mutations (1 equivalent); CI 1763 passed
## Observations
- 2026-11-23 · one of my mutants was equivalent (.replace('', '') changes nothing): a 'missed' needs checking that the mutant changes behaviour before it counts as a test gap
- 2026-11-23 · my Q-87 framing ('the fit stands 0.5 widths off its mode') read a Gauss-Newton distance as the fit's; exact curvature says 0.003 (L-27 again: the diagnostic measured GN widths, not the mode)
## Pending question
- none (Q-88 auto-picked option 1 at 05:23 UTC)
## Handoff
