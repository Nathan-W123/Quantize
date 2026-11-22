# Current task
Status: done
Request: T-48 (D-114, D-115): why split-objective widths depart from the split estimator's spread (N7); then say so per fit
## Brief
- Outcome: the cause measured (split fits stall short of their fixed point); a per-fit check and an honest warning; split answers unchanged.
## Progress
- N1-N3 · diagnosis (t48/lr.py, probe.py, nullgap.py, sched.py) · coupling-only linear response refuted; null-space gradient left at the stop; rank cut sv_min_abs = 1/sigma_x vs data-scoring acceptance; answer moves up to 1.47 widths with the trust radius
- N4 · split_stall + split_inconsistency (stall first) + README · done; 5 tests, 4/4 mutations; CI 1718 passed
## Observations
- 2026-11-22 · my first hypothesis (the coupling N7 named) was refuted by computing it, not argued; the fixed-point residuals found the real cause
- 2026-11-22 · a mutation (gradient at the reference point) passed because the fixture sits on its quantum point; moving the test geometry off it made the test bite (L-15)
- 2026-11-22 · re-counting a docstring figure against the table caught 12 -> 11 (L-18)
## Pending question
- Q-75 (asked 22:02 UTC, deadline 22:17 UTC): what next? 1 (Recommended) independent verifier on 58d159a (T-45 fixes) and 9c1ad3c (split stall), neither checked by anyone but me; 2 ozone's 0.006-width stop; 3 N34 dimer in joint mode; 4 Done for now. Deadline takes 1.
## Handoff
