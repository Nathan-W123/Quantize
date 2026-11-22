# Current task
Status: clarifying
Request: T-48 (D-114): why split-objective widths depart from the split estimator's spread (N7)
## Brief
- Outcome: the cause, measured; then a proposal. No change before the cause is shown.
## Progress
- N1 · linear response (t48/lr.py, Cartesian split, 9 molecules) · the range-null coupling K alone does not reproduce T-40's clouds: lr/cloud 0.02-4.4 against target/cloud 0.02-5.7; water and ozone (K = 0) match (1.16, 1.00); my range-only response reproduces the engine's target where the data reach every direction, so units and noise model are right
- N2 · fixed-point check (t48/probe.py, nullgap.py) · split fits do not reach the split fixed point: the range condition holds (1e-6) but the null-space energy gradient does not vanish; one Newton step to the null minimum moves an atom 2.9-16.6 mA (FA 4.2, FF 16.6, VF 10.8, CFM 2.9, FE 9.3; HNCO 2.2 A, its quadratic meaningless); every fit ends with its null steps rejected at the 1e-4 A trust floor; four FF refits likewise (null gradient 0.02-0.12)
- N3 · cause (t48/sched.py) · the rank cut is sv_min_abs = 1/sigma_x: directions the data constrain less than the prior (singular values 0.2-6% of the largest, FF/VF/CFM) go to the energy-only null step, but the acceptance test still scores the data there (FF: the null step takes 69 -> 180 MHz), so it is refused and the fit stalls; the split answer then depends on the step schedule: trust radius 0.05 / 0.2 instead of 0.1 moves it up to 1.47 of its reported width (FF 11.6 mA, bond RMS 20.19 -> 9.55 mA)
- Not checked: internal-mode split (same mechanism expected; T-40's internal clouds depart alike)
## Observations
- 2026-11-22 · my first hypothesis (the coupling N7 named) was refuted by computing it, not argued; the fixed-point residuals found the real cause
## Pending question
- Q-74 (asked 21:38 UTC, deadline 21:53 UTC): what to do about the split stall? 1 (Recommended) per-fit check + honest split warning (reporting only, split answers unchanged); 2 also withhold split widths when it fires; 3 fix the split solver (changes split answers: your call); 4 Done for now. Deadline takes 1.
## Handoff
