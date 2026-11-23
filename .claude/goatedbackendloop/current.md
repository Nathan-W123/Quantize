# Current task
Status: done
Request: T-55 (D-125): measure the split posterior's unprojected prior-target pull, then propose
## Brief
- Outcome: how far the pull moves the split posterior's mode, notes and widths; a proposal.
## Progress
- N1 · measure (t55/pull.py) · distance to mode: current 0.023-0.998, step-consistent Pn H d 0.002-0.030, posterior-consistent Pn H Pn d 0.126-1.54; widths unchanged in all; no in-repo user of split + target
- N2 · document only (D-126, auto-picked Q-82): _posterior_gradient and split_off_mode docstrings, README split note; no behaviour change
## Observations
- 2026-11-23 · the reviewer's "0.030 after removing the range part" was one of two projections; the other (the split posterior's own definition) goes the other way (1.54): measure both before choosing
## Pending question
- Q-83 (asked 02:44 UTC, deadline 02:59 UTC): 1 (Recommended) N31 dimer: measure why the water dimer in joint internal mode ends 0.29 widths from its mode (measure first, no code change before); 2 report on the 13 MolecularOptimizer settings nothing outside quantize.py passes (reachable, tested, worth keeping; nothing deleted); 3 N29 angle-row units in the Cartesian internal prior (user's call: accuracy moves both ways); 4 Done for now. Deadline takes 1.
## Handoff
