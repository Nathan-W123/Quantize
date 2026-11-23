# Current task
Status: verifying
Request: T-72 (D-146 to D-149): Hessian refresh in quantize run; use_dihedrals auto default; multistart dihedrals when any start needs them; remove null_convergence_requires_energy. Results table rebuild waits (user: "5 wait on").
## Brief
- Each is a default or API change the user approved; measure each against its page figures after the change.
## Progress
- N1 · D-149 option removed; D-147 auto default (constructor, config schema, translate_config); D-148 multistart probes every start; D-146 run_multistart(refresh_hessian) per start, run_generic.main(refresh_hessian=True), --no-hessian-refresh on quantize run and run_from_config; README + docstrings
- N2 · tests: two default tests now assert auto; dimer note test pins dihedrals=False; 3 multistart tests (Cs then no-point-group -> both get dihedrals); test_run_hessian_refresh.py (2); end-to-end water test pins the no-refresh bounds and checks the refreshed run (bond 2.42 -> 0.77 mA, B target worse than B0 -- flag to user); CI 1796 passed
- T-71 still open in the background: HF base+offsets on/off comparison (t71/on, t71/off), then the interleaved timing run (t71/timing.py).
## Observations
- My "what is waiting" answer listed 4 of the page's 12 items; corrected in chat (items 1, 3, 5, 7, 8 and 10-12 still waiting).
## Pending question
## Handoff
