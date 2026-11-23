# Current task
Status: done
Request: T-67 (D-141): tests for the geometry guardrails (N35); test-only
## Brief
- Unit: _geometry_validity at a sound geometry, a crushed bond, a stretched bond, a clash, the violation allowance, guardrails off.
- Loop: a forced invalid trial is refused (history), damping and trust radius move by the configured factors (against factors of 1), accept_requires_geometry_valid=False accepts it.
- Mutations on each rule.
## Progress
- N1 · dev/tests/test_geometry_guardrails.py: 9 tests; 13/13 mutations; CI 1772 passed; lint clean
## Observations
## Pending question
- Q-98 (asked 08:49 UTC, deadline 09:04 UTC): 1 (Recommended) the exact-correction trial (the reviewer's t65v ideal prototype, with the reach rule) on the other seven no-point-group dimer starts, Cartesian and internal, off against on, so page item 9 rests on more than seed 3; scratchpad only; 2 remove null_convergence_requires_energy (user's call: an API change); 3 adopt the step fix (user's call: changes answers); 4 Done for now. Deadline takes 1; never 2 or 3.
## Handoff
