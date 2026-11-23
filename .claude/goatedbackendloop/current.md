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
## Handoff
