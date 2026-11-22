# Current task
Status: clarifying
Request: T-39 (N29, D-95): measure how much the internal prior's 57.3x angle-row Jacobian moves the fits that use it, then propose the unit fix
## Brief
- Outcome: who reaches InternalPriorEngine (runner presets, library flags),
  how much its unit error moves their fits (geometry, iterations, the ratio
  test), and a proposed fix measured before it is written.
- Approach: find every path that stacks it; run fits with the shipped rows and
  with the Jacobian put in degrees (a monkeypatch), same starts.
- Success: numbers per path, the population named; a fix proposed, not
  implemented, until agreed.
## Progress
- N1 · reach · which configs stack the Cartesian internal prior · pending
- N2 · measure · shipped vs unit-consistent rows · after: N1 · pending
## Observations
## Pending question
## Handoff
