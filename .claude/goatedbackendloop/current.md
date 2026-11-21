# Current task
Task: T-18
Status: building
Request: measure how far the theory geometries sit from tight minima of their own surface (backlog N21; D-56, auto-picked at Q-23); measurement only, any change proposed before it is made.

## Brief
- Outcome: per molecule and level, the RMS distance and the largest bond
  change from the cached theory geometry to a tight minimum of the same
  surface (the backend's default-grid gradient), and whether any README
  figure would move.
- How: one gradient at each theory geometry; a Newton step on the cached,
  sum-rule-repaired Hessian with rigid modes projected out gives the
  displacement to the minimum (exact for a quadratic surface). Check it
  against real tight optimisations on two molecules.
- Tradeoff: the theory leg itself is scored at this geometry, so a tighter
  optimisation changes the "theory alone" column directly, and the hybrid
  through the prior centre and alpha.
- Success: the per-molecule table; Newton estimate agreeing with a tight
  optimisation; a proposal (or not).

## Graph
- N1 · diagnose · gradient + Newton step at every theory geometry, both levels · after: none · running · handoff: t18/
- N2 · verify · tight optimisation on two molecules (one per level) vs the Newton estimate · after: N1 · pending · handoff: -
- N3 · gate · propose (or not) tighter theory geometries · after: N1, N2 · pending · handoff: -

## Progress

## Observations

## Pending question

## Handoff
