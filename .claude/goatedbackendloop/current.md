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
- N1 · diagnose · gradient + Newton step at every theory geometry, both levels · after: none · RHF done; B3LYP running (t18/newton_b3lyp.log) · handoff: t18/newton_hf.json
- N2 · verify · tight optimisation on two molecules (one per level) vs the Newton estimate · after: N1 · pending · handoff: -
- N3 · gate · propose (or not) tighter theory geometries · after: N1, N2 · pending · handoff: -

## Progress
- N1 RHF (12 geometries; t18/newton_hf.json): |g|max 2e-6 to 1.8e-5 Ha/A
  (gtol 2e-5); Newton distance <= 1.2e-4 A RMS except H2O2 4.8e-4 (left out
  at RHF, trans) -- torsions; largest bond change 0.008 mA, theory-column
  RMS change <= 0.005 mA. RHF theory geometries are converged for bonds.
- N2 RHF: tight L-BFGS-B (gtol 1e-7) of fluoroethane from its theory
  geometry lands 5.0e-7 A from the Newton point (move 1.19e-4 A); bond
  change 0.0053 vs 0.0054 mA. The Newton estimate holds.

## Observations

## Pending question

## Handoff
