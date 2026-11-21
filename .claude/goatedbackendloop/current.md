# Current task
Task: T-17
Status: building
Request: measure the acoustic-sum-rule repair of the B3LYP Hessian (backlog N19 b-ii; D-54, auto-picked at Q-21); measurement only, any engine change proposed before it is made.

## Brief
- Outcome: per molecule, how far the repair moves the B3LYP alpha (all
  isotopologues) and every B3LYP benchmark row; whether it moves alpha toward
  a converged grid on a molecule other than acetyl fluoride.
- How: the repair subtracts each atom's row sums from its diagonal block,
  then symmetrises (T-15 verifier's asr()). Prototype: applied in the
  benchmark's hessian_fn after the disk cache (cache stays raw), so alpha
  tables and the joint objective's prior curvature both see it, as an
  engine fix in the backend would. B3LYP only (the defect is the DFT grid).
- Tradeoff: the repair also changes the vibrational block (it is meant to);
  rows that move are not automatically better -- the grid-converged
  reference decides.
- Success: N1 per-molecule alpha table; N2 113 B3LYP rows with/without;
  N3 independent converged-grid check on a second molecule; N4 a proposal.

## Graph
- N1 · diagnose · per-molecule sum-rule violation, rigid eigenvalues, harmonic alpha shift at the B3LYP theory geometry (cached Hessians) · after: none · running · handoff: t17/diag
- N2 · build · prototype in a git-archive copy, rerun the 113 B3LYP rows · after: N1 · pending · handoff: -
- N3 · verify · fresh agent: converged-grid alpha on a second molecule (fluoroethane or formyl fluoride), raw vs repaired, own Hessians · after: N1 · pending · handoff: -
- N4 · gate · propose (or not) an engine change · after: N2, N3 · pending · handoff: -

## Progress

## Observations

## Pending question

## Handoff
