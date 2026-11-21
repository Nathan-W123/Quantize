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
- N1 · diagnose · per-molecule sum-rule violation, rigid eigenvalues, harmonic alpha shift at the B3LYP theory geometry (cached Hessians) · after: none · done · handoff: t17/diag.json, diag_hf.json
- N2 · build · prototype in two git-archive copies (repair after the disk cache, B3LYP only), rerun base + offsets for all 9 (18 rows; the full 113 only if adopted) · after: N1 · running since 03:36 (t17/logs, chains a/b, timeout 7000 s) · handoff: t17/proto_a, proto_b
- N3 · verify · fresh agent: converged-grid alpha on formyl fluoride and ozone (levels 3/5/7, fixed geometry + re-optimised control), raw vs repaired, own Hessians, 1 thread · after: N1 · running · handoff: t17/verify
- N4 · gate · propose (or not) an engine change · after: N2, N3 · pending · handoff: -

## Progress
- N1 (cached Hessians at each B3LYP theory geometry, harmonic alpha through
  the engine, projected): every B3LYP Hessian breaks the sum rule -- row
  sums 2e-4 (water) to 1.9e-2 Ha/bohr^2 (CH2ClF) -- with raw rigid
  eigenvalues to 23-117 cm-1; repaired, <= 20 cm-1 (rotations keep a few).
  CH2ClF's lowest vibration moves 381.5 -> 376.2 cm-1. Alpha on observed
  components moves <= 0.81 MHz (CD3-AF A, CH2ClF Cl-35 A), <= 0.8% of its
  sigma (CH2ClF 0.19%, the rest <= 0.07%). Unobserved A of linear FA /
  HNCO is garbage either way (FA's A diverges). RHF Hessians satisfy the
  rule to ~1e-9: the repair is a no-op there.
- Rows will move little, so N2 measures base + offsets only (18 rows); the
  full 113 rerun waits for adoption.

## Observations

## Pending question

## Handoff
