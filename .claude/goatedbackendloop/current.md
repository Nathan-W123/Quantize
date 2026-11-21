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
- N2 · build · prototype in two git-archive copies (repair after the disk cache, B3LYP only), rerun base + offsets for all 9 (18 rows; the full 113 only if adopted) · after: N1 · running: first launch 03:23 died at the harness's 30-min default (VF base done, 5.2307 -> 5.24); resumed 03:54 with base first, then offsets (tags a_base/b_base/a_off/b_off, 2 h limit) · handoff: t17/proto_a, proto_b
- N3 · verify · fresh agent: converged-grid alpha on formyl fluoride and ozone (levels 3/5/7/9, fixed geometry + re-optimised control), raw vs repaired, own Hessians · after: N1 · done: verdict (a) at fixed geometry, with a geometry caveat (see Progress) · handoff: t17/verify
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

- N3 verifier (own pyscf Hessians, levels 3/5/7/9, 1 thread): at the
  fixed theory geometry the repair takes level-3 alpha to within 0.032 MHz
  (HCOF) / 0.012 (O3) of level 9; projection alone 0.40 / 0.49; 12/12 and
  15/15 components improve; levels 7 and 9 agree to <= 0.005. All <= 1.2e-3
  sigma. Not asked, found: (i) projection WITHOUT the repair is worse than
  no projection on O3 (0.12 -> 0.49 MHz; A) -- repair + projection is the
  right pair at a DFT default grid; (ii) the B3LYP theory geometries are not
  tight minima: HCOF |P g_mw| 7.5e-5, 2.6e-4 A from the tight level-3
  minimum (gradient also lacks grid response) -- moves A_e 17 MHz and alpha_A
  0.42 MHz, as large as the grid effect; O3 within 8e-6 A. Level-9 minimum vs
  theory geometry: alpha 0.36 (HCOF) / 0.70 MHz (O3).
- N2 so far (repaired, base): VF 5.2307 -> 5.24 hybrid; FF hybrid 2.5558 ->
  2.52, ME 0.963 -> 0.94; AF hybrid 8.33 = 8.33, ME 5.677 -> 5.72 (toward
  the pre-projection 5.74).

## Observations
- 2026-11-21 · a background launch without an explicit timeout got the harness's 30-min default, not the inner `timeout 7000`; the chains died with 1 of 18 rows. L-09's limit check must include the harness's own default, not just the job's.

## Pending question

## Handoff
