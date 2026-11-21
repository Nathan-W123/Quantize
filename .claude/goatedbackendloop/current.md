# Current task
Task: T-17
Status: building
Request: (now D-55, engine change) measure the acoustic-sum-rule repair of the B3LYP Hessian (backlog N19 b-ii; D-54, auto-picked at Q-21); measurement only, any engine change proposed before it is made.

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
- N2 · build · prototype in two git-archive copies (repair after the disk cache, B3LYP only), rerun base + offsets for all 9 (18 rows; the full 113 only if adopted) · after: N1 · done 05:30 (t17/compare.txt) · handoff: t17/proto_a, proto_b
- N3 · verify · fresh agent: converged-grid alpha on formyl fluoride and ozone (levels 3/5/7/9, fixed geometry + re-optimised control), raw vs repaired, own Hessians · after: N1 · done: verdict (a) at fixed geometry, with a geometry caveat (see Progress) · handoff: t17/verify
- N3b · verify · CH2ClF level-7 Hessian at its theory geometry: does the repaired level-3 vibrational block / alpha match it (the one molecule the repair moves) · after: N2 · done: yes (see Progress) · handoff: t17/cfm_grid.json, cfm_L7.npz
- N4 · gate · propose (or not) an engine change · after: N2, N3, N3b · done: option 1 auto-picked (D-55) · handoff: -
- N5 · build · engine helper + call sites + test (fails on the old code) + full suite on 3.12 · after: N4 · done (ba67cfe) · handoff: -
- N6 · build · rerun every row at both levels from the commit (B3LYP 113, RHF 135; shared raw cache) · after: N5 · done 09:00 (all ba67cfe, archived) · handoff: t17/final, reports/accuracy_upgrades
- N6b · verify · engine rows reproduce the prototype's 18 (all consumers covered) · after: N6 · done: max |d| 8.6e-9 mA over 18 rows (the helper's second application) · handoff: -
- N7 · verify · fresh agent: README/docstring prose vs the JSON, own code · after: N8 · done: 2 wrong, 3 misleading, fixed (see Progress) · handoff: t17/verify_n7
- N8 · build · README table/provenance + benchmark and module docstrings · after: N6 · done (75ef5fa) · handoff: t17/final/figures.txt, table_best.txt

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

- N2 result (18 rows, repaired vs b29cec3 archive; t17/compare.txt):
  theory leg identical; every other leg <= 0.04 mA except CH2ClF -- base
  hybrid 10.842 -> 10.525, vs r_e (Vogt r_e^SE) 9.237 -> 8.945; offsets
  8.174 -> 8.077, vs r_e 5.992 -> 5.910; ME base 16.54 -> 16.49. Other r_e
  moves: VF offsets 0.996 -> 0.990, ozone base 0.064 -> 0.068, offsets
  1.021 -> 1.025, water unchanged. Base hybrid mean 4.634 -> ~4.59.
  CH2ClF is the molecule with the worst violation (1.9e-2) and the
  vibrational shift (381.5 -> 376.2 cm-1), and an underdetermined fit (2
  isotopologues) that leans on the prior's curvature (N16's mechanism).

- N3b (my own check, 153 s at level 7, 4 threads): CH2ClF lowest modes
  L3 raw 381.01 / 729.46, L3 repaired 376.24 / 727.39, L7 376.15 / 727.43
  cm-1 (all modes within 0.1 after repair); alpha max vs L7 0.82 MHz raw,
  0.006 repaired. The repair corrects the curvature, not only alpha, so
  CH2ClF's 0.3 mA move toward its r_e is the converged-grid direction.

- N5 (ba67cfe): impose_translational_sum_rule in base_backend; applied in
  QuantumState.__init__ (every optimizer Hessian: _run_hessian both
  branches, load_orca) and in harmonic_alpha (_project_rigid_mw, so both
  mode analyses; _rigid_directions_with_derivatives; the displaced Hessians
  of cartesian_cubic_force_field and normal_mode_hessian_derivatives).
  Prototype arithmetic (0.5(H+H^T) is bit-exact on the backend's symmetric
  Hessians). Test (analytic H-N-C surface + a 3.5e-3 on-site error): with
  the helper set to identity each test fails on its own assertion (lines
  63, 90, 99); alpha without it moves -1672 / -3.6 / +1.7 MHz, with it 0.0.
  CI selection on 3.12: 1615 passed.

- N6 B3LYP (113 rows, ba67cfe) vs b29cec3: every row moves; CH2ClF base
  family 10.842 -> 10.525 (vs r_e 9.237 -> 8.945), offsets family 8.174 ->
  8.077 (5.992 -> 5.910), track_direct 10.992 -> 10.692; everything else
  <= 0.049 (AF lam ME). Best-of 5.31 / 4.05 / 3.76 (was 4.06 / 3.77), base
  9.19 / 5.40 / 4.59 (was 5.41 / 4.63); wins 5/9, 5/9, 7/9 unchanged; RHF ->
  B3LYP improves 8/9 (CH2ClF 5.43 -> 8.08); FE theory-over-hybrid margin
  0.027 -> 0.002. r_e table B3LYP: CH2ClF ME 14.34 / hybrid 8.95; ozone
  hybrid 0.068; VF ME 3.86. Engine = prototype to 8.6e-9 mA (N6b).

- RHF so far (120 of 135, ba67cfe vs b29cec3): 116 rows within 1e-4 mA
  (39 bit-identical below 1e-6). FA's tracked rows move: track_quadratic
  13.632 -> 13.591 (the T-15 prototype's value -- the fit has two
  solutions and tiny perturbations pick one), track 1.1e-3, offsets+track
  7e-4. No best-of row is a FA tracked row. FE (R3, wb) left.
- B3LYP band scan (n6 logs): no check in [3,5]; low side water base family
  0.217 and ozone offsets family 0.226, as before.

- RHF final (135, ba67cfe): 131 rows < 1e-4 mA from b29cec3 (39 < 1e-6);
  ozone track_direct 1.3e-4; FA track 13.6354 -> 13.6343, offsets+track
  4.3641 -> 4.3648, track_quadratic 13.6318 -> 13.5909. Every quoted RHF
  figure unchanged. RHF band scan: 0 rows in band.
- N8: README table (B3LYP 4.05 / 3.76, base 5.40 / 4.59), CH2ClF 8.08, r_e
  table (ozone 0.07, VF ME 3.86, CH2ClF 14.34 / 8.95), Provenance rewritten
  (D-52 + D-55); docstring table = --table (checked), D-55 paragraph, base
  means, FE tie 6.630/6.631, ozone 0.07, vintages. Self-check caught one
  overclaim (ozone track_direct is also among the four RHF rows > 1e-4).

- N7 fact-check of 75ef5fa (own code): every table cell, mean, win count,
  RHF/B3LYP move, band-scan and converged-grid figure verified. Wrong: (1)
  CH2ClF "0.32 / 0.10 toward its r_e" -- those are its effective-reference
  moves; vs r_e 0.29 (9.24 -> 8.95) and 0.08 (5.99 -> 5.91), README and
  docstring (and 75ef5fa's commit message); (2) _project_rigid_mw "CD3-AF
  A 0.8 -> 0.04" -- 0.04 was the parent's; CD3 0.832 -> 0.018. Misleading:
  "rigid-mode eigenvalues 23-117" is each molecule's largest, at the theory
  geometries (off them, to 579 cm-1); "two changes ... behind alpha and the
  prior" (D-52 is alpha only); QuantumState "every Hessian the optimizer
  uses" (the correction-level table is repaired inside harmonic_alpha).
  Rounding: "within 0.03" was 0.0319. All fixed; CH2ClF's C-Cl/C-F +7.4 /
  -7.2 and 0.44 sentence marked as measured at c8904da.

## Observations
- 2026-11-21 · L-18 again: the fact-check found I had quoted chlorofluoromethane's effective-reference move (0.32) as its move against r_e (0.29) -- the yardstick, not just the leg, has to travel with the number.
- 2026-11-21 · L-09 slipped: I wrote the new test's bounds and "measured" comments before measuring; two of three failed on first run. Measure, then write the bound -- the comment is a record, not a guess.
- 2026-11-21 · a background launch without an explicit timeout got the harness's 30-min default, not the inner `timeout 7000`; the chains died with 1 of 18 rows. L-09's limit check must include the harness's own default, not just the job's.

## Pending question

## Handoff
