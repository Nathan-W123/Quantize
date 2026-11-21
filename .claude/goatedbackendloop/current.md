# Current task
Task: T-19
Status: done
Request: (D-58, auto-picked at Q-25) measure the grid-weight term pyscf's default RKS gradient leaves out of run_gradient, and what it does to the fit; measurement only, any change proposed before it is made.

## Brief
- Outcome: per B3LYP molecule, |g_default - g_grid| at the theory geometry,
  the shift it puts on the fit's quantum-prior centre (H^-1 dg, bonds in
  mA), whether the benchmark rows can see it, and the gradient's cost with
  and without.
- How: the code path first. Benchmark fits pass prior_target_coords, and
  the step then uses H (x - x_target) in place of the gradient
  (_prior_gradient) -- so rows should not move; the CLI never sets a
  target, so its prior's minimum is x_ref - H^-1 g (the quadratic model the
  step solves), and a gradient error dg moves it by H^-1 dg exactly under
  that model. Then: both gradients at every theory geometry, the Newton
  shift on the cached sum-rule-repaired Hessian, and an independent check
  through real fits (one benchmark row with the gradient perturbed; a
  no-target joint fit with each gradient).
- Tradeoff: a change touches every CLI DFT fit's gradient; the benchmark
  would not move.
- Success: the table; the benchmark-insensitivity shown by experiment, not
  by reading; a no-target fit's move against the predicted centre shift; a
  proposal (or not).

## Graph
- N1 · diagnose · which entry points read run_gradient into the answer · after: none · done by reading (benchmark: target mode, gradient replaced; CLI: no target, centre x_ref - H^-1 g) · handoff: -
- N2 · diagnose · dg and the centre shift at all 12 B3LYP theory geometries; gradient cost · after: N1 · done (t19/gridg.json) · handoff: -
- N3 · verify · fresh agent, own experiments: a benchmark row with run_gradient perturbed; a no-target fit with each gradient vs N2's shift · after: N2 · done (t19/verify; see Progress) · handoff: -
- N4 · gate · propose (or not) the grid response in run_gradient (and run_hessian's gradient) · after: N2, N3 · done: flagged, not changed (D-59, auto-picked at Q-26) · handoff: -
- N5 · build · README + run_gradient docstring note · after: N4 · done (a20b3e4) · handoff: -
- N6 · verify · fresh agent fact-check of the note against t19 JSON, own code · after: N5 · done: 1 wrong (FE 1.8e-4 -> 1.7e-4, rounded twice), 3 misleading (the target bypass is Cartesian-only; the move came through run_hessian's gradient; references are effective structures), and the 7-14% cost was an artefact of the verifier's wrapper (a second gradient per call); fixed in the next commits · handoff: -
- N7 · verify · re-time with exactly the change (t19/cost: lean wrapper, alternating, 2 runs each) · after: N6 · done: FF 55.5 / 55.4 -> 53.5 / 53.1 s, FE 122.8 / 128.2 -> 126.4 / 135.8 s (+4.5% on the means; repeats 4%); answers equal the verifier's grid_both to 5e-13 A · handoff: - · handoff: -

## Progress
- N2 (t19/gridg.json; T-18 geometries, which are grid-response minima):
  run_gradient == the default gradient (<= 4e-15). |g_default - g_grid|
  max: HCOF 4.4e-4, FE 3.3e-4, VF 2.8e-4, AF 1.8e-4, CH2ClF 1.2e-4 Ha/A
  (6-22x gtol), ozone 5.2e-5, the other six <= 2.3e-5. Part of dg is rigid
  (up to 1.0e-4: the default gradient's forces do not sum to zero); the
  projection removes that part. No-target prior centre x - H^+ P g moves,
  default against grid: max bond FE 0.274, HCOF 0.263, AF 0.118, VF 0.078,
  CH2ClF 0.031, rest <= 0.005 mA; Cartesian up to 7.0e-4 A RMS (AF, one
  atom 1.8e-3). The default-gradient centre lands near the OLD (pre-D-57)
  geometries: max bond vs old FE 0.043, HCOF 0.020, AF 0.051, VF 0.167.
  Theory bond RMS at that centre: HCOF 5.659 -> 5.539, FE 6.620 -> 6.655,
  AF 9.479 -> 9.445, VF 4.146 -> 4.150, CH2ClF 25.108 -> 25.090.
  Cost, best of two per gradient (s, default / grid): FE 1.2 / 1.8, VF
  0.8 / 1.0, AF 2.0 / 2.1, CH2ClF 1.3 / 1.5, HCOF 0.6 / 0.6; the SCF is
  1.2-3.9 s, so per gradient job (SCF + gradient) at most +13% (FE).

- N3 (fresh agent, t19/verify, worktree at b81a762): (a) benchmark
  formyl fluoride base, B3LYP, through main(): run_gradient default / grid
  / +N(0,1e-3) noise -> fitted coordinates bit-identical (0.0 A), row
  5.6599 / 0.9873 / 2.4885 in all three -- the target prior replaces the
  gradient. (b) through the CLI (cli.main run, pyscf_hf, BALANCED, no
  target, the benchmark's base correction table passed in): run_gradient
  alone changes nothing -- the selected start (theory geometry) makes one
  run_hessian call and no run_gradient call, and the gradient it uses is
  run_hessian's. With run_hessian's gradient also on the grid response:
  HCOF moves 1.53e-4 A (C=O -0.068, C-F +0.064 mA), bond RMS 2.532 ->
  2.486; FE 1.75e-4 A (C-C -0.139, C-F +0.132), 6.677 -> 6.612; internal
  mode the same to 0.002. The CLI's grid geometry for HCOF is 5.7e-6 A from
  the benchmark hybrid's (default 1.56e-4). Repeats reproduce to 8e-10 A.
  Wall time (whole CLI run, 3 starts): HCOF 63 -> 67 s, FE 135 -> 154 s
  (repeat 131.5); the selected start alone HCOF 13.4 -> 13.4, FE 32.9 ->
  34.0. Jittered starts call run_gradient 2-6 times and move too (FE start
  2: 1.26e-2 A, a ~1 deg methyl torsion).
- Side findings (N3): the validator's inertial-defect warning fires on
  |defect| > 1 for any molecule, but I_c - I_a - I_b = -2 P_c is large and
  negative for every non-planar one (fluoroethane -6.3 to -7.7) -- a false
  positive (backlog F3). The plateau exit prints "Converged to rank-N
  spectral floor" for every plateau (wording only).

## Observations
- 2026-11-21 · I quoted the verifier's wall times as the change's cost without reading its wrapper, which computed a second gradient per call (7-14% reported; the change itself costs 0-4.5%). The cost reached a gate question (Q-26) and a decision (D-59) before the fact-check caught it.
- 2026-11-21 · the verifier, told only to measure through the CLI, found that the gradient the CLI's answer uses is run_hessian's, not run_gradient's -- my brief named the wrong function; patching only it would have shipped a no-op.

## Pending question

## Handoff
