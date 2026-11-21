# Current task
Task: T-19
Status: building
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
- N3 · verify · fresh agent, own experiments: a benchmark row with run_gradient perturbed; a no-target fit with each gradient vs N2's shift · after: N2 · running (t19/verify) · handoff: -
- N4 · gate · propose (or not) the grid response in run_gradient (and run_hessian's gradient) · after: N2, N3 · todo · handoff: -

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

## Observations

## Pending question

## Handoff
