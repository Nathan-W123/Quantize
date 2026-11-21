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
- N2 · diagnose · dg and the centre shift at all 12 B3LYP theory geometries; gradient cost · after: N1 · running (t19/gridg.py) · handoff: -
- N3 · verify · fresh agent, own experiments: a benchmark row with run_gradient perturbed; a no-target fit with each gradient vs N2's shift · after: N2 · todo · handoff: -
- N4 · gate · propose (or not) the grid response in run_gradient (and run_hessian's gradient) · after: N2, N3 · todo · handoff: -

## Progress

## Observations

## Pending question

## Handoff
