# Current task
Task: T-18
Status: building
Request: (now D-57, a change) measure how far the theory geometries sit from tight minima of their own surface (backlog N21; D-56, auto-picked at Q-23); measurement only, any change proposed before it is made.

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
- N1 · diagnose · gradient + Newton step at every theory geometry, both levels · after: none · done (t18/newton_{hf,b3lyp}.json) · handoff: -
- N2 · verify · tight optimisations vs the Newton estimate · after: N1 · done: RHF Newton exact; B3LYP needs grid-response optimisations (all 12 run, t18/gr_*.json) · handoff: t18/gr_a.log, gr_b.log
- N3 · gate · propose (or not) tighter theory geometries · after: N1, N2 · done: option 1 auto-picked (D-57) · handoff: -
- N4 · build · optimise(): grid-response gradient for DFT + convergence check; test failing on the old code; CI selection on 3.12 · after: N3 · done (4614cd0; 1619 passed) · handoff: -
- N5 · build · regenerate the B3LYP theory geometries, rerun every B3LYP row (113), archive · after: N4 · geometries regenerating (t18/wg at 4614cd0, t18/regen.py, regen.log) · handoff: -
- N6 · build · README + docstrings · after: N5 · pending · handoff: -
- N7 · verify · fresh agent: prose vs JSON · after: N6 · pending · handoff: -

## Progress
- N1 RHF (12 geometries; t18/newton_hf.json): |g|max 2e-6 to 1.8e-5 Ha/A
  (gtol 2e-5); Newton distance <= 1.2e-4 A RMS except H2O2 4.8e-4 (left out
  at RHF, trans) -- torsions; largest bond change 0.008 mA, theory-column
  RMS change <= 0.005 mA. RHF theory geometries are converged for bonds.
- N2 RHF: tight L-BFGS-B (gtol 1e-7) of fluoroethane from its theory
  geometry lands 5.0e-7 A from the Newton point (move 1.19e-4 A); bond
  change 0.0053 vs 0.0054 mA. The Newton estimate holds.

- N1 B3LYP (t18/newton_b3lyp.json): the optimiser stopped well above its
  gtol (2e-5 Ha/A): |g|max VF 3.2e-4, FE 1.4e-4, HCOF 1.3e-4, CH2ClF
  9.3e-5, AF 5.9e-5 (water, FA, calibration ones <= 1.8e-5). Newton
  step: VF theory bond RMS 4.220 -> 4.150 mA (max bond 0.168), FE 1.1e-3 A
  (torsion, bonds 0.019), CH2ClF 25.102 -> 25.090. But at B3LYP the Newton
  step understates: HCOF Newton 2.6e-5 A RMS vs the T-17 verifier's tight
  minimum 1.03e-4 (bonds: verifier's C-H +0.044 vs Newton max 0.014) --
  the default-grid gradient and Hessian both lack grid response, so they
  are not consistent with each other or with the energy.
- Found: pyscf_backend.optimise() never checks sol.success / sol.message --
  an unconverged L-BFGS-B result is returned silently (it reports only a
  hit step bound). Suspect: line-search failure on a gradient that is not
  the energy's derivative (no grid response).

- N2 VF (t18/opt_vinyl_fluoride.json), L-BFGS-B from the theory geometry:
  backend gradient (no grid response), gtol 2e-5 or 1e-6: stops on ftol
  ("relative reduction of F") after 10 iterations with |g| 2.0e-4 Ha/A,
  theory bond RMS 4.220 -> 4.100 mA (max bond 0.349) -- the gradient is not
  the energy's derivative, so the run ends where the energy stops falling,
  and that depends on the start. Grid-response gradient (exact derivative
  of the default-grid energy), gtol 1e-6: converges, 71 iterations, |g|
  6e-6, theory bond RMS 4.147 (max bond 0.210). The cached theory geometry
  is 0.21 mA (worst bond) from the true minimum of its own energy.

- N2 all 12 B3LYP (grid-response gradient, gtol 1e-6, from the cached
  theory geometry): converge to |g| 1.7e-6 to 1.4e-5 Ha/A (start 7.9e-6 to
  3.2e-4). Largest bond change: FE 0.241, HCOF 0.236, VF 0.208, AF 0.146,
  CH2ClF 0.059, ozone 0.016, FA 0.014, HNCO 0.008, water 0.003, calibration
  <= 0.009 mA. B3LYP base theory column: VF 4.220 -> 4.146, AF 9.438 ->
  9.478, FE 6.630 -> 6.620, HCOF 5.542 -> 5.657, CH2ClF 25.102 -> 25.079,
  ozone 7.586 -> 7.570, rest <= 0.007; mean 9.193 -> 9.196.
- Toward the converged grid: against the T-17 verifier's level-9 minima,
  HCOF bond RMS 0.093 (old theory geometry) -> 0.047 mA (new), ozone 0.010
  -> 0.006.

- N4 (4614cd0): _energy_gradient(grid_response=) and BaseBackend.is_dft;
  optimise() uses the grid-response gradient for DFT, records last_optimise
  (message, iterations, |g| max Ha/A, gtol, converged) and warns above gtol.
  Tests on a quadratic surface: the b3lyp grid-response, unconverged-report
  and converged-record tests fail on the old code, the HF control passes on
  both; measured: an offset gradient stops (abnormal line search, 3 it) at
  |g| 3.4e-4, 17x gtol; clean |g| 2.7e-10. CI selection 3.12: 1619 passed.

## Observations
- 2026-11-21 · the same slip as T-17 within the hour: a test comment said "Measured" before the measurement. Write measured-value comments only after the run that prints them.

## Pending question

## Handoff
