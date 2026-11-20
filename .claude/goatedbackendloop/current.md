# Current task
Task: T-14
Status: building
Request: diagnose fluoroacetylene's tracked fits (backlog N19 b; D-49,
auto-picked at Q-16); diagnosis first, any fix proposed before it is made.

## Brief
- Outcome: the cause of the 2nd / 3rd tracked fits' reduced chi2 16.00
  (dof = n_obs) and of the tracked rows being worse (13.59 -> 19.16 mA),
  and what a fix would move.
- How: instrument the benchmark's tracking path in a git-archive copy of
  HEAD (t14/wt, t13 caches), logging slopes, shifts, sigmas, residuals and
  each alpha call's parts; then a fresh-agent check of the mechanism and a
  prototype fix in a second copy (t14/fix), every row rerun.
- Success: a per-number trace from cause to the 16.00, independently
  reproduced; the list of rows a fix changes.

## Graph
- N1 · build · trace the tracked fits · after: none · done · handoff: cause found (below)
- N2 · verify · fresh agent: continuity scan across the cut + rerun with the cut changed · after: N1 · running · handoff: -
- N3 · build · prototype fix in t14/fix, all RHF rows + B3LYP fluoroacetylene · after: N1 · running (task b4tjegik3) · handoff: -
- N4 · gate · propose the fix to the user (engine change, rows move) · after: N2, N3 · pending · handoff: -

## Progress
- 18:45 UTC: t14/trace.py (+ trace2 / trace3, more probes). HF
  fluoroacetylene, configs track and track_direct; rows reproduce the
  archive (19.1634, 13.6626).
- The `track` scheme extrapolates the stationary-point table along
  d(alpha)/ds from a central difference at coords +/- step/2 (step 0.01554
  A, under the 0.020 gate). Slopes came out at 4.6e8 - 8.2e8 MHz/A, so the
  shifts were 7.1e6 - 1.3e7 MHz on alphas of 30 - 41 MHz; the extrapolated
  B targets were 3.7e6 - 6.4e6 MHz (observed ~9700), with sigma = 0.25 x
  |half the shift|.
- Fits 2 and 3: every weighted residual exactly 4.0 (= 1/0.25), chi2 16.00,
  dof 6.000; the x4 widening gives 1.000. The data have no leverage, the
  fit returns to the prior (3e-5 A away) -> row 19.16 = theory.
- Per-term breakdown (compute_harmonic_alpha): Coriolis and anharmonic
  parts are normal at all three points; the Watson harmonic ("centrifugal")
  term is -7.47e6 MHz at the minus point only. Same frequencies, same skip
  count, is_linear True at all three.
- Cause: that term keeps 1/I for I > 1e-8 amu A^2 (absolute). I_A: centre
  6.1e-9, plus 1.5e-9, minus 1.37e-8. The module's linearity test
  (is_linear, I_min / I_max < 1e-6, i.e. I_A < ~5e-5 here) says linear at
  all three, so the minus point is "linear" for its modes but divides by
  its axial moment in the harmonic term. The theory geometry itself sits
  at 6.1e-9, a factor 1.6 under the cut.
- track_direct is unaffected (its tables are built at fitted geometries
  whose I_A stays under the cut); B3LYP fluoroacetylene's tracked fits had
  chi2 <= 0.8 in T-13 (not affected or gated).
- Prototype (t14/fix, scratch copy only): in the Watson term, zero the
  axial inverse moment when _rigid_mode_count says linear, the same
  decision the normal modes and the Coriolis pair use. Running: all 9
  molecules x 13 configs at RHF, and fluoroacetylene x 11 configs at
  B3LYP, to list every row it moves.

## Observations
- 2026-11-20 · my first guess (the degenerate bend pair passing the 1 cm^-2 Coriolis guard) was wrong; logging each term at the three points named the harmonic term in one run (L-06 applied).

## Pending question

## Handoff
