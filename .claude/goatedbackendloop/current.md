# Current task
Task: T-14
Status: verifying
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
- N2 · verify · fresh agent: continuity scan across the cut + rerun with the cut changed · after: N1 · done · handoff: confirmed; clean-bend step 7-10 MHz, blow-up needs rotation-contaminated off-stationary Hessians; caught my prototype's axis bug
- N3 · build · prototype fix in t14/fix, all RHF rows + B3LYP fluoroacetylene · after: N1 · done (corrected) · handoff: 3 HF FA rows move (track, offsets+track, track_quadratic); nothing else; best-of unchanged
- N4 · gate · propose the fix to the user (engine change, rows move) · after: N2, N3 · decided (Q-17 auto-pick, D-50) · handoff: fix applied

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
- (Superseded by the prototype below: track_direct IS affected, mildly,
  at both levels -- a direct table built at a fitted geometry can cross the
  cut without a catastrophic alpha. My reading "unaffected" came from the
  sane chi2, not from a run.)
- Prototype (t14/fix, scratch copy only): in the Watson term, zero the
  axial inverse moment when _rigid_mode_count says linear, the same
  decision the normal modes and the Coriolis pair use. Running: all 9
  molecules x 13 configs at RHF, and fluoroacetylene x 11 configs at
  B3LYP, to list every row it moves.
- 18:51 UTC (prototype done, ~15 min): exactly 5 rows move, all FA tracked.
  HF track 19.1634 -> 13.6402, offsets+track 4.9633 -> 4.3627,
  track_quadratic 19.1653 -> 13.6391, track_direct 13.6626 -> 13.4793;
  B3LYP track_direct 3.9776 -> 4.0929 (worse: its direct table at a fitted
  geometry had crossed the cut too, and the contaminated value was luckier).
  The other 113 HF + 10 B3LYP rows are bit-identical to the HEAD runs
  (t13/wt4, t13/b3lyp_t4_merged, t12/run1). README best-of means unchanged
  (RHF 10.39 / 9.85 / 5.34, B3LYP 5.31 / 4.05 / 3.78; FA's best rows are
  offsets at RHF and offsets+track at B3LYP, both untouched), so are the win
  counts and the base table. Fixed tracked rows now sit at or near the
  untracked ones (13.64 vs base 13.59; 4.36 vs offsets 4.36).
- 19:05 UTC N2 report (own HEAD copies; fix `inv[0] = 0` when linear):
  A. bend scan from an exactly linear HF geometry: the Watson part steps
  -16.04 -> -23.61 MHz (H end; -26.24 F end) exactly where I_A crosses 1e-8,
  is_linear True throughout; flat with the fix. A clean bend gives a 7-10
  MHz step, not a blow-up: the benchmark's 1e6-1e9 needs bends carrying
  rotation (the +-step/2 points are off-stationary, |grad| 0.012 / 0.068
  Ha/bohr; rotation eigenvalues +-1.4e-4, bend-rotation overlap 1-18%);
  projecting rotations out gives -22.96 / -22.66. B. track 19.1634 ->
  13.6402, offsets+track 4.9633 -> 4.3627, track_quadratic 19.1653 ->
  13.6391 (its targets were -3.7e6: curvature -1.2e11), track_direct
  unchanged (its fitted geometries are exactly linear); "= 16" 6x -> 0x.
  Separate: at the is_linear boundary itself (I_A ~5.3e-5) n_vib 7 -> 6 and
  alpha_B jumps +24.6 MHz through Coriolis / anharmonic; not touched.
- My prototype was wrong for track_direct: argmin over finite moments,
  but rotational_constants_mhz turns a moment <= 1e-10 into inf (A = 0), so
  on an exactly linear geometry it zeroed B or C. Corrected to inv[0] (A is
  index 0, A >= B >= C) and FA rerun at both levels: exactly 3 HF rows move
  (track -5.523, offsets+track -0.601, track_quadratic -5.526 mA); every
  other FA row at both levels bit-identical to HEAD. Non-FA rows cannot move
  (branch fires only for a linear geometry; they were bit-identical even
  under the faulty variant). Best-of tables unchanged.
- Test plan if the fix is approved: dev/tests has only CO and water
  analytic surfaces; add a linear-triatomic valence force field and assert
  alpha_sum B is continuous as the bend takes I_A across 1e-8 (fails on HEAD).

- 19:30 UTC (D-50, Q-17 auto-pick): fix in harmonic_alpha.py (inv[0] = 0
  when _rigid_mode_count says linear; comment with the measurements) and
  dev/tests/test_linear_axial_moment.py (analytic HCN-like valence field,
  fresh FD Hessian per geometry, I_a 5e-9 / 2e-8, premises asserted).
  Measured: fixed -88.76798 / -88.76798 / -88.76800 MHz (2e-5 apart); old
  code -202.016 above the cut (113 MHz, harmonic term). Test fails on the
  HEAD copy at exactly that assertion, passes on 3.11 and on 3.12 (venv from
  requirements.txt, as CI). 182 related alpha / correction tests pass.
  Commit 2eb9b6c.
- 19:4x UTC: clean worktree at 2eb9b6c (removed after), archived HF file
  minus FA's 3 rows, rerun -> vintage 2eb9b6c, 13.6402 / 4.3627 / 13.6391
  (= prototype and verifier); the other 132 rows byte-identical in the
  archive. --table unchanged (10.39 / 9.85 / 5.34; 5.31 / 4.05 / 3.78; win
  counts 4/9 6/9 7/9, 5/9 5/9 7/9). README provenance names the vintage;
  test_benchmark_summary_table + test_calibration_references pass. Commit
  aace5d0. Full CI suite running (task b0766romz).

## Observations
- 2026-11-20 · my first guess (the degenerate bend pair passing the 1 cm^-2 Coriolis guard) was wrong; logging each term at the three points named the harmonic term in one run (L-06 applied).
- 2026-11-20 · my prototype took argmin over an array whose 'absent' entry is encoded as inf, and zeroed the wrong axis on exactly linear geometries; the verifier's independent edit disagreed on one row and exposed it. Check how the input encodes the missing case before selecting from it (L-P04 extension candidate).

## Pending question

## Handoff
