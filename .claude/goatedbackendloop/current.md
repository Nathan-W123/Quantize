# Current task
Task: T-15
Status: building
Request: measure whether projecting translations and rotations out of the
Hessian before the alpha normal-mode analysis moves any benchmark row
(backlog N19 b-i; D-51, auto-picked at Q-18); measurement only, any engine
change proposed before it is made.

## Brief
- Outcome: per row (both levels, non-torsion configs), hybrid_rms_ma with
  and without the projection, and whether any README figure moves.
- How: prototype in a git-archive copy of HEAD (t15/proj): a mass-weighted
  rigid-mode projector applied to the Hessian at both _normal_modes call
  sites in harmonic_alpha.py (compute_harmonic_alpha and the cubic
  normal-mode derivative). Reference: HEAD rows from t14/fix (HF, all 117,
  FA from the corrected run) and t13/b3lyp_t4_merged + t12/run1 (B3LYP).
- Tradeoff to watch: at a stationary point the projection should be a
  no-op; any change there is noise or a defect in the projector.
- Success: the per-row table, a sanity check at stationary points, and an
  independent check of anything that moves.

## Graph
- N1 · build · prototype projector + stationary-point sanity check · after: none · done · handoff: corrected once (see Progress)
- N2 · build · all rows at both levels · after: N1 · RHF done; B3LYP 12/99 done, rest running in two halves (t15/inst_proj: HNCO, AF, FE, FF; t15/b3b: FA, CH2ClF, water, ozone) · handoff: RHF table below
- N2b · build · the 4 RHF torsion rows with the torsion applied (AF, FE) · after: N2 · pending · handoff: -
- N2c · diagnose · why HNCO offsets moves 20 mA · after: N2 · done · handoff: see Progress 22:xx
- N3 · verify · fresh agent: continuity scan of HNCO alpha theory -> offset geometry with its own Hessians and projector; own edit reproducing the row · after: N2c · done · handoff: t15/verify (scripts/scan.py, run_bench.py, variants.sh; results/)
- N4 · gate · propose (or not) an engine change · after: N3 · done: option 1 auto-picked (D-52) · handoff: -
- N5 · build · engine projector + test (fails on the old code) + full suite on 3.12 · after: N4 · running · handoff: -
- N6 · build · rerun every row at both levels from the commit (worktrees, warm cache, chi2 logged) · after: N5 · pending · handoff: -
- N7 · verify · fresh agent: the committed rows and README against the JSON · after: N6 · pending · handoff: -
- N8 · build · README table/provenance + benchmark docstring (rescale caveat) · after: N6 · pending · handoff: -

## Progress
- 19:52 UTC: t15/proj from HEAD (129ca37 code = 2eb9b6c engine), caches
  from t13/wt3 + t14/fix. Projector: translations sqrt(m) e_a, rotations
  sqrt(m) (w x r), P F_mw P, back to Cartesian.
- Sanity check 1 (HF, theory geometries): water alpha moves 0.1-0.6 MHz of
  1e4 (noise at a stationary point); fluoroacetylene LOST a mode (7 -> 6
  frequencies, a 440.9 cm^-1 bend component became 765.6) and alpha_B went
  41.26 -> 64.82. Cause: my projector kept the rotation about the molecular
  axis (norm ~1e-5 of the others at I_a 6e-9) as a sixth rigid column --
  the very case quantize.py's _rigid_mode_projector docstring warns about.
- Corrected: rotations about the principal axes, the smallest-moment axis
  dropped when _rigid_mode_count says linear. Re-check: FA 7 modes, alpha_B
  41.2591 vs 41.2584 at the theory geometry; 43.955 at the slope's minus
  point (verifier, unprojected: 43.966); FA track fits 13.5909 / 13.6366 /
  13.6399 (HEAD 13.6402). Water unchanged (0.5593).
- 19:55 UTC: full run started: HF 9 x 13 configs, B3LYP 9 x 11 configs.

- 21:00 UTC: RHF 117/117 rows done (65 min, not the 15-20 I said: every
  cubic-term displacement follows the projected modes, so every displaced
  Hessian is a cache miss -- 745 new HF, 242 new B3LYP). B3LYP hit its
  6800 s timeout at 12/99 rows (VF all, AF base): moves <= 3e-3 mA on VF,
  1e-4 on AF base.
- RHF per row vs t14/fix (same environment, HEAD engine): stationary-point
  rows (base family) move <= 2e-3 mA; offsets families <= 0.04 mA except
  HNCO; track_direct rows 0.003-0.02 mA except HNCO; HNCO offsets family
  8.137 -> 28.555 mA (6 rows), HNCO track_direct 22.130 -> 18.044.
- Side finding: archived reports rows for AF, FE and ozone (vintage
  5cfc5d4) differ from today's HEAD-engine run by 1e-5 to 4.6e-4 mA;
  below every README figure, but a reason to compare runs only with runs.
- HNCO diagnosed (logged, L-06): one rescale pass each, chi2 3.47 (HEAD)
  vs 1.85 (proj), both under 4 -- not the rescale. The offsets table is
  built at offset_coords, a non-stationary point. There the raw mass-
  weighted Hessian's rotation block reads -187 / 63 / 77 cm^-1 (2-6 at the
  theory geometry); the kept modes stay the right six, but the Watson
  inertia-derivative ("centrifugal") term for B goes -31.7 (theory geom)
  -> -131.1 MHz (offset geom, raw) vs -33.5 projected; alpha_B parent
  -26.2 -> -125.8 raw / -28.1 projected (DNCO -25.3 -> -143.2 / -27.0).
  So HEAD's 8.14 mA came from a table ~100 MHz off in alpha_B; the
  projection removes it. Probe: t15/probe/{capture,analyze,terms2}.py.
- RHF table with the projection (torsion rows whose torsion is not applied
  take their twin's projected row; AF/FE applied torsion rows still the
  archive's): hybrid mean 5.34 -> 6.44 (HNCO 8.14 -> 18.04, its best row
  now track_direct), ME 9.85 -> 9.86, theory unchanged; win counts 4/9,
  6/9, 7/9 unchanged.
- 22:12 UTC: verifier (own pyscf Hessians, own projector, own patch)
  confirms every number: alpha_B parent -26.17 -> -125.80 raw / -28.13
  projected, DNCO -143.15 / -27.04; the move is all in the a^{ab}/I_a
  piece of the inertia-derivative term (bend's |a^{ab}| 0.840 raw = 0.271
  vibrational + 0.570 from a 0.71% rigid c-rotation); row 8.1373 ->
  28.5553 reproduced to 1e-9, chi2 3.468 / 1.853, one pass each.
  Refuted my brief's prediction: the raw error is not proportional to the
  gradient (superlinear, a spurious rotation at 48 -> 99 cm-1 climbing
  toward a softening bend; sign-odd linear part, +5.8 MHz at lambda -0.5).
  Found: the corrected row is recipe-dependent -- every rotation-free
  recipe gives 23.2-28.65 mA (mass-weighted projection 28.56; unweighted
  Cartesian 23.20; alpha from the stationary geometry 23.36), because the
  HNC bend force constant is not unique off a stationary point (334 ->
  279 cm-1 with the mass-weighted / valence offset-force surfaces, 324-336
  with the others). The raw 8.14 has no defensible recipe behind it. The
  mass-weighted projection is the Watson/Eckart-consistent one and agrees
  with the valence-coordinate offset-force surface (tiltA) to 0.12 MHz.
- B3LYP at 22:11: 14 rows in the two halves in 16 min (the benchmark runs
  molecules in its own order; HNCO is last in half A).

## Observations
- 2026-11-20 · the stationary-point sanity check caught my projector removing a real bend on a near-linear molecule before any long run -- the same linearity trap as T-14, a second time in one day (L-P04 applied; candidate lesson: every new rigid/linear treatment goes through rigid_mode_count, never its own threshold).
- 2026-11-20 · the "no-op at stationary points" framing hid where the projection matters: the offsets configs evaluate alpha off-stationary by design; my ETA (15-20 min) missed that projected modes move every cubic displacement off the cache.
- 2026-11-20 · L-10 again: told to use its own projector and its own surfaces, the verifier refuted my proportional-to-gradient prediction and showed the corrected value spans 23.2-28.65 mA by recipe -- a range I would have reported as one number.
- 2026-11-20 · a fit that T-13 found threshold-fragile turned out to sit on a ~100 MHz alpha error -- a row that flips under a small perturbation deserved a look at its inputs, not just its switch.

## Pending question

## Handoff
