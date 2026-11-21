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
- N5 · build · engine projector + test (fails on the old code) + full suite on 3.12 · after: N4 · done (b29cec3) · handoff: -
- N6 · build · rerun every row at both levels from the commit (worktrees, warm cache, chi2 logged) · after: N5 · done: RHF 135, B3LYP 113, all b29cec3 · handoff: t15/final (merged JSON, figures.txt, tables), reports/accuracy_upgrades
- N6b · verify · fresh agent: is AF B3LYP's stationary-point alpha shift (0.3-1.3 MHz) removed noise or a projector defect -- own Hessians, orientation and grid tests · after: N6 · done: removed contamination, projector correct (see Progress) · handoff: t15/verify_af
- N7 · verify · fresh agent: the committed rows and README against the JSON · after: N6 · pending · handoff: -
- N8 · build · README table/provenance + benchmark docstring (rescale caveat) · after: N6 · done (uncommitted -> next commit) · handoff: -

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
- 22:45 UTC: engine change b29cec3 (prototype arithmetic exactly; module
  differs from the prototype only in comments). New test (analytic bent
  H-N-C stretched 0.015 A; reference = the surface tilted by its own
  gradient): unprojected 23.5 MHz off, projected 0.92; fails with an
  identity projector; the linear test fails if the axial rotation is kept
  (n_vib 3). Full suite: 2 failures. (1) water end-to-end: the bond bound
  (0.65x) had come from contaminated corrections -- B's target 2028 MHz
  from B_e(r_e), worse than B_0's 1724; projected 253 (C 458 -> 382; A
  wrong either way, no bend anharmonicity on that surface); bond 0.61 ->
  2.42 mA vs raw 2.99. Test now checks B, C against B_e(r_e) (old engine
  fails it) and the bond at 0.9x (measured 0.81). (2) test_prior_target
  moved_ang == 0.0 reads 6.97e-11 -- identical on clean trees at HEAD and
  2eb9b6c, with or without the change: pre-existing here, backlog N20.
  Rerun: 1611 passed + N20.
- Prototype B3LYP runs stopped (VF 11 + AF 8 + FA 11 + CH2ClF 1 rows done,
  their Hessians cached); every row now reruns from b29cec3.
- 00:39 UTC: chain A done (RHF 135/135; B3LYP VF/AF/water/HNCO 50 rows);
  chain B hit its 7000 s limit at FE 8/11. Resumed 00:40 as chains C (wta:
  FF, FA + twins) and D (wtb: FE rest, CH2ClF, ozone + twins), task
  bbps4q960, logs t15/logs/run_<tag>.log (run_commit2.sh).
- RHF b29cec3 vs the prototype: 60/117 identical, 57 move 1e-6 to 2e-5 mA,
  FA track 13.6399 -> 13.6354 and track_quadratic 13.5909 -> 13.6318. Code
  identical (docstrings only). The prototype code rerun now (t15/proj_rerun,
  warm cache) gives 13.635415978521591 / 13.631784424046522 -- the commit's
  values to every digit: the difference was cache state (fresh Hessians vs
  ones read back through the 1e-9 A key), amplified by FA's fragile track
  fits, not code. HEAD had 13.6402 / 13.6391: the prototype's 13.59 was the
  outlier.
- RHF b29cec3 vs HEAD rows (t14/fix + archive torsion): HNCO offsets family
  8.1373 -> 28.5553 (6 rows incl. offsets+torsion), HNCO track_direct
  22.1304 -> 18.0436; every other row <= 0.09 mA (FE offsets+torsion 7.1015
  -> 7.0119, AF offsets+torsion 9.6669 -> 9.6449, both vs 5cfc5d4 archive).
- RHF table from the committed rows (--table): best-of theory 10.39, ME
  9.84 (README 9.85: FE's best ME is now offsets+torsion 8.12), hybrid 6.43
  (README 5.34); wins 4/9, 6/9, 7/9 unchanged. Base config unchanged
  (21.47 / 17.35 / 11.03; 5/9, 5/9, 9/9).
- Figure generator t15/figs/figures.py (every figure README + docstring
  quote); on the archived rows it reproduces all of them (10.39 / 9.85 /
  5.34, 11.28, FF 0.08, ozone 0.04 / 1.02, r_e table). Found stale already:
  print_summary_table's docstring says hybrid-picks-the-row flatters ME by
  2.9 mA; the data gave 11.28 - 9.85 = 1.43. RHF from b29cec3: one-row ME
  mean 11.10; cd-section offsets mean 5.38 -> 7.64; FF margin 0.07; CH2ClF
  rows now carry r_e scores themselves (45.68 / 28.16 / 10.80, = README's
  footnoted values). RHF chi2 band scan (t15/scan/band.py, any check in
  [3,5] or [0.2,1/3], every pass): 0 of 135 rows (HEAD: HNCO offsets 3.47).
- B3LYP b29cec3, 59 rows so far vs HEAD rows: HNCO offsets family 3.7128 ->
  3.4947; AF base family hybrid +1e-4 but ME 5.7397 -> 5.6773 (stationary
  point -- my 21:00 note gave the hybrid leg only); FE offsets 7.6563 ->
  7.6930; AF offsets+track 6.4450 -> 6.4242 (new AF best). Commit = the
  prototype on VF 11 + AF 8 rows (16 identical, 3 cache noise <= 8e-5).
  Probing AF's stationary-point ME move (t15/probe/capture_af.py,
  alpha_af.py).
- AF B3LYP at its theory geometry (stationary): the raw mass-weighted
  Hessian's six rigid eigenvalues are -41.6 / -14.2 / 15.9 / 35.5 / 42.3 /
  51.8 cm-1 (RHF: 2-6), against the methyl torsion at 136.8 (CD3 101.7).
  Projection moves alpha by 0.29-0.48 MHz (parent) to 0.96-1.31 (CD3) --
  that is the 0.06 mA ME move. Reading: DFT-grid / residual-gradient rigid
  residue mixed into the torsion, removed -- not a projector defect. To
  test by another mechanism (N7): a finer grid should shrink the raw rigid
  block and bring raw alpha toward the projected value.

- AF verifier (t15/verify_af, own pyscf Hessians, 4 frames, grids L3-L7):
  verdict (a). The unprojected alpha is contaminated and the projection
  removes it; the cause is pyscf's analytic RKS Hessian lacking the
  grid-weight response (rotations leak into the vibrational eigenvectors,
  changing zetas), not the residual gradient (predicted 4.7e-6 au, actual
  1.2e-4; re-optimising to 30x smaller gradient changed nothing). Projected
  default-grid alpha is within 0.07/0.04/0.01 MHz (parent A/B/C) of the
  converged grid (L7), unprojected off by 0.46/0.52/0.29; CD3 0.83 vs 2.14
  (A). Both approach the same limit at L5/L7. Left: the same defect in the
  vibrational block (CD3 A -0.83) -- an acoustic-sum-rule repair of the raw
  Hessian (free) or grids.level >= 5 removes it (within 0.04 MHz). Not
  implemented: an accuracy/cost option (D-03) -> backlog. Docstring note:
  at B3LYP's default grid the rigid residue is +-40 cm-1, not "numerical".

- B3LYP rerun done 02:30 (chains C/D + E/F in a third worktree wtc on the
  same cache; 15 ozone/CH2ClF rows run twice, bit-identical). 113 rows, all
  b29cec3, every row moved, max 0.28 (HNCO offsets family 3.71 -> 3.49).
  Best-of 5.31 / 4.06 / 3.77 (was 3.78), base 9.19 / 5.41 / 4.63, wins
  5/9, 5/9, 7/9 unchanged; RHF -> B3LYP improves 8/9 (CH2ClF 5.43 -> 8.17).
  B3LYP band scan: no check in [3,5]; low side water base family 0.217 and
  ozone offsets family 0.226 -- both the same at HEAD, where T-13 measured
  the extra refit moving the rows < 1e-5. Stationary-config max moves: RHF
  hybrid 1.8e-3, ME 8e-4; B3LYP ME 0.078 (AF lam), hybrid 6.6e-3 (FE defect).
- N8: README table, prose, r_e table (CH2ClF footnote dropped: rows carry
  r_e scores now), Provenance rewritten for b29cec3; benchmark docstring
  table (= --table output, checked), 11.10, edge paragraph rewritten, 6.66,
  7.64; print_summary_table's stale 2.9 -> 9.84 -> 11.10; projector
  docstring's "numerical residue" corrected with the AF verifier's numbers;
  monofluoro_references D-43 figures marked as measured at c8904da.
  Archived (scripts/archive_benchmark_results.py; old output/ JSON kept as
  *.pre-t15). figures.py on reports/ = on t15/final; tests 16 passed.

## Observations
- 2026-11-21 · L-09 ETA: chain D's CH2ClF tracked rows would have run into its timeout; a third worktree on the shared cache finished the twins and ozone in parallel and its duplicates came out bit-identical -- parallel resumption on a shared exact cache is safe.
- 2026-11-21 · L-10 again: the AF verifier confirmed the projection but refuted my mechanism (grid-weight response, not residual gradient) and found the docstring's "numerical residue" claim wrong at B3LYP.
- 2026-11-20 · the stationary-point sanity check caught my projector removing a real bend on a near-linear molecule before any long run -- the same linearity trap as T-14, a second time in one day (L-P04 applied; candidate lesson: every new rigid/linear treatment goes through rigid_mode_count, never its own threshold).
- 2026-11-20 · the "no-op at stationary points" framing hid where the projection matters: the offsets configs evaluate alpha off-stationary by design; my ETA (15-20 min) missed that projected modes move every cubic displacement off the cache.
- 2026-11-20 · a test bound set by measurement (water e2e 0.65x) had measured the defect: when a fix makes a test fail, check the test's quantity against an exact reference (B_e(r_e)) before calling either side wrong.
- 2026-11-20 · L-10 again: told to use its own projector and its own surfaces, the verifier refuted my proportional-to-gradient prediction and showed the corrected value spans 23.2-28.65 mA by recipe -- a range I would have reported as one number.
- 2026-11-20 · a fit that T-13 found threshold-fragile turned out to sit on a ~100 MHz alpha error -- a row that flips under a small perturbation deserved a look at its inputs, not just its switch.

## Pending question

## Handoff
