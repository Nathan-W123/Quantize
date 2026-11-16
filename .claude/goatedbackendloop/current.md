# Current task
Status: done (step 8); next N1
Request: Verify the user's 7-item list (5 confirmed + 2 possible); fix where true and needed.

## Brief
Verified by workflow wf_e5e2e0fe-c99 (7 verifiers + critic; verdicts saved in
scratchpad/verify/verdicts/*.json). All seven mechanisms real; 1, 3, 5, 6
confirmed, 2, 4, 7 partly. Fixes are coupled; implementing in the critic's
order, one commit per step, each with before/after measurement and tests:
1 item 5 subspace inverse -> 2 fit-side consistency (row parity, qfix, joint
ratio gate, trust reset, Cartesian alpha) -> 3 library prior off when QM on ->
4 one posterior (item 2) -> 5 GN honesty (item 7 fix B) -> 6 correlated
systematic (item 1) after resolving double counting -> 7 config defaults
(item 4) -> 8 bootstrap (item 3).

User decisions 2026-11-12: Cartesian alpha in internal mode (accepts +29% wall
in no-symmetry internal runs); library internal prior off by default when the
quantum prior is on, honours configured widths when used.

## Progress
- DONE 1dbac6f item 5: Laplace inverted over physical directions. Water floor
  off: 4 refusals + 4 wrong (angle 0.0) -> all 12 exact to <=4e-6;
  floored runs move <1e-6; geometry_uncertainty 15% faster (basis cache).
- DONE 1987406 step 2: qfix, joint ratio gate, row parity (my a6bc9ed bug),
  Cartesian alpha + no autoconfig retune, trust reset, autoconfig after trial.
  Restart moves 15.5 -> 0.04 mA (FF default), VF 80 it not converged -> 12 it,
  4.3x faster. Same-objective second pass moves 0.0 mA everywhere.
- CORRECTED: default analytic-water uncertainty fixture was frozen (0/3 steps)
  by the row bug -> earlier GN "26,012%" / "-372,388 saddle" numbers were at
  the start geometry. Real-molecule cost record (f5c0b12) measured nothing.
- DONE a699c1a step 3: internal prior off by default with a QM prior; widths
  as given win over the library's. FF internal mode now lands where the
  Cartesian benchmark does (12.25 vs 12.26 mA; was 4.4 mA apart).
- DONE 379a5e6 step 4: one posterior (_posterior_blocks) for Laplace, GN
  check, log_probability, report; v1 compute_uncertainty deleted. Config-run
  report widths 2.6x / 4.6x wider (v1 chi2 "inflation" ran backwards, x0.25).
- DONE 9465e88 step 5: GN check reports a failure only where the widths are
  taken (stationarity gate, relevance filter, stationary-saddle refusal).
- DONE step 6 (item 1): model error propagated as shared across
  isotopologues per constant type (rho 0.99, Q = I) and counted once (sandwich
  difference; reference sigmas flagged, systematic floor, auto fill and
  correction sigma all taken out of the measurement part). Report-only: fitted
  geometries move <=1.5e-14 A. Ozone vs r_e z 2.31/1.42 -> 1.47/0.90; FF C-H
  x0.78 (shared error cancels in isotopic differences); fixture double count
  1.364 -> 1.0007 at rho 0. Also: component/robust weights now in the
  propagation, CD rows no longer drop the term, chi2 rescaling floors only
  where sigma carries model error. +65-120 us per call; geometry_uncertainty
  unchanged within noise. Suite 1426 passed.
- DONE 27b97d3 step 7 core: config runs fit the engine/benchmark objective
  (joint, level-calibrated sigma, alpha 1, symmetry auto when the key is
  absent, D-09 library prior rule, engine conv_freq, enforce only for split;
  explicit split keeps the legacy set-up). normalize_config no longer injects
  prior_weight 1 / widths 0.04 A / 2 deg (it had defeated D-09 and step 3 on
  every CLI run). 9 molecules HF/6-31G internal: restart-stationary 2/9 ->
  8/9, bond 13.29 -> 9.51 mA, angle 4.05 -> 0.97 deg, QM cost -14%.
- DONE 60f1e62 step 7 decisions (user): start anchor off while the QM prior
  is on (D-11); shipped configs drop prior_weight (D-12). Default cartesian
  config path now bit-identical to the anchor-off measurement on 9 molecules.
- DONE b71a261 step 8 (item 3), verification workflow wf_3a613cec-119 running:
  error-model draws (constant_error_covariance, shared with
  _systematic_covariance), rto (quantum-prior centre shift), frozen
  corrections in the worker, convergence-certificate gate (termination
  reason + distance_to_mode <= 0.25), polish loop; quantum_prior_weight
  cache cleared on each quantum update (was stale: 2097 vs 1848 on water).
  Before: water cloud 3.06/0.62 vs target 5.70/1.11, 23/48 usable; FF
  9/48 usable, 1.5-2x narrow. After: water 1000 draws 0.99/1.05 (error-model),
  0.97/1.03 (rto); FF 400 draws 0.96-1.04; all usable. FF rto angles
  +12-17% (n=100) under test. Correction to the commit message: FF refit
  median 0.093 s after vs 0.083 s before (+12%), not ~0.05 s.
- VERIFIED wf_3a613cec-119 (2 lenses, 67 min): claims 1, 3, 5, xi-shape and
  the gate on shipped settings hold. Refuted: my FF rto "+17% angles walk the
  real surface" = Monte Carlo noise (n=100). Real: live rto refits refresh the
  quantum gradient past orca_update_thresh while xi and the target use the
  base's start-geometry Hessian -> FF C-H 7% narrow, water bond 2.6% narrow;
  frozen reference reproduces C_total paired within 2.2%, 15-21x faster.
  My seeds were 11 (FF n=100, water n=1000) and 23 (FF n=400), not recorded;
  verifiers used 7. Separately: rto xi draws are NOT reproducible across BLAS
  threads (eigh sign flips; measured 1.1-2.0 rel diff at 2e-14 coord diff).
  Other confirmed: CLI prints the error-model cloud as "95% Credible
  Interval" (up to 13x narrow on FF) and mixes it with MCMC in --mode both;
  split rto not refused and inconsistent; _proj_ref_terms_cache stale after a
  quantum update; chi2_rescale and mass-dependent B0->Be refit not frozen;
  draw_kept blind to Be target/sigmas and taken before polish; max_iter
  refusal ignores the certificate; 0.25 limit in posterior widths (up to 3.2
  target sds on FF); CLI default engine 'legacy' exits after the base fit;
  rto on spectral-only = traceback after the base fit; ddof 0 vs 1.
  Pre-existing: CD rows sign (r = pred - obs vs J = dpred/dx) and the
  sanitizer drops cd_observed; reported posterior uses a start-geometry
  Hessian (adaptive schedule; water config 10.8 mA away, alpha -14% if
  refreshed) -> flag, accuracy/efficiency trade (D-03).
- PLAN step 8b commits: (1) freeze quantum reference in refits + reproducible
  draws (symmetric square roots) + doc corrections; (2) refit gaps: chi2/md
  freeze, draw_kept, max_iter via certificate, split rto refused, limit in
  the cloud's own units, caches; (3) CLI interval from the posterior unless
  rto, no mixing, v2 default, early strategy check, ddof; (4) CD sign +
  sanitizer carry-through + CD rows drawn.
- DONE 2cbf948 fix 1: refits hold the base's quantum model (orca_update_thresh
  inf, joint only after fix 2); draws via symmetric square roots (seed-
  reproducible across BLAS). FF rto paired C-H 0.935 -> 1.001, all six 0.92-1.02
  -> 0.98-1.00 (seeds 7, 2026); 400 draws 282-294 s -> 20 s; refit 1.92 -> 0.091 s.
  Water rto paired 0.977/1.011 -> 1.003/1.004, em angle 1.022 -> 1.004.
- DONE f4428b4 fix 2: chi2/rm1 frozen; draw_kept = targets + sigmas + eff sigma
  after polish; max_iter judged by certificate (0/20 -> 20/20 usable); rto
  refuses split; gate scaled to the cloud (FF 0.075); _proj_ref_terms_cache
  cleared on quantum updates; reference held under joint only (fix 1 had
  broken split error-model 12/12 -> 8/12; restored). Suite 1464.
- DONE f734263 fix 3: CLI interval = MCMC if usable, else rto refits, else
  reported-posterior draws (2000); never pooled; v2 default, legacy exits
  before load; spectral-only rto exits before the fit; ddof 1;
  stacked_unweighted(want_jacobian=False) PPC 3.27 -> 0.26 s (FF, 2000).
  Water CLI interval 5.36 mA/1.04 deg (cloud) -> 6.5/1.13 (posterior 6.52/1.16).
- DONE 5f1842a fix 4: CD rows obs - pred (cos -1 -> +1), sanitizer keeps
  cd_observed (default path had 0 rows), error-model/rto draw CD observables
  (cd_row_sigmas). Fit distance 1.29 -> 7e-5, 5/40 -> 40/40 usable. Suite 1476.
- VERIFIED wf_688e5c55-db0 (20 min): reproducibility, CLI pooling/source,
  posterior draws, want_jacobian callers, CD sign consistency supported.
  New: split posterior omits the step's range-null coupling (error-model
  0.04x target on the angle); CLI quoted a bimodal rto cloud at a saddle;
  harmonic CD froze the CD rows (J_cd = 0); pred.sigma made the CD weight
  geometry-dependent (fit 4.19 widths off its optimum at cd_weight 1e4);
  YAML dropped cd_observed; small rto clouds; ridge flag ignored; fit loop
  finite-differenced a Jacobian it discarded; stale Hessian unreported.
- DONE fa9a145 fix 5: every bootstrap strategy refused under split; split
  report warns; docstring range corrected; gate fallback note.
- DONE 6909553 fix 6: CLI refuses rto at a saddle (bundle written, stderr),
  rto < 200 refits -> posterior interval + note, ridge reaches interval and
  analytic column (1e4: 6.5/1.13 -> 5.4/0.83).
- DONE 961ec3e fix 7: CD predicted at the trial geometry (harmonic CD: J_cd
  0 -> 974), default sigma 40% of observed fixed at construction (distance
  4.19 -> 2.6e-4 at 1e4/20%), cd_residuals_mhz from _cd_rows, YAML carries
  cd_observed.
- DONE 471662a fix 8: fit loop residuals without the Jacobian, bit-identical;
  refit median water 37.4 -> 27.4 ms, FF 88.1 -> 57.4 ms.
- DONE 9e78f59 fix 9: hessian_distance + posterior warning past
  orca_update_thresh (water cfg 10.77 mA, FF 5.12 mA warn; water pyscf 3.23
  no) + bootstrap summary line. Suite 1490 passed, 21 skipped.
- VERIFIED wf_d6294e4a-4b4 (13 min): saddle, small rto, fit-loop bit-identity
  and speedup, Hessian tracking supported. New: split refusal too broad (full
  rank is exact); given cd_sigma lacked model error, partial blocks 5%;
  delta_J/delta_K unvalidated (water delta_K wrong sign); CD rows invisible in
  report/CLI; ridge note missing for MCMC/rto; two refusals on stdout without
  bundle; my "finite-differenced Jacobian" claim wrong (analytic).
- DONE 33f8d01 fix 10: split refusal/warning only when split_coupling > 1e-8
  (ABC 4.5e-16: 200 usable 0.999/0.988; C only 0.085 refused); rto always.
- DONE 2327cfd fix 11: CD sigma sqrt(meas^2 + (40% obs)^2), partial keys get
  model error, rows DJ/DJK/DK only (+warning), unused pred_cd warned. 1%
  sigma: pulls 140 -> 0.84, shift 0.41-0.62 -> 0.00 widths.
- DONE 2b973b7 fix 12: distortion rows in posterior, report.md table,
  residuals.csv, CLI; chi-square labelled with row counts.
- DONE ca6dc1a fix 13: ridge note for MCMC/rto intervals, mcmc/post-fit
  refusals to stderr with bundle, six Jacobian comments corrected.
- DONE d3932fd fix 14: CD predictions skip the reduction residual: CD fit
  1.43 -> 0.70 s, bit-identical. Suite 1500 passed, 21 skipped.
- VERIFIED wf_588822aa-f67 (fourth check, ~35 min/lens): HIGH split_coupling
  read the posterior's null space, not the step's (internal-prior C-only split
  passed at 2.7e-16, 0/400 usable refits); its size did not track the error;
  --mode both lost MCMC on a post-fit refusal; CD 40% only at cd_weight 1,
  'auto' reduction re-decided per call, cd_model_sigma recomputed per build;
  rebuilt report dropped CD rows; HTML/plots/CSV mixed kinds; MCMC reason on
  stdout only; ridge note on any non-default ridge; 'not fitted' per build.
- DONE 867e925 fix 15: split_coupling = null response |K| (C-only 0.46,
  two-iso asym 0.054, B,C 0, C1 ABC 2e-11; tol 1e-6) + split_inconsistency
  (base distance_to_mode > 0.25): internal-prior C-only (2.2 widths) and
  two-iso C-only (3.8 widths: step rank 1 vs posterior rank 2) now refused
  and warned in the posterior and the Laplace summary; full-rank still
  bootstrapped; --mode both runs MCMC after a post-fit refusal (exit 2).
  +1 ms per split posterior call, none on joint.
- DONE d1f79db fix 16: cd_reduction (A/S) or auto pinned once per fit
  (98 -> 1 decisions); reduction on rows and in the report; stated/carried
  cd_model_sigma kept (rebuilt engine 1.3x -> 1.0); cd_weight warning; README
  CD note rewritten (was "unvalidated, 100%"); 'not fitted' once per process
  and in posterior warnings. A vs S on water: geometry moves < 1e-4 widths.
- DONE 5faf56c fix 17: payload carries the uncertainty block (rebuild 39 ->
  57 lines, CD table kept; old payloads fall back to residual_rows); CSV kind
  column; HTML distortion table; MHz plots rotational only.
- DONE ab1ceff fix 18: MCMC reason on stderr; ridge note only when a width
  moves > 1e-3 (ridge 0: 1 -> 0 notes, 1e4 kept). Suite 1512 passed, 21 skip.
- NOT DONE (flagged): per-key CD model fractions (one molecule is not a
  validation set; backlog N8); rebuilt report lacks rovib/conformer
  sections (pre-existing, N9); split could adopt the step's rank instead of
  refusing (N7).
- VERIFIED wf_4fb65e30-399 (fifth check, 11.5 min): fix 15's |K| threshold
  refuted -- it refused accurate C1 split checks (forced clouds matched
  controls at |K| up to 0.058) and |K| does not track the error (0.0195 ->
  refits 1.3-1.4x off); off-mode test blamed internal mode for an early
  Cartesian stop; CLI quoted an MCMC interval without the split caveat;
  exit 2 ambiguous; --chains 1 message; rebuild cov path absolute; HTML no
  warnings; cd_weight warning assumed 40%; cd_model_sigma unvalidated;
  'auto' from the start geometry (multistart starts could differ).
- DONE 55c7472 fix 19: split bootstraps run (rto and unformed posteriors
  refused); summary flags cloud/analytic departures beyond 3x Monte Carlo;
  |K| a worded detector in the warnings; CLI interval carries the split
  caveat (output, warnings, cloud block); robust try/except.
- DONE 72ece8a fix 20: split refits certified by a restart (step's fixed
  point != posterior mode by the coupling itself): CLI split_int 14/40 ->
  40/40 usable, angle flagged 0.041x; off-mode fits no longer refused
  (internal-prior C 199/200 angle 0.65x; two-iso C 199/200 angle 0.10x);
  accurate cases 200/200 no flag. +1 step run per split refit.
- DONE 22df96e fix 21: exit 3 for partial (bundle partial/exit_status;
  codes in --help and README); --chains 1 message; ACF estimator (3 -> 0
  RuntimeWarnings); relative cov path in the payload; HTML warnings;
  ridge check 2.25 -> 1.34 ms.
- DONE e927d67 fix 22: 'auto' from the observed A,B,C (150-degree start:
  S -> A); cd_model_sigma must be finite > 0 (ValueError names key);
  per-key cd_weight warning. Suite 1526 passed, 21 skipped.
- VERIFIED wf_706bb93b-253 (sixth check, 18 min): HIGH restart certificate
  passed stalled refits (restart rejecting every step moved 0.0 at
  max_iter 2-8); HIGH CLI 150-degree start still pinned S (corrections had
  replaced obs_constants, A < 0); Gaussian MC error 3x small for kurtosis
  ~20 clouds; no size floor; off-mode summary blamed a zero coupling;
  stale bundle after pre-fit refusal; exit 3 missed refused MCMC.
- DONE 090737a fix 23: unmeasured only if the restart accepted no step AND
  ended on max_iter (requiring an accepted step had dropped good cases to
  0-13/200); vs true fixed point, 0/192 passed far; kurtosis-aware MC
  error; flag needs >= 2% move of the reported width; off-mode note with
  Newton shift (+4.38 deg).
- DONE dd362b4 fix 24: observed ABC kept at construction for 'auto' (CLI
  150 deg: S -> A); bundle cleared at start, pre-fit refusal writes one;
  exit 3 for any requested part with nothing usable; cd_model_sigma and
  cd_reduction checked by validate; floor named; caveat quotes verdict;
  old/Windows covariance paths mapped. Suite 1539 passed, 21 skipped.
- DONE step 8, wrapped 2026-11-16 at the user's choice (round-6 fixes not
  re-verified: backlog N10). Next task chosen by the user: N1 (which
  quantum Hessian the prior's curvature uses).

## Observations
- Workflow sizing: 4-CPU container caps concurrent agents at 2; 7 broad
  verifiers (75-170 tool calls each) took 3.6 h wall. User asked why.
- Critic caught cross-item dependencies none of the verifiers saw (alpha
  discontinuity, double counting, qfix+gate coupling) -- worth its cost.
- Fixing an optimizer exposed test values tuned to its defects (water sigma
  0.005 under q-space alpha; GN flip at a frozen start). Ablate before
  re-pinning: the ablation showed which change moved each number.
- Worker restart killed a foreground suite mid-run (exit 137): run long
  checks in the background with output to a file.
- Accuracy yardstick: 7 of 9 references are r_s/effective (+/-5 mA); 1-3 mA
  differences cannot rank fixes. Use restart-stationarity and convergence as
  the correctness evidence; report reference deltas as indicative only.
- Step 6: a numpy rigid rotor on the repo's own r_e data (ozone) gave a
  physics test of the correlation model independent of the engine -- cheaper
  and more decisive than re-running the verifier's level ladders.
- A 5-sample timing median showed a false +43% regression; 40 samples showed
  noise. Time with >= 30 repetitions before calling a regression.
- Step 7: edited run_generic while a measurement queue ran against the live
  tree; part of queue C ran mixed code. Measure "after" against a frozen
  worktree, never the tree being edited.
- Step 8: a polish loop "fixed" nothing because the samples were converged;
  the real defect was a stale alpha cache making the certificate wrong.
  When a remedy changes nothing, check the measurement, not the fit.
- Step 8: wrote a timing figure into a commit message from memory (FF ~0.05 s)
  instead of the log (0.093 s). Quote numbers from the measurement file.

- Step 8b: my step-8 numbers used seeds 11/23 that the commit did not record,
  and the verifiers could not reproduce them; separately the rto draws were
  not reproducible across BLAS threads at all. Record seed, threads and code
  hash with every number; make draws a function of the seed alone.
- Step 8b: fix 1 (hold the reference) broke split error-model (12/12 -> 8/12);
  caught only because I re-ran the verifiers' split script on both versions.
  Re-run every mode a reproduction script covers, not just the one fixed.
- Step 8b: two 40-draw readings (CLI rto 0.77, CD error-model 0.84) looked
  like defects and were noise at 400 (0.99, 0.97). Confirm with >= 400 draws
  or a paired check before acting on a cloud ratio.

- Round 2: each adversarial round found medium defects in my previous
  round's fixes (round 1: 5, round 2: 4), mostly in modes or paths my own
  measurements did not exercise (split, saddle, harmonic CD, no cd_sigma).
  Keep rounds narrow and run them until one comes back clean.
- Round 4: 1 high + 3 medium in round 3's fixes, again in paths my checks
  skipped (internal-prior split, multi-isotopologue symmetric start, the
  `quantize report` rebuild). A test that measured the quantity's meaning
  (null response vs the linear-response error) beat a threshold on a proxy.
- Round 5 (correction to round 4's note): the "meaningful" |K| was still a
  proxy -- forced bootstraps showed it neither separates good from bad nor
  scales with the error. Three rounds tuned a predictor of what the
  bootstrap measures directly; the fix was to let the bootstrap run and
  flag departures. When a check keeps needing new thresholds, ask whether
  the thing it predicts can simply be measured.
- Round 5: my own after-measurement used the API on analytic water where
  the split gate passed 200/200; the CLI config (internal mode) showed
  14/40. Measure through the user's entry point too.

## Handoff
