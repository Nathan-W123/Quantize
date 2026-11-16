### D-04 · Gauss-Newton stays the reported basis, but must announce itself (2026-11-10)
- Decision: keep JtJ as the Hessian behind reported uncertainties, and add a
  directional check that warns when the omitted sum_i r_i Hess(r_i) dominates
  the curvature it perturbs or flips its sign. Separately, refuse MCMC
  intervals when R-hat/ESS say the chains have not mixed.
- Why: Gauss-Newton is standard for nonlinear least squares and arguably the
  object you want -- the curvature of the model, not of one noisy draw.
  Switching the reported basis changes published numbers and needs a story
  for why a direction the fit appears to constrain is reported unconstrained.
  The silence is fixable now and is strictly an improvement under any later
  choice.
- Alternatives considered: (a) treat the exact posterior as authoritative and
  report that direction as prior-limited; (b) point MCMC at the Gauss-Newton
  objective so all three engines agree -- rejected because it demotes MCMC to
  a sampler of the approximation, removing the reason to have one.
- Source: stated by user (chose "Detect and warn first" from four options)
- Revisit if: a molecule's reported width is dominated by a flagged
  direction, or the systematic-error model lands and residuals drop to the
  few-sigma range where Gauss-Newton is valid anyway.
### D-05 · Do not weight the fit with the model error (2026-11-10)
- Decision: `systematic_in_weighting` exists, defaults off, and stays off.
  The model gap is a reporting term only.
- Why: it fixes the statistics (residuals 11.87 -> 0.51 sigma, monotonic
  intervals) and degrades fitted accuracy ~2.3x on the angle in 5 of 5
  starting geometries. Accuracy is a hard goal; this is the flagged-trade
  case the user asked to be told about rather than shipped.
- Alternatives considered: quadrature addition instead of a floor
  (numerically identical here, and asserts independence the coherent error
  does not have); full GLS with a correlated systematic covariance (correct
  in principle, needs a correlation model the inertial defect does not
  provide).
- Source: recommended and approved in advance by standing instruction D-03
- Revisit if: the B0-versus-Be correction lands and residuals fall to the
  few-sigma range on their own, at which point the floor would rarely bite;
  or if a correlated-covariance treatment is built.

### D-06 · Ship the row-consistency fix and restate the water claim (2026-11-12)
- Decision: fix the step-acceptance row mismatch and rewrite
  `test_calibrated_quantum_prior_improves_the_weakly_determined_angle` to
  assert the measured trade, with a 1.05x ceiling on the angle.
- Why: the gate was provably not a descent test (zero-step sides differed
  25-30%). Bond improves 3.9x, angle gives up 9.6% and lands 0.8% past
  theory's bar; the repo's own combined metric goes 1.384 -> 1.127 (2.0 =
  theory alone). Re-tuning cannot recover the angle -- nine values of
  quantum_prior_sigma_ang from 0.002 to 0.030, none beats theory on both, and
  0.005 is still best, so the calibration was not stale.
- Alternatives considered: keep the regression as an xfail; hold the fix and
  flag only; apply it to the bootstrap path alone (two acceptance rules in one
  optimiser).
- Source: stated by user (chose "ship it, update the test" when asked).
- Revisit if: the angle drifts past 1.05x theory's error, or a config is found
  where the fix costs real quantum chemistry.

### D-07 · Measure cost on a real molecule before tuning it (2026-11-12)
- Decision: establish whether the 2.4x iteration cost held with real quantum
  chemistry before touching convergence criteria or the trust-region schedule.
- Why: the 2.4x came from a 2-parameter analytic surface. Measured on three
  real molecules at HF/6-31G in internal mode, it is zero -- identical
  iterations and identical answers. Tuning would have been chasing an artefact.
- Alternatives considered: chase the iteration count immediately; log it and
  move on.
- Source: stated by user.
- Revisit if: a production config is found whose zero-step ratio sits near one,
  where the fix would change the accept/reject pattern.

### D-08 · Quantum weight from the Cartesian Hessian in internal mode (2026-11-12)
- Decision: compute the calibrated alpha's lambda_bar from the projected
  Cartesian Hessian in internal mode too; alpha_quantum = 1 and no heuristic
  or autoconfig rescaling when the prior sigma is calibrated.
- Why: the q-space definition mixes Hartree/A^2 and Hartree/rad^2 and is
  discontinuous in planarity (formyl fluoride 3438 / 76 / 1.4 at 0 / 1e-4 /
  1e-3 A out of plane); the Cartesian one stays 1911-1931.
- Cost accepted: +29% wall in no-symmetry internal config runs (92 vs 71 s
  over 9 molecules, 6 more Hessians, mostly vinyl fluoride); neutral with
  symmetry imposed.
- Alternatives considered: keep q-space alpha and warn.
- Source: stated by user (chose "Implement it").
- Revisit if: the extra Hessians turn out to be overhead rather than the
  quantum term doing work.

### D-09 · Library internal prior off when the quantum prior is on (2026-11-12)
- Decision: the internal-coordinate library prior defaults off whenever the
  quantum prior is active; it stays available and honours configured widths
  (no silent replacement) when switched on.
- Why: research design is spectra + QM in one objective; the library term
  silently replaced configured widths (0.12 A / 5 deg -> 0.015 A / 2 deg),
  has biased centres (vinyl fluoride H-C-H 109.5 vs 120.1 deg), and with it
  off formyl fluoride's angle error fell 2.03 -> 1.08 deg.
- Alternatives considered: keep on but honour widths; leave as is.
- Source: stated by user (chose "Off when the QM prior is on").
- Revisit if: a molecule is found where the QM prior alone leaves a
  direction unconstrained that the library prior used to hold.

### D-10 · Model error: correlated across isotopologues, counted once (2026-11-13)
- Decision: propagate the systematic as S R S with R_ab = Q[comp_a, comp_b] x
  (1 if same isotopologue else rho), rho = 0.99, Q = I by default (both
  constructor args); report the sandwich difference
  A [D (Sigma_meas + S R S) D - I] A^T so model error already in the weights
  (reference sigmas via sigma_constants_include_systematic, systematic floor,
  auto-filled sigma, correction sigma) is counted once; a vibrationally
  corrected row charges its correction sigma, not the removed gap.
- Why: measured correlation 0.94-1.00 across isotopologues (4 molecules,
  theory and experiment), no consistent sign across A/B/C; ozone's real gap
  vs r_e: z 2.9/3.5 independent -> 1.3/1.6 correlated (numpy rigid rotor),
  1.47/0.90 in the engine (was 2.31/1.42 double-counted).
- Alternatives considered: independent (shipped); uniformly coherent (refuted,
  z 62 on water's angle); per-isotopologue coherent (refuted); rho = 1
  (ill-conditioned FF spectral-only z 4.55); GLS weighting with the
  correlated error (fit-side, untested -- backlog).
- Source: physics decision delegated to Claude by the user ("Which model is
  right is a physics decision for you to make").
- Revisit if: a molecule's measured cross-isotopologue correlation falls well
  below 0.9, or near-prolate B-C coupling proves general (kappa-gated Q).

### D-11 · Start-anchored prior off when the quantum prior is on (2026-11-13)
- Decision: run_generic passes the preset's use_internal_priors (anchor each
  fit to its own start, 0.04 A / 2 deg; acts in cartesian mode only) only for
  spectral-only runs.
- Why: with it on, restarts and multistart seeds fit different objectives;
  9 molecules cartesian HF/6-31G: on 11.02 mA / 1.38 deg, 4/9 stationary
  (max 8.25 mA); off 9.53 mA / 0.97 deg, 8/9 (max 0.77) = internal mode.
  QM cost equal on 8/9 (HNCO converges only with it off).
- Alternatives considered: keep the presets as they are.
- Source: stated by user (chose "Off when the QM prior is on").
- Revisit if: a spectral+QM molecule needs a start anchor to stay in its
  basin.

### D-12 · Shipped configs do not state prior_weight (2026-11-13)
- Decision: remove internal_coordinates.prior_weight from the 10 shipped
  configs that stated it (template value); widths stay as documentation.
- Why: a stated weight is honoured (D-09), so the examples kept the library
  prior on; as shipped it measured mean bond 12.88 vs 9.51 mA, angle 2.34 vs
  0.97 deg, 2.3x iterations, +23% wall (9 molecules, internal mode).
- Alternatives considered: prior_weight: 0 explicitly; leave as is.
- Source: stated by user (chose "Remove prior_weight from them").
- Revisit if: a shipped molecule is shown to need the library prior.

### D-13 · The bootstrap checks the reported covariance (2026-11-13)
- Decision: default bootstrap strategy "error-model" draws the constants from
  constant_error_covariance (measurement part + correlated model error, the
  matrix the reported covariance propagates); "rto" adds the quantum-prior
  centre; "isotopologue-gaussian" kept as legacy. The reported uncertainty
  stays the analytic posterior; the bootstrap prints its analytic target
  beside the cloud as a refitting check.
- Why: the legacy draw (quoted sigma, independent) matched neither the fit's
  weights nor the reported covariance, so the cloud described a third model.
- Alternatives considered: keep the legacy draw and only relabel it; make
  rto the reported interval (16x cost on FF, quadratic-proxy dependent).
- Source: assumption, following the critic's step-8 plan ("label the
  bootstrap as the sandwich; RTO only as validation") and D-10.
- Revisit if: a user needs the bootstrap to include the correction chain's
  geometry dependence (frozen here).

### D-14 · Bootstrap refits hold the base's quantum model, under joint only (2026-11-13)
- Decision: _freeze_corrections sets orca_update_thresh = inf when the
  objective is joint; split refits keep refreshing. Also frozen: chi2_rescale,
  the rm1 mass-dependent refit. Draws use symmetric square roots.
- Why: the analytic column, its target and the rto shift are all built from
  the base's quadratic model; live refits fitted another model (FF C-H 7%
  narrow) at 21x the cost. Under split the step's gradient Pn(g + H dx) and
  the posterior's Pn(g + H Pn dx) agree only near the reference, so holding
  it there refused 4 of 12 converged refits.
- Alternatives considered: refresh the base Hessian at the fit before drawing
  (accuracy/efficiency trade on the base fit; flagged as backlog N1).
- Source: assumption (physics decision delegated: "Which model is right is a
  physics decision for you to make"), following the verifiers' recommendation.
- Revisit if: N1 changes which Hessian the base reports with.

### D-15 · The CLI interval is a sample of the posterior (2026-11-13)
- Decision: interval/PPC/plots from the MCMC chains when usable, else rto
  refits, else draws of the reported posterior; never pooled; v2 is the CLI
  default and "legacy" exits before loading the config.
- Why: the default cloud (error-model) is the posterior less its prior part
  (up to 13x narrow on FF) and was printed as a 95% credible interval.
- Alternatives considered: print no interval for error-model; analytic
  +/-1.96 sd without draws (loses the PPC and plots).
- Source: assumption, following D-13 ("the reported uncertainty is the
  analytic posterior").
- Revisit if: the MCMC target is shown to differ from the reported posterior.

### D-16 · Distortion rows: observed minus predicted, kept by the sanitizer (2026-11-13)
- Decision: fix the CD residual sign; sanitize_isotopologues carries
  cd_observed/cd_sigma; bootstrap draws the observed CD constants.
- Why: the old sign pushed every CD fit away from the data; the sanitizer
  made fit_cd_constants a silent no-op on the default path.
- Alternatives considered: refuse CD fitting until validated.
- Source: assumption (pre-existing correctness bug; opt-in feature, no shipped
  config supplies cd_observed, so no shipped result moves).
- Revisit if: CD predictions from the harmonic force field prove too crude to
  weight at the default cd_sigma.

### D-17 · No bootstrap under the split objective (2026-11-13) — superseded by D-21
- Decision: BootstrapEngine refuses every strategy when objective_mode is
  not joint; internal_coordinate_posterior warns under split that its widths
  are the posterior blocks', not the split estimator's spread.
- Why: the split posterior's quantum term Pn H Pn leaves out the step's
  range-null coupling; split error-model refits read 0.04x their target on
  the angle (step's own linear response predicts 0.05x).
- Alternatives considered: build the split target from the step's linear
  response (K_c Sigma K_c^T); correct the split covariance itself (backlog
  N7).
- Source: assumption ("the main solving method is hybrid joint method not
  svd"; split is legacy).
- Revisit if: someone needs uncertainty for split fits.

### D-18 · CD rows: geometry-dependent prediction, observed-based default sigma (2026-11-13)
- Decision: _predict_cd_for_iso uses the Hessian at the trial coordinates
  whenever one is set (attached pred_cd only without a Hessian); the default
  CD sigma is max(40%, fraction) of the OBSERVED value, fixed at engine
  construction; YAML configs may carry cd_observed / cd_sigma.
- Why: the attached prediction gave zero-Jacobian rows; a prediction-based
  sigma moved the weight with the geometry and left fits off their optimum.
- Source: assumption (correctness; opt-in path).
- Revisit if: a better (anharmonic) CD prediction is attached and should be
  used as an offset on the harmonic geometry dependence.


### D-19 · Split refusals test the meaning, not a proxy (2026-11-13) — superseded by D-21
- Decision: split_coupling is the null response |K| over the physical
  directions (tol 1e-6), and a split fit standing more than 0.25 posterior
  widths from its posterior's mode is refused by the bootstrap and warned in
  the report; --mode both still runs MCMC after a post-fit refusal.
- Why: the raw coupling |Pn H Pr| neither tracked the error (0.043 for 6%,
  0.085 for 95%) nor saw rank disagreements between step and posterior
  (internal-prior split: 0/400 usable at coupling 2.7e-16).
- Alternatives considered: build the posterior's null space from the step's
  own rank decision (backlog N7); compare ranks directly.
- Source: assumption (split is legacy; D-17's scope).
- Revisit if: split fits need uncertainty rather than a refusal.

### D-20 · Distortion rows keep one reduction and a stated model sigma (2026-11-13)
- Decision: cd_reduction ('A'/'S') names the observed block's reduction;
  without it 'auto' is decided once at the first prediction and kept. A
  cd_model_sigma that arrives with an entry is kept. cd_weight keeps its
  meaning (rows x sqrt(w)); the posterior says what the 40% becomes when
  w != 1. Uniform 40% kept (per-key fractions need more than water).
- Why: DJ/DJK/DK differ between reductions (water up to 3.8%); a per-call
  'auto' could step rows mid-fit; recomputed sigmas moved with draws.
- Source: assumption (opt-in path; no shipped config sets cd_weight).
- Revisit if: a second molecule validates per-key model errors.

### D-21 · Split fits are bootstrapped; the check reports departures (2026-11-16)
- Decision: under split only rto and an unformable posterior are refused;
  split refits are certified by a restart (unmeasured if it accepted no
  step and hit max_iter); the summary flags a cloud/analytic departure
  beyond 3 kurtosis-aware Monte Carlo errors that moves the reported width
  by >= 2%; |K| and the off-mode distance are warnings.
- Why: no threshold on a predictor separated accurate from broken split
  checks; the refits are the estimator, so the cloud is the measurement.
- Alternatives considered: refuse on |K| or distance (D-17, D-19); adopt the
  step's rank in the split posterior (backlog N7).
- Source: assumption (split is legacy; physics decisions delegated).
- Revisit if: split fits become a supported path needing correct widths.

### D-22 · `quantize uncertainty` exit codes 0/1/2/3 (2026-11-16)
- Decision: 0 interval and every part ran; 1 no interval; 2 usage error or
  refused before the fit; 3 interval reported but a requested part produced
  nothing usable (bundle partial: true, reasons). Every path writes
  exit_status; a stale bundle is removed at start.
- Why: exit 2 after a full MCMC interval was indistinguishable from a
  refusal where nothing ran.
- Source: assumption, unconfirmed (CLI contract).
- Revisit if: automation depends on the old 0 for partial runs.

### D-23 · `quantize uncertainty` recomputes the quantum Hessian at the answer (2026-11-16)
- Decision: the uncertainty command always recomputes the Hessian once at
  the converged geometry and re-converges (repeating only past the 5 mA
  drift rule) before any width is computed; `quantize run` and the library
  default keep the fit's own schedule and the Hessian-age warning.
  `--no-hessian-refresh` skips it (assumption: an escape hatch for
  expensive methods).
- Why: measured on 9 HF/6-31G molecules + water cfg: Hessians 4-20 mA old;
  refreshing moves widths up to 7%, alpha -14..+24%, the geometry under
  0.8 mA (below reference resolution) at +20-83% wall; restarts become
  stationary (<= 0.001 mA).
- Alternatives considered: every fit by default; opt-in only; keep as is;
  refresh only past 5 mA.
- Source: stated by user (policy "uncertainty only", threshold "always once
  at the end"); opt-out flag is an assumption.
- Revisit if: a method where one Hessian dominates the run makes the
  default too slow.
