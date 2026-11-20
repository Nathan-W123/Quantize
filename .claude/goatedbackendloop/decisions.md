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

### D-24 · Split null step cut to the smaller of the two trust radii; split certificate kept (2026-11-16)
- Decision: the split step cuts its quantum-weighted null step to
  min(null_trust_radius, trust_radius) before the whole-step cut. The
  split bootstrap keeps the restart certificate (no trust floor, no Newton
  probe).
- Why: with the whole radius below an explicitly given null one (unset,
  the null radius is half the whole one everywhere; corrected 2026-11-16 by
  round 9) the null step
  filled the step and the whole-step cut starved the range step 500x:
  fits zigzagged at the cap (water C1 B,C at 1e-3: 0.28 widths off, 107
  MHz, 342 iterations; after 0.006, 0.03 MHz, 34) and their refits passed
  4.5-8.6 widths unconverged. A Newton-probe certificate measures a point
  the split procedure never reaches (its spectral acceptance test stops it
  first; internal-prior water's base fit is 0.35 widths from it), and
  restart trust floors moved refits to other stopping points.
- Alternatives considered: Newton-probe certificate; restart trust floor
  1e-4 / 1e-3; validator rule null <= trust (does not cover autoconfig).
- Source: assumption (physics/numerics delegated); no accuracy/efficiency
  trade (both improve; default radii unchanged).
- Revisit if: split becomes a supported path -- its stopping points depend
  on step size (backlog N12).

### D-25 · A target-centred prior keeps the target's Hessian (2026-11-16)
- Decision: with prior_target_coords (Cartesian), the Hessian that sets the
  prior's precision is computed at the target (in the current frame) once
  and rotated into the frame afterwards; energy and gradient stay the
  current geometry's; hessian_distance measures against the target.
- Why: a prior must be fixed before the data; H from wherever the fit had
  got to made it move with the fit (isocyanic acid HF/6-31G: 97 mA per
  restart, no fixed point under refreshes). Every shipped script starts at
  its target, so the benchmark answers are unchanged (<= 1e-8 mA); a fit
  that recomputes its Hessian on schedule changes (split + target, no
  shipped script: isocyanic acid 0.91 mA).
- Alternatives considered: tighten the plateau rule (the fits were already
  converged for their model); refresh the Hessian at the answer like D-23
  (wrong model for a target prior, and worse scores: 23.98 -> 40.13 mA).
- Source: assumption (physics delegated); no accuracy/efficiency trade.
- Revisit if: a target prior is wanted whose curvature follows the fit
  (that would be a different prior, and needs its own fixed point).

### D-26 · Config runs state their model error; unstated runs are labelled (2026-11-16)
- Decision: configs can state the B0-versus-Be model error
  (sigma_systematic_mhz per isotopologue, sigma_includes_systematic,
  model_error: {auto, isotopologue_correlation}); a run whose rows carry
  none keeps measurement-only widths and says so in the report and bundle.
- Why: raw-B0 config widths were 25x narrow on water (0.28 vs 6.9 mA) with
  no way to state the term; the auto estimate is 3-10x large (F13), so it
  stays opt-in.
- Alternatives considered: auto estimate by default; keys only.
- Source: stated by user ("Keys + honest label").
- Revisit if: F13's estimator is calibrated (then auto could be default).

### D-27 · A correction's sigma replaces a stated model error on its rows, and says so (2026-11-17)
- Decision: where a B0 -> Be correction with its own sigma is applied, that
  sigma is the row's model error and a stated sigma_systematic_mhz (or the
  engine's estimate) for the row is not used -- unchanged numbers -- but the
  label reads "corrections (stated replaced)" and the Laplace warnings name
  the rows and the knob that widens them (the correction's sigma).
  sigma_includes_systematic alone is refused.
- Why: both code paths (apply_corrections_to_isotopologues,
  _systematic_covariance) already do this by design: the correction removes
  the gap the stated value stood for, and its sigma is the error left. The
  defect was silence (round 10 B-7: a stated 1% / 10% left corrected water's
  widths unchanged with no message).
- Alternatives considered: add the two in quadrature (double counts when the
  user's value was meant as the total); stated wins (discards the
  correction's own error estimate).
- Source: assumption, unconfirmed (physics decision delegated; numbers unchanged).
- Revisit if: users state model error meaning "residual after corrections".

### D-28 · Earlier runs' outputs are named in the bundle, never deleted (2026-11-17)
- Decision: `quantize uncertainty` lists, under files_from_earlier_runs with
  one warning, every entry of its output folders whose newest file predates
  the run's start; it deletes nothing but its own stale bundle (as before).
- Why: a refused run (a typo such as --samples 0) left the last run's
  posterior and bootstrap summaries, samples and plots beside a new exit-2
  bundle (round 10 B-6); deleting them would cost hours of PySCF refits.
- Alternatives considered: delete them at run start; move them aside.
- Source: assumption, unconfirmed (the reviewer offered both; the non-destructive one chosen).
- Revisit if: a coarse-timestamp filesystem (2 s FAT) misfiles this run's own files.

### D-29 · The regenerated table leaves out `elec` and `all` (2026-11-17)
- Decision: N13's runs use every same-level configuration except `elec`
  and `all` (which includes elec); the README says they were not run.
- Why: their rotational g-tensor needs pyscf.prop (pyscf-properties), which
  the container lacks; installing it was blocked by the permission check.
  The benchmark's own notes say elec costs more than it returns on this set
  (ozone 2.32 -> 3.23 mA), and the base-only table does not depend on it.
- Alternatives considered: install pyscf-properties 0.1.0 --no-deps and run
  them as an extra pass at the same vintage.
- Source: stated by user ("Leave them out").
- Revisit if: pyscf.prop becomes available (one g-tensor per molecule).

### D-30 · A separate correction level, at its own minimum, built once, opt-in (2026-11-18)
- Decision: correction_method / correction_basis (constructor and
  rovibrational_corrections keys) build the Hessian-built corrections at that
  level's own minimum, once, kept across refreshes; off by default; refused
  without harmonic_from_hessian, with a backend that cannot optimise
  (pyscf_hf only), spectral-only, or with a torsion replacement.
- Why: on water the fit level's corrections set the floor (CCSD/cc-pVDZ
  -0.58 mA, data-only the same); MP2/cc-pVTZ corrections gave -0.03 mA for
  ~3 min. The benchmark's corr protocol (own minimum) is the measured one;
  evaluating another level's Hessian off its minimum gave an artefact there.
- Alternatives considered: the other level's Hessian at the fit's geometry
  (refreshed); making it the default (adds cost to every run: D-03).
- Source: recommended and approved ("Correction-level option").
- Revisit if: a default correction level is wanted (cost per molecule:
  ~7 Hessians at the correction level for a triatomic).


### D-31 · Vinyl fluoride's deuterated A0 carry a stated measurement sigma (2026-11-19) — superseded by D-32
- Decision: new optional `Isotopologue.measured_sigma_mhz` (measurement side:
  max with the quoted step, survives a vibrational correction, not in the
  systematic part). Vinyl fluoride 784/787/788/789/791 A get
  |Delta0 - Delta0(parent)| A^2/K rounded to 10 MHz: 910, 160, 1350, 280,
  70 MHz. Benchmark vinyl fluoride rows re-run at the branch head (only this
  change affects them since 5cfc5d4; base config reproduced exactly before
  it) and merged into the archive; the table states the vintage mix.
- Why: the compilation's round-number A values are inconsistent (negative
  defects for 784/788); data-only justification, no reference used. Effect:
  RHF hybrid 18.34 -> 13.90 mA vs ref (11.05 vs SE100), B3LYP 6.20 -> 5.48,
  matching hand-dropping 784/788 (13.63 / 5.42) without discarding data.
- Alternatives considered: drop the rows (loses B/C-consistent information,
  and dropping all D A made RHF worse, 15.75); sigma_rel (model error --
  removed by the correction, so no effect on corrected fits); robust losses
  (measured, no fix); sizing to SE100's misses (tunes to the yardstick).
- Source: recommended and approved ("Down-weight those A rows").
- Revisit if: the original A0 papers (NBS refs 804, 875, 948) or a modern
  measurement are found; or a component-aware defect floor lands in the
  engine (would make the stated values redundant).

### D-32 · Vinyl fluoride's constants are Hayashi & Inagusa's 1989 measurements (2026-11-19)
- Decision: replace NBS 1968 (entry 780) with Hayashi & Inagusa, J. Mol.
  Spectrosc. 138, 135 (1989), as tabulated in Piccardo et al. 2015's SI, for
  the module's eight species (adds A for both 13C species: 24 constants);
  each stated uncertainty becomes measured_sigma_mhz. D-31's defect-sized
  sigmas go. Rerun vinyl fluoride at both levels (also adds its r_e scoring).
- Why: the 1968 deuterated A were off by 118-1350 MHz; with the 1989 set
  every inertial defect is normal and Piccardo's r_e^SE reproduces all 24
  constants (with B3LYP corrections) to 0.06%.
- Alternatives considered: keep D-31's stated sigmas (estimates of errors
  now measured); also add the two CD2 isotopologues (offered; not chosen).
- Source: recommended and approved ("Yes, replace them").
- Revisit if: the original paper's table can be read directly (the SI is a
  secondary transcription; 791's B and C lack the source footnote).

### D-33 · Per-bond prior widths are not adopted (2026-11-20)
- Decision: keep the quantum prior's single sigma_x (the RMS that
  prior_sigma_for_molecule builds); do not widen the prior only along
  uncalibrated bond classes. The study is kept as
  scripts/archive/perbond_prior_widths.py.
- Why: exact per-class widening (added variance / s^2 = 1.0, alpha pinned)
  fixes CH2ClF with offsets on (17.64 -> 2.44 mA RHF, 11.22 -> 4.76 B3LYP vs
  Vogt r_e^SE) but makes HNCO worse at both levels (20.01 -> 24.81 mA and
  angle 1.49 -> 16.47 deg at RHF; 2.03 -> 3.62 at B3LYP, vs SE100 and r_s
  alike). Bond-only widths cannot cover an angle error (HF's H-N-C); the
  blended RMS width loosens the angles too. Without offsets it was mixed
  (CH2ClF worse at RHF, better at B3LYP; water/ozone worse at B3LYP).
- Alternatives considered: angle-aware widening (a third variant designed
  after seeing results; not run); calibration molecules for C-Cl, N-H, C-N
  (the root fix; backlog N16).
- Source: auto-picked, no answer in 15 min (Q-01, recommended option)
- Revisit if: calibration molecules cover these classes, or an angle-aware
  width is tested on molecules outside this set.

### D-34 · Next task: calibration molecules for the uncovered bond classes (2026-11-20)
- Decision: T-07 looks for calibration molecules for C-Cl, N-H and C-N
  (backlog N16's root fix), literature first, no engine change; candidates
  and sources are reported before any molecule is added.
- Why: per-bond widths failed (D-33); a class seen once cannot be calibrated
  by any width rule, only by more members.
- Alternatives considered: F13 (auto systematic sigma), N2 (sparse-data rows).
- Source: auto-picked, no answer in 15 min (Q-02, recommended option)
- Revisit if: the user picks another next step.

### D-35 · Calibration molecules: calibration-only, five classes, the 0.5 mA rule (2026-11-20)
- Decision: new molecules enter the bond-offsets pool as calibration-only
  entries (reference structure + theory geometry; not fitted or scored);
  they are sought for C-Cl, N-H, C-N, O-H and O-O; a reference must be r_e
  or r_e^SE with a stated uncertainty of 0.5 mA or better.
- Why: the offsets need only a reference and a theory geometry; scoring new
  molecules would change the 9-molecule comparison; O-H and O-O are
  singletons too (water, ozone); the rule keeps the yardstick independent.
- Alternatives considered: full benchmark molecules; only the three classes
  first named; accepting r_s / effective references.
- Source: auto-picked, no answer in 15 min (Q-03, recommended options)
- Revisit if: the user wants new molecules scored as well, or no N-H source
  meeting the rule turns up.

### D-36 · Add CH3Cl, HCN, H2O2 as calibration-only references behind calibration=extended (2026-11-20)
- Decision: dev.monofluoro_references.CALIBRATION_ONLY holds methyl chloride
  (Melosso 2025 r_e^SE), hydrogen cyanide (Carter, Mills & Handy 1992 r_e)
  and hydrogen peroxide (Baraban 2018 r_e^SE). The benchmark adds them to the
  offsets pool only with calibration=extended, writes those rows to a
  separate results file (_calext), and chooses the pool by key
  (offsets_pool). Default runs are unchanged.
- Why: measure what the extra calibration does before changing the archived
  tables; a flag keeps the README reproducible from the default code.
- Alternatives considered: add without HCN (C#N pooled with N=C); search
  more for N-H first; stop.
- Source: auto-picked, no answer in 15 min (Q-04, recommended option)
- Revisit if: the measurement says the extended pool helps, at the next gate
  (make it the default and rerun the tables), or it hurts (drop it).

### D-37 · Measure the extended calibration on all nine before deciding the default (2026-11-20)
- Decision: rerun only the offsets configuration for the nine scored
  molecules at RHF/6-31G and B3LYP/6-31G(d), default pool and
  calibration=extended back to back, in a frozen worktree at 41f13eb with
  copies of the 5cfc5d4 theory-geometry and Hessian caches; compare per
  molecule. No default, README or archived-table change.
- Why: four molecules measured so far; the other five's prior widths move,
  so their rows would too; a same-code default rerun makes the comparison
  pool-only.
- Alternatives considered: drop H2O2's O-O (bond-order mismatch with ozone;
  chosen after seeing results); stop and keep it opt-in.
- Source: auto-picked, no answer in 15 min (Q-05, recommended option)
- Revisit if: the rerun shows the five getting worse, or the user picks
  the O-O exclusion.

### D-38 · The extended offsets pool is the benchmark default (2026-11-20)
- Decision: the benchmark pools bond offsets from the scored molecules plus
  CALIBRATION_ONLY by default; calibration=scored restores the earlier pool
  and writes to *_calscored.json (the token is "scored", not "default", so
  the name says what it does). Each offsets row records its pool
  (offsets_pool). Every offsets-family row is rerun at both levels and the
  README tables regenerated.
- Why: nine molecules x two levels, offsets config: mean hybrid bond RMS vs
  primary references 8.56 -> 5.37 mA (RHF), 4.96 -> 4.13 (B3LYP); vs r_e
  where one exists 6.00 -> 2.30 and 3.33 -> 2.12; CH2ClF and HNCO gain most;
  the five fluorides move <= 0.2 mA. Cost: ozone vs r_e 0.37 -> 1.02 mA at
  B3LYP (H2O2's single O-O against ozone's 1.5-order bond); HNCO and CH2ClF
  angles worse at RHF.
- Alternatives considered: keep it opt-in with a README note; drop H2O2's
  O-O first (chosen after seeing results).
- Source: auto-picked, no answer in 15 min (Q-06, recommended option)
- Revisit if: the full offsets-family rerun reverses the offsets-config
  picture, or more O-O / N-H references arrive (bond-order split).

