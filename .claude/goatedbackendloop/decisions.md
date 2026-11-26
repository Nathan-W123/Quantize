### D-04 · Gauss-Newton stays the reported basis, but must announce itself (2026-11-10) · superseded by D-145 where the exact curvature is the larger
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


### D-39 · Next task: why chlorofluoromethane is worse at B3LYP (2026-11-20)
- Decision: T-08 diagnoses why CH2ClF's offsets hybrid is worse at B3LYP
  than at RHF (6.00 vs 2.91 mA vs Vogt r_e^SE; 8.18 vs 5.42 vs its primary
  reference). Diagnostic only, no engine change.
- Why: it is the one molecule that does not improve RHF -> B3LYP, an
  exception the README now states as unexplained; the per-bond pattern from
  T-07 (B3LYP C-Cl +7.4 / C-F -7.2 mA after offsets) fits backlog N16's
  mechanism but was not tested.
- Alternatives considered: an N-H calibration reference search; re-scanning
  the README's sparse-data sigma rows (N2).
- Source: auto-picked, no answer in 15 min (Q-07, recommended option)
- Revisit if: the user picks another option.

### D-40 · Next task: chlorofluoromethane's other isotopologues (2026-11-20)
- Decision: T-09 searches the literature for CH2ClF isotopologues beyond the
  benchmark's two Cl species (13C, D, CHD, CD2) with published ground-state
  constants. Literature first; adding data to the reference entry is a
  separate gate (it changes a benchmark molecule's input and its rows).
- Why: T-08 found the B3LYP error survives because two Cl isotopologues
  cannot see a C-Cl long / C-F short combination; more substituted species
  are the direct test. The benchmark's constants are also 1950s values (NBS
  Monograph 70).
- Alternatives considered: a second C-Cl calibration reference; an N-H
  calibration reference.
- Source: auto-picked, no answer in 15 min (Q-08, recommended option)
- Revisit if: the user picks another option.

### D-41 · Next task: a second C-Cl calibration reference (2026-11-20)
- Decision: T-10 looks for further C-Cl calibration-only references
  (CHF2Cl, CH2Cl2, CHCl3; r_e / r_e^SE with a stated sd <= 0.5 mA, D-35's
  rule), literature first, then measures whether they fix chlorofluoro-
  methane's C-Cl offset miss. Adding entries is a separate gate.
- Why: T-08 -- methyl chloride, the C-Cl class's only other member, misses
  CH2ClF's own C-Cl error by 12.4 mA (RHF, over) / 12.8 (B3LYP, under);
  T-09 -- more isotopologue data would not help under the engine's
  weighting.
- Alternatives considered: correlated (GLS) weighting on CH2ClF (F14); an
  N-H calibration reference.
- Source: auto-picked, no answer in 15 min (Q-09, recommended option)
- Revisit if: the user picks another option.

### D-42 · Measure C-Cl candidates on all nine molecules before adding (2026-11-20)
- Decision: run the offsets fits for all nine molecules at both levels with
  the current pool, the pool + CH2Cl2, and the pool + CHF2Cl (separately),
  from scratch theory geometries; nothing is added to the repository.
  Adding CHF2Cl would also need the user's say-so (it breaks D-35's 0.5 mA
  rule: C-Cl 0.9, C-H 1.1 mA).
- Why: only CHF2Cl narrows CH2ClF's C-Cl miss at both levels (3.1 / 5.4
  from 12.4 / 12.8 mA); CH2Cl2 fixes RHF's only; a better C-Cl centre made
  T-08's RHF fit worse, and new members also move every molecule's C-H
  (and, for CHF2Cl, C-F) offsets (L-17).
- Alternatives considered: add CH2Cl2 now; stop and record the fluorine
  dependence.
- Source: auto-picked, no answer in 15 min (Q-10, recommended option)
- Revisit if: the measurement shows either candidate hurting another molecule.

### D-43 · No second C-Cl calibration reference (2026-11-20)
- Decision: add neither CH2Cl2 nor CHF2Cl to CALIBRATION_ONLY; record the
  measurement (dev/monofluoro_references.py comment, backlog N16).
- Why: offsets fits on all nine molecules (T-10, one entry per unique
  bond, verified by a fresh agent): mean hybrid RHF 5.38 -> 5.57 (+CH2Cl2)
  / 5.70 (+CHF2Cl), B3LYP 4.13 -> 4.11 / 3.91. Neither improves both
  levels; CHF2Cl also breaks D-35 (C-Cl sd 0.9, C-H 1.1 mA). B3LYP's C-Cl
  error grows ~13-15 mA with a fluorine on the carbon, so only a
  fluorinated chloride transfers to CH2ClF at B3LYP.
- Alternatives considered: add CHF2Cl (rule exception; B3LYP -0.22, RHF
  +0.32); add CH2Cl2 (meets D-35; net loss).
- Source: auto-picked, no answer in 15 min (Q-11, recommended option)
- Revisit if: a fluorinated chloride with an r_e sd <= 0.5 mA appears, or
  uncalibrated-class widths stop depending on every class mean (N16).

### D-44 · Next task: decoupled widths for uncalibrated bond classes (2026-11-20)
- Decision: measure, in scratch only, a fallback width for uncalibrated
  classes that does not move with unrelated calibration members, on every
  molecule it reaches at both levels; no engine change without the user.
- Why: T-10 found HNCO's RHF row moves 8.14 -> 8.73 from a C-Cl member
  alone, because uncorrected_class_bias is the RMS of all class means and
  C-Cl's (~80 mA at HF) dominates it.
- Alternatives considered: correlated (GLS) weighting study (F14); r_e^SE
  references for the five r_s molecules (N3).
- Source: auto-picked, no answer in 15 min (Q-12, recommended option)
- Revisit if: the user picks another option.

### D-45 · Drop the robust fallback width; flag the chi2-rescale edge (2026-11-20)
- Decision: keep uncorrected_class_bias as the RMS of class means (no
  engine change); state in the benchmark docstring that isocyanic acid's
  RHF offsets figure (8.14) sits 1.9 mA of sigma_x above the chi2-rescale
  switch; backlog N19 for the switch.
- Why: T-11 (verified): the robust rule (1.4826 x median |class mean|)
  decouples only rows that barely moved (<= 0.13 mA) and costs HF HNCO 3
  mA without decoupling it; HF HNCO's row is a function of sigma_x alone,
  with a jump at ~17.3 mA where the first fit's reduced chi2 passes 4.0.
- Alternatives considered: backlog only (no docstring sentence).
- Source: auto-picked, no answer in 15 min (Q-13, recommended option).
  Option 1 said "README sentence"; the figure is stated in the benchmark
  docstring, not the README, so the sentence went there.
- Revisit if: N19 changes how the rescale fires.

### D-46 · Next task: how many benchmark rows sit near the chi2-rescale switch (2026-11-20)
- Decision: run the benchmark's base and offsets configurations at both
  levels with the optimiser instrumented (first-pass reduced chi2, dof,
  passes) and count rows near a threshold; measurement only.
- Why: T-11 found HF HNCO's offsets row 1.9 mA of sigma_x above the switch
  (N19); whether other published rows are as fragile is unknown.
- Alternatives considered: F14 GLS study; N3 references.
- Source: auto-picked, no answer in 15 min (Q-14, recommended option)
- Revisit if: the user picks another option.

### D-47 · Next task: the chi2-threshold check on the other configurations (2026-11-20)
- Decision: run the remaining benchmark configurations (HF: lam, bob,
  offsets+bob, defect, offsets+defect, cd, offsets+cd, track,
  offsets+track, track_direct, track_quadratic; B3LYP the same minus the
  two cd rows, which were never run there) at thresholds 4 / 3 / 5 with
  the T-12 instrumentation; torsion rows after, if time allows; measurement
  only.
- Why: T-12 covered base / offsets (1 of 36 fragile); the README's best-of
  table takes each leg's best over all configurations.
- Alternatives considered: F14 GLS study; N3 references.
- Source: auto-picked, no answer in 15 min (Q-15, recommended option)
- Revisit if: the user picks another option.

### D-48 · Record T-13's count in the benchmark docstring (2026-11-20)
- Decision: extend the docstring's rescale caveat (scripts/accuracy_upgrades_benchmark.py) from "the only base or offsets row" to the T-13 count: HNCO's six RHF offsets-family rows are one fit and flip together at 3.0 (best-of 5.34 -> 5.93, no win count change); every other row run stays within 1e-5 mA. No engine or row change.
- Why: the README's best-of table takes each leg's best over all configurations, so the base / offsets-only statement understated the reach (four more rows, one best-of figure).
- Alternatives considered: leave the docstring as is (true but incomplete); a README note (the README does not quote 8.14).
- Source: assumption, unconfirmed (within D-47's measurement-only scope)
- Revisit if: the rescale becomes smooth (N19 a) or the rows are rerun.

### D-49 · Next task: diagnose fluoroacetylene's tracked fits (2026-11-20)
- Decision: find why fluoroacetylene's 2nd and 3rd tracked hybrid fits (HF track / offsets+track / track_quadratic) start at reduced chi2 16.00 with dof = n_obs, and why tracking makes the row worse (13.59 -> 19.16 mA); diagnosis first, any fix proposed before it is made.
- Why: an exact 16.00 (= the x4 clamp squared) on three configurations points at a mechanism, possibly a defect, in a configuration the README's best-of table uses.
- Alternatives considered: F14 GLS study; B3LYP torsion rows for T-13's count.
- Source: auto-picked, no answer in 15 min (Q-16, recommended option)
- Revisit if: the user picks another option.

### D-50 · Fix the linear-molecule axial moment in the Watson harmonic term (2026-11-20)
- Decision: in compute_harmonic_alpha's Watson term, drop the axial inverse moment (index 0, A) whenever _rigid_mode_count says the geometry is linear -- the decision the normal modes and the Coriolis pair already use -- instead of only below an absolute I > 1e-8 amu A^2. Add a regression test (linear triatomic bent across I_A = 1e-8) that fails on the old code; rerun and archive fluoroacetylene's RHF track, offsets+track and track_quadratic rows.
- Why: T-14: on a geometry bent by ~1e-4 A the two linearity rules disagree; near the minimum the harmonic term steps 7-10 MHz, and at the tracking slope's off-stationary points it reached -7.5e6 MHz, leaving the tracked fits data-free (chi2 16.00) and the rows at theory (19.16 mA). The fix moves exactly those 3 rows (19.16 -> 13.64, 4.96 -> 4.36, 19.17 -> 13.64); every other row at both levels is bit-identical; best-of tables unchanged.
- Alternatives considered: also project rotations out of off-stationary Hessians in the slope (may move other tracked rows; not now); record only.
- Source: auto-picked, no answer in 15 min (Q-17, recommended option)
- Revisit if: the user picks another option; or the separate jump at the is_linear boundary itself (I_A ~5e-5, +25 MHz) matters for a molecule.

### D-51 · Next task: measure rotation projection in off-stationary alpha Hessians (2026-11-20)
- Decision: measure, on every benchmark row a projection can reach, what projecting translations and rotations out of the Hessian before the alpha normal-mode analysis would change (T-14 verifier: the tracking slope's +-step/2 points are off-stationary and their bends carry 1-18% rotation). Measurement only; any engine change proposed before it is made.
- Why: that contamination amplified the T-14 defect ~1e5x; whether it moves rows on its own is unmeasured.
- Alternatives considered: F14 GLS study; N2 sparse-data sigma re-scan.
- Source: auto-picked, no answer in 15 min (Q-18, recommended option)
- Revisit if: the user picks another option.

### D-52 · Project rigid motions out of the alpha Hessian in the engine (2026-11-20)
- Decision: in harmonic_alpha.py, project translations and rotations (mass-weighted, about the principal axes; the axial rotation left in when _rigid_mode_count says linear) out of the Hessian before both alpha normal-mode analyses (compute_harmonic_alpha and the cubic normal-mode derivative), with the prototype's exact arithmetic so its cached Hessians are reused. Add a test that alpha off a stationary point is rotation-free, rerun every row at both levels from the commit, and update the README and the benchmark docstring.
- Why: T-15: off a stationary point the raw Hessian's rotation leaks into the vibrational eigenvectors and, through the inertia-derivative term's a^{ab}/I_a piece, moved HNCO's offsets-table alpha_B by ~100 MHz (-125.8 vs -28.1 projected, -26.2 at the stationary point); the README's 8.14 mA RHF HNCO best-of came from it. Verified independently (own Hessians, projector, patch). At stationary points rows move <= 2e-3 mA.
- Alternatives considered: also take the offsets configs' alpha at the stationary geometry (removes the 23.2-28.65 mA recipe spread for HNCO offsets; changes what the configs measure; left as a backlog item); leave the engine and document.
- Source: auto-picked, no answer in 15 min (Q-19, recommended option)
- Revisit if: the user picks another option; or a molecule's row depends on the off-stationary recipe (mass-weighted vs valence vs stationary alpha) by more than its README figure's precision.

### D-53 · Next task: the failing test_prior_target check (backlog N20) (2026-11-21)
- Decision: find why test_a_target_prior_keeps_the_target_s_curvature's exact-zero refresh assertion fails, then bound it by a measured scale and show the bound still catches the defect it guards.
- Why: Q-20 option 1 (recommended): small, keeps the CI selection green; cause was unknown.
- Alternatives considered: sum-rule repair of the B3LYP Hessian (N19 b-ii); offsets configs take alpha at the stationary geometry (N19 b-i); done for now.
- Source: auto-picked, no answer in 15 min (Q-20)
- Revisit if: the user picks another option.

### D-54 · Next task: measure the acoustic-sum-rule repair of the B3LYP Hessian (backlog N19 b-ii) (2026-11-21)
- Decision: prototype the translational sum-rule repair on every B3LYP/6-31G(d) Hessian the benchmark consumes (after the disk cache, so the cache stays raw), measure per-molecule alpha and every B3LYP row, check against a converged-grid reference on a second molecule, and propose (or not) an engine change before making one.
- Why: Q-20/Q-21 option 1 (recommended): T-15's AF verifier found pyscf's analytic RKS Hessian lacks the grid-weight response; projection removes the rigid leak but CD3-AF A stays 0.8 MHz off the converged grid, and the repair brought it within 0.04 at no cost.
- Alternatives considered: more r_e^SE references (N3); offsets alpha at the stationary geometry (N19 b-i); done for now.
- Source: auto-picked, no answer in 15 min (Q-21)
- Revisit if: the user picks another option.

### D-55 · Impose the translational sum rule on every Hessian the engine consumes (2026-11-21)
- Decision: an engine helper that subtracts each atom's row sums from its diagonal block and symmetrises (exact for a symmetric input's symmetric part; a no-op on an invariant Hessian, idempotent to ~1e-18), applied where a Hessian enters: the optimizer's quantum state (_run_hessian) and the alpha module's consumers (the rigid-mode projector both mode analyses use, the rigid-direction derivatives, and the displaced Hessians of both cubic schemes). All levels. Test failing on the old code; full suite on 3.12; rerun every row at both levels from the commit; engine rows must reproduce the T-17 prototype's 18; README/docstrings updated.
- Why: T-17: every B3LYP/6-31G(d) Hessian from pyscf's analytic RKS (no grid-weight response) breaks the rule (row sums 2e-4 to 1.9e-2 Ha/bohr^2); the repaired level-3 Hessian matches level 7-9 on alpha (HCOF 0.032, O3 0.012, CH2ClF 0.006 MHz vs 0.40 / 0.49 / 0.82 raw) and CH2ClF's vibrations (within 0.1 cm-1 vs 5); RHF Hessians satisfy it to ~1e-9. Benchmark: 17 of 18 measured rows move <= 0.04 mA; CH2ClF improves vs its r_e^SE (base 9.24 -> 8.95, offsets 5.99 -> 5.91).
- Alternatives considered: alpha only (leaves the prior's curvature wrong -- CH2ClF's 5 cm-1); apply in the pyscf backend (the source; would mix repaired and raw entries in the shared disk cache unless every Hessian were recomputed); don't adopt, document.
- Source: auto-picked, no answer in 15 min (Q-22, recommended option)
- Revisit if: the user picks another option; a backend's Hessian breaks the rotational rule in a way the projection does not cover; or the B3LYP theory geometries are re-converged (N3 found HCOF 2.6e-4 A from its tight minimum, as large an alpha effect).

### D-56 · Next task: measure how far the theory geometries sit from tight minima (backlog N21) (2026-11-21)
- Decision: measure, per molecule at both levels (the nine scored plus the calibration-only ones), the distance from the cached theory geometry to a tight minimum of the same surface -- a Newton step on the cached (sum-rule-repaired) Hessian, checked against real tight optimisations -- and the bond-length changes; then, only if they matter, what they do to the rows. Propose before any engine or benchmark change.
- Why: Q-23 option 1 (recommended): the T-17 grid verifier found formyl fluoride's B3LYP theory geometry 2.6e-4 A from its tight minimum (optimiser gtol 2e-5 Ha/A on the largest component), moving A_e 17 MHz and alpha_A 0.42 MHz -- as large as the Hessian fix.
- Alternatives considered: more r_e^SE references (N3); offsets alpha at the stationary geometry (N19 b-i); done for now.
- Source: auto-picked, no answer in 15 min (Q-23)
- Revisit if: the user picks another option.

### D-57 · DFT geometry optimisation on the energy's own gradient, with a convergence check (2026-11-21)
- Decision: pyscf_backend.optimise() takes, for a DFT mean field, the gradient with the grid-weight response (the exact derivative of the energy it minimises), and after L-BFGS-B checks the final gradient against gtol, warning and recording (last_optimise) when it stopped above it. HF and correlated methods unchanged. Regenerate the B3LYP theory geometries, rerun every B3LYP row, update README/docstrings.
- Why: T-18: without the grid response the gradient is not the energy's derivative, so L-BFGS-B stops on ftol where the energy stops falling -- B3LYP theory geometries left |g| up to 3.2e-4 Ha/A (gtol 2e-5), bonds up to 0.24 mA off the energy's minimum (FE, HCOF, VF), silently; with the grid response all 12 converge to <= 1.4e-5 and HCOF / ozone land nearer the level-9 minimum (bond RMS 0.093 -> 0.047, 0.010 -> 0.006 mA). RHF geometries are converged to 0.008 mA and do not change.
- Alternatives considered: convergence check only; document as a limitation; grid response in run_gradient too (changes the gradient every fit sees -- not measured, left out).
- Source: auto-picked, no answer in 15 min (Q-24, recommended option)
- Revisit if: the user picks another option; or a fit's own gradient term turns out to need the grid response too.


### D-58 · Next task: measure the grid-response term the fit's gradient leaves out (backlog N22) (2026-11-21)
- Decision: measure, at every B3LYP/6-31G(d) theory geometry, the difference between pyscf's default RKS gradient (what run_gradient returns) and the grid-response gradient, and what it does to the fit: in the benchmark's target-prior mode (expected nothing: the step replaces the gradient with H(x - x_target)) and in the CLI's no-target mode (the prior's centre is x_ref - H^-1 g, so it moves by H^-1 dg). Gradient cost with and without. Propose before any change.
- Why: Q-25 option 1 (recommended): D-57 fixed optimise() but left run_gradient on the default gradient, unmeasured.
- Alternatives considered: more r_e^SE references (N3); offsets alpha at the stationary geometry (N19 b-i); done for now.
- Source: auto-picked, no answer in 15 min (Q-25)
- Revisit if: the user picks another option.

### D-59 · The fit's DFT gradient: flagged, not changed (2026-11-21)
- Decision: leave run_gradient and run_hessian's gradient on pyscf's default RKS gradient (no grid-weight response); document the effect in the README's results provenance and the backend's docstrings; backlog N22 stays open as a flagged trade-off for the user.
- Why: T-19: the benchmark cannot see it (target prior; rows bit-identical with the gradient replaced or corrupted); CLI fits (no target) centre the quantum prior up to 0.27 mA (bond) off the energy's minimum, and with the grid response on both gradients the two measured CLI answers move <= 0.14 mA (bond RMS HCOF 2.53 -> 2.49, FE 6.68 -> 6.61) for 7-14% more wall time. The user's stated preference: accuracy-efficiency trade-offs are flagged, not implemented.
- Alternatives considered: implement (needs the user's say-so); measure a cost-neutral variant first (one grid-response gradient per Hessian, its correction reused between); done for now.
- Source: auto-picked, no answer in 15 min (Q-26; the only option consistent with the stated preference)
- Revisit if: the user chooses to implement it, or a cost-neutral variant is measured.
- Correction (2026-11-21, T-19 fact-check): the 7-14% cost in Q-26 and above was wrong -- the verifier's wrapper computed a second gradient on every patched call. Timed with exactly the change (two alternating CLI runs each): formyl fluoride 55.5 / 55.4 -> 53.5 / 53.1 s (no slower), fluoroethane 122.8 / 128.2 -> 126.4 / 135.8 s (~4%, the repeats' own spread). Still a cost, so still the user's call under the stated preference; the decision stands until they choose.

### D-60 · Inertial-defect check: against the config's own structure (2026-11-21)
- Decision: the validator's inertial-defect warning compares each isotopologue's observed defect with the one its config geometry gives (geometry.method coords, the isotopologue's masses), warning beyond 1.0 amu*A^2 + 10% of the structure's own; without coordinates it warns only on a defect above +1.0 (no structure gives one). A separate warning when the labelled constants are not ordered A >= B >= C (what a label swap produces, and what the old two-sided rule caught as a large negative defect).
- Why: Q-27 option 1. The old rule warned on |defect| > 1, but I_c - I_a - I_b = -2 P_c is large and negative for every non-planar molecule (fluoroethane -6.3 to -7.7: false positive, T-19 N3). Measured T-20: observed vs reference structure, 37 benchmark isotopologues, gap -0.070 to +0.113; vs the benchmark's start geometries <= 0.26 (7.5%); fluorobenzene's bad constants +5.6.
- Alternatives considered: one-sided only (misses label swaps and negative errors on planar molecules); build every geometry form (smiles, pubchem, bonds) at validation (needs RDKit / the network).
- Source: auto-picked, no answer in 15 min (Q-27)
- Revisit if: a real config's start geometry misses its own defect by more than the tolerance.

### D-61 · Next task: re-measure the README's "which objective" rows (backlog N2) (2026-11-21)
- Decision: re-measure, at HF/6-31G with each script's own data, start and settings, the fluorobenzene (1 and 8 isotopologues) and monofluoro (parent only, all species) rows of the README's "Which objective to use" table, with the objective passed explicitly: split, joint at the calibrated default, joint at 0.003-0.050 A. Then correct the table and its prose; fix the scripts whose "split" leg passes nothing.
- Why: Q-28 option 1. The rows' 0.005 A predates the Cartesian-alpha units fix; and MolecularOptimizer's default objective became joint at 7216d87 (2026-08-14) while fluorobenzene_full_data.py, monofluoro_benchmark.py and monofluoro_tune.py still get their "split" legs by passing no objective_mode -- so a "split" number from them since then may be joint at the calibrated width.
- Alternatives considered: auto_systematic_sigma magnitudes (F13); implement N22 (user's call); done for now.
- Source: auto-picked, no answer in 15 min (Q-28)
- Revisit if: the user picks another option.

### D-62 · Next task: measure how far plain joint fits stop from their optimum (backlog N23) (2026-11-21)
- Decision: measure, at HF/6-31G on the nine benchmark molecules from theory's minimum and from a 0.03 A displaced start, the distance from a plain joint fit (no target prior, as the CLI runs) to the answer restarts settle on (three passes in a row under 1e-5 A RMS), and what each remedy recovers and costs: the engine's refresh_hessian_at_answer (one Hessian; quantize uncertainty already calls it) and orca_update_thresh 0.001. Propose before any change; a remedy that adds cost to every run is the user's call (stated preference).
- Why: Q-29 option 1. T-21: fluorobenzene from a displaced start stops 3.7-4.1 mA (largest atom) and 1.3-1.4 mA of bond RMS from its settled answer; the plain fit skips the at-answer refresh by design ("it costs one Hessian").
- Alternatives considered: auto_systematic_sigma magnitudes (F13); implement N22 (user's call); done for now.
- Source: auto-picked, no answer in 15 min (Q-29)
- Revisit if: the user picks another option.

### D-63 · The plain fit's stop short of its optimum: flagged, not changed (2026-11-21)
- Decision: leave `run` (the plain fit) without the at-answer Hessian refresh; document the gap in the README's Running section and refresh_hessian_at_answer's docstring; backlog N23 stays open as a flagged trade-off.
- Why: T-22: through `cli run` (HF/6-31G, HCOF and CH2ClF, warm and displaced starts) the plain answer is 0.03-0.6 mA rms (1.4 max atom) from its own optimum and its bond RMS moves up to 0.5 mA either way; the refresh reaches the optimum and removes a 0.5 mA thread-count variation for one Hessian (+15-21% wall time). The user's stated preference: accuracy-efficiency trade-offs are flagged, not implemented.
- Alternatives considered: refresh at the end of every plain run; an opt-in config key (default off); done for now.
- Source: auto-picked, no answer in 15 min (Q-30; the only option consistent with the stated preference)
- Revisit if: the user chooses either alternative.

### D-64 · Next task: the multistart's stale-energy ranking (backlog N24) (2026-11-21)
- Decision: measure, per start of a `cli run`, the energy the selection uses (the last quantum point's), the fit's own quadratic-model energy at the answer (free) and the true SCF energy at the answer, and which start each ranks first; then propose a fix before making it.
- Why: Q-31 option 1 (recommended). T-22's CLI verifier: reported per-start energies off by up to 6.3e-4 Ha against real differences of ~1e-5; formyl fluoride picked the wrong start and its restart chain oscillates.
- Alternatives considered: auto_systematic_sigma magnitudes (F13); implement N22 / N23 (user's call); done for now.
- Source: auto-picked, no answer in 15 min (Q-31)
- Revisit if: the user picks another option.

### D-65 · The multistart's stale per-start energy: flagged, not changed (2026-11-21)
- Decision: keep select_best_result's ranking; document in its docstring (and at _run_single_start) that each start's `energy` is the last quantum point's, with T-23's measurement; reword backlog N24 to the underlying issue.
- Why: T-23, 10 CLI runs (5 molecules x warm/cold, 30 starts): the stale energy is 8.6e-5 Ha (median), up to 1.6e-3, from the energy at the answer, yet it put the lowest-at-the-answer start first in 7/10; the model energy at the answer (free) 8/10; every miss of either was between starts within 1.4e-6 Ha, 0.15-12 mA apart along flat directions, where no energy ranking is meaningful. Re-ranking would change reported answers by up to 12 mA with no measured accuracy gain.
- Alternatives considered: rank and report by the model energy at the answer (free); by the true energy (one SCF per start); done for now.
- Source: auto-picked, no answer in 15 min (Q-32)
- Revisit if: a case shows starts whose energies at their answers differ by more than the model's error but are misranked; or the report's per-start energy is used for anything beyond ranking.

### D-66 · Next task: auto_systematic_sigma's magnitudes (backlog F13) (2026-11-21)
- Decision: measure the inertial-defect estimate (defect_model_sigma) behind model_error.auto against per-species B0-Be gaps -- the benchmark's own B3LYP VPT2 correction (0.5 |alpha_sum|) on nine molecules, and the r_e^SE structures in the repo as an independent yardstick -- and propose before any change.
- Why: Q-33 option 1 (recommended): F13 recorded it 3-10x off on water A and B.
- Alternatives considered: more r_e^SE references (N3); implement N22 / N23 (user's call); done for now.
- Source: auto-picked, no answer in 15 min (Q-33)
- Revisit if: the user picks another option.

### D-67 · model_error.auto (and the no-sigma fallback) use the flat relative default (2026-11-21)
- Decision: where a species gives no systematic sigma and auto_systematic_sigma is on, or gives no sigma at all, the engine fills in DEFAULT_SIGMA_REL_ABC (1% of A, 0.5% of B and C) instead of the inertial-defect estimate (defect_model_sigma, kept as a diagnostic). README, docstrings and the runner's comment corrected; a test that fails on the old code; CI selection; fact-check.
- Why: T-24: against the benchmark's B3LYP VPT2 gaps on seven molecules the defect estimate ran 0.17-8.9x (A), 0.05-6.1x (B), 0.04-1.06x (C) -- too small on B and C everywhere but water (fluoroethane 20-25x) -- and against the r_e^SE gaps 0.14-0.40 on B and C; the flat default 1.0-2.5 / 0.58-1.17 / 0.23-0.83 (VPT2) and 0.68-1.05 / 0.58-0.80 on B / C (r_e^SE). No compute cost; only opt-in users' widths (and API species with no sigma) change.
- Alternatives considered: flag only; remove model_error.auto; done for now.
- Source: auto-picked, no answer in 15 min (Q-34)
- Revisit if: a per-species estimate that beats the flat fractions on both yardsticks is found (the VPT2 correction itself already supplies its sigma when one is computed).

### D-68 · Next task: the other two defect-based corrections (2026-11-21)
- Decision: check defect_bias_floor and defect_bias_scale for the per-component split D-67 removed from the model-error estimate; propose before any behaviour change.
- Why: Q-35 option 1 (recommended).
- Outcome (T-25, by reading the code): defect_bias_scale multiplies all three components' sigmas by one ratio |defect after| / |defect before| -- no per-component split, nothing to change. defect_bias_floor floors each sigma at defect_model_sigma of the corrected constants (the split), but it is off by default and unreachable from MolecularOptimizer and the CLI (only resolve_corrections(defect_bias_floor=True), in tests), and its own docstring records it as superseded by the scale. Docstring note added; no measurement needed.
- Alternatives considered: make model_error.auto the default (user's call); implement N22 / N23 (user's call); done for now.
- Source: auto-picked, no answer in 15 min (Q-35)
- Revisit if: defect_bias_floor is wired into the engine.

### D-69 · Next task: say when multistart answers are near-tied but far apart (backlog N24) (2026-11-21)
- Decision: measure the tie and distance thresholds on T-23's ten CLI runs, propose the exact reporting behaviour, then build it; no extra quantum calls, ranking unchanged (D-65).
- Why: Q-36 option 1 (recommended). T-23: near-tied starts (within 1.4e-6 Ha true) sit up to 12 mA apart, so the reported answer is effectively arbitrary, and nothing says so.
- Alternatives considered: F3 hygiene; model_error.auto as default, N22, N23 (user's call); done for now.
- Source: auto-picked, no answer in 15 min (Q-36)
- Revisit if: the user picks another option.

### D-70 · Report multistart near-ties (2026-11-21)
- Decision: build the near-tie report as proposed in T-26 (model energy at each start's answer; near_tied_starts with 5e-5 Ha and 2 mA; stored on the best result, printed, listed in the md / html reports under "Best start"; "Final energy" relabelled as the last quantum point's; ranking unchanged).
- Why: T-26 thresholds from the T-23 data (model energy error 90th percentile 2.5e-5 Ha; flags 5 of 10 runs, the methyl-torsion molecules and VF cold, 8.7-20.6 mA apart).
- Alternatives considered: bundle only; configurable thresholds; done for now.
- Source: auto-picked, no answer in 15 min (Q-37)
- Revisit if: the flag fires on cases where the starts are not genuinely undecided.

### D-71 · Next task: F3 hygiene (BOMs, scripts package, rigid-mode test) (2026-11-21)
- Decision: confirm and fix three F3 items: the UTF-8 BOMs, `scripts/` as a namespace package, and the `all(s < 1.0)` proxy in test_rigid_modes_do_not_leak_into_the_widths.
- Why: Q-38 option 1 (recommended).
- Outcome (T-27): BOMs were in 57 tracked .py files (F3 named two, quantize.py and spectral.py, which had them too); every one failed ast.parse on utf-8-decoded text; stripped (first line only), all 243 parse. `scripts/` as a namespace package lost to a regular `scripts` package placed later on sys.path (measured: ImportError on scripts.uncertainty_plots, which cli.py imports); `scripts/__init__.py` added, after which ours wins. The test now asserts the covariance's rigid share < 1e-10 (measured 4e-17) and bond widths < 1e-3 A (measured 2.8e-4); with the rigid directions let into physical_basis it fails (share 1.0, widths 1.2e-2 A), where the old assertion passed.
- Alternatives considered: N19 chi-square rescale switch; model_error.auto as default, N22, N23 (user's call); done for now.
- Source: auto-picked, no answer in 15 min (Q-38)
- Revisit if: -

### D-72 · Next task: test-suite hygiene (writes into the repo, force-field tests' partial file) (2026-11-22)
- Decision: find every test that writes into the repository instead of tmp_path and make it write to tmp_path; make the force-field multireference tests skip, not KeyError, on a partial gitignored output/theory_geometries_hf_6-31g.json.
- Why: Q-39 option 1 (recommended). F3: the suite left 21 run directories in output/runs on the T-27 run alone.
- Alternatives considered: N19 chi-square rescale switch; model_error.auto as default, N22, N23 (user's call); done for now.
- Source: auto-picked, no answer in 15 min (Q-39)
- Revisit if: -

### D-73 · Next task: the chi-square rescale switch (N19 a) -- measure, then propose (2026-11-22)
- Decision: measure how close the current benchmark fits sit to the rescale's switch points (reduced chi2 4 and 1/4), then propose; no behaviour change without a proposal; any accuracy/efficiency trade-off flagged, not implemented.
- Why: Q-40 option 1 (recommended).
- Alternatives considered: embed the multireference test geometries; model_error.auto as default, N22, N23 (user's call); done for now.
- Source: auto-picked, no answer in 15 min (Q-40)
- Revisit if: -

### D-74 · Keep the chi-square rescale switch; record its checks (2026-11-22)
- Decision: keep the switch (threshold 4, two-sided) unchanged; MolecularOptimizer.run keeps one record per rescale check (chi2_rescale_checks: reduced chi-square, dof, outcome, factor) and its dof gate reuses the dof reduced_chi_square() computed; accuracy_upgrades_benchmark's hybrid_fit hands the checks back through an `info` dict and each row stores them; the benchmark docstring's rescale caveat gives the 03e7d03 margins. No fit's answer changes.
- Why: T-29 measured that no current result is decided by the switch: the nearest fits sit at 1.853 (HF HNCO offsets, 2.2x below 4) and 0.2175 / 0.2267 (B3LYP water base / ozone offsets, 13% / 10% below 1/4), and turning the rescale off moves those rows by 9e-10 and 1e-5 mA. A continuous rescale would move every firing fit (340 of 362 at T-15) for no measured benefit. Recording the checks lets the next rerun show a near-switch fit without instrumentation.
- Alternatives considered: docstring only; a continuous rescale (flagged, not recommended); done for now.
- Source: auto-picked, no answer in 15 min (Q-41)
- Revisit if: a rerun's chi2_rescale_checks show a check in [3, 5] or [1/5, 1/3] whose rescale-off row differs by more than 1e-3 mA.

### D-75 · Next task: the rescale's high-side widening while the sigma floor binds (N25) -- measure, then propose (2026-11-22)
- Decision: measure on real fits how often a widening refit runs on unchanged weights and how far the widening falls short, then propose (skip a refit whose weights did not change; or widen the weighting sigma); no behaviour change without a proposal; flag any accuracy/efficiency trade-off.
- Why: Q-42 option 1 (recommended).
- Alternatives considered: embed the multireference test geometries; model_error.auto as default, N22, N23 (user's call); done for now.
- Source: auto-picked, no answer in 15 min (Q-42)
- Revisit if: -

### D-76 · Chi-square rescale: a pass that moves no weight does not refit (N25 (a)) (2026-11-22) · superseded by D-77
- Decision: after scale_sigma, if the stated sigmas changed but every weighting sigma (_effective_sigma) is bit-identical, record "weights unchanged" and go to the next pass without refitting; keep scaling exactly as before. Only when nothing else in the fit reads the stated sigma (autoconfig off, no mass-dependent correction).
- Why: Q-43 option 1 (recommended). Departure from its wording, which said "stops": a stop would change answers (5 MHz water: the third x4 pass crosses C's 278 MHz floor; stopping at pass 1 would lose it). Skip-and-continue keeps "no result change", which the option promised: 17 analytic-water cases bit-identical (coords, sigmas, chi2 sequence, widths), 0-3 fewer fits each. Real fits: 0 of 86 script fits and 0 of 362 benchmark fits widen, so their answers cannot change.
- Alternatives considered: widen the weighting sigma itself (accuracy change, user's call, still open as N25 (b)); docstring only; done for now.
- Source: auto-picked, no answer in 15 min (Q-43); the skip-and-continue refinement is an assumption, unconfirmed
- Revisit if: a fit path reads the stated sigma outside _effective_sigma (the guard must then include it).

### D-77 · Revert D-76: keep the refit on unchanged weights (2026-11-22)
- Decision: restore run() as before 5eb7554 and add a comment saying why a pass whose weights did not change still refits; the skip's tests go with it.
- Why: the independent verifier refuted "bit-identical": a refit on identical weights is a no-op only after a fit that stopped at a stationary point. After a plateau or max_iter stop it keeps optimising (reproduced: water 5 MHz from a shifted start, chi2/nu 201.96614 -> 201.96841 -> 201.96874; final chi2 132.27626 refitting vs 132.27695 skipping; pyscf HF water 1.3e-10 A). So the skip trades convergence for time, which the stated preference says to flag, not implement. It never fires in a measured real fit (0 of 86 script fits, 0 of 35 benchmark fits), so it saved nothing real.
- Alternatives considered: skip only after a "converged" stop (the plateau stop is what these fits end on, so it would rarely fire, and bit-identity is still not guaranteed; diagnostics such as history length and iteration counts differ even then); keep D-76 as is (implements a trade-off).
- Source: assumption, unconfirmed (my call on the verifier's finding: reverting restores the behaviour the user had)
- Revisit if: the rescale becomes reachable from the CLI or config, or a real fit is seen refitting on unchanged weights.

### D-78 · Next task: does the reported width cover the multistart ambiguity? (N24) (2026-11-22)
- Decision: measure, for T-23's ten CLI runs rerun at HEAD, whether the selected start's reported 1-sigma bond and angle widths cover the distance to its near-tied rivals (|dq| / sigma per coordinate); then propose; no behaviour change without a proposal.
- Why: Q-44 option 1 (recommended). D-65 measured the ranking and D-70 reports near-ties; what is left of N24 is whether the arbitrariness is already inside the stated uncertainty or outside it.
- Alternatives considered: N8 (b) distortion sigma floor; N3 references; done for now.
- Source: auto-picked, no answer in 15 min (Q-44)
- Revisit if: -

### D-79 · The near-tie note says how far the reported coordinates move, in sigma (2026-11-22)
- Decision: near_tied_starts adds max_shift_in_sigma and max_shift_coordinate per listed start (largest |q_other - q_selected| / sigma_selected over the bonds, angles and dihedrals both posteriors carry; dihedrals wrapped; None without both posteriors); near_tie_note prints it. No extra compute; ranking and answers unchanged.
- Why: Q-45 option 1 (recommended). T-31: the 7 near-tied pairs in 10 CLI runs are 8.6-20.6 mA apart at the largest atom but move the reported bonds and angles by at most 0.29 sigma (median over the pairs 0.13; the commit message's 0.11 is the bonds-only median), so the note's atom distance alone overstated what a near-tie changes.
- Alternatives considered: docstring only; records only; done for now.
- Source: auto-picked, no answer in 15 min (Q-45)
- Revisit if: a run lists a start whose shift is above 1 sigma (then the reported width does not cover the ambiguity).

### D-80 · Next task: does the 0.01 MHz distortion sigma floor swamp heavy molecules' rows? (N8 (b)) (2026-11-22)
- Decision: measure, for the reference molecules, the harmonic DJ / DJK / DK the engine predicts (HF/6-31G Hessian) against the row sigma's 0.01 MHz floor: how many rows the floor sets, and by what factor it cuts their weight; then propose; no behaviour change without a proposal.
- Why: Q-46 option 1 (recommended). No reference molecule carries observed distortion constants (the benchmark docstring says the gate is data), so the predicted constants stand in for the observed ones the floor is applied to.
- Alternatives considered: N3 references; N7 split widths; done for now.
- Source: auto-picked, no answer in 15 min (Q-46)
- Revisit if: -

### D-81 · Distortion rows: flag the measured hazard, keep the floor (2026-11-22)
- Decision: no behaviour change. README's centrifugal-distortion note and a comment at the floor in SpectralEngine._cd_rows give T-32's numbers; backlog N8 updated.
- Why: Q-47 option 1 (recommended). T-32, as shipped: with synthetic distortion data the rows slow the fit (no rows 11-12 iterations; with rows 4 of 9 stopped after 59-133, 5 still moving at 200) and constants 30% off the prediction move bonds up to 42 mA, up to 2.9 of a bond's width; removing the 0.01 MHz floor moved bonds 0.2-31 mA, closer to the reference in only 3 of 9. (Commit 37cc82a's message and first README text said 59-199, 11-13, 6 of 18 and 45 mA, mixing in the no-floor runs and miscounting; corrected after the fact-check. T-33 then showed the 42 mA came from autoconfig-off fits that had not converged; with autoconfig on, the default, the moves were 1.1-3.5 mA; README corrected again.) No observed distortion constants for these molecules exist in the repo to settle the weighting.
- Alternatives considered: measure where the pull goes and what sigma keeps a model-sized mismatch inside the widths (accuracy-side, the user's call); remove the floor (accuracy change, not auto-picked); done for now.
- Source: auto-picked, no answer in 15 min (Q-47)
- Revisit if: published distortion constants for a reference molecule are added, or a user fits distortion rows on a heavy molecule.

### D-82 · Next task: why distortion-row fits do not converge (2026-11-22)
- Decision: diagnose the non-convergence T-32 found (as shipped 5 of 9 fits with distortion rows still taking full steps at 200 iterations, 11-12 without rows): the rows' Jacobian against finite differences, the prediction moving at Hessian refreshes, oscillation; then propose.
- Why: Q-48 option 1 (recommended). A fit that does not reach its own optimum makes every number it reports a stopping point, not an answer.
- Alternatives considered: N3 references; N7 split widths; done for now.
- Source: auto-picked, no answer in 15 min (Q-48)
- Revisit if: -

### D-83 · Next task: measure a growing trust radius on the scripts' fits (N26) (2026-11-22)
- Decision: prototype the standard trust-region growth (ratio > 0.75 with the step at the radius doubles it, capped at the start radius) by monkeypatch, and measure on fits without distortion rows, autoconfig off as the scripts run them: iterations, stop reason and answers against the shipped code; measurement only, no behaviour change.
- Why: Q-49 option 1 (recommended). T-33: adapt_lambda never enlarges the radius, so autoconfig-off fits crawl after early rejections; any fix changes every such fit's path, so its effect on the fits that already converge must be known first.
- Alternatives considered: settle the optimum in the distortion cases; N3 references; done for now.
- Source: auto-picked, no answer in 15 min (Q-49)
- Revisit if: -

### D-84 · Close N26: keep the trust rule as it is (2026-11-22)
- Decision: no code change. The standard trust-radius growth (rho > 0.75 with the step at the radius doubles it, capped at the pass's start radius) is not adopted; backlog N26 closed as measured, the narrowed cause of the distortion crawl recorded in N8 (f).
- Why: Q-50 option 1 (recommended). T-34: on nine molecules' plain joint fits (HF/6-31G, autoconfig off, no distortion rows) the rule never enlarged the radius -- its condition fired only before any rejection, at the cap -- and answers were bit-identical at 40 and 200 iterations; on the distortion crawl (formyl fluoride x1.0) it grew the radius 25 times and 24 of the doubled steps were refused (30 rejections against 6), still 200 iterations. The crawl is the step model failing beyond ~2 mA, not the radius bookkeeping.
- Alternatives considered: diagnose why steps beyond ~2 mA fail on distortion-row fits; N3 references; done for now.
- Source: auto-picked, no answer in 15 min (Q-50)
- Revisit if: a fit without distortion rows is seen crawling at a shrunken radius (fluoroacetylene crawls at ~95% of its radius and is the nearest case).

### D-85 · Next task: why distortion-row steps beyond ~2 mA fail the ratio test (2026-11-22)
- Decision: along the steps a distortion-row fit takes and refuses, split the objective change into the rotational rows, the distortion rows and the quantum term, and compare each with its Gauss-Newton prediction; check the rows' curvature; then propose.
- Why: Q-51 option 1 (recommended). T-33/T-34 ruled out the Hessian refresh, the rows' Jacobian and the trust bookkeeping; the step model failing beyond ~2 mA is what is left.
- Alternatives considered: N7 split widths; N3 references; done for now.
- Source: auto-picked, no answer in 15 min (Q-51)
- Revisit if: -

### D-86 · Distortion-row crawl: document the cause, no solver change (2026-11-22) · superseded by D-87
- Decision: README's distortion note and backlog N8 (f) say the crawl comes from the step model: the distortion rows pull along directions the rotational constants do not see to first order, where the rotational residuals' curvature (dropped by Gauss-Newton) outweighs their linear change beyond ~2 mA. No solver change.
- Why: Q-52 option 1 (recommended). T-35: on 56 refused steps of four formyl fluoride fits the rotational rows' change departs from the linear model by a median 1.9x the linear change (0.21 on accepted steps under 2 mA); the distortion rows' terms are 15-187x smaller; the steps lie entirely in the null space after iteration 20 (FF x1.0). Distortion fitting is opt-in and no reference molecule has observed constants to test a fix against.
- Alternatives considered: prototype a second-order term for the rotational rows along the step and measure it (a solver change; the user's call); N7 split widths; done for now.
- Source: auto-picked, no answer in 15 min (Q-52)
- Revisit if: observed distortion constants for a reference molecule are added, or a fit without distortion rows shows the same refused-step pattern.

### D-87 · The crawl is an orientation defect in the distortion prediction; README corrected (2026-11-22)
- Decision: correct the README note (D-86's mechanism was wrong) and propose a fix (Q-53) rather than leave it documented.
- Why: the fact-check of 5722ff2 (fc35) found the steps are rigid rotations (share >= 0.9998), the structural part of each step is linear (<= 0.002), and the distortion prediction changes under rotation; confirmed with my own check (t35/rotcheck.py): a 0.01 rad rotation moves the predicted constants up to 5.0e-3 relative, A/B/C 1e-15. The "curvature" D-86 cited is the stretch of a straight-line step along a rotation.
- Alternatives considered: keep D-86's documentation (wrong); fix without asking (a behaviour change to agreed doc-only work: asked as Q-53).
- Source: assumption, unconfirmed (my reading of the verifier's finding, which I reproduced)
- Revisit if: the fix shows the rotation is not the whole cause.

### D-88 · Fix: predict the distortion constants in the Hessian's own frame (2026-11-22)
- Decision: SpectralEngine.set_hessian_for_cd takes the geometry the Hessian was computed at (MolecularOptimizer passes _hessian_ref_coords at all three hand-offs); _predict_cd_for_iso aligns the trial geometry onto it (unweighted Kabsch, as _run_hessian rotates a held Hessian) before predicting. Without a stated geometry (direct engine use, tests) nothing changes.
- Why: Q-53 option 1 (recommended). T-35: the prediction kept the Cartesian Hessian in its own frame, so a 0.01 rad rotation moved DJ/DJK/DK up to 0.5% while A/B/C did not move; the fit turned the molecule to fit the rows. A correctness fix, opt-in distortion fits only.
- Alternatives considered: project rigid modes out of the rows' Jacobian (leaves the prediction orientation-dependent); leave it documented; done for now.
- Source: auto-picked, no answer in 15 min (Q-53)
- Revisit if: the distortion fits still crawl after the fix.

### D-89 · Next task: linear molecules' distortion prediction (N27) (2026-11-22)
- Decision: measure what compute_cd_constants and the distortion rows do for a linear molecule (at linear and slightly bent geometries), and what the rows should be (a linear molecule has one quartic constant, D); then propose; no behaviour change without a proposal.
- Why: Q-54 option 1 (recommended). T-36's verifier: fluoroacetylene's predicted DJ ~1e8 MHz at bent geometries, and rotation-dependent by up to 35x with an exactly linear reference.
- Alternatives considered: N7 split widths; N3 references; done for now.
- Source: auto-picked, no answer in 15 min (Q-54)
- Revisit if: -

### D-90 · Linear molecules: D only, from the perpendicular axes (2026-11-22)
- Decision: compute_cd_constants treats a geometry with I_min/I_max < CD_LINEAR_REL_TOL (1e-3) as linear and returns D = -(tau'_bb + tau'_cc)/8 as DJ (DJK, DK, the asymmetry parameters NaN, reduction "linear"); the distortion rows fit DJ alone for such a prediction and drop a stated DJK / DK with a note. README says to give D as DJ.
- Why: Q-55 option 1 (recommended; deadline passed with no answer). T-37: the asymmetric-top path gave DJ 1e11-1e13 MHz for fluoroacetylene and HCN bent 0.5-3 deg; the perpendicular-axis D is smooth (<= 0.35% at 5 deg); the ratio separates the reference set by 13x (HNCO 1.3e-2 against <= 5.6e-4 for the linear two bent 5 deg). Non-linear predictions bit-identical on one Hessian (nine molecules).
- Alternatives considered: refuse distortion rows for linear molecules; document only; done for now.
- Source: auto-picked, no answer in 15 min (Q-55)
- Revisit if: a quasi-linear molecule (I_min/I_max near 1e-3) is fitted.

### D-91 · Pin the linear branch once per isotopologue (fixes a regression from D-90) (2026-11-22) · rule revised by D-92
- Decision: the distortion rows decide linear-or-not once per isotopologue, with the reduction (SpectralEngine._cd_reduction): "linear" when no isotopologue observes A and the first geometry's I_min/I_max < CD_LINEAR_PIN_REL_TOL (2e-2), or when cd_reduction: linear is stated. compute_cd_constants switches by geometry only for reduction "auto" (CD_LINEAR_REL_TOL 1e-3); an explicit A, S or linear is honoured. The report says when stated DJK / DK are dropped; the harmonic printout gives D for a linear molecule.
- Why: the verifier of 5731f0c found D-90's per-geometry switch broke fits: just past 1e-3 the prediction took the asymmetric-top path while finite-difference steps fell in the linear branch (DJK, DK NaN), the Jacobian filled with NaN and the SVD failed -- a CLI HCN + DCN fit aborted where the parent commit completed (multistart jitter puts 12-15% of starts past 1e-3). Fixed: its reproducer gives one finite DJ row and the SVD succeeds at the threshold; the same CLI config completes with both isotopologues pinned linear and the dropped keys reported.
- Alternatives considered: refuse distortion rows for linear molecules; a single per-geometry threshold (crashes at the boundary); an explicit flag only (misses the common case).
- Source: assumption, unconfirmed (a fix of my own regression, within D-90's scope)
- Revisit if: a symmetric top with only B observed and I_min/I_max under 2e-2 is fitted (it would be pinned linear; state cd_reduction A or S).

### D-92 · The linear pin: B alone observed and atoms near the axis; diatomics; the table pinned too (2026-11-22)
- Decision: pin "linear" only when every isotopologue observes B alone and every atom of the first geometry lies within CD_LINEAR_PIN_OFF_AXIS_ANG (0.4 A) of the molecule's axis (replacing D-91's no-A / inertia-ratio rule); the linear branch also serves a pinned diatomic; the harmonic report table takes the pinned reduction.
- Why: the verifier of 86c99da refuted D-91: HNCO (B and C observed, no A) and short B-only symmetric tops (CH3C#CCN, ratio ~1.3e-2) were pinned linear, and a diatomic pinned linear raised "Unknown reduction 'LINEAR'" (the branch needed 3 atoms; at 5731f0c CO built its rows). Measured: 20000 jittered HCN / fluoroacetylene starts reach 0.21 A off-axis; every bent reference molecule has an atom >= 0.44 A off (ozone), methyl hydrogens ~1.0 A. The diatomic's D matches 4B^3/omega^2 to 1e-3 on an analytic Hessian.
- Alternatives considered: a tighter inertia ratio (2x margins at best, fails on HNCO's observed C); an explicit flag only.
- Source: assumption, unconfirmed (fix of my own regression within D-90's scope)
- Revisit if: a linear molecule's isotopologue lists C, or a molecule with every atom within 0.4 A of its axis is not linear.

### D-93 · Next task: diagnose fluoroacetylene's crawl (N28) (2026-11-22)
- Decision: T-38 diagnoses why fluoroacetylene's plain joint fit crawls (57-58 iterations, last steps 3-6e-3 A), starting from the hypothesis that the solver treats a linear molecule's 5 rigid modes as 6; then propose a fix.
- Why: Q-56's recommended option; diagnosis is reversible and the crawl means the reported geometry is not an optimum.
- Alternatives considered: N7 split widths; N3 reference search; done for now.
- Source: auto-picked, no answer in 15 min (Q-56)
- Revisit if: the user picks another option.

### D-94 · Cartesian steps use the Jacobian with its rigid part removed (2026-11-22)
- Decision: in Cartesian mode the stacked Jacobian the step, its rank and the ratio test use is J P_rigid (the unweighted rigid projector at the current geometry), under the same project_rigid_modes flag as the quantum terms.
- Why: T-38: near-linear molecules (B ~ C, gap/scale < 1e-4) take the finite-difference Jacobian, which leaks 1.5e-5 of |J| into rigid motion; the joint matrix has only the 1e-8 floor and lambda there, so the step became 95% translation/rotation and the fit crawled (fluoroacetylene 57 iterations, stopped 2.6 mA from its optimum). The true Jacobian has no rigid part, so the projection is exact; prototype: fluoroacetylene 19 iterations (same as the analytic Jacobian), 8 other molecules unchanged to 1e-6 mA.
- Alternatives considered: the analytic Jacobian nearer degeneracy (Q-57 option 2, not chosen); a README note only.
- Source: auto-picked, no answer in 15 min (Q-57)
- Revisit if: a Jacobian row is found that legitimately depends on the molecule's position or orientation. Found (T-38 verifier): the conformer mixture (lab-frame offsets); the projection is skipped for it (rows unprojected, results as before). _chi2_dof counts the same projected Jacobian.

### D-95 · Next task: measure the internal prior's angle-row units (N29) (2026-11-22)
- Decision: T-39 measures how much InternalPriorEngine's angle/dihedral Jacobian (57.3x too small against its degree residuals) moves the fits that use it, then proposes the unit fix.
- Why: Q-58's recommended option; a confirmed defect found by the T-38 verifier; measuring first is reversible.
- Alternatives considered: N7 split widths; N3 references; done for now.
- Source: auto-picked, no answer in 15 min (Q-58)
- Revisit if: the user picks another option.

### D-96 · The internal prior's angle units: documented, not fixed (2026-11-22)
- Decision: document the defect (README note under Running, a comment at InternalPriorEngine.stacked, backlog N29 open and flagged); leave the code.
- Why: T-39 measured it (Cartesian spectral-only with the BALANCED/STRICT prior, 9 molecules): the angle rows' Jacobian is 57.3x too small, 58-67 rejected steps, no uncertainty for 3/9 fits, angle widths 38-64 deg for 3 others; the fix gives 2-4 rejections and 0.3-2.1 deg widths but moves accuracy both ways (noisy starts, autoconfig off: bonds 24.16 -> 23.00 mA, angles 1.578 -> 1.922 deg; reference starts: 2.92 -> 3.27 mA, 0.217 -> 0.193 deg), which is the user's call.
- Alternatives considered: fix the units (Q-59 option 1, recommended, left for the user); move on to N7.
- Source: auto-picked, no answer in 15 min (Q-59); option 1 left for the user
- Revisit if: the user decides on option 1, or a calibration of the preset's angle sigma is wanted.

### D-97 · Next task: measure the split estimator's widths against its spread (N7) (2026-11-22)
- Decision: T-40 measures, for objective_mode split (and joint as the control), the reported geometry widths against the Monte Carlo spread of refits under the noise the posterior assumes; then proposes.
- Why: Q-60's recommended option; N7 says split widths are not the split estimator's (one-constant water: 0.05x); measuring is reversible.
- Alternatives considered: N29 fix (user's call, not auto-picked); N3 references; done for now.
- Source: auto-picked, no answer in 15 min (Q-60)
- Revisit if: the user picks another option.

### D-98 · Next task: why joint internal-mode fits stop short (N31) (2026-11-22)
- Decision: T-41 diagnoses why joint internal-coordinate fits (the CLI default) stop 0.03-1.0 posterior widths short of their own optimum on acetyl fluoride, fluoroethane and fluoroacetylene, then proposes.
- Why: Q-61's recommended option; default path; it leaves the CLI bootstrap with 0-27 of 50 usable refits.
- Alternatives considered: N7 split covariance; N30 spawned workers; done for now.
- Source: auto-picked, no answer in 15 min (Q-61)
- Revisit if: the user picks another option.

### D-99 · Complete the internal coordinate set when bonds and angles miss a direction (2026-11-22)
- Decision: in internal mode, when ic_use_dihedrals is off and bonds + angles do not span every direction the fit can move (the physical basis: not rigid, not symmetry-forbidden), add dihedrals, decided once per fit and said so; molecules already spanned unchanged.
- Why: T-41: acetyl fluoride and fluoroethane miss the methyl torsion (14/15, 17/18); the fit cannot move along it and stops 0.16-0.90 posterior widths from its optimum; with dihedrals both reach it (<= 1.7e-6) and acetyl fluoride agrees with the Cartesian fit (14.584 vs 14.602 mA).
- Alternatives considered: dihedrals on everywhere; a warning only.
- Source: auto-picked, no answer in 15 min (Q-62)
- Revisit if: any molecule already spanned changes, or an answer moves by more than the stuck fits' difference (0.17-0.22 mA bond RMS).

### D-100 · Automatic dihedrals committed as an opt-in (2026-11-22)
- Decision: internal_coordinates.use_dihedrals / ic_use_dihedrals accept "auto" (dihedrals added once, at the start geometry, when bonds and angles miss a direction the fit can move); the default stays False; README note on free torsions; the posterior records use_dihedrals and dihedrals_reason.
- Why: T-41 measured (9 molecules, HF/6-31G, joint internal): 7 unchanged; acetyl fluoride and fluoroethane reach their optimum (0.19 -> 1.7e-6, 0.16 -> 1.6e-7 widths) and keep 42/50 and 50/50 bootstrap refits (0, 4 before); but bond RMS moves -0.19/-0.08 (AF) and +0.22/+0.37 mA (FE, autoconfig off/on), beyond D-99's 0.22 bound, and AF costs one more Hessian (13 -> 27 s).
- Alternatives considered: auto as the default (Q-63 option 1, recommended, left for the user); drop it.
- Source: auto-picked, no answer in 15 min (Q-63); default change left for the user
- Revisit if: the user wants "auto" as the default.

### D-101 · Dihedral follow-ups from the T-41 verifier (2026-11-22)
- Decision: (A) the acceptance test's q-space prior rows wrap dihedral differences (_ic_prior_trial_rows); (B) the bootstrap's analytic reference takes dihedral spreads as wrapped deviations from the base; (C) multistart resolves "auto" once on the first start and passes the bool to every start; (D) "auto" counts a direction reached only where the damped inverse realises over half a step (s^2 > ic_damping), and adds dihedrals only when they reach more. Docstring figure corrected (AF 0.19, not 0.90).
- Why: the verifier of 4209882 (default path bit-identical on 6 molecules + a CLI run) found: crossing steps refused (spectral-only FE with dihedrals, 17/17), dihedral cloud spreads of 98-181 deg near +-180, per-start sets in a fluorobenzene multistart (24 vs 0 dihedrals), fluorobenzene's barely reached direction (stall at 0.056 widths), and notes/posteriors for dihedrals that reach nothing (linear FA "0.0 +- 0.0", planar HCOF). A and B were already wrong for use_dihedrals: true.
- Alternatives considered: decide the set from the bonding graph (not attempted); leave A/B as pre-existing.
- Source: assumption, unconfirmed (verifier findings on a change already chosen; each fix exact or opt-in only)
- Revisit if: "auto" with symmetry off on a near-linear molecule (FA start adds a dihedral of an almost linear chain) shows nonsense widths.

### D-102 · Next task: bootstrap workers without fork-after-threads (N30) (2026-11-22)
- Decision: T-42 measures, then proposes running single-worker bootstraps inline and multi-worker ones in spawned processes, so no refit is forked from a multi-threaded process.
- Why: Q-64's recommended option; T-40 saw a forked refit deadlock (pyscf, OMP 4); Python warns on every such fork (the CI selection's 37 warnings).
- Alternatives considered: auto dihedrals as default (user's call); N7 split covariance; done for now.
- Source: auto-picked, no answer in 15 min (Q-64)
- Revisit if: the measurement shows a time cost (then propose before committing).

### D-103 · Warn before forking several workers from a threaded in-process backend (2026-11-22)
- Decision: when a bootstrap or multistart pool forks more than one worker, the backend runs in-process and OMP_NUM_THREADS is not 1, print a warning naming the hang and the remedy; behaviour unchanged.
- Why: T-42: a forked refit hung 3/3 with pyscf at four threads; one worker now runs in-process (4f23b17), several still fork; forkserver costs +0.5 s and needs a __main__ guard (the user's call).
- Alternatives considered: forkserver pools (Q-65 option 2, not auto-picked); README only.
- Source: auto-picked, no answer in 15 min (Q-65)
- Revisit if: the user picks forkserver, or Python's default start method changes (3.14: forkserver).

### D-104 · Next task: fluoroacetylene's internal-mode refits (N31 rest) (2026-11-22)
- Decision: T-43 diagnoses why fluoroacetylene's joint internal-mode bootstrap keeps 27/50 refits (its coordinates span its directions), then proposes.
- Why: Q-66's recommended option; the default path's bootstrap.
- Alternatives considered: N7 split covariance; N3 references; done for now.
- Source: auto-picked, no answer in 15 min (Q-66)
- Revisit if: the user picks another option.

### D-105 · Internal mode: retake a step the back-transform could not realise in Cartesians (2026-11-22)
- Decision: T-43 implements and measures a Cartesian fallback: an internal-mode step refused whose back-transform failed is retaken as a Cartesian joint step on the same objective (rigid-free J, the q-space prior rows through Wilson B, the projected Cartesian quantum terms); propose before committing if anything other than the stalled fits moves.
- Why: T-43: fluoroacetylene's optimum is exactly linear; a valence angle cannot step through 180 deg, so the internal step back-transforms with a 1.6e-3 error, is refused, and is retried unchanged until the plateau rule stops the fit at 179.995 deg (0.008 widths); the bootstrap then keeps about half its refits. A Cartesian fit from there reaches 180.0000 deg (7e-11 widths) in 11 iterations.
- Alternatives considered: linear-bend coordinates (changes reported names); documentation only.
- Source: auto-picked, no answer in 15 min (Q-67)
- Revisit if: the fallback changes any fit that did not stall.

### D-106 · The Cartesian fallback committed as an opt-in (2026-11-22)
- Decision: ic_cartesian_fallback / internal_coordinates.cartesian_fallback (default false): a joint internal-mode step refused whose back-transform error exceeds max(0.1 |dq|, 1e-6) is retaken as a Cartesian step on the same objective, and the rest of the pass steps in Cartesians; history records cartesian_fallback; README note.
- Why: T-43 measured (9 molecules, HF/6-31G, joint): fluoroacetylene reaches its optimum (0.008 -> 1.9e-6 widths, bootstrap 27 -> 50 of 50, atoms <= 2.6e-4 A); acetyl fluoride's base fit 0.19 -> 2.8e-6 with bond RMS -0.18 mA and one more Hessian (13 -> 24 s), refits 9/50; seven others bit-identical. Without the 1e-6 floor, converged fits' zero steps switched on round-off; without the switch, acetyl fluoride ended at 0.40 widths. A free torsion is use_dihedrals: auto's case, not this one.
- Alternatives considered: as the default (moves AF's answer, costs a Hessian: the user's call, per D-100's precedent); linear-bend coordinates; documentation only.
- Source: assumption, unconfirmed (opt-in per Q-67's own condition: something other than the target fit moved)
- Revisit if: the user wants it on by default, or linear bends are preferred.

### D-107 · Next task: internal-mode water from a far start (N33) (2026-11-22)
- Decision: T-44 diagnoses why internal-mode water (analytic backend, start r 1.25 A / 80 deg, joint) refuses zero-length steps and ends 820 widths from its mode, then proposes.
- Why: Q-68's recommended option; the default path.
- Alternatives considered: a page of the decisions waiting for the user; N7; done for now.
- Source: auto-picked, no answer in 15 min (Q-68)
- Revisit if: the user picks another option.

### D-108 · Join disconnected fragments in the bond graph (2026-11-22)
- Decision: _detect_bonds adds, while the covalent-radii graph has more than one fragment, the shortest inter-fragment atom pair as a bond; a connected molecule's bonds are unchanged.
- Why: T-44: water started at 1.25 A (O-H threshold 1.164 A) had no bonds and no angles, so internal mode's steps were zero and the fit stopped at its start, 820 widths from its mode, silently; joined, it reaches 6.5e-6 widths. A weakly bound complex is disconnected the same way, so its intermolecular motion is unreachable in internal mode today; joining adds an intermolecular distance coordinate.
- Alternatives considered: warn only; join and warn.
- Source: auto-picked, no answer in 15 min (Q-69)
- Revisit if: a complex's report should not list the joining distance.

### D-109 · Next task: verify the four unchecked changes (2026-11-22)
- Decision: T-45 runs an independent verifier on 4f23b17 (one-worker bootstrap in-process), dda22df (fork warning), 23ca969 (opt-in Cartesian fallback), 81edad9 (fragment joining).
- Why: Q-70's recommended option; no one but me has checked them (L-10: every verifier round so far found something).
- Alternatives considered: a page of the decisions waiting for the user; N7; done for now.
- Source: auto-picked, no answer in 15 min (Q-70)
- Revisit if: the user picks another option.

### D-110 · T-45 verifier: one commit held, three fixed (2026-11-22)
- Decision: (1) Fragment joins are named `contact i-j` (angles and dihedrals through one are flagged) and get no rule-book prior; where the internal prior is on they are held to their start. (2) The Cartesian fallback uses the internal step's quantum weight, and the history records the step taken: its prior RMS, model change, own range/null split (rigid-free J with the prior rows), no back-transform error, and refused Cartesian steps after the switch. (3) fork_hang_warning counts psi4_num_threads; multistart without a backend key warns for Psi4, the default. (4) Wording: FA's optimum lies within 0.05 deg of linear, not on it; the fallback is joint-only (README and docstring).
- Why: independent verifier (L-10). Held: 4f23b17 inline bootstrap (bit for bit on five cases); dda22df for its stated rule; 23ca969's numbers (FA 8.0e-3 -> 1.9e-6 widths, bootstrap 27 -> 50/50; AF 0.189 -> 2.8e-6; seven bit-identical); 81edad9 for far starts and 34 connected sets. Refuted: 81edad9 for complexes -- the O-H rule pulled a dimer's O...H from 1.953 to 1.459 A (fixed: 1.949); Ar...HF came out 0.19 A asymmetric at chi2 44 (fixed: 1e-4 A, chi2 1.4e-5). Fallback: weight 2.667 against 2.2 (legacy rule), history fields from the refused internal step. Warning: silent on Psi4's own threads and on the default backend. My own check: the new split fields first broke the convergence test (None), caught by reading every reader (L-25).
- Alternatives considered: give contacts a soft van der Waals prior (no library for it); drop the fallback's history fields (multistart scoring reads prior_wrms).
- Source: assumption, unconfirmed (defect fixes within the agreed T-45 scope)
- Revisit if: a contact should carry its own prior, or a complex in joint mode needs converging (N34).


### D-111 · Next task: one page of the decisions waiting for the user (2026-11-22)
- Decision: T-46 gathers every open "user's call" from the records into one page: what each changes, the measured evidence, the options, and a recommendation.
- Why: Q-71's recommended option; about nine flagged trade-offs have piled up and none can be auto-picked.
- Alternatives considered: N7 split-objective widths; N34 dimer in joint mode; done for now.
- Source: auto-picked, no answer in 15 min (Q-71)
- Revisit if: the user picks another option.

### D-112 · Next task: measure automatic dihedrals and the Cartesian fallback together (2026-11-22)
- Decision: T-47 runs the nine molecules (HF/6-31G, joint, internal mode) in four settings -- neither, use_dihedrals: auto, cartesian_fallback, both -- and reports distance to each fit's own optimum, bond RMS, Hessians and wall time; no default changes.
- Why: Q-72's recommended option; decision page item 3 asks for this before either becomes a default (both act on acetyl fluoride, never measured together).
- Alternatives considered: N7 split-objective widths; N34 dimer in joint mode; done for now.
- Source: auto-picked, no answer in 15 min (Q-72)
- Revisit if: the user answers the decision page.

### D-113 · Automatic dihedrals and the Cartesian fallback do not interact (2026-11-22)
- Decision: no code change. Decision page items 2-4 updated with T-47's numbers; item 3's "after one check" condition is met (recommend yes for 2 and 3 together, still the user's call).
- Why: T-47 (nine molecules, HF/6-31G, joint, internal; t47/): with both on, every molecule's answer is bit-identical to the setting that acts -- dihedrals on AF and FE (the fallback takes 0 steps there), the fallback on FA (no dihedrals added); the other six bit-identical in all four settings, autoconfig off and on. Hessians and bootstraps equal the single setting's (AF 42/50, FE 50/50, FA 50/50; neither 0, 4, 27). The remaining 0.3-4.7 mA (worst atom) between the dihedral, fallback and Cartesian answers is the Hessian's expansion point: refreshed at each answer they agree to 0.01-0.15 mA (<= 1.2e-3 widths), bond RMS AF 14.5956 / 14.5954 / 14.5954, FE 12.502 / 12.504; so dihedrals' bond moves are the fit reaching the coordinate-free answer.
- Alternatives considered: none (measurement only).
- Source: assumption, unconfirmed (measurement within D-112's scope)
- Revisit if: the user switches either default.

### D-114 · Next task: split-objective widths against the split estimator's spread (N7) (2026-11-22)
- Decision: T-48 diagnoses why split-objective widths depart from the split estimator's bootstrap spread (T-40: 7 of 9 molecules, ratios 0.05-58.7), then proposes; no change before the cause is shown.
- Why: Q-73's recommended option; the largest confirmed width error left in the backlog.
- Alternatives considered: ozone's 0.006-width stop; N34 dimer in joint mode; done for now.
- Source: auto-picked, no answer in 15 min (Q-73)
- Revisit if: the user answers the decision page or picks another option.

### D-115 · Split fits: measure the stall and say so (2026-11-22)
- Decision: a per-fit check (MolecularOptimizer.split_stall) measures how far the split fit stopped from the minimum of its null-space energy (one Newton step on the fit's quadratic model, in posterior widths and Angstrom); split_inconsistency reports it first, with the measured cause, ahead of the coupling note. Reporting only: split answers unchanged.
- Why: T-48: split fits stall where their acceptance test refuses the energy-only null step on directions the data still weigh (rank cut sv_min_abs = 1/sigma_x); the null-space minimum lies 2.9-16.6 mA away on 5 of 7 molecules, the answer moves up to 1.47 widths with the trust radius, and no covariance formula reproduces the refit spread (coupling-only linear response 0.02-4.4x). Today's warning blames the coupling, which was refuted.
- Alternatives considered: also withhold split widths when it fires (Q-74 option 2); fix the split solver (option 3: changes split answers, the user's call).
- Source: auto-picked, no answer in 15 min (Q-74)
- Revisit if: the user wants split widths withheld or the split solver fixed.
- Outcome (2026-11-22, T-48): split_stall reads 0 where the data reach every direction (water, ozone, the five split-water test fixtures) and 0.10-43 widths on the seven benchmark molecules with a null space; 11 of their 14 Cartesian and internal fits pass the 0.25 limit (FA Cartesian 0.17, CFM 0.21 / 0.10 keep the coupling note). It is close to distance_to_mode on most of them [refuted by the T-49 verifier for AF and HNCO: 12.1 vs 4.0, 43 vs 4.25, a Hessian-floor artefact], so it replaces the off-mode note's rank-disagreement attribution, which was wrong in Cartesian mode (step and posterior null spaces identical). README split section corrected (it said sv_min_abs defaults to 0; it is 1/sigma_x with the quantum prior on). Tests: dev/tests/test_split_stall.py, 4/4 mutations caught (one only after moving the Newton-step test off the quantum point); CI selection 1718 passed, 15 skipped.


### D-116 · Next task: verify 58d159a and 9c1ad3c (2026-11-22)
- Decision: T-49 runs a fresh independent verifier on 58d159a (the T-45 fixes: contacts, fallback weight and history, warning gaps) and 9c1ad3c (split_stall and the split warning), trying to refute their claims and figures; defects found are then fixed.
- Why: Q-75's recommended option; neither commit has been checked by anyone but me (L-10: every verifier round so far found something).
- Alternatives considered: ozone's 0.006-width stop; N34 dimer in joint mode; done for now.
- Source: auto-picked, no answer in 15 min (Q-75)
- Revisit if: the user answers the decision page.

### D-117 · T-49 verifier: 58d159a held; split_stall rebuilt on the step's own record (2026-11-22)
- Decision: (1) split_stall measures on what the split step last solved with (_last_split_step: its own null space and rank, its gradient after any target pull, its Hessian, q-space in internal mode), and only when the last step was refused at this geometry; negative-curvature null directions are counted and left out; the atom figure is the physical part. (2) split_inconsistency gives the stall note only where the step's null space agrees with the posterior's (sine of the largest principal angle < 0.1, _split_null_mismatch); otherwise the off-mode note. (3) The bootstrap summary picks units by coordinate kind, not name (contacts were printed in degrees). (4) README: the config runner's split keeps sv_min_abs 0 (original paragraph restored) and the API defaults it to 1/sigma_x; stall figures by coordinate mode; the dimer sentence says the monomers keep their rule-book priors.
- Why: T-49 verifier (report: scratchpad t49v/REPORT.md). 58d159a held: contacts (dimer O...H 1.4589 -> 1.9491 A; 48 connected coordinate sets bit-identical), fallback weight and history, both warning gaps. 9c1ad3c: false stall on internal fits at their own fixed point (two-constant water 4.06 widths against 0.028 on the step's own null space; acetyl fluoride internal 3.64 vs 0.015), "over an Angstrom" a Hessian-floor artefact (negative null curvature, -0.042 / -0.119 Hartree/A^2), false stall with a prior target (0.85 widths at a fixed point), README sv_min_abs wrong for the config runner, contact units regression, 6 of 10 mutations uncaught. Re-measured (t48/stall9.py): the stall note now fires on 7 of 18 fits (HNCO, FF, VF both modes, FE Cartesian): 2.9-16.6 mA Cartesian, 0.8-8.8 internal, HNCO 28 mA + 1 negative direction; the reviewer's cases read 0.028 (off-mode note) and None (target fits ended on an accepted step). Tests: 16 split-stall + 3 fragment; 10 of 11 mutations caught (the 11th is equivalent in Cartesian mode); CI 1732 passed, 15 skipped.
- Alternatives considered: drop split_stall (the off-mode note alone); keep the posterior's null space and gate on rank only (equal ranks disagreed in internal mode: one-constant water with an internal prior).
- Source: assumption, unconfirmed (defect fixes within T-49's scope)
- Revisit if: a split fit's last step is accepted while it is stalled (the check then reads None), or the user wants split widths withheld.


### D-118 · Next task: a narrow check of af66799 (2026-11-22)
- Decision: T-50 runs a fresh independent verifier on af66799 only (split_stall on the step's own record, the null-space agreement rule, negative curvature, the physical atom figure, summary units by kind, README), trying to refute its claims and figures; defects found are then fixed.
- Why: Q-76's recommended option; af66799 replaced most of 9c1ad3c's logic and nobody but me has checked it (L-10).
- Alternatives considered: ozone's 0.006-width stop; N34 dimer in joint mode; done for now.
- Source: auto-picked, no answer in 15 min (Q-76)
- Revisit if: the user answers the decision page.

### D-119 · T-50 verifier: one rule for both null spaces; the record carried over the last step (2026-11-23)
- Decision: (1) _split_null_mismatch cuts the step's null space and the posterior's by one rule (eigenvalues of B^T P B above 0.5, P each null space's Cartesian projector). (2) _split_record carries the last record to the fit's geometry when the last step was accepted (g + H d, d that step in the step's parameters) instead of returning None. (3) split_off_mode names the range-null coupling, not a disagreement, when the two null spaces agree. (4) Docstrings and README: per-mode figures, the agreement condition, the internal-mode atom figure is a B+ linearisation; the "both rank 2" docstring corrected.
- Why: T-50 verifier (report: scratchpad t50v/REPORT.md): the record is what the step solved with in 10 modes; the parent's defects are fixed. Defects: cut by two rules, HDO under C2v read 1.0 for projectors equal to 2e-16 (stall 29.8 widths reported as a disagreement; HDO B-only's 0.456 as a coupling; ozone 1.0 harmlessly); "None after an accepted step" hid acetyl fluoride Cartesian (4.03 widths / 43.8 mA, after 11 refusals at the trust floor) and fluoroacetylene internal (1.34); the off-mode note blamed a disagreement where the null spaces agree (a prior target at sigma_x 5 mA, vinyl fluoride after a refresh); internal atom figures are linearised (FF 8.82 vs 6.82 realised); 5 of 16 mutations uncaught. Re-measured (t48/stall9.py): mismatch ~0 for HDO and ozone; the note fires on 9 of 18 fits (adds AF Cartesian 4.03 / 43.8 mA + 1 negative direction, FA internal 1.34 / 115 mA linearised); FE internal keeps the off-mode note (its step cannot turn the torsion: 15 of the posterior's 16 null directions). Tests 24 in test_split_stall.py; 11 of 11 mutations caught (incl. the verifier's 5 misses); CI 1740 passed, 15 skipped.
- Alternatives considered: compute a fresh record by running one more iteration (side effects on real backends); keep None after an accepted step and document it.
- Source: assumption, unconfirmed (defect fixes within T-50's scope)
- Revisit if: a fit ends several accepted steps from a refusal while still stalled (the carry covers one step), or the user wants split widths withheld.


### D-120 · Next task: make the multireference tests run in a fresh clone (F3) (2026-11-23)
- Decision: T-51 embeds the 3-atom geometries the two multireference tests read from the gitignored benchmark cache, so they run in CI; they must still pass and must fail when the property they test is broken.
- Why: Q-77's recommended option; the tests always skip in a fresh clone (F3), so CI never runs them.
- Alternatives considered: a narrow check of c4eb2b6; N34 dimer in joint mode; done for now.
- Source: auto-picked, no answer in 15 min (Q-77)
- Revisit if: the user answers the decision page.
- Outcome (2026-11-23, T-51): the ozone and water HF/6-31G minima regenerated exactly as the benchmark caches them (pyscf_hf optimise from start_geometry; O-O 1.25067, O-H 0.94963 A) give max T2 0.17935 and 0.04713 against the tests' 0.1794 and 0.0471, and are now written in the test; the cache reader is gone. Both tests run (0.5 s, 0.06 s); a threshold of 0.04 or 0.2 fails one each. CI selection 1742 passed, 13 skipped (was 1740 / 15).


### D-121 · Next task: unused imports and dead locals (F3) (2026-11-23)
- Decision: T-52 removes unused imports and dead local assignments in .github/backend, runner and cli.py (not scripts/archive, not tests), found with a linter; anything another module, a test or a script imports from there, or a test monkeypatches, stays; a dead assignment whose right side has effects keeps its call.
- Why: Q-78's recommended option; F3 lists 79 unused imports and ~15 dead locals from the 2026-11-19 audit.
- Alternatives considered: a narrow check of c4eb2b6; N34 dimer in joint mode; done for now.
- Source: auto-picked, no answer in 15 min (Q-78)
- Revisit if: the user answers the decision page.
- Outcome (2026-11-23, T-52): nothing to remove. ruff 0.17 (rules F401/F841, and the whole F set) finds no unused import or dead local in .github, runner, cli.py, scripts or dev; a planted unused import is found, so the check works. efabab3 (2026-11-19, "Remove dead code found by the bloat audit") had already acted on this audit: seven genuinely unused imports and the dead locals removed, the other 71 "unused" imports kept on purpose (backend registration, re-exports; marked noqa). The backlog row had not been updated; it is now.


### D-122 · Next task: a narrow check of c4eb2b6 (2026-11-23)
- Decision: T-53 runs a fresh independent verifier on c4eb2b6 only (one cut rule for both null spaces, the record carried over the last accepted step, the off-mode note naming the coupling, the docs' figures); defects found are then fixed.
- Why: Q-79's recommended option; c4eb2b6 is a code change nobody but me has checked (L-10).
- Alternatives considered: a status check of F3's remaining audit items; N34 dimer in joint mode; done for now.
- Source: auto-picked, no answer in 15 min (Q-79)
- Revisit if: the user answers the decision page.

### D-123 · T-53 verifier: the off-mode note names no cause it has not measured (2026-11-23)
- Decision: (1) split_off_mode, when the two null spaces agree, says the posterior is not centred on the fit and names no cause. (2) Docs: the carry is exact in Cartesian coordinates and first-order in internal ones; "ten refused steps, the last accepted 1e-4 A step at the trust floor" (c4eb2b6's commit message said eleven refusals at the floor); the unconverged back-transform figures (6.8, 186 mA) removed; the 9-of-18 count is from start_geometry (7 from the benchmark's warm starts); the check reads the quantum side only (ozone at trust radius 0.2 moves 0.63 widths with a stall of 0.001). (3) Tests for the carry over a range-and-null step in both modes, the dihedral wrap, and the wording.
- Why: T-53 verifier (report: scratchpad t53v/REPORT.md). Held: the one-rule cut (HDO reads 0, Cartesian fresh records <= 4.4e-14), 9 of 18, the mA ranges, AF Cartesian 4.03 / 43.8, FA internal 1.34, the record never from another quantum point (144+ fits). Refuted: the coupling attribution (it explained 1 of 11 off-mode cases; a prior target's unprojected pull and an unfinished range explained the others), "exact" in internal mode (AF 0.028 carried vs 0.015 fresh), the back-transform figures and the refusal count; 6 of 18 mutations survived (now 5 of 6 retested caught; the 0.99-cut one needs a null vector between 50 and 99% physical, which no fixture has). CI 1745 passed, 13 skipped.
- Alternatives considered: decompose the posterior gradient to name the cause (more machinery on a secondary objective, the user's call); drop the split diagnostics.
- Source: assumption, unconfirmed (defect fixes within T-53's scope)
- Revisit if: the user wants the off-mode cause measured, or the split diagnostics simplified.


### D-124 · Next task: status of F3's remaining audit items (report only) (2026-11-23)
- Decision: T-54 checks which of the 2026-11-19 audit's remaining items (unreferenced defs, unreferenced investigation scripts, backend modules unreachable from the CLI/runner, test/script-only functions, never-passed optimizer kwargs) are still open, with evidence per item; deletes nothing.
- Why: Q-80's recommended option; L-26 (T-52 found part of this row already done).
- Alternatives considered: measure the split posterior's unprojected prior-target pull; N34 dimer in joint mode; done for now.
- Source: auto-picked, no answer in 15 min (Q-80)
- Revisit if: the user answers the decision page.
- Outcome (2026-11-23, T-54; scratchpad t54/graph.py, kwargs.py): done -- unused imports and dead locals (efabab3, ruff clean), the five named unreferenced defs (0 references), ten investigation scripts archived (a2aafb9; build_methods_pdf.py kept), five unreachable modules deleted (988bad5: model_selection, prior_sensitivity, geometric_bridge and two more). Still there, by design: a static import graph from cli.py and runner/ reaches 65 of 72 backend modules; the other 7 (1,922 lines) serve benchmark scripts, tests and dev (two engine mentions are comments); 13 of 144 MolecularOptimizer parameters are passed by nothing outside quantize.py (isotropic_prior, chi2_rescale_threshold, spectral_analytic_jacobian, spectral_jacobian_degeneracy_tol, null_convergence_requires_energy, six guardrail tunables, accept_requires_geometry_valid, mass_dependent_frac_sigma). Not re-measured: the 47 test/script-only functions. Nothing deleted.


### D-125 · Next task: measure the split posterior's prior-target pull (2026-11-23)
- Decision: T-55 measures how far the split posterior's unprojected prior-target pull (gradient H(x - t) and energy 1/2 d^T H d, against a Pn H Pn curvature) moves its mode, distance_to_mode, the off-mode notes and widths, on analytic water and wherever split runs with a target; then proposes. No code change before the measurement.
- Why: Q-81's recommended option; T-53 found it (1.0 of 1.0 widths off mode at sigma_x 5 mA).
- Alternatives considered: N34 dimer in joint mode; a review of the 13 never-passed settings; done for now.
- Source: auto-picked, no answer in 15 min (Q-81)
- Revisit if: the user answers the decision page.
- Measured (2026-11-23, T-55; scratchpad t55/pull.py; analytic one-constant water, Cartesian split, sigma_x 20/10/5 mA, targets 5.6 and 15.5 mA from the fit): distance to the posterior's mode 0.023-0.998 widths as the code stands (full H(x - t)); 0.002-0.030 with the step's own left-projected pull Pn H (x - t); 0.126-1.54 with the split posterior's own definition Pn H Pn (x - t) (the coupling dropped, as without a target). Reported widths identical to 1e-6 in all three (the curvature is Pn H Pn throughout). No script, config or CLI path combines split with prior_target_coords (the benchmark's targets are joint-only). Proposal pending (Q-82).


### D-126 · Prior-target pull on the split posterior: document only (2026-11-23)
- Decision: no behaviour change. _posterior_gradient's and split_off_mode's docstrings and one README sentence say that with a prior target the split posterior keeps the target's pull along the data's directions while its curvature (alpha Pn H Pn) does not, so its mode lies 0.02-1.0 widths off the split fit (analytic water, sigma_x 20-5 mA), widths unaffected, and that no script or config combines the two.
- Why: Q-82's recommended option. Nothing in the repo runs split with a target (re-checked: the five scripts that pass prior_target_coords run joint; the config runner never passes one), widths do not move, and the two candidate fixes pull the mode distance opposite ways (0.030 vs 1.54), so neither is plainly right.
- Alternatives considered: make the target branch follow the split posterior's own definition, Pn H Pn (mode diagnostics move, widths do not); N34 dimer in joint mode; done for now.
- Source: auto-picked, no answer in 15 min (Q-82)
- Revisit if: a script or config starts combining split with prior_target_coords, or the user answers the decision page.

### D-127 · Next task: measure why the water dimer in joint internal mode stops off its mode (2026-11-23)
- Decision: T-56 measures why the hydrogen-bonded water dimer, joint objective, internal mode, quantum prior on, ends 0.29 posterior widths from its posterior's mode (backlog N31, ex-N34). No code change before the measurement.
- Why: Q-83's recommended option; it is the default path (joint), unlike the split work of T-48 to T-55.
- Alternatives considered: a report on the 13 MolecularOptimizer settings nothing outside quantize.py passes; N29 angle-row units (user's call); done for now.
- Source: auto-picked, no answer in 15 min (Q-83)
- Revisit if: the user answers the decision page.
- Measured (2026-11-23, T-56; scratchpad t56/dimer_reach.py, dimer_dirs.py, dimer_reach_seeds.txt; water dimer, five isotopologues, HF/6-31G pyscf, joint internal, 15 starts at 0.02 A noise): the start decides the point group. 7 of 15 get Cs (8 directions): bonds, angles and the contact reach 8 of 8, the fits end 8e-5 to 0.023 widths from their mode, and auto adds nothing. 8 of 15 get none (12 directions): the 10 coordinates reach 10 of 12 (singular value exactly 0 on the other 2: the torsions about the near-linear O...H-O bond, dihedrals 6-4-5-1 and 2/3-1-5-4); the fits end 0.068 to 4.0 widths from their mode (seed 8 at max_iter), yet within the 10 directions they reach 5e-5 to 0.041 widths (seed 11: 0.32, not explained), and their answers differ from the Cartesian one by 9 to 62 mA, almost all of it along the two torsions. With use_dihedrals auto: 13 coordinates reach 12 of 12, 0.003 to 0.080 widths, within 0.3 to 7.4 mA of the Cartesian answer (seed 23: 23.5 mA with both at their mode; NOT a second minimum, T-57: a 40.6 mA-old Hessian). The report says nothing below 2 widths (Gauss-Newton's "not the mode"); the bootstrap would refuse such refits (0.25 limit). Same cause as T-41's free torsions (D-99, D-100). Proposal pending (Q-84).

### D-128 · Report an internal set that cannot reach every direction (2026-11-23)
- Decision: report only, no fit changes. When internal mode runs without dihedrals and bonds and angles reach fewer directions than the fit can move (the reach test _resolve_ic_dihedrals already makes), the run and the uncertainty warnings say how many they reach, that the fit cannot move the rest, and that use_dihedrals: auto adds dihedrals (where dihedrals do reach more).
- Why: Q-84's recommended option. T-56: on the C1 water dimer the two H-bond torsions are out of reach and the fits end 0.068-4.0 widths from their mode with no note below 2 widths.
- Alternatives considered: make use_dihedrals auto the default (the user's call since D-100; never auto-picked); seed 23's second minimum; done for now.
- Source: auto-picked, no answer in 15 min (Q-84)
- Revisit if: the user makes auto the default (the note would then fire only for an explicit False).
- Outcome (2026-11-23, T-56; 6920019 then the follow-up commit; verifier report scratchpad t56v/REPORT.md): fits bit-identical (11 fits parent vs child, coordinates, history, Laplace widths; only dihedrals_reason's text differs, its "(a torsion has no bond or angle of its own)" dropped). Verifier findings fixed: the remedy now names use_dihedrals: true (multistart turns auto into the first start's True/False, so "auto adds them" was false for a later start); the note covers a set with dihedrals that still falls short (auto-added or True: an exactly linear H-O-O-C#C-H reaches 10 of 12 with them); the wording names what is measured (the step realises at most half of a move along a missed direction) instead of "a torsion" (fluorobenzene's missed direction is seen faintly; a planar start's is a wag); the counts are said to be the start geometry's; the note survives a saddle refusal and a Laplace summary with no covariance; the reach test cannot raise in the constructor (coincident atoms); the docstring's "only" claim corrected (acetone 22 of 24 and fluorobenzene 29 of 30 fire too). Tests 7 -> 17 cases in the section; 17 of 17 mutations caught (t56/mutations2.txt). Pushed commit messages are not rewritten: 6920019's "fires for ... only" is superseded here.

### D-129 · Next task: measure the water dimer's second minimum (2026-11-23)
- Decision: T-57 measures seed 23's two answers (internal with dihedrals and Cartesian, joint HF/6-31G, both at their mode, 23.5 mA apart): which is lower in the joint objective with real energies, whether each is a minimum (refit from the other's answer), how often starts land in each, and whether the report says anything. No code change before the measurement.
- Why: Q-85's recommended option; the reported answer would depend on the start and the coordinate mode.
- Alternatives considered: a report on the 13 MolecularOptimizer settings nothing outside quantize.py passes; multistart adding dihedrals when any start needs them (user's call: changes fits); done for now.
- Source: auto-picked, no answer in 15 min (Q-85)
- Revisit if: the user answers the decision page.
- Measured (2026-11-23, T-57; scratchpad t57/minima.py, refresh.py, age_note.py; C1 water dimer, joint HF/6-31G, 8 starts): no second minimum. Seed 23's Cartesian answer C lies below the internal answer A on both terms (E_HF 2.7e-5 Eh lower, 1/2|r|^2 0.0045 lower), so for any alpha, and refits from A move to C. The plain fits' Hessians were 4.5 to 40.6 mA from their answers (seed 23's A: 40.6) and their weights alpha = 1/(1/2 mean positive curvature sigma_x^2) ranged 2999 to 3712 with the start; one refresh_hessian_at_answer (what quantize uncertainty runs, D-63) moved A 0.6 to 14.4 mA rms and C 0.2 to 5.7, brought alpha to 3036-3048 and A to within 0.7 to 2.6 mA of C (from 0.3 to 23.5), and every answer to 13.1 to 13.6 mA from the truth (A up to 28.4 before). The posterior's Hessian-age warning fires (40.61 mA against the 5 mA threshold) but says only that the widths carry the old curvature; the Laplace summary carries no age warning. Proposal pending (Q-86).

### D-130 · The Hessian-age warning says the answer was fitted on that curvature (2026-11-23)
- Decision: wording and coverage only, no behaviour change. The Hessian-age warning says the answer itself was fitted on the old curvature and the weight alpha_q it sets, gives the C1 water dimer's measured figures, names refresh_hessian_at_answer / quantize uncertainty as the remedy, and the Laplace summary carries it too.
- Why: Q-86's recommended option. T-57: the plain dimer fits' Hessians were 4.5-40.6 mA old; one refresh moved answers up to 14.4 mA and alpha_q up to 18%, while the warning spoke of widths only and the Laplace summary had none.
- Alternatives considered: refresh the Hessian at the answer in the plain fit, or on drift (user's call: changes answers, costs a Hessian per fit; never auto-picked); a report on the 13 never-passed settings; done for now.
- Source: auto-picked, no answer in 15 min (Q-86)
- Revisit if: the user makes the plain fit refresh its Hessian.
- Outcome (2026-11-23, T-57): hessian_age_warnings() shared by internal_coordinate_posterior and both Laplace summary paths (with and without a covariance); wording says the answer was fitted on that curvature and weight, with the dimer's figures and the remedy; README sentence in the uncertainty section. The existing test extended (both channels, the geometry asked about, the no-covariance path); 7 of 7 mutations caught (t57/mutations.txt); CI 1759 passed. No independent verifier this time: wording and one extra channel, every line mutation-checked.

### D-131 · Next task: why a refreshed fit can stand 0.5 widths off its mode while "settled" (2026-11-23)
- Decision: T-58 measures why the water dimer's seed-3 Cartesian fit (joint HF/6-31G) stands 0.505 posterior widths from its mode after refresh_hessian_at_answer, which reports settled. No code change before the measurement.
- Why: Q-87's recommended option; `quantize uncertainty` quotes widths at that geometry, and the bootstrap's gate is 0.25 widths.
- Alternatives considered: a report on the 13 never-passed settings; refresh the Hessian in the plain fit or on drift (user's call; never auto-picked); done for now.
- Source: auto-picked, no answer in 15 min (Q-87)
- Revisit if: the user answers the decision page.
- Measured (2026-11-23, T-58; scratchpad t58/settled.py, soft.py, report.py; dimer, joint HF/6-31G, after refresh_hessian_at_answer): "settled" checks the Hessian's age and the answer's last move, not the distance to the mode. Seed 3 Cartesian's 0.505 widths lie along one direction, the concerted torsion about the near-linear hydrogen bond (dihedrals 2-1-5-4, 3-1-5-4, 6-4-5-1). There the raw HF Hessian has its one negative physical eigenvalue at every refreshed answer measured (-0.0007 to -0.0017 Eh/A^2), floored to 4.8e-8 by _spd_hessian, so the Gauss-Newton curvature is the data's alone: 0.0077 at seed 3 (1.5-3.0 at the others) against an exact 251.6 (188 at seed 23), the omitted residual term (84 MHz residuals) dominating. In exact curvature the torsion's component is 0.003 widths (the distance is 0.054, a stiffer direction's: T-60); its Gauss-Newton Newton step is 1.36 A rms. The reported widths of angles 2-1-5 and 3-1-5 are 227 deg (seed 3) and 17.6 deg (seed 23), about 1.3-1.6 deg with the exact curvature; the report warns "Gauss-Newton is doubtful along physical direction 0 ... treat that width as order-of-magnitude" (it is 181x and 11x). D-04's revisit condition ("a molecule's reported width is dominated by a flagged direction") is met. Proposal pending (Q-88).

### D-132 · The Gauss-Newton warning says by how much a width is off (2026-11-23)
- Decision: wording only; D-04 stands (Gauss-Newton widths are reported). The "doubtful" warning states by what factor and which way the width along that direction is off, sqrt(exact / Gauss-Newton curvature), instead of "treat that width as order-of-magnitude"; refresh_hessian_at_answer's docstring says "settled" is the Hessian's age, not the distance to the mode.
- Why: Q-88's recommended option. T-58: on the water dimer the flagged widths are 181x (seed 3) and 11x (seed 23) too wide by the exact curvature, more than the order of magnitude the warning allowed.
- Alternatives considered: revisit D-04 (exact-curvature or prior-limited widths along flagged directions; the user's own choice, never auto-picked; its revisit condition is met); a report on the 13 never-passed settings; done for now.
- Source: auto-picked, no answer in 15 min (Q-88)
- Revisit if: the user revisits D-04.
- Outcome (2026-11-23, T-58): _gn_width_factor gives sqrt(exact/Gauss-Newton) and its direction in the doubtful warning (a no-finite-width case kept); refresh_hessian_at_answer's docstring says settled is not the mode. Existing test asserts the factor; a parametrised test covers narrower, wider, the dimer's 181x and the guard. 5 of 6 mutations caught; the sixth was an equivalent mutant (no behaviour change). CI 1763 passed; lint unchanged.

### D-133 · The open-decisions page carries the calls T-56 to T-58 left (2026-11-23)
- Decision: update the open-decisions page (artifact 144eMMauzGEeS3WLmW51Db) with four calls for the user, each with its measured evidence and trade-off: revisit D-04 (Gauss-Newton widths), refresh the Hessian in the plain fit, use_dihedrals auto as the default, multistart adding dihedrals when any start needs them. No code change.
- Why: Q-89's recommended option; four user calls had accumulated in the records only.
- Alternatives considered: a report on the 13 never-passed settings; revisit D-04 now (user's call; never auto-picked); done for now.
- Source: auto-picked, no answer in 15 min (Q-89)
- Revisit if: the user answers any of them.
- Outcome (2026-11-23, T-59): page version 3 (https://claude.ai/artifact/144eMMauzGEeS3WLmW51Db; source scratchpad t46/quantize-open-decisions.html): twelve items. New: 6 multistart dihedrals (recommend yes, with item 2), 9 revisit D-04 (your call). Water-dimer rows added to item 2 (auto dihedrals) and item 4 (Hessian refresh in quantize run); items renumbered; footer sources D-04, D-59 to D-132. Checked: tags balanced, numbering 1-12, no horizontal scroll at 390 px.

### D-134 · Next task: does the bootstrap's gate refuse refits at their exact mode? (2026-11-23)
- Decision: T-60 runs error-model bootstraps of the refreshed water-dimer fits (seeds 3 and 23, Cartesian, joint HF/6-31G) and records each refit's Gauss-Newton distance_to_mode (what the 0.25-width gate reads) and its exact-Hessian distance, and how many refits the gate keeps. No code change before the measurement.
- Why: Q-90's recommended option; T-58 found the Gauss-Newton distance 0.505 where the exact one is 0.003, along the hydrogen-bond torsion.
- Alternatives considered: a report on the 13 never-passed settings; done for now.
- Source: auto-picked, no answer in 15 min (Q-90)
- Revisit if: the user revisits D-04.
- Measured (2026-11-23, T-60; scratchpad t60/gate.py, gate.txt, base_dirs.txt): error-model bootstraps (16 refits, one in-process worker) of the refreshed seed-3 and seed-23 Cartesian fits keep 8 of 16 each, exactly the usable minimum. Gate limits 0.068 and 0.050 (scaled to the cloud). Refused refits read 0.25 to 2.71 Gauss-Newton widths; for every one whose exact-corrected distance was recorded (8 of 8 at seed 3, 4 of 8 at seed 23: the recorder keyed on id(opt), which Python reuses), the exact curvature along the probed directions still puts them 0.18 to 1.39 widths short. So the gate is not refusing refits at their exact mode: half the refits stop short. Correction to T-58: the seed-3 base fit's 0.003 was the torsion's component only; with the exact curvature there the distance is 0.054, a stiffer direction's (docstring, page item 9 corrected; 07ea55c's message stands as pushed).

### D-135 · Next task: why half the dimer refits stop short (2026-11-23)
- Decision: T-61 measures, for refused error-model refits of the refreshed seed-3 dimer fit, the trust radius, how each loop ended, the last steps, the polish passes, and which direction carries the distance. No code change before the measurement.
- Why: Q-91's recommended option; T-60 found 8 of 16 refits refused, 0.18-1.39 widths short with the exact curvature too.
- Alternatives considered: a report on the 13 never-passed settings; done for now.
- Source: auto-picked, no answer in 15 min (Q-91)
- Revisit if: the user answers the decision page.
- Measured (2026-11-23, T-61; scratchpad t61/stopshort.txt, steps.txt; refreshed seed-3 dimer, 16 error-model refits, seed 7): every refused refit ends on the plateau rule (steps under 1 mA, residual flat within 1 MHz over 10 iterations) with the trust radius at 0.78 mA, its 1.6 mA steps refused and 0.78 mA ones accepted. With the exact curvature the remaining distance lies along the stiffest direction (curvature 1.4e7, width 0.27 mA); the Gauss-Newton distance the gate reads lies along the torsion. Cause, measured on the base fit's continued run: the proposed steps point 97.6-100% along the Gauss-Newton softest direction (the H-bond torsion; stiffest share 1e-9 to 1e-6), raw solves of 0.01-0.3 A refused until the trust radius is 0.8-3.1 mA, so the stiff direction barely moves; each run (and each polish pass) restarts at the base trust radius of 0.1 A. With 20 polish passes instead of 3 the exact-corrected distance falls slowly (0.729 -> 0.397, 1.346 -> 0.111, 1.7 -> 0.002, 0.224 -> 0.001) and the gate still keeps 8 of 16: two refits at 0.001-0.002 by the exact Hessian read 0.13 and 0.61 Gauss-Newton widths and are refused. So the Gauss-Newton gap of D-04 also steers the step. Page item 9 updated (v5). Proposal pending (Q-92).

### D-136 · Prototype: the joint step with the exact curvature on flagged directions (scratchpad only) (2026-11-23)
- Decision: T-62 monkeypatches, in a scratch script, the joint step to add the measured omitted curvature (exact minus Gauss-Newton) along the directions the Gauss-Newton check flags, and measures the water dimer's fit and error-model refits and the nine benchmark molecules' joint HF/6-31G Cartesian answers and run time against the code as it stands. Nothing goes into the repository.
- Why: Q-92's recommended option; evidence for decision page item 9 (D-04).
- Alternatives considered: a report on the 13 never-passed settings; done for now.
- Source: auto-picked, no answer in 15 min (Q-92)
- Revisit if: the user decides item 9.
- Measured (2026-11-23, T-62; scratchpad t62/proto.py, dimer.py, bench.py, formyl.py and their .txt; nothing in the repository): the prototype adds (exact - Gauss-Newton) v v^T along flagged directions to the joint step (class-level patch; Cartesian mode). Dimer seed 3: plain fit 32 -> 16 iterations, exact-corrected distance 0.026 -> 0.0045; after refresh 0.054 -> 0.0099 (GN 0.505 -> 0.082); error-model bootstrap keeps 15 of 16 refits (8 of 16 off), exact distances <= 0.017, 66-109 iterations per refit (91-210 off). Nine benchmark molecules, Cartesian joint HF/6-31G from benchmark starts: it acts on acetyl fluoride (10 of 19 steps), formyl fluoride (2/12), fluoroacetylene (8/19); bond RMS unchanged to 0.004 mA on eight; formyl fluoride 12.4701 -> 14.1354 mA, its old answer 1.4 mA out of plane (0.31 new) and above the new one on the real objective (Phi -0.020 at alpha 1803: E_HF 4.1e-5 Eh lower, 1/2|r|^2 0.054 higher); restarts from either answer with the code as it stands land 0.16-0.17 mA from the new one (bond RMS 13.76-13.78), so the old answer was a stopping point. Cost: 2-12% run time (a Gauss-Newton check per step, unoptimised). Internal mode untested. Page item 9 updated (v6).

### D-137 · Prototype in internal mode (scratchpad only) (2026-11-23)
- Decision: T-63 extends T-62's scratch prototype to internal mode, the default: the flagged Cartesian directions' added curvature carried into the q-space step, measured on the water dimer (seed 3, internal, dihedrals auto) and the nine benchmark molecules in internal mode, off against on. Nothing goes into the repository.
- Why: Q-93's recommended option; T-62 covered Cartesian mode only.
- Alternatives considered: a report on the 13 never-passed settings; adopting the step fix (user's call; never auto-picked); done for now.
- Source: auto-picked, no answer in 15 min (Q-93)
- Revisit if: the user decides page item 9.
- Measured (2026-11-23, T-63; scratchpad t63/proto.py, dimer.py, bench.py, compare.py and their .txt; nothing in the repository): internal mode, the correction carried into q-space as D (B+^T v)(B+^T v)^T. Dimer (seed 3, dihedrals auto): plain 36 -> 34 iterations, exact 0.0059 -> 0.0042; after refresh exact 0.024 -> 0.0006 (GN 0.014 -> 0.045); bootstrap 12 -> 14 of 16 (the two refused at exact 0.005, GN 0.55 and 0.75). Nine molecules, internal defaults: six unchanged (no step corrected); formyl fluoride 13.47 -> 13.76 mA (where the Cartesian restarts landed); fluoroacetylene GN 0.008 -> 0.262, bond RMS 15.36 -> 17.14, its answer higher on the real objective (Phi +0.107 at alpha 1639) and 6.2 mA from where restarts land (they land on the old answer); acetyl fluoride (no dihedrals) GN 0.189 -> 0.594, Phi +0.004, both answers stopping points (restarts land 3.6 mA from both). So, as tried, the correction harms the molecules whose bonds and angles miss directions (a near-linear chain, a free torsion). Page item 9 updated (v7).

### D-138 · Prototype refined: internal mode corrects only directions the q-space step reaches (scratchpad only) (2026-11-23)
- Decision: T-64 refines T-63's scratch prototype: in internal mode a flagged direction is corrected only where the q-space step realises at least half of a move along it (v^T B+ B v > 0.5, the reach rule of _resolve_ic_dihedrals), and reruns the internal comparison (dimer and nine molecules), with the real-objective check for any that move. Nothing goes into the repository.
- Why: Q-94's recommended option; T-63's version harmed fluoroacetylene and acetyl fluoride, whose bonds and angles miss directions.
- Alternatives considered: a report on the 13 never-passed settings; adopting the step fix for Cartesian mode (user's call; never auto-picked); done for now.
- Source: auto-picked, no answer in 15 min (Q-94)
- Revisit if: the user decides page item 9.
- Measured (2026-11-23, T-64; scratchpad t64/*.txt; nothing in the repository): with the reach rule (correct in q-space only where v^T B+ B v > 0.5): dimer as T-63 (every flagged direction realised 1.0); fluoroacetylene 0.021 mA from the code as it stands, Phi lower by 2e-5, restarts land on both (T-63's harm gone; 6 of 20 steps corrected, 1 skipped); formyl fluoride 13.76 mA as T-63; six unchanged; acetyl fluoride (dihedrals off) Phi +0.0076, 0.84 mA from off, both stopping points (restarts land 3.6 and 4.3 mA away); acetyl fluoride with use_dihedrals auto: 0.29 mA apart, both at their mode (GN 1.7e-6, 1.7e-7), Phi within 0.0003, bond RMS 14.584 -> 14.579 mA, 11 of 19 steps corrected, run time 26.2 -> 24.7 s. Page item 9 updated (v8).

### D-139 · Independent check of the prototype evidence on page item 9 (2026-11-23)
- Decision: T-65: a fresh background subagent checks the T-62 to T-64 prototype (scratchpad t62/, t63/, t64/): whether the patch does what it claims (the added curvature, the q-space image, the reach rule), whether the comparisons are fair, and whether the figures on page item 9 follow from the outputs. No repository change.
- Why: Q-95's recommended option; L-10; the user's decision on item 9 would rest on these figures.
- Alternatives considered: a report on the 13 never-passed settings; adopting the step fix (user's call; never auto-picked); done for now.
- Source: auto-picked, no answer in 15 min (Q-95)
- Revisit if: the check finds a defect.
- Outcome (2026-11-23, T-65; scratchpad t65v/REPORT.md and outputs): the check held the q-space image, the reach quantity, the correction's size, the owner wiring, the fairness of the comparisons (bit-identical reruns) and the T-61 figures. Defect: the prototype added the correction before _spd_hessian's floor and so also undid the floor along the torsion (Cartesian dimer: 729 of the intended 771 added; step within 6%) and inflated the damping scale (internal dimer: x13 at iteration 0, step changes median 21%, up to 200%). With the correction added exactly (t65v ideal runs): Cartesian results hold (dimer 15/16 kept, refreshed 0.0107; formyl 14.1354); internal dimer 0.0237 -> 0.0107 (not 0.0006), 14/16 kept; internal formyl fluoride 13.47 -> 13.56 mA, restarts 0.40 mA away (not 'reaches them'); fluoroacetylene neutral with the reach rule (0.027 mA), harmed without (4.1 mA); acetyl fluoride (dihedrals off) Phi +0.001 between stopping points. Overstated: cost '2-12%' (-3 to +15% per fit, the dimer bootstrap +50%, concurrent runs timed to whole seconds); 'the two whose bonds and angles miss directions' (fluoroacetylene reaches 5 of 5 at its start; its harm came through one probe realised 0.07); 'objective within 0.0003' (+0.00034, on higher); geometries moved up to 0.64 mA where bond RMS moved < 0.004; on the internal dimer the Gauss-Newton distance gets worse (0.014 -> 0.020) while the exact one improves. Page item 9 rewritten from the check's figures (v9).

### D-140 · Report on the 13 never-passed MolecularOptimizer settings (2026-11-23)
- Decision: T-66 reports, for each MolecularOptimizer parameter nothing outside quantize.py passes (D-124's list), whether any entry path (API, config runner, CLI, scripts) can reach it, whether a test covers it, what it does, and whether it is worth keeping. Nothing deleted, no behaviour change.
- Why: Q-96's recommended option, long pending.
- Alternatives considered: more dimer starts for the prototype; adopting the step fix (user's call; never auto-picked); done for now.
- Source: auto-picked, no answer in 15 min (Q-96)
- Revisit if: the user wants any removed or exposed.
- Outcome (2026-11-23, T-66; scratchpad t66/REPORT.md, never.txt, guard.txt): still 13 of 144. Keep 11 (isotropic_prior's negative result is documented on purpose; the Jacobian switches and chi2/rm1 knobs are tested at the engine level or at their defaults). null_convergence_requires_energy: candidate for removal (never set, untested). The geometry guardrails, whose six tunables and accept_requires_geometry_valid are in the list, are ON in the config runner, presets and run_water.py but have no test at all; on the nine benchmark fits they refuse 0 steps and change nothing (bit-identical). Backlog N35: a guardrail test. Nothing deleted.

### D-141 · A test of the geometry guardrails (2026-11-23)
- Decision: T-67 adds tests (dev/tests) for the geometry guardrails: the validity check's three rules (bond too short, too long, non-bonded clash) and its violation allowance, off by default (nothing refused), and in the fit loop a refused step with damping raised and trust radius shrunk by the configured factors, and accept_requires_geometry_valid. Test-only, no behaviour change, mutation-checked.
- Why: Q-97's recommended option; backlog N35: on in the config runner, no test.
- Alternatives considered: the corrected trial fix on more dimer starts; removing null_convergence_requires_energy (user's call: an API change); done for now.
- Source: auto-picked, no answer in 15 min (Q-97)
- Revisit if: a test shows the guardrails misbehave (then it is a fix, and asked).
- Outcome (2026-11-23, T-67): dev/tests/test_geometry_guardrails.py, 9 tests (each rule and its scale, the violation allowance, off by default, bonds from the start only, a refused trial in the loop with the boost and shrink against factors of 1 and of 3/0.5, accept_requires_geometry_valid both ways); 13 of 13 mutations caught (t67/mutations.txt); CI 1772 passed; lint clean. No behaviour change. Backlog N35 done.

### D-142 · The exact-correction trial on seven more dimer starts (scratchpad only) (2026-11-23)
- Decision: T-68 runs the reviewer's exact-correction prototypes (t65v/ideal, t65v/ideal63) on the other seven no-point-group water-dimer starts (seeds 4, 5, 6, 8, 11, 12, 23), Cartesian and internal with dihedrals auto, off against on: iterations, exact-corrected distance after the refresh, bootstrap kept. Nothing goes into the repository.
- Why: Q-98's recommended option; page item 9 rested on seed 3 alone.
- Alternatives considered: removing null_convergence_requires_energy (user's call: an API change); adopting the step fix (user's call); done for now.
- Source: auto-picked, no answer in 15 min (Q-98)
- Revisit if: the user decides page item 9.
- Measured (2026-11-23, T-68; scratchpad t68/*.txt, spread.txt; prototype t65v/ideal/proto.py: exact correction, reach rule): seven no-point-group dimer starts (4, 5, 6, 8, 11, 12, 23). Cartesian: plain-fit iterations 24-69 -> 13-54; exact-corrected distance after the refresh 0.016-0.062 -> 0.002-0.013; bootstrap 8/16 on every start -> 13-15/16; truth distance 13.21-13.62 -> 13.13-13.16 mA. Internal (dihedrals auto): iterations 33-106 -> 13-106; exact 0.007-0.066 -> 0.0001-0.052; bootstrap 8-13 -> 14/16 on every start. Start-to-start answer spread: Cartesian 0.12-3.00 mA rms (median 0.90) -> 0.03-1.06 (0.48); internal 0.10-0.96 (0.45) -> 0.03-0.77 (0.38); Cartesian vs internal from one start 0.41-1.51 -> 0.03-0.93. The Gauss-Newton distance (reported, gated) is larger with the fix on most starts (up to 0.534). Page item 9 updated (v10).

### D-143 · Measure the trial fix's real cost (scratchpad only) (2026-11-23)
- Decision: T-69 times the exact-correction prototype (t65v/ideal/proto.py) against the code as it stands, sequentially and repeated, on the nine benchmark molecules and the water dimer, plus a variant that runs the Gauss-Newton check once per quantum Hessian, with that variant's effect on the answers. Nothing goes into the repository.
- Why: Q-99's recommended option; cost is the open trade-off on page item 9 and the earlier figures came from concurrent runs timed to whole seconds.
- Alternatives considered: adopting the step fix (user's call); removing null_convergence_requires_energy (user's call); done for now.
- Source: auto-picked, no answer in 15 min (Q-99)
- Revisit if: the user decides page item 9.
- Measured (2026-11-23, T-69; scratchpad t69/timing.txt, timing_cached.txt, summary.txt; sequential, median of 3, internal defaults plus dimer seed 4 in both modes): fits whose answers are bit-identical across variants (vinyl fluoride, fluoroethane, chlorofluoromethane, water, ozone, isocyanic acid) vary 10-20% in wall time between runs, so differences below that are noise. Every-step check: acetyl fluoride 13.2 -> 16.2 s (+22%), formyl fluoride 3.35 -> 3.93 s (+17%), fluoroacetylene 7.7 -> 7.3 s; nine molecules 59.1 -> 60.8 s in all; dimer fits 2.3-2.4 -> 1.9-2.2 s (fewer iterations); dimer bootstraps 10.2 -> 12.7 s (internal, kept 13 -> 14) and 4.9 -> 7.1 s (Cartesian, kept 8 -> 15). Once-per-Hessian check (rows cached until the Hessian changes; 41 checks, 2721 reuses): dimer answers 7.1-7.4 mA from the every-step ones, the Cartesian dimer bootstrap keeps 0 of 16, fluoroacetylene moves 1.4 mA: not viable (directions go stale as the fit moves). The first attempt at the cached run failed on a wrong module name and was rerun alone. Page item 9 cost sentence updated (v11).

### D-144 · Fact-check of page item 9's T-68 and T-69 rows (2026-11-23)
- Decision: T-70: a fresh background subagent checks the "Seven more starts" row and the cost sentence of "A trial fix" on page item 9 against the outputs and scripts in scratchpad t68/ and t69/, reporting any figure that does not follow. No repository change.
- Why: Q-100's recommended option; L-10/L-18: the T-65 check found five overstated figures.
- Alternatives considered: adopting the step fix (user's call); removing null_convergence_requires_energy (user's call); done for now.
- Source: auto-picked, no answer in 15 min (Q-100)
- Revisit if: the check finds an error (then the page is corrected).
- Outcome (T-70): the checker confirmed the distances, kept counts, GN 'up to 0.53' and the +17-22%; found the +45% was the Cartesian dimer under an 'internal mode' label, the bootstrap timed once, fluoroacetylene (-5%) left out, the 10-20% noise band taken against a later run (adjacent runs -9% to +2%, water +62%), the spread quoted per component without saying so (per atom: Cartesian 0.21-5.2 -> 0.05-1.8 mA), 8 of 42 pairs moving apart, and every corrected refit within 0.019 exact widths (refusals on the GN reading only). Main session reproduced each (t70v/REPORT.md); page item 9 rewritten (version 12).

### D-145 · Exact curvature where Gauss-Newton is flagged: in the step and in the report (2026-11-23)
- Decision: T-71. Along the probed soft directions the Gauss-Newton check flags (finite difference stable, no sign flip, discrepancy over 50%, exact curvature above Gauss-Newton), the joint step adds the difference to its matrix after the floor and the damping (internal mode: the q-space image B+^T v, only where the step reaches the direction, v^T B+ B v > 0.5), and the reported widths, the distance to the mode and the bootstrap gate use the exact curvature there, while the geometry stands within 2 widths of its mode. The check itself keeps comparing against Gauss-Newton. Supersedes D-04.
- Why: water dimer (T-58 to T-70): Gauss-Newton kept 0.0077 of an exact 252 along the H-bond torsion; widths 181 and 11 times too wide; refits refused on the Gauss-Newton reading only; with the step fix the answers settle (exact distance 0.002-0.013, kept 13-15 of 16).
- Alternatives considered: the step fix only (widths stay Gauss-Newton); leave it open.
- Source: recommended and approved ("okay go ahead", after "My recommendation is both")
- Revisit if: a flagged direction with the exact curvature below Gauss-Newton turns up (kept on Gauss-Newton with the existing warning), or the cost per fit exceeds the measured -5% to +22%.
- Refined (T-71): the report uses the exact curvature either way along a flagged direction (the exact one smaller widens the width); the step only where it is the larger. Found on the dimer, internal seed 4: 64 by Gauss-Newton against an exact 31 along direction 1, which kept that start's widest angle at 6.77 deg against 9.58 with it. This is the user's 'the reported uncertainty also uses the true stiffness there'.
- Measured (T-71, new code, 8 starts x 2 modes): bootstrap keeps 16 of 16 on all 16 runs (8-13 before either change); distance after the refresh 0.002-0.013 (Cartesian), 0.00006-0.052 (internal); widest angle 6.6-9.6 deg against Gauss-Newton's 13.5-331 at the same answers; Gauss-Newton keeps 0.002-2.2% of the exact torsion curvature (~170). Seed 4 plain and refreshed answers bit-identical to the T-68 prototype.

### D-146 · quantize run refreshes the Hessian at its answer (2026-11-23)
- Decision: T-72. `quantize run` takes one more quantum Hessian at the converged geometry and re-converges on it (refresh_hessian_at_answer), as `quantize uncertainty` does; `--no-hessian-refresh` skips it. Page item 4, backlog N23, revisits D-63.
- Why: a plain run stops on the Hessian from its start: 0.03-0.6 mA RMS from its own optimum on formyl fluoride and chlorofluoromethane, a 0.5 mA thread-count difference; the water dimer's answers moved up to 14.4 mA and internal/Cartesian answers 23.5 -> 2.6 mA apart.
- Alternatives considered: keep the age note only (today).
- Source: stated by user ("1-4 go", item 1 of the list: refresh the quantum Hessian at the end of a plain quantize run fit)
- Revisit if: the cost (+15-21% wall measured) grows on larger molecules.

### D-147 · use_dihedrals: auto is the default (2026-11-23)
- Decision: T-72. Internal-coordinate fits add dihedrals when bonds and angles miss a direction the fit can move (D-100's auto) unless the configuration says otherwise. Page item 2.
- Why: acetyl fluoride 0.19 -> 1.7e-6 widths from its optimum, fluoroethane 0.16 -> 1.6e-7, bootstraps 0 -> 42 and 4 -> 50 of 50; 7 of 9 unchanged; the water dimer 0.07-4.0 -> 0.003-0.08 widths.
- Alternatives considered: keep false.
- Source: stated by user ("1-4 go", item 2)
- Revisit if: an auto-added dihedral set makes a fit worse than bonds and angles alone.

### D-148 · Multistart adds dihedrals when any start needs them (2026-11-23)
- Decision: T-72. With use_dihedrals: auto, multistart checks every start and gives all starts dihedrals when any one's bonds and angles miss a direction dihedrals reach: still one coordinate set for every start (D-101's reason). Page item 6, backlog N30.
- Why: a dimer with a Cs start first and a no-point-group start second ran both without dihedrals; the second ended 0.44 widths from its optimum.
- Alternatives considered: first start decides (today).
- Source: stated by user ("1-4 go", item 3)
- Revisit if: starts that did not need dihedrals move materially.

### D-149 · Remove null_convergence_requires_energy (2026-11-23)
- Decision: T-72. Delete the MolecularOptimizer option (never set by any caller, untested, T-54/T-66) and its branch.
- Why: dead option on the public constructor.
- Alternatives considered: keep it.
- Source: stated by user ("1-4 go", item 4)
- Revisit if: a caller outside the repository used it.
- Measured (T-71/T-72, HF/6-31G benchmark rows, Cartesian, base and offsets): 15 of 18 rows identical, acetyl fluoride and fluoroethane base +/-0.003 mA, isocyanic acid +0.26 (base) and +0.45 mA (offsets) against its r_s reference, landing on another stationary point 1.4-2.0 mA away whose objective is higher by 0.044-0.050 in log p (t71/chk/hnco_check.json). Reported to the user as a cost.
- Timed (T-71, interleaved on/off, median of three, current defaults incl. dihedrals auto; t71/timing.txt): where it acts -4.3% (acetyl fluoride), +5.0% (formyl fluoride, 21 iterations against 14), -3.5% (fluoroacetylene); where it does not +0.1% to +5.0% wall (CPU +1.7% to +6.1%), the per-step check; dimer fit -8.3% (internal) / +0.6% (Cartesian), bootstrap +63% (internal, kept 16 vs 13) / +1% (Cartesian, 16 vs 8). Answers bit-identical on vs off where it does not act, and identical to the T-69 prototype runs on every molecule the dihedral default leaves alone.
- Reviewed (T-72, fresh subagent, t72v/REPORT.md): seven findings, all fixed except the seeding-rank one (backlog N39). The bootstrap's error-model and rto targets now also leave out cov K cov (no refit spreads by K); the distance beyond 2 exact widths is the larger of the two readings (it read 0.21 for an exact 2.1 and passed the gate); the note says whether the step takes the exact curvature (split, spectral-only, unreached internal directions do not); relevance also judged on the reported variance; a refresh that fails after moving reports where the fit stands; quantize uncertainty decides dihedrals over every start; stale comments. Tests strengthened so nine mutations (the four that survived the review among them) are each caught. CI 1806 passed.
- Re-measured after the review fixes (T-72, t72/dimer): kept 14-16 of 16 (Cartesian 16,16,16,14,15,16,14,15; internal 16,16,15,16,16,15,15,15), distances and widths unchanged. The corrected error-model target is narrower along the torsion, so the gate's scale is 0.046 (limit 0.0115 widths); Cartesian seed 6 refuses two refits at 0.0138 and 0.0144, 0.30-0.31 cloud widths short. The 16/16 figures of T-71 came from the too-wide target.

### D-150 · Measure the ratio test judging the corrected step on its own model (2026-11-23)
- Decision: T-73. In a scratch copy only, make _joint_step_ratio's predicted reduction include the step's curvature term (pred -= 1/2 dp^T K dp, K the term the step added), and measure whether isocyanic acid's RHF rows return to the lower stationary point (off: 23.98 / 28.56 mA; on: 24.23 / 29.00) while the dimer (8 starts x 2 modes) and the other benchmark rows stay where D-145 put them. No repository change.
- Why: Q-102's recommended option; the ratio test still judges the corrected step on the Gauss-Newton model (D-145 docstring), the one inconsistency left between the step and its acceptance.
- Alternatives considered: reuse the step's Jacobian in the check (user's call: efficiency change); decide more page items (user's call); done for now.
- Source: auto-picked, no answer in 15 min (Q-102)
- Revisit if: it moves the dimer or other rows, or does not change isocyanic acid.
- Outcome (T-73, scratch t73/): the patched ratio test fired on the corrected steps everywhere (isocyanic acid 2 of 2, dimer 13-97 per fit) and flipped no acceptance: all 18 RHF benchmark rows identical to 4 decimals (isocyanic acid 24.2325 / 29.0037), the dimer answers, iteration counts and kept counts bit-identical on every start measured. Isocyanic acid's landing comes from the step's direction, not its acceptance; the patch is inert where measured, so it is not proposed. Backlog N36 keeps the question open.

### D-151 · Where the exact-curvature check's per-step time goes (2026-11-23)
- Decision: T-74. Profile, in scratch only, the per-step check (_step_curvature_correction -> gauss_newton_discrepancy -> _posterior_blocks, finite-difference residuals, eigen-decompositions) on HF/6-31G fits where it acts and where it does not, and estimate what a bit-identical reuse of the step's own quantities would save. No repository change.
- Why: Q-103's recommended option; T-71 measured +0.1% to +5% per fit from the check, a cost on every joint fit.
- Alternatives considered: decide more page items (user's call); rebuild the results table (user: wait); done for now.
- Source: auto-picked, no answer in 15 min (Q-103)
- Revisit if: the check is under 1% of a fit (nothing to save), or a saving needs a non-identical change (then it is the user's call).
- Outcome (T-74, t74/profile.json): profiled, the check is 0.3-2.2% of a benchmark molecule's fit (chlorofluoromethane 0.3%, acetyl fluoride 0.4%, vinyl fluoride 1.1%, ozone 1.8%, formyl fluoride 2.2%) and 4.0-4.1% of the dimer's; of it, finite-difference residuals ~65-75%, the spectral Jacobian inside _posterior_blocks ~15-20%, the physical basis ~0. Reusing the step's own Jacobian (the only bit-identical reuse) would save 0.01-0.02 s per fit, under 1% of the dimer's and 0.3% elsewhere: not worth the code, not proposed. Correction to T-71: its +0.1% to +5% wall differences where the correction does not act are mostly not the check (chlorofluoromethane +4.3% against a 0.3% check); I had named the check as their cause without measuring it (L-27). Cause of the rest not measured; T-71 ran 'on' first in every pair.

### D-152 · Focus on the four accuracy items, validation molecules first (2026-11-26)
- Decision: answer to Q-104. Work items 1-4 of the production list in turn: (1) the Cartesian internal-prior angle-unit bug (page item 1, N29); (2) more validation molecules with true equilibrium structures, then a results-table rebuild; (3) molecules the data cannot pin down (N16); (4) the open defaults (page items 3, 5, 7). First task: find more validation molecules (T-75).
- Why: the user, "We should focus on 1-4 ... can we find more validation molecules".
- Alternatives considered: packaging, backend CI, scaling, user guide (items 5-9), left for later.
- Source: stated by user
- Revisit if: the user reorders.

### D-153 · Validation-molecule scope and reference bar (2026-11-26)
- Decision: Q-105 option 1: any semirigid molecule of 12 atoms or fewer with a dipole (fluorine and non-fluorine). Q-106 option 1: keep RE_REFERENCE_MAX_UNCERTAINTY_MA = 0.5 mA on bonds for headline scoring; candidates within 1 mA listed as a second tier for the user.
- Why: the most candidates without weakening the bar the code already enforces.
- Alternatives considered: fluorine only; accept 1 mA now; accept SE100 without stated uncertainty.
- Source: auto-picked, no answer in 15 min (Q-105, Q-106)
- Revisit if: the user answers otherwise; the fit sigmas prove to understate real error (N3 search: HNCO's two structures differ 1.0-2.9 mA beyond their sigmas).

### D-154 · Six validation molecules, as a separate validation set (2026-11-26)
- Decision: Q-107 option 1: formaldehyde, thioformaldehyde, ketene, SO2, thiophene, pyrimidine. Q-108 option 1: a separate validation set, scored with bond-class offsets pooled from the benchmark molecules only; the nine benchmark rows unchanged.
- Why: open data for all six, 3-10 atoms; a held-out test leaves the published table as it is.
- Alternatives considered: all eight tier-1 (adds thiazole, pyridine); the four smallest only; merging into the benchmark pool.
- Source: auto-picked, no answer in 15 min (Q-107, Q-108; the wake-up fired at 00:57, two minutes before the stated 00:59)
- Revisit if: the user answers otherwise.

### D-155 · Reference uncertainties are compared at 1-sigma; SO2 uses its three species with stated sigmas (2026-11-26)
- Decision: RE_REFERENCE_MAX_UNCERTAINTY_MA (0.5 mA) applies to the 1-sigma bond uncertainty; a source quoting 2-sigma is halved and the entry says so (ketene: 0.69 -> 0.345 mA). SO2 is wired with 32S, 33S, 34S16O2 (fitted constants with sigmas); the Kivelson-Wilson species without sigmas are recorded in the draft but not scored.
- Why: the existing references (ozone, vinyl fluoride, CH2ClF) quote 1-sigma; mixing constant types with and without sigmas would weight unknown-quality values by their printed digits.
- Alternatives considered: compare at 2-sigma (ketene fails the bar); include all 19 SO2 species.
- Source: assumption, unconfirmed
- Revisit if: the user prefers 2-sigma, or more SO2 species are needed to locate O.

### D-156 · Keep the module's 18O mass for the validation entries (2026-11-26)
- Decision: the validation entries use dev/monofluoro_references.py's M_O18 = 17.9991610 u, 1.39e-6 u above NIST's 17.99915961286; noted in the new module.
- Why: it moves 18O species' constants by ~3.5e-8 relative (H2C18O B 0.0012 MHz), against fit sigmas floored at 0.5-1% model error (hundreds of MHz); updating the shared constant would change existing benchmark rows for no measurable gain. Both verifiers (T-75 N9a, N9b) found it and nothing else.
- Alternatives considered: update M_O18 everywhere (changes archived rows at ~1e-6 mA); a separate NIST value for new entries only (two masses for one isotope).
- Source: assumption, unconfirmed
- Revisit if: a fit ever weights 18O constants at their measurement sigma.

### D-157 · Validation-set wiring: scored= flag, whole-pool offsets, cross-ring distances are not bonds (2026-11-26)
- Decision: the benchmark scores the validation set with `scored=validation` into accuracy_upgrades_<level>[_calscored]_validation.json, with offsets from the whole pool (benchmark nine plus calibration), never leave-one-out on itself and never pooled; a run refuses if a validation key reaches the pool. Pyrimidine's cross-ring C2...C5 parameter is kept out of `bonds` (it would be scored as a bond and pooled as a C-C offset) and checked on the geometry in the tests; its 2-sigma 0.45 mA is read as 0.225 at 1-sigma (D-155).
- Why: D-154's separate set; `set=` is already read at import by scripts/monofluoro_benchmark.py.
- Alternatives considered: `set=validation` (collides); scoring C2...C5 as a bond.
- Source: assumption, unconfirmed
- Revisit if: a validation molecule should also calibrate (then it leaves the set).

### D-158 · Thiophene is scored on the 2021 S-reduced constants of the species used in the 2021 structure (2026-11-26)
- Decision: structure from Esselman et al. 2025 Table IV (46-isotopologue r_e^SE, 2-sigma 0.3 mA, read as 0.15 at 1-sigma); constants from Orr et al. 2021 Table II (S reduction, printed uncertainties), the 24 species that paper used, not the 2025 Table II constants (49 species, determinable A/S averages, no uncertainties printed).
- Why: D-155's rule that scored constants carry stated sigmas; the 2021 set is the same group's fitted constants; 24 species for a 9-atom molecule is ample.
- Alternatives considered: the 2025 49-species set floored at its printed digits; both sets.
- Source: assumption, unconfirmed
- Revisit if: the 2025 SI's S-reduced constants with sigmas are obtained, or the user prefers the larger set.

### D-159 · Fix the Cartesian internal prior's angle units (2026-11-26)
- Decision: InternalPriorEngine scales its angle and dihedral Wilson B rows by 180/pi so each row's Jacobian is in the units of its residual and sigma (degrees), and wraps dihedral residuals to [-180, 180). Supersedes D-96 (documented, not fixed).
- Why: the user put it first in the focus list (D-152, item 1; page item 1). Measured at HEAD (T-76, the T-39 harness): finite differences confirm the rows were 57.2958x small; the fix equals the T-39 prototype to 4.5e-11 on all 18 fits; rejections 58 -> 3 (autoconfig off), angle widths none / up to 64 deg -> 0.3-2.1 deg; accuracy both ways (bonds 24.16 -> 23.00 mA, angles 1.578 -> 1.922 deg; fluoroacetylene keeps its start's bend, 1.12 -> 3.86 deg, because the start-anchored prior now holds it at its stated 2 deg).
- Alternatives considered: leave it (D-96); also re-tune the presets' 2-deg angle sigma (a separate calibration, not done).
- Source: stated by user (focus item 1, D-152); the accuracy trade was on the page the user answered
- Revisit if: the start-anchored angle prior's 2-deg sigma should be calibrated (fluoroacetylene).


### D-160 · The offsets prior holds each equivalent bond and angle separately (2026-11-26)
- Decision: offset_corrected_geometry targets every symmetry-equivalent pair (and angle triple) at its own theory value minus the class offset, weighted 1/sqrt(n) per name, instead of the name's average; benchmark rows record bond_pair_spread_ma (largest difference between equivalent bonds) for theory, mixed estimation and hybrid.
- Why: found by the T-77 width-coverage check (|z| up to 24 on pyrimidine's offsets fit). The averaged targets left antisymmetric distortions to the 1e-4 tether, and on pyrimidine and thiophene (four equivalent pairs) the offsets prior came back 96-217 mA from symmetric while every averaged bond sat on target; the averaged score (5-6 mA) hid it. Measured (t78/compare.py): rings symmetric to the theory's own 0.003-0.013 mA; the nine benchmark molecules' priors move <= 0.0015 mA and the other four validation molecules' <= 0.0005, so no published row changes. A ring cannot take unequal class offsets with its angles held, so ring bonds now take 44-68% of their offsets and angles move <= 0.74 deg (the old code took 100% by breaking the ring).
- Alternatives considered: guard only (refuse a broken prior, keep the builder); impose the theory's symmetry after the solve; score per pair instead of averaged (changes every row's convention).
- Source: assumption (defect fix inside the validation work; no published number moves), unconfirmed
- Revisit if: ring offsets should be taken in full (would need the angle terms to give way, a calibration question), or the scores should move to per-pair.

### D-161 · Mixed estimation keeps the molecule's symmetry (2026-11-26)
- Decision: mixed_estimation_fit keeps its symmetry-averaged predicates and adds equality terms (weight 1e4 x the predicate's 1/sigma) holding each equivalent bond and angle at its theory offset from its name's mean -- symmetry imposed, as fitting symmetry-unique parameters does.
- Why: T-79 (found by the T-78 review): the averaged predicates left antisymmetric distortions to the rotational constants, and the baseline broke symmetry where the isotopologues cannot see it -- chlorofluoromethane's C-H pair 964 mA apart (averaged score 4.26 mA, offsets, the README's 'ME beats the hybrid' there), fluoroethane 63-67, acetyl fluoride 8-17, SO2 591, pyrimidine 79. HF nine (t79/out/me_hf_6-31g_benchmark_all.final.json): CH2ClF 31.38 -> 26.00 / 4.26 -> 6.03, fluoroethane 12.66 -> 11.28 / 8.24 -> 9.07, acetyl within 0.02, six unchanged. A synthetic CH2ClF case with constants from the reference structure breaks the old code too (383 mA; dev/tests/test_mixed_estimation_symmetry.py). The new code reproduces the study's 'sym' variant (t79/me_hf_bench_verify.log).
- Alternatives considered: soft per-pair predicates (D-160 recipe; equals sym except fluoroethane base 12.00, spread 7 mA); keep the baseline and report the spread beside it.
- Source: auto-picked, no answer in 15 min (Q-160, recommended option)
- Revisit if: the user prefers the soft variant or the unchanged baseline.

### D-162 · README: correction note now, table rebuild stays on hold (2026-11-26)
- Decision: withdraw the 'ME beats the hybrid on chlorofluoromethane' sentence and add a dated correction note with the measured HF changes; the tables themselves keep the old baseline until the rebuild the user put on hold ("5 wait on").
- Why: the claim rested on a broken structure; a note fixes the false statement without rebuilding tables the user asked to hold.
- Alternatives considered: leave the README until the rebuild.
- Source: auto-picked, no answer in 15 min (Q-161, recommended option)
- Revisit if: the user wants the note removed or the tables rebuilt now.

### D-163 · Cartesian fallback on by default (2026-11-26)
- Decision: ic_cartesian_fallback / internal_coordinates.cartesian_fallback default true (decision page item 3); false restores D-106's opt-in.
- Why: the user said yes ("Yes on all suggestions", page items 3, 5, 7). Re-measured at the new defaults (T-81, t81/defaults.py; nine molecules, HF/6-31G, joint, internal, dihedrals auto): acts on fluoroacetylene (9 fallback steps; 3.5e-3 -> 6.9e-7 widths from its optimum, bond RMS 15.3955 -> 15.3849 mA) and formyl fluoride (8 steps; answer 0.50 mA from the fallback-off one without a refresh; with the CLI's Hessian refresh 0.018 mA from the Cartesian fit against 0.072 without, Cartesian restarts differing by 0.053: t81/cart_check.py); the other seven bit-identical, wall time unchanged. The page's 'other seven bit-identical' (T-47) predates formyl fluoride's change.
- Alternatives considered: keep opt-in.
- Source: stated by user
- Revisit if: a fit the fallback acts on lands further from the Cartesian answer than without it.

### D-164 · DFT fits take the gradient with the grid response (2026-11-26)
- Decision: pyscf_hf's run_gradient and run_hessian's gradient include the grid-weight response for DFT (decision page item 5); supersedes D-59's flag.
- Why: the user said yes. Measured: on displaced water at B3LYP/sto-3g the gradient matches a central finite difference of the energy to 3.7e-7 Ha/bohr against 4.8e-6 without (6-31G 4.2e-7 / 3.0e-6); dev/tests/test_dft_fit_gradient.py fails on the old code. T-19's CLI figures stand (answers <= 0.14 mA, formyl fluoride no slower, fluoroethane ~4%); not re-timed. Benchmark rows unaffected (target prior). The Hessian itself still has no grid response (D-55).
- Source: stated by user

### D-165 · Model error in reported widths by default (2026-11-26)
- Decision: auto_systematic_sigma / model_error.auto default true (decision page item 7): a species with no stated systematic sigma gets the flat estimate (1% A, 0.5% B and C); model_error: {auto: false} for measurement precision only. The config validator treats an absent auto as on.
- Why: the user said yes. Measured: analytic water raw B0 O-H width 0.20 -> 1.76 mA, angle 0.03 -> 0.32 deg, geometry bit-identical; the nine benchmark molecules' fits unchanged (their inputs state a model error). Found while making it the default and fixed with it: (a) every corrected run warned 'the estimated model error is not used' -- an estimate a correction takes the place of is no longer marked replaced (a stated one still is); (b) an indefinite Laplace covariance was no longer refused once the systematic block was added (29214 A widths on the guard test) -- geometry_uncertainty and internal_coordinate_posterior now check the Laplace part before adding it; (c) test_geometry_uncertainty_agrees_on_scale's bond filter looked for the word 'angle', which no key holds, so it compared degrees with Angstrom.
- Source: stated by user
- Revisit if: a molecule class shows the flat estimate badly off (N42: rings at HF are a different cause).
- Review (2026-11-26, independent check of 122cc0f, scratchpad review81/): two defects, both fixed in the follow-up commit. (1) The engine wrote its fills into the caller's isotopologue dicts, so the runner's dihedral probe made the real fit see the estimate as stated: every default corrected CLI run warned 'the stated model error is not used' -- fills now work on copies. (2) The geometry claim was false: compute_corrected_targets took the estimate out of a quoted sigma on vibrationally corrected rows (0.98 mA on water with a 2000 MHz quoted sigma) -- an estimate beside a user-quoted sigma is now marked sigma_systematic_estimated and not taken out (where sigma_constants is filled from the same estimate it is, as before). Also: sigma_includes_systematic without a stated split is refused again unless auto: true is explicit (the estimate exceeded a 0.2 MHz sigma_mhz), and the engine skips the default fill for such a species (auto_systematic_sigma=None is the default, True/False explicit); stale opt-in comments rewritten; _indefinite_ratio refuses a matrix with no positive eigenvalue. Remaining measured difference: a fit with no quoted sigma whose chi2 rescale would tighten it stops at the estimate's floor (1.5e-5 A). New tests fail on 122cc0f. CI 1868 passed.

### D-166 · Next: measure fixes for N41 and N42 (2026-11-26)
- Decision: T-82 measures, in scratch only, remedies for N41 (offsets ignore bond order: pyrimidine's aromatic C-N takes HNCO/HCN's multiple-bond offset at HF) and N42 (ring widths too narrow at HF), and proposes before changing anything; the B3LYP validation ring rows go into the README when they finish.
- Why: Q-166's recommended option.
- Source: auto-picked, no answer in 15 min (Q-166)
- Revisit if: the user picks the table rebuild or ends the loop.

### D-167 · While B3LYP runs: the HF width-factor scan (2026-11-26)
- Decision: run the nine at HF/6-31G with sigma_x (and the offsets configs' per-molecule widths) x1.5 and x3, scratch only (t82/n42_nine.py f1.5, f3), so page item 13 has x1 / x1.5 / x2 / x3 at HF. Measurement only; no default changes (item 13 stays the user's call).
- Why: Q-167's recommended option.
- Source: auto-picked, no answer in 15 min (Q-167)
- Revisit if: the user picks N40 or ends the loop.

### D-168 · Next after T-82: N40, straight-line angles (2026-11-26)
- Decision: take backlog N40 as T-83: measure in scratch, fix, independent verify, report any moved rows before the README.
- Why: Q-168's recommended option; N40 is a confirmed defect in what every report says about a 180-degree angle.
- Source: auto-picked, no answer in 15 min (Q-168, 14:53 UTC).
- Revisit if: the user picks another option.

