# Current task
Task: T-06
Status: done
Request: per-bond prior prototype (diagnostic, no engine change) -- loosen the
quantum prior only on bond classes the calibration does not cover, score
CH2ClF and vinyl fluoride against their r_e^SE, water and ozone against r_e.

## Brief
- Outcome: does widening the prior along uncalibrated bonds fix the
  underdetermined split (N16) without costing the determined molecules?
- Approach: scratchpad perbond/perbond2.py. Benchmark base recipe (same
  caches as the 5cfc5d4 worktree). For each uncalibrated class
  (bond_offsets.leave_one_out_offsets) add s^2 = uncorrected_class_bias^2
  of variance to its bonds' marginal prior covariance, shared within the
  class, everything conditional on the bonds unchanged:
  Sigma' = Sigma + Sigma B M^-1 S M^-1 B^T Sigma, M = B^T Sigma B.
  Applied to the rigid-projected Hessian the fit reads
  (_project_quantum_terms, on the instance), alpha_q pinned to the
  unwidened Hessian's value (lookup; an unknown Hessian raises).
- Check: the engine's own accessors (_projected_reference_terms,
  quantum_prior_weight) must show added variance / s^2 = 1 on each widened
  bond, ~0 elsewhere.
- Tradeoff: diagnostic only; s is estimated from the other bond classes.

## Graph
- N1 · build · per-bond widening on the base recipe (perbond2.py) · after: none · done · handoff: mechanism exact; mixed result
- N2 · build · same with offsets on (perbond3.py) · after: N1 · done · handoff: CH2ClF fixed at both levels, water/ozone flat
- N3 · verify · HNCO, the only other molecule it changes · after: N2 · done · handoff: worse at both levels; verdict not adoptable

## Progress
- First version (perbond.py) widened the raw Hessian and rescaled alpha by
  a raw-matrix lambda_bar ratio: check failed (2.15 / 2.27 / 1.11). A probe
  showed the fit reads the rigid-projected Hessian (lambda_bar 3.05 vs 1.52
  raw, both Hartree/A^2), so the raw-space update and the ratio were both
  wrong. Its numbers were discarded.
- perbond2.py: check passes everywhere (1.0 on the widened bond; C-F 0.01-
  0.02, C-H 0.0005 through prior correlation); alpha unchanged to 1e-5.
- Results, bond RMS vs r_e (mA), base -> per-bond [data only, from
  balance/]: RHF (s_x 20 mA): water 0.56 -> 0.63; ozone 3.13 -> 2.75;
  CH2ClF 10.80 -> 12.42 [8.56] (C-Cl +7.63 -> -7.57, C-F -0.50 -> +13.39);
  vinyl fluoride 11.39, unchanged (no uncalibrated class: C-F, C-H, C=C are
  all calibrated). B3LYP (s_x 7 mA): water 0.05 -> 0.81; ozone 0.06 ->
  0.43; CH2ClF 9.24 -> 5.21 [3.48] (C-Cl +10.03 -> +3.33, C-F -11.13 ->
  -5.39); vinyl fluoride 3.36 unchanged. Uniform x20 loosening (balance/):
  CH2ClF 8.68 RHF, 7.51 B3LYP.
- s underestimates the one class that needs it: C-Cl's theory error is
  73 mA (RHF) / 38 mA (B3LYP) against s = 21.2 / 6.6. Water and ozone are
  widened too: O-H and O-O are singleton classes in the 9-molecule set.
- Verdict: not adoptable. Helps CH2ClF at one level of two, worsens it at
  the other, costs water/ozone 0.4-0.8 mA at B3LYP. The RHF loss is in C-F,
  whose class is calibrated but not corrected here (base recipe, no
  offsets): loosening C-Cl hands the split to a biased C-F prior.

- 2026-11-20, user chose "retry with offsets on": perbond3.py. Prior
  centre = offset-corrected geometry (leave-one-out), table built there, as
  the benchmark's `offsets` config. Fits: (a) offsets as is (sigma_x = the
  RMS of per-bond widths, prior_sigma_for_molecule); (b) tight = sigma_x at
  the calibrated spread (residual_sigma_after_offsets); (c) per-bond = (b)
  plus s^2 on the uncalibrated classes only -- the per-bond prior the RMS
  collapses. Water RHF: (a) reproduces the archived offsets row (0.6461 mA
  primary); (c) 0.65 -> 0.14 mA, angle 0.278 -> 0.130 vs r_e; added var /
  s^2 = 1.0. Full runs: perbond/off_{hf,b3}.log.
- Results, bond RMS vs r_e (mA), offsets as is -> offsets + per-bond
  [tight, unwidened]: RHF water 0.65 -> 0.14 [0.54]; ozone 2.86 -> 2.88
  [6.41] (angle 0.068 -> 0.162 deg); CH2ClF 17.64 -> 2.44 [18.03] (C-Cl
  +23.2 -> +0.3, C-F -18.5 -> +2.0, C-H -7.3 -> -3.7); vinyl fluoride 2.84
  (nothing widened). B3LYP water 0.69 -> 0.81 [1.04]; ozone 0.37 -> 0.40
  [0.66] (angle 0.060 -> 0.007); CH2ClF 11.22 -> 4.76 [11.28] (C-Cl +14.1
  -> +5.4, C-F -13.3 -> -5.9); vinyl fluoride 1.03. Every offsets row
  reproduces its archived primary-reference value to 4 decimals; added
  var / s^2 = 1.0 on every widened bond.
- Reading: with the calibrated classes corrected, widening only the
  uncovered class fixes CH2ClF at both levels (best of every variant: base
  10.80 / 9.24, data only 8.56 / 3.48) and leaves water / ozone within
  0.5 mA. Limits: one molecule carries the evidence; 5 of 9 have no
  uncovered class (unchanged by construction); HNCO (C-N, N-H uncovered)
  has only r_s / SE100-unstated references; this was the second variant
  tried (named in backlog N16 before the run). Offsets themselves are
  benchmark-only (bond_offsets is not on the CLI/config path).
- HNCO (user chose it as the second test), uncovered C-N and N-H, scored
  vs SE100 (trans-bent, matches r_s to 4 mA) and r_s: offsets -> per-bond
  [tight]: RHF 20.01 -> 24.81 [27.02] mA (r_s 21.62 -> 23.08), H-N-C/N-C-O
  angle RMS 1.49 -> 16.47 deg [20.05]; B3LYP 2.03 -> 3.62 [1.62] (r_s 4.23
  -> 4.91), angle 0.68 -> 1.29. Offsets rows reproduce the archive (21.6184,
  4.2259); added var / s^2 = 1.0 on N-H, N-C.
- Why: HF/6-31G gets HNCO's H-N-C angle ~30 deg wrong. The as-is offsets
  width (27 mA RMS, blending the uncovered bonds in) loosens the angles too,
  so the data fix the angle (1.49 deg); the per-bond prior keeps angles at
  the tight calibrated width and the angle stays near theory. Bond-only
  widths cannot cover an angle error. Verdict: not adoptable (helps CH2ClF,
  hurts HNCO at both levels).
- Q-01 (what next) had no answer by 01:14 UTC: auto-picked option 1. D-33
  recorded; the study is scripts/archive/perbond_prior_widths.py (archive
  rather than dev/, the repo's place for one-off studies), checked to
  reproduce the prototype rows exactly (offsets: water 0.65/0.14, CH2ClF
  17.64/2.44, HNCO 20.01/24.81; base: CH2ClF 10.80/12.42, archived primary
  13.3838 matched).

## Observations
- 2026-11-19: I estimated N17's saving ("2 wasted refits") without
  measuring; measured, it was ~0.2 s and no QC calls. Measure cost before
  pitching an efficiency fix.
- 2026-11-19: a rescale test passed for the wrong reason (the fixture
  widened, the test assumed tightening); asserting the premise in the test
  caught it (L-15 again).
- 2026-11-19: classifying bonds vs angles by counting '-' in their names
  garbled C=O / O=C-F rows; caught by eye (a 451 'mA' angle). Use the
  molecule's own bonds / angles keys, never the label's spelling.
- 2026-11-19: the hypothesis I offered (prior holds back determined fits)
  came from an SE-type yardstick; the independent r_e reversed it. Rank the
  yardsticks by independence before proposing an engine change.
- 2026-11-19: reproducing the archived base rows to the digit before
  reporting the pull (L-P04) is what lets the drop-row numbers be read as
  the benchmark's own, not a script's.
- 2026-11-19: the SE100 consistency check, run per constant rather than as
  one RMS, separated a reference that is fine (all B, C at 0.04%) from bad
  input data (five A rows). An aggregate 0.67% would have read as "SE100
  disagrees". Break a failed consistency check down by row before judging
  the reference.
- 2026-11-19: I wrote a verification claim into a comment ("swapping the
  labels costs 0.3%") from a different computation; caught on re-read
  (the label swap is 5.8%). Tie each number in a comment to the run that
  produced it.
- 2026-11-19: a secondary table (NQCC) was the only readable source for
  the CH2FCl r_e; checking its numbers against the repo's own B0 and the
  engine's corrections (0.06%, inside the corrections' sigma) is what made
  it safe to wire in. Use a data-consistency check whenever the primary is
  not readable.
- 2026-11-17: code_vintage() reads `git status --porcelain`, so editing the
  loop records (tracked) while a benchmark starts stamps it dirty. Caught
  from the log header within a minute; runs moved to a frozen worktree.

- 2026-11-17: the internal-coordinate posterior gives angle widths in
  radians with no unit field; my script printed them as degrees (0.005 for
  0.29). Check a quantity's units against a known value before reporting.
- 2026-11-19: the first per-bond numbers looked plausible and were wrong
  (the patch widened a matrix the fit never reads). The end-to-end check
  through the engine's own accessors (added variance / s^2, expect 1) is
  what caught it; a probe of what _calibrated_alpha_q receives found why.
- 2026-11-19: I told the user water and ozone would be unchanged before
  checking which classes are singletons (O-H, O-O are); corrected in the
  brief before running. Check the data a claim rests on before making it.

- 2026-11-20: CH2ClF alone said "fixed" (17.64 -> 2.44); the second molecule
  (HNCO) reversed it at both levels. The user's over-fitting worry was the
  right test; run a remedy on every molecule it changes before calling it.

## Pending question

## Handoff
