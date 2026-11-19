# Current task
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

## Handoff
