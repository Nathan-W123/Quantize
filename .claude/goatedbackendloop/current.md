# Current task
Task: T-10
Status: building
Request: a second C-Cl calibration reference (CHF2Cl, CH2Cl2, CHCl3) so the
C-Cl offset is not one molecule's; literature first, then measure; adding
entries is a separate gate (D-41, auto-picked at Q-09).

## Brief
- Outcome: candidate C-Cl references meeting D-35's rule (r_e / r_e^SE,
  stated sd <= 0.5 mA, readable source), and a measurement of what they do
  to CH2ClF's C-Cl offset miss (12.4 RHF over / 12.8 B3LYP under) and to
  every other molecule whose classes they join (L-17: C-F, C-H too).
- How: NQCC pages (Bailey) tabulated Vogt 2014's r_e^SE for CH2FCl
  readably; the same pages for CHF2Cl, CH2Cl2, CHCl3 likely do too. Theory
  geometries at both levels (one optimisation each), conformer guard,
  offsets measured as for T-07's entries.
- Expectation to test, not assume: C-Cl's theory error moves with
  fluorination (CH3Cl +85.6 / +25.4 mA vs CH2ClF +73.2 / +38.2), so a
  CHF2Cl member could bracket CH2ClF; CH2Cl2 / CHCl3 may not.
- Gate: adding entries changes every offsets-family row (rerun ~2.5 h).

## Graph
- N1 · explore · NQCC CHF2Cl / CH2Cl2 / CHCl3 pages: r_e^SE and sources · after: none · done · handoff: CH2Cl2 meets D-35; CHCl3 C-H 0.6; CHF2Cl C-Cl 0.9 / C-H 1.1
- N2 · explore · theory geometries + offsets for candidates, both levels; C-Cl miss for CH2ClF with each pool · after: N1 · running · handoff: done; only CHF2Cl helps both levels (miss 3.1 / 5.4 from 12.4 / 12.8)
- N3 · gate · Q-10: measure, add, or stop? · after: N2 · done · handoff: D-42 (auto-picked): measure first
- N4 · verify · offsets fits, 9 molecules x 2 levels x 3 pools (current, +CH2Cl2, +CHF2Cl) · after: N3 · done · handoff: RHF mean 5.38 -> 5.66 / 5.70; B3LYP 4.13 -> 4.10 / 3.91
- N4v · verify · fresh agent checks pools.py recipe and both result files · after: N4 · running · handoff: -
- N5 · gate · Q-11: add which, if any? · after: N4 · pending · handoff: -

## Progress
- NQCC pages (fetched 2026-11-20): CH2Cl2 r_e^SE (Vogt 2014) C-Cl
  1.76425(3), C-H 1.0816(2), ClCCl 112.166(3), HCH 111.722(4) -- meets
  D-35. CHCl3 r_e^SE (Demaison, JMS 251, 217 (2008)) C-Cl 1.7595(1), C-H
  1.0818(6), HCCl 108.080(16), ClCCl 110.825(16) -- C-H 0.6 mA, just over.
  CHF2Cl r_e^SE (Vogt 2014) C-H 1.0850(11), C-Cl 1.7560(9), C-F
  1.3363(5), HCF 109.97(4), HCCl 109.60(6), ClCF 109.62(4), FCF 108.06(6)
  -- C-Cl 0.9 and C-H 1.1 mA, over the rule.
- RHF (t10/cl.py, cl_hf.json; geometries optimised from the r_e^SE;
  builders check: CHF2Cl FCF 108.039 vs 108.06, CHCl3 ClCCl 110.826):
  theory - r_e^SE C-Cl: CH3Cl +85.6, CH2ClF +73.2, CH2Cl2 +67.3, CHCl3
  +60.7, CHF2Cl +54.7 (C-H -12.5 / -14.7 / -17.9; CHF2Cl C-F +26.4).
  CH2ClF's C-Cl miss by pool: CH3Cl -12.37; +CH2Cl2 -0.13; +CHCl3 +6.34;
  +CHF2Cl +3.08; +CH2Cl2+CHCl3 +6.23; all four +7.98 (class mean per
  bond, so CHCl3's three bonds weigh three).
- B3LYP (cl_b3lyp.json): C-Cl CH3Cl +25.4, CH2Cl2 +26.3, CHCl3 +27.9,
  CH2ClF +38.2, CHF2Cl +40.2 (C-H +6.1 / +3.9 / +5.7; CHF2Cl C-F +4.9).
  A fluorine on the carbon adds ~13-15 mA to B3LYP's C-Cl error; chlorine
  substitution barely moves it. CH2ClF's C-Cl miss by pool: CH3Cl +12.78;
  +CH2Cl2 +12.13; +CHCl3 +10.91; +CHF2Cl +5.35; +CH2Cl2+CHCl3 +11.21; all
  four +9.31. Only CHF2Cl helps at both levels (RHF +3.08) and it breaks
  D-35's rule (C-Cl sd 0.9, C-H 1.1). Its C-F (+4.9 at B3LYP vs CH2ClF's
  own -1.2) would not help C-F. A better C-Cl centre is not a better fit by
  itself: T-08's RHF C-Cl oracle made the fit worse (2.91 -> 4.16).
- N4 runs (t10/pools.py <level>; pools current / +CH2Cl2 / +CHF2Cl; a
  molecule whose offsets and sigma_x are unchanged takes its archived
  c8904da row; resumable): started 08:1x UTC, HF and B3LYP in the
  background (logs t10/pools_{hf,b3lyp}.log). First RHF +CH2Cl2 refits
  (C-H offsets move <= 0.35 mA): VF 6.375 -> 6.337 (vs r_e 2.796 ->
  2.811), AF 7.274 -> 7.269, FE 7.217 -> 7.211, FF 7.154 -> 7.032.
- RHF done (pools_hf.json), hybrid vs primary (/ vs r_e), current ->
  +CH2Cl2 -> +CHF2Cl: CH2ClF 5.42/2.91 -> 7.15/4.02 -> 7.52/4.42 (C-Cl
  +5.3 / +6.1 vs r_e: the over-correction that cancelled the fit's pull is
  gone, as T-08 predicted); HNCO 8.14 -> 9.05 -> 9.14; FF 7.15 -> 7.03 ->
  6.98; FE 7.22 -> 7.21 -> 7.08; AF 7.27 -> 7.27 -> 7.33; FA 4.36 -> 4.39
  -> 4.46; VF 6.37/2.80 -> 6.34/2.81 -> 6.33/2.84; water 0.64 all; ozone
  1.80 -> 1.82 -> 1.83. Mean 5.376 -> 5.656 -> 5.701. HNCO, water, ozone
  move with no class change: at RHF their N-H / O-H / O-O are uncalibrated
  (H2O2 excluded), so their sigma_x is uncorrected_class_bias = RMS of all
  class means, and a C-Cl member lowers the C-Cl mean (85.6 -> ~70),
  narrowing them (HNCO 19.2 -> 18.0 / 17.9 mA).
- B3LYP done 10:20 UTC (pools_b3lyp.json), same layout: CH2ClF
  8.18/6.00 -> 8.05/5.85 -> 6.45/4.29 (bonds vs r_e: current C-Cl +7.4 /
  C-F -7.2 from T-08; +CH2Cl2 +7.2 / -7.0; +CHF2Cl +5.0 / -5.3); HNCO
  3.71 -> 3.69 -> 3.72; FF 3.34 -> 3.36 -> 3.22; FE 7.66 -> 7.66 -> 7.61;
  AF 6.45 -> 6.40 -> 6.28; FA 2.09 -> 2.00 -> 2.09; VF 5.24/1.00 ->
  5.25/0.98 -> 5.23/1.03; water 0.46 -> 0.47 -> 0.47; ozone 0.04/1.02 ->
  0.00/1.06 -> 0.07/1.13. Mean 4.129 -> 4.098 -> 3.905.
- Net: CH2Cl2 is +0.28 at RHF, -0.03 at B3LYP (a loss). CHF2Cl is +0.33
  at RHF, -0.22 at B3LYP, and breaks D-35's sd rule. Neither improves
  both levels; the RHF losses come from CH2ClF (its old cancellation) and
  HNCO (uncorrected_class_bias coupling, not chemistry).

## Observations

## Pending question

## Handoff
