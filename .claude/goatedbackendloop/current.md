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
- N2 · explore · theory geometries + offsets for candidates, both levels; C-Cl miss for CH2ClF with each pool · after: N1 · running · handoff: RHF done; B3LYP running
- N3 · gate · Q-10: add which entries? · after: N2 · pending · handoff: -

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

## Observations

## Pending question

## Handoff
