# Current task
Task: T-11
Status: building
Request: measure a width for uncalibrated bond classes that does not move
with unrelated calibration members (N16 D-03 flag; D-44, auto-picked at
Q-12); scratch only, no engine change.

## Brief
- Outcome: for each molecule with an uncalibrated class (RHF: HNCO N-H,
  water O-H, ozone O-O; B3LYP: HNCO N-H), the offsets row under the current
  fallback (RMS of all class means) and a robust one (1.4826 x median
  |class mean|, the MAD estimate of the same spread), each with the current
  pool and the two T-10 candidate pools. Two numbers per rule: accuracy at
  the current pool, and how far the row moves when a C-Cl member is added.
- Why this rule: it estimates the same quantity (sd of class biases) but
  one outlier class (C-Cl ~80 mA at HF) cannot dominate it. Fallback values
  (HF, HNCO excluded): RMS 32.3 / 30.8 / 29.5 mA across the pools, robust
  16.9 / 17.0 / 17.3.
- Trade-off to expect: the robust value is about half as wide at HF, and
  N16 found HNCO's HF fit leans on a wide width (H-N-C angle error), so
  stability may cost HNCO accuracy. That is a D-03 trade: measure and flag.
- Out of scope: the within-class spread (also pooled over every class;
  vinyl fluoride's sigma_x 6.92 -> 7.06 / 7.88 at HF with the candidates).
- Success: both rules' rows for the four (molecule, level) pairs x 3
  pools, the current-rule rows reproducing T-10's, and a fresh-agent check.

## Graph
- N1 · build · t11/fallback.py: robust-rule fits, affected molecules x 3 pools x 2 levels · after: none · running · handoff: -
- N2 · verify · fresh agent checks the patch reaches the fit and recomputes the table · after: N1 · pending · handoff: -
- N3 · gate · Q-13: adopt, flag, or drop? · after: N2 · pending · handoff: -

## Progress
- Fallback widths (mA), pool current / +CH2Cl2 / +CHF2Cl. HF: excl HNCO
  RMS 32.33 / 30.79 / 29.49, robust 16.94 / 17.02 / 17.30; excl water
  34.19 / 32.73 / 31.51 vs 32.98 (all); excl ozone 33.49 / 32.01 / 30.75
  vs 26.10 (all). B3LYP excl HNCO RMS 13.36 / 12.51 / 14.00, robust 6.61
  / 6.66 / 6.66. Class means HF (excl HNCO): C-Cl 82.0, C-F 27.8, O-O
  -21.0, C-C -11.7, C-H -11.1, C-N -9.1, H-O -8.1, C-O 4.0.

## Observations

## Pending question

## Handoff
