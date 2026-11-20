# Current task
Task: T-11
Status: done
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
- N1 · build · t11/fallback.py: robust-rule fits, affected molecules x 3 pools x 2 levels · after: none · done · handoff: decouples water/ozone/HNCO-B3LYP; HNCO HF 8.14 -> 11.18, angle 1.84 -> 9.92 deg, still pool-sensitive
- N2 · verify · fresh agent checks the patch reaches the fit and recomputes the table · after: N1 · done · handoff: patch sound, rows bit-exact; HF HNCO row = f(sigma_x) alone, with a chi2-rescale switch at 17.3 mA
- N3 · gate · Q-13: adopt, flag, or drop? · after: N2 · done · handoff: D-45 (auto-picked): drop; docstring caveat; backlog N19

## Progress
- Fallback widths (mA), pool current / +CH2Cl2 / +CHF2Cl. HF: excl HNCO
  RMS 32.33 / 30.79 / 29.49, robust 16.94 / 17.02 / 17.30; excl water
  34.19 / 32.73 / 31.51 vs 32.98 (all); excl ozone 33.49 / 32.01 / 30.75
  vs 26.10 (all). B3LYP excl HNCO RMS 13.36 / 12.51 / 14.00, robust 6.61
  / 6.66 / 6.66. Class means HF (excl HNCO): C-Cl 82.0, C-F 27.8, O-O
  -21.0, C-C -11.7, C-H -11.1, C-N -9.1, H-O -8.1, C-O 4.0.
- N1 done 13:24 UTC (t11/robust_{hf,b3lyp}.json; RMS-rule rows from
  t10/pools2_*.json). Hybrid mA (vs r_e; angle deg), current / +CH2Cl2 /
  +CHF2Cl. HF water RMS 0.645 / 0.641 / 0.637, robust 0.641 all.
  HF ozone RMS 1.801 (2.861) / 1.816 / 1.829, robust 1.898 (2.958) all.
  HF HNCO RMS 8.137 (1.84) / 8.730 (1.86) / 9.109 (1.88), sigma_x 19.23 /
  18.42 / 17.91; robust 11.183 (9.92) / 10.955 (9.73) / 10.195 (8.99),
  sigma_x 10.82 / 10.94 / 11.44 -- spread across pools 0.97 -> 0.99, so
  not decoupled: its C-N / C-O width (the within-class spread) is pooled
  over every class too. B3LYP HNCO RMS 3.713 (0.33) / 3.697 / 3.722,
  robust 3.584 (0.17) / 3.585 / 3.582.
- Reading: HF HNCO's row tracks its sigma_x (8.14 at 19.2 mA, 11.2 at
  10.8), whichever rule sets it -- consistent with N16 (HF's H-N-C angle
  error needs a wide bond prior). To be checked by N2, not assumed.
- N2 (fresh agent, 13:40 UTC): patch reaches all 12 ME and hybrid calls and
  is undone; reruns byte-identical; rule reaches exactly HF water / ozone /
  HNCO and B3LYP HNCO. HF HNCO: centre bitwise equal in all pools, and the
  current pool with sigma_x forced to each of the six values reproduces all
  six rows bit-exactly -- the row is a function of sigma_x alone (B3LYP
  too). Sweep (hybrid mA / deg): 8 18.04/14.9, 10 12.95/11.3, 12 9.66/8.2,
  15 10.99/5.2, 17 12.68/3.9, 18 9.04/1.87, 20 7.60/1.82, 25 5.52/1.80,
  30 6.35/1.83, 40 10.41/1.91. The step at ~17.3 mA is the chi2 rescale
  (benchmark hybrid_fit: chi2_rescale=True, 3 passes; threshold 4.0):
  reduced chi2 after the first fit crosses 4.0 there (17.2: 4.02, refit,
  12.80 / 3.77; 17.4: 3.96, one pass, 9.50 / 1.90). The archived 8.14 sits
  at 19.2 (chi2 3.5-3.8). Without the rescale the robust row is 13.25 /
  2.85 deg, not 11.18 / 9.92. The rows are non-monotonic in sigma_x even
  without it (minimum ~5.5 near 25 mA). B3LYP's sweep is smooth and
  monotone (3.51 at 4 mA -> 3.96 at 30).
- Conclusion: the robust rule does not decouple HF HNCO (its C-N / C-O
  width is the pooled within-class spread) and costs it 3 mA; elsewhere
  it changes rows by <= 0.13 mA. Drop it. Separate finding: the README's
  HNCO RHF offsets figure (8.14) is 1.9 mA of sigma_x above a chi2-rescale
  switch; below it the same fit reads ~12.8.

## Observations
- 2026-11-20 · asking the verifier an experiment ("is the row a function of sigma_x alone? reproduce by forcing it") instead of my reading found the chi2-rescale switch, which my reading had missed.

## Pending question
- Q-14 (asked 13:56 UTC, deadline 14:11 UTC; send_later trig_01ESRyfjLYehdVNfy7Qybuoy): what's next?
  1 N19: first-pass chi2 of every benchmark row, count near the switch
  (Recommended; auto-pick) · 2 F14 correlated (GLS) weighting study · 3 N3
  r_e^SE references · 4 Done for now.

## Handoff
