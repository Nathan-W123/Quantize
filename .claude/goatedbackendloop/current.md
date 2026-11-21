# Current task
Task: T-24
Status: done
Request: (D-66, auto-picked at Q-33) auto_systematic_sigma's magnitudes (backlog F13): measure against per-species B0-Be gaps; propose before any change.

## Brief
- Outcome: per molecule and component, the ratio of each sigma estimate
  (the defect-based auto, the flat 1% / 0.5% / 0.5% default, the
  benchmark's reference model) to the B0-Be gap, on two yardsticks.
- How: t24/gaps.py (B3LYP VPT2 tables from cached Hessians, 0 computed),
  t24/se_check.py (gaps against Be of the r_e^SE / r_e structures).
- Tradeoff: none in compute; a change alters widths for model_error.auto
  users (and API species given no sigma).
- Success: the table; a proposal.

## Graph
- N1 · diagnose · ratios on 9 molecules (VPT2) and 3 (r_e^SE) · after: none · done (t24/gaps.json) · handoff: -
- N2 · gate · propose · after: N1 · done: option 1 (D-67, auto-picked at Q-34) · handoff: -
- N3 · build · flat estimate; README, docstrings, runner comment; 5 water tests re-measured, 2 fail on the old code; CI selection 1625 passed · after: N2 · done · handoff: -
- N4 · verify · fresh agent fact-check of the new prose vs t24 data and the tests · after: N3 · done: main claim holds (flat closer in 19/21 VPT2, 11/12 r_e molecule-components); fixed: the report warning still said 'inertial defect, 3-10x', ranges were medians unlabelled, old-estimate figures wrong (8.1/4.2/2.7 -> 8.5/4.4/2.8%, 43/62 -> 60/87x, 0.0455->0.0180 -> 0.0160->0.0159), stale docstrings and the trade table (re-measured), ozone's r_e is not r_e^SE, water's r_e left out, VPT2 weak on water and ozone A; 'only opt-in widths change' is wrong with systematic_in_weighting (the fit moves) · handoff: -

## Progress
- Reachability: model_error.auto is opt-in (CLI `model_error: {auto: true}`,
  API auto_systematic_sigma=True; default off). The same estimate fills
  sigma_constants when a species has none -- the CLI requires sigma_mhz,
  so only API callers reach that.
- N1 VPT2 yardstick (median ratio estimate / gap per molecule; FA and HNCO
  have no species with all of A, B, C, so auto falls back to flat):
  auto A 0.17-8.9 (water 8.9, ozone 5.2, HCOF 3.0, VF 1.2, CH2ClF 0.74,
  AF 0.60, FE 0.17), B 0.05-6.1 (water 6.1; the other six 0.05-0.45), C
  0.04-1.06 (water 1.06; others 0.04-0.28). Flat A 1.0-2.5, B 0.58-1.17,
  C 0.23-0.83. The reference model = flat except water (3.2 / 4.4 / 1.2).
- N1 r_e^SE yardstick: VPT2/gap_SE 0.89-1.03 on VF and CH2ClF (so VPT2 is
  a fair yardstick there); ozone A 3.4. auto/gap_SE: VF 1.11 / 0.40 / 0.23,
  CH2ClF 0.76 / 0.20 / 0.14, ozone 16.6 / 0.36 / 0.26; flat: VF 1.04 / 1.05
  / 0.72, CH2ClF 1.04 / 1.02 / 0.80, ozone 8.6 / 0.68 / 0.58.
- The README says the auto estimate "runs 3-10x large on A and B"; on B it
  is too small for every molecule but water. defect_model_sigma's docstring
  compares formyl fluoride's 0.21% with fluoroacetylene's VPT2 0.23% -- two
  molecules; against formyl fluoride's own VPT2 gap its B is 0.45x.

## Observations
- 2026-11-21 · the fact-check found I had re-measured only the new values and carried the old estimate's figures over from memory and from a different data set (gaps.json's B0 vs the test's) -- four wrong. Measure both sides of a before/after before writing either.
- 2026-11-21 · five water tests had been tuned to the old estimate's water numbers (43x / 62x, 7.9-71 GHz, anti-monotonic); one of them was the stated reason the option is off by default, and it no longer holds. Measured each value before re-pinning it (L-09).

## Pending question
- Q-35 (asked 22:03 UTC, deadline 22:19): what's next. 1 the other two defect-based corrections, defect_bias_floor / defect_bias_scale (recommended) · 2 make model_error.auto the default (user's call) · 3 implement N22 or N23 (user's call) · 4 done for now.

## Handoff
