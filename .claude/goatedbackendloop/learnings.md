# goatedbackendloop learning reference
Sessions: 0 · Last consolidated: never
## Stated preferences
- Accuracy and efficiency are both hard goals; anything trading one for the other is flagged, not implemented · scope: all projects · stated 2026-11-11
- Use AskUserQuestion for choices; keep the loop running until told to stop · scope: all projects · stated 2026-11-11
## Workflow lessons
### L-01 · Size multi-agent workflows to the machine
- Observation: 7 parallel verifiers ran 2 at a time on a 4-CPU container; 3.6 h wall, user asked why.
- Evidence: 2026-11-12 · wf_e5e2e0fe-c99, nproc=4, cap = CPUs-2 = 2; verifiers 29-79 min each, 75-170 tool calls.
- Adjustment: run `nproc` before designing a fan-out; on <=4 CPUs scope each agent narrowly and expect serial rounds; state the expected wall time to the user up front.
- Scope: workflow
- Status: provisional
- Outcomes:
### L-02 · A number is not reported until its seed, threads and code hash are
- Observation: commit b71a261 quoted FF clouds from seeds 11 and 23 without
  saying so; the verifiers ran seed 7 and could not reproduce them, and the
  rto draws turned out not to be reproducible across BLAS thread counts even
  at one seed.
- Evidence: 2026-11-13 · wf_3a613cec-119 F2; repro_draws.py (OMP 1 vs 4:
  xi differs by 1.1-2.0x its size).
- Adjustment: put seed, thread count and commit hash next to every measured
  number in commit messages and docstrings; make random draws a function of
  the seed alone (no factorisation-dependent draws).
- Scope: workflow
- Status: provisional
- Outcomes: +2026-11-16 helped (rounds 3-6 reproduced the quoted numbers); -2026-11-16 one CLI ratio (72ece8a) quoted without its seed
### L-03 · When a check keeps needing new thresholds, measure the thing it predicts
- Observation: three verification rounds tuned a predictor (split coupling,
  then a null response with a tolerance) of whether the split posterior
  describes the split estimator; each threshold refused good cases and
  passed bad ones. The bootstrap measures that directly.
- Evidence: 2026-11-13/16 · fa9a145, 33f8d01, 867e925 refuted by
  wf_d6294e4a, wf_588822aa, wf_4fb65e30 (forced clouds on target at |K|
  0.058, 1.4x off at 0.018); 55c7472 let the bootstrap run and flag.
- Adjustment: when a gate on a proxy needs a second retuning, ask whether
  the quantity it predicts can be measured cheaply instead, and gate or
  report on the measurement.
- Scope: workflow
- Status: provisional
- Outcomes:
### L-04 · Measure fixes through the user's entry point, with every mode a script covers
- Observation: my after-measurements ran the API on analytic water and
  looked clean; the CLI config path (internal mode, multistart, corrections)
  showed 14/40 usable refits and a reduction chosen from corrected targets.
- Evidence: 2026-11-13/16 · 55c7472 (API 200/200 vs CLI 14/40); e927d67
  unit test passed, CLI 150-degree start still S (wf_706bb93b F0).
- Adjustment: for every fix, run one before/after through the CLI on a real
  config as well as the API check, and re-run every mode the reproduction
  script covers, not only the one fixed.
- Scope: workflow
- Status: provisional
- Outcomes:
## Retired
## Changelog
- 2026-11-12 created; L-01 added.
- 2026-11-13 L-02 added (reproducible numbers).
- 2026-11-16 L-03 (measure instead of tuning a proxy) and L-04 (CLI entry point) added after step 8; L-02 outcome +2026-11-16 (seeds quoted in every later commit; a verifier still caught one CLI number without its seed).
