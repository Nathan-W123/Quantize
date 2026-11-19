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
- Outcomes: +2026-11-17 (round 10: all 19 findings reproduced through validate/build/CLI before fixing; one, the replaced stated value, only showed after the fit's Hessian corrections, not at construction)
### L-05 · Before fixing to a reference, check what the reference changes
- Observation: twice in one day a "the fit is not converged" verdict came
  from a comparison that changed more than one thing: round 8's reference
  was a terminus reached from another start (the procedure has several),
  and N6's restart was a fresh optimizer, i.e. also a fresh Hessian, which
  changed the model being fitted.
- Evidence: 2026-11-16 · Quantize round 8 (two fixes tuned to the wrong
  reference, one doubled the "far" count) and N6 (same-object restart
  <= 0.0003 mA vs fresh restart 97 mA on one molecule).
- Adjustment: when a check says "not at the optimum", rerun it changing one
  thing at a time (same object / same start / same model) and ask what the
  reference procedure does from the point being judged before changing code.
- Scope: workflow
- Status: provisional
- Outcomes: +2026-11-16 (N6: one 4-minute run named the cause)
### L-06 · Instrument the failure before hypothesising about it
- Observation: a library call failed ("CCSD did not converge"); three guessed
  causes (cycles, DIIS space, start geometry) were each tested and wrong.
  Logging the call's inputs found it in one run: the optimiser's unbounded
  first step had stretched the bonds 1.23 -> 2.22 A.
- Evidence: 2026-11-10 · Quantize (was project lesson L-P05).
- Adjustment: when a failure surfaces from inside a library call, log the
  inputs to that call across the failing sequence before forming a theory
  about the library's settings.
- Scope: workflow
- Status: provisional
- Outcomes:
### L-07 · Classify a tool's findings before quoting its count
- Observation: a dead-code report quoted a raw tool count to the user; most of
  it was intentional on inspection.
- Evidence: 2026-11-19 · pyflakes' "79 unused imports" was 67 backend
  registrations + 9 re-exports / noqa lines; 7 were real. Vulture's whole-repo
  list likewise mixed decorators, dataclass fields and library callbacks in.
- Adjustment: before reporting how much of something a static tool found,
  sort its hits into real / intentional / false positive and quote the real
  number; say what the tool cannot see (registries, string dispatch, fixtures).
- Scope: workflow · Status: provisional

## Retired
## Changelog
- 2026-11-12 created; L-01 added.
- 2026-11-13 L-02 added (reproducible numbers).
- 2026-11-16 L-03 (measure instead of tuning a proxy) and L-04 (CLI entry point) added after step 8; L-02 outcome +2026-11-16 (seeds quoted in every later commit; a verifier still caught one CLI number without its seed).
- 2026-11-17 L-06 moved here from the project's lessons (workflow scope); lessons.md consolidated 219 -> ~110 lines (L-10 into L-14, L-P01 and L-13 into L-15); L-04 outcome +2026-11-17.
