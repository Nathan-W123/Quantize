# goatedbackendloop learning reference
Sessions: 0 · Last consolidated: 2026-11-20
## Stated preferences
- Accuracy and efficiency are both hard goals; anything trading one for the other is flagged, not implemented · scope: all projects · stated 2026-11-11
- Use AskUserQuestion for choices; keep the loop running until told to stop · scope: all projects · stated 2026-11-11
## Workflow lessons
### L-01 · Size multi-agent workflows to the machine
- Observation: 7 parallel verifiers ran 2 at a time on a 4-CPU container; 3.6 h wall, user asked why.
- Evidence: 2026-11-12 · wf_e5e2e0fe-c99, nproc=4, cap = CPUs-2 = 2.
- Adjustment: run `nproc` before designing a fan-out; on <=4 CPUs scope each agent narrowly and expect serial rounds; state the expected wall time up front.
- Scope: workflow · Status: provisional

### L-02 · A number is not reported until its seed, threads and code hash are
- Observation: quoted FF clouds (seeds 11, 23) could not be reproduced at seed 7, and rto draws differed across BLAS thread counts at one seed.
- Evidence: 2026-11-13 · wf_3a613cec-119 F2; OMP 1 vs 4 moved xi by 1.1-2.0x its size.
- Adjustment: put seed, thread count and commit hash next to every measured number; make random draws a function of the seed alone.
- Scope: workflow · Status: provisional
- Outcomes: +2026-11-16 (rounds 3-6 reproduced the numbers) -2026-11-16 (one CLI ratio without its seed)

### L-03 · When a check keeps needing new thresholds, measure the thing it predicts
- Observation: three rounds retuned a proxy gate for "does the split posterior describe the split estimator"; each refused good cases and passed bad ones; the bootstrap measures it directly.
- Evidence: 2026-11-13/16 · fa9a145, 33f8d01, 867e925 refuted; 55c7472 let the bootstrap run and flag.
- Adjustment: when a proxy gate needs a second retuning, ask whether the predicted quantity can be measured cheaply, and gate or report on the measurement.
- Scope: workflow · Status: provisional

### L-04 · Measure fixes through the user's entry point, with every mode a script covers
- Observation: API checks on analytic water looked clean; the CLI config path showed 14/40 usable refits.
- Evidence: 2026-11-13/16 · 55c7472 (API 200/200 vs CLI 14/40); e927d67 (unit test green, CLI start still wrong).
- Adjustment: for every fix, run a before/after through the CLI on a real config as well as the API, and every mode the reproduction script covers.
- Scope: workflow · Status: provisional · Outcomes: +2026-11-17 (round 10: 19 findings reproduced through the CLI first)

### L-05 · Before fixing to a reference, check what the reference changes
- Observation: two "not converged" verdicts came from comparisons that changed more than one thing (another start; a fresh optimizer, i.e. a fresh Hessian).
- Evidence: 2026-11-16 · round 8 (two fixes tuned to the wrong reference); N6 (same-object restart <= 0.0003 mA vs fresh 97 mA).
- Adjustment: rerun the check changing one thing at a time (same object / start / model) before changing code.
- Scope: workflow · Status: provisional · Outcomes: +2026-11-16 (N6: one run named the cause)

### L-06 · Instrument the failure before hypothesising about it
- Observation: three guessed causes of a CCSD non-convergence were wrong; logging the call's inputs found an unbounded first step (1.23 -> 2.22 A).
- Evidence: 2026-11-10 · Quantize (was project lesson L-P05).
- Adjustment: when a failure surfaces inside a library call, or a runtime patch gives odd numbers, log what that call receives across the failing sequence before theorising.
- Scope: workflow · Status: provisional · Outcomes: +2026-11-19 (a patched prior: the fit read a rigid-projected Hessian, found in one probe) +2026-11-20 (T-14: a guessed Coriolis cause was wrong; logging each alpha term at the three slope points named the harmonic term in one run)

### L-07 · Classify a tool's findings before quoting its count
- Observation: pyflakes' "79 unused imports" was 67 registrations + 9 re-exports; 7 were real.
- Evidence: 2026-11-19 · Quantize dead-code audit.
- Adjustment: sort a static tool's hits into real / intentional / false positive and quote the real number; say what the tool cannot see.
- Scope: workflow · Status: provisional

### L-08 · A `cd` or variable at the head of a backgrounded chain reaches only its first job
- Observation: `cd X && (A) & (B) & wait` ran B in the wrong directory / with SP empty.
- Evidence: 2026-11-20 · twice in one session.
- Adjustment: put the `cd` and variables inside each job's parentheses, or export them on a separate line first.
- Scope: stack: bash · Status: provisional · Outcomes: +2026-11-20 (T-13: per-job `export` in a loop; both halves ran)

### L-09 · Ending a turn on background work: say what runs, how far, how long
- Observation: the user twice asked why the loop had stopped while a benchmark ran in the background.
- Evidence: 2026-11-20 · "why did the loop stop" (Quantize T-07).
- Adjustment: when a turn ends to wait on background work, say what is running, its progress (n of N), a rough ETA, and that the loop resumes on its own.
- Scope: workflow · Status: provisional · Outcomes: +2026-11-20 (T-10, T-13: no "why did it stop" since)

### L-10 · Give a verifier an experiment that settles the question, not your reading
- Observation: asked to force sigma_x and reproduce, a verifier found a chi2-rescale switch my reading had put down to the width itself.
- Evidence: 2026-11-20 · Quantize T-11 (t11/verify/fine_hf.json).
- Adjustment: state the claim as an experiment the verifier can run (hold X fixed, vary Y, reproduce the rows, by a different mechanism than yours), not as a conclusion to review.
- Scope: workflow · Status: provisional · Outcomes: +2026-11-20 (T-12: "move the threshold to 3 and 5" gave the flip point and the floor) +2026-11-20 (T-13: a source edit instead of my monkeypatch reproduced every row bit for bit and showed the 5.93 best-of needs six rows, not the four I named) +2026-11-20 (T-14: told to make its own edit, the verifier's fix disagreed with my prototype on one row and exposed my prototype's bug)

## Retired
## Changelog
- 2026-11-12 created; L-01 added.
- 2026-11-13 L-02 added (reproducible numbers).
- 2026-11-16 L-03 and L-04 added; L-02 outcome +.
- 2026-11-17 L-06 moved here from the project's lessons; lessons.md consolidated 219 -> ~110 lines; L-04 outcome +.
- 2026-11-20 L-09 outcome +; project L-P04 extended (hand-built stand-ins for a library function's output).
- 2026-11-20 L-10 added (verifiers get experiments, not readings).
- 2026-11-20 L-10 outcome + (T-13).
- 2026-11-20 L-06, L-10 outcomes + (T-14); project L-P04 extended (sentinel encodings).
- 2026-11-20 consolidated: learnings.md 127 -> ~75 lines (observations and evidence compressed, no lesson dropped); lessons.md 161 -> ~80 (L-08 into L-07, validated lessons compressed); L-08 outcome + (T-13).
