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
- Scope: workflow · Status: provisional · Outcomes: +2026-11-17 (round 10: 19 findings reproduced through the CLI first) +2026-11-21 (T-22: through the CLI the verifier found a multistart selection bug and a thread-count non-determinism the direct-optimizer measurement could not see)

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
- Adjustment: when a turn ends to wait on background work, say what is running, its progress (n of N), an ETA from a measured per-item rate (not a guess), and that the loop resumes on its own; if a job will meet a time limit, start its remainder in parallel rather than wait for the timeout; pass the harness's own timeout explicitly on every background launch (its default is 30 min, whatever the inner `timeout` says).
- Scope: workflow · Status: provisional · Outcomes: +2026-11-20 (T-10, T-13: no "why did it stop" since) -2026-11-20 (T-15: a guessed 15-20 min ETA missed by hours -- projected modes moved every cubic displacement off the cache) +2026-11-21 (T-15: remainder started in a parallel worktree before a 2 h timeout; no rows lost) -2026-11-21 (T-17: a launch without an explicit timeout died at the 30-min default with 1 of 18 rows; resumed from the disk cache)

### L-10 · Give a verifier an experiment that settles the question, not your reading
- Observation: asked to force sigma_x and reproduce, a verifier found a chi2-rescale switch my reading had put down to the width itself.
- Evidence: 2026-11-20 · Quantize T-11 (t11/verify/fine_hf.json).
- Adjustment: state the claim as an experiment the verifier can run (hold X fixed, vary Y, reproduce the rows, by a different mechanism than yours), not as a conclusion to review.
- Scope: workflow · Status: provisional · Outcomes: +2026-11-20 (T-12: "move the threshold to 3 and 5" gave the flip point and the floor) +2026-11-20 (T-13: a source edit instead of my monkeypatch reproduced every row bit for bit and showed the 5.93 best-of needs six rows, not the four I named) +2026-11-20 (T-14: told to make its own edit, the verifier's fix disagreed with my prototype on one row and exposed my prototype's bug) +2026-11-21 (T-15, three verifiers: own projector refuted my gradient-proportional prediction and found the 23.2-28.6 recipe range; own Hessians, frames and grids confirmed the fix but refuted my mechanism; own code on the JSON found 5 misleading statements my figure script could not) +2026-11-21 (T-17: the grid verifier found two things I had not asked -- projection alone worse than none on ozone, theory geometries not tight minima; the fact-checker found two wrong figures) +2026-11-21 (T-18: the fact-checker found 3 wrong figures and the leg / reference / margin gaps my figure script could not) +2026-11-21 (T-19: the CLI verifier found the gradient the answer uses is run_hessian's, not the run_gradient my brief named; the fact-checker found the cost artefact) +2026-11-21 (T-21: the fact-checker found the non-stationary split cells, the shallow-clone date and the reversed derived-data claim) +2026-11-21 (T-23/T-24: measuring what a bug changes, not only that it exists, turned a fix into a note; the fact-checker found the user-facing warning text the change missed) +2026-11-22 (T-29: loading the old module beside the new and diffing 42 fits found the one ordering difference my tests had missed) +2026-11-22 (T-30: own shifted starts and single-threaded pyscf refuted my bit-identical claim, which held only from the fixture's exact minimum; the change was reverted) +2026-11-22 (T-31: end-to-end CLI diff old vs new and own internal-coordinate code confirmed the figure; it also found a wrong median in my docstring and a stale sentence in the fallback note)

### L-11 · Time the change itself, not an instrumented stand-in
- Observation: a verifier's monkeypatch computed both variants per call "for the record"; I quoted its wall times as the change's cost.
- Evidence: 2026-11-21 · T-19: 7-14% reported; re-timed with exactly the change, 0% (formyl fluoride) and 4.5% (fluoroethane, repeats 4%). The wrong figure had already reached a gate question.
- Adjustment: before quoting a time as a cost, read what the timed code does per call; time the exact change with nothing recording inside the loop, alternating with the baseline, at least two runs each, and state the repeats' spread beside it.
- Scope: workflow · Status: provisional · Outcomes: -

### L-12 · Date a change from the full history, not a shallow clone
- Observation: `git log --follow` in a shallow clone showed a default flipping at a grafted root commit; the real change was a day earlier on GitHub.
- Evidence: 2026-11-21 · T-21: README and a commit message said 7216d87 (in .git/shallow); the fact-checker found 6f43c5a, and the wrong date had supplied a false cause.
- Adjustment: before dating a change or naming its cause from history, check `.git/shallow`; when the commit found is listed there, read the history from the remote (the GitHub tools) instead.
- Scope: workflow · Status: provisional · Outcomes: -

### L-13 · Quote the shipped configuration's figures, counted by script
- Observation: two tasks in a row, summary prose mixed populations: a bonds-only median quoted for bonds and angles (T-31); ranges taken across the shipped and an experimental variant, and "6 of 18" counted by eye for 7 (T-32).
- Evidence: 2026-11-22 · both caught only by the independent fact-check (T-31 verifier; T-32 fc32: 4 wrong, 6 imprecise claims in one README paragraph).
- Adjustment: before writing a figure into prose, compute it by script from the data files, for the configuration the reader will run, and name the population it covers (which fits, which coordinates, weight or sigma).
- Scope: workflow · Status: provisional · Outcomes: -

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
- 2026-11-21 L-09 revised (measured-rate ETA, parallel remainder before a limit); L-10 outcome + (T-15); project L-21 added, L-P04 / L-09 / L-18 extended.
- 2026-11-21 project L-P02 validated (T-16: a library-version cause found by running on both interpreters); L-09 (project) outcome +.
- 2026-11-21 L-09 revised (explicit harness timeout on background launches); L-10 outcome + (T-17); project L-09 / L-17 / L-18 / L-21 outcomes.
- 2026-11-21 L-10 outcome + (T-18); project L-09 / L-18 outcomes -, L-17 +.
- 2026-11-21 L-11 added (time the change itself); L-10 outcome + (T-19); project L-18 outcome -.
- 2026-11-21 project L-12 / L-15 outcomes + (T-20).
- 2026-11-21 L-12 added (shallow-clone history); L-10 outcome + (T-21); project L-22 added.
- 2026-11-21 L-04 outcome + (T-22).
- 2026-11-21 L-10 outcome + (T-24); project L-18 outcome -.
