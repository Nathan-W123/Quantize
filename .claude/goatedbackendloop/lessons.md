# Project lessons
Last consolidated: 2026-11-20 (161 -> ~80 lines: L-08 merged into L-07;
validated lessons compressed to adjustment + one line of evidence; outcome
lists shortened). Earlier: 2026-11-17 (219 -> ~110). Reasoning before
2026-11-10 lives in the branch's commit messages (this directory was wiped
once by a container swap).

### L-P02 · Cross-check on the CI interpreter before pushing a numerics fix
- Adjustment: for a change touching linear algebra, conditioning or tolerances, run the affected tests under CI's Python before pushing; when a test "passed before, fails now" with no code change, run it on every interpreter and numpy build first, and record the versions beside a measured tolerance.
- Evidence: 2026-11-10 · a nan-asserting fix passed on 3.11, failed on 3.12. +2026-11-21 (T-16: N20's "cause unknown" was numpy 2.4.6 vs 2.5.5 -- T-14's suite ran on 3.11, T-15's on 3.12; found in one run on both). · Scope: project · Status: validated

### L-P03 · Long background runs get reaped; bank the expensive part to disk
- Adjustment: scope a run to one container lifetime, make the expensive intermediate durable (hessian_cache) first, and make the run resumable (saved rows, a part-N log per resume).
- Evidence: 2026-11-10 · the MP2 benchmark died four times; cached Hessians let it resume. +2026-11-20 (T-13: a container restart and a 2 h timeout cost only the row in progress). · Scope: project · Status: validated

### L-P04 · Verify a computed quantity against a value already in the repo
- Adjustment: before adding a derived quantity -- including a hand-built stand-in for a library function's output -- compute it for a case whose answer is in the repo and compare values, shape and multiplicity; no such case is itself a reason for suspicion.
- Evidence: 2026-11-10 · a Costain r_s floor gave 852 mA for ozone against the docstring's 4.1. -2026-11-20 (T-10: candidate offsets one per atom vs measure_offsets' one per bond; a verifier caught it). -2026-11-20 (T-14: a prototype took argmin over moments where "absent" is encoded as inf and zeroed the wrong axis on exactly linear geometries; the verifier's own edit disagreed on one row). -2026-11-20 (T-15: the new projector's own linearity threshold projected out part of a real bend; caught by the stationary-point check before any long run). Extension: before selecting from a library array (argmin / argmax / index), check how it encodes the missing case (inf, 0, NaN); any new rigid / linear treatment goes through rigid_mode_count, never its own threshold. · Scope: project · Status: validated

### L-P06 · A library module with only a script consumer is misplaced
- Adjustment: before committing a module under backend/, list its importers; if none are in backend/, wire it in or move it.
- Evidence: 2026-11-10 · force_field_diagnostics had only the benchmark. · Scope: project · Status: provisional

### L-07 · A finite uncertainty is not a checked uncertainty; two paths must agree
- Adjustment: assert a bound in the quantity's own units (a bond sigma below the bond length), not finiteness; where two paths compute one quantity, make them share the source and assert they agree in scale -- the assertion is the deliverable. (Merged L-08, 2026-11-20.)
- Evidence: 2026-11-10 · sigma 112 A on a 0.958 A bond through a green suite; it showed only once v2 delegated to the method (2871x disagreement). · Scope: project · Status: provisional

### L-09 · Bound tests by a measured scale, never by a round number
- Adjustment: print the quantity and its scale before writing a tolerance; set the bound from the measurement with stated headroom.
- Evidence: 2026-11-10 · atol=1e-18 on a norm-3.1e4 projector. +2026-11-20 (T-15: a bound set by measurement -- water e2e 0.65x -- had measured the defect; when a fix makes such a test fail, check its quantity against an exact reference, here B_e(r_e), before calling either side wrong). +2026-11-21 (T-16: an exact-zero assertion on a re-converged geometry broke under a numpy upgrade; bounded by the measured restart scale, with the defect's own move measured on the other side). -2026-11-21 (T-17: I wrote a new test's bounds and "measured" comments before measuring; two of three failed on first run -- measure, then write the bound). -T-18 (the same slip again in the convergence tests' comments, caught before commit). +T-27 (the rigid-mode test's 1.0 bound passed a leak that widened the bonds 42x; replaced with the measured share 4e-17 and width 2.8e-4 A; F3's own "2 BOM files" was unmeasured -- 57). · Scope: project · Status: validated

### L-11 · A fix in one objective mode can break another
- Adjustment: when a fix touches the step, the certificate or the bootstrap, rerun every reproduction script the verifiers left, on both versions, for every objective mode they cover.
- Evidence: 2026-11-13 · holding the quantum reference fixed joint and broke split (12/12 -> 8/12 usable). · Scope: project · Status: provisional

### L-12 · Check what a "far from the truth" verdict is measured against
- Adjustment: before fixing to a reference, ask what the reference procedure does from the point being judged; if it does not move, the point is valid and the comparator is wrong.
- Evidence: 2026-11-16 · split termini 17-25 widths apart, both stationary; two fixes tuned to the wrong comparator failed. +2026-11-21 (T-20: |defect| > 1 compared every molecule with a planar one; comparing with the config's own structure removed the false positive and kept the catches). · Scope: project · Status: provisional

### L-14 · One value, one reader: validator, builder and CLI share the rule
- Adjustment: when a fix adds a check for a config key, grep every reader and route them through one helper; read provenance from the raw file, never after normalization; read optional numbers with `is not None`, never `or`; test through load_config.
- Evidence: 2026-11-17 · round 10: rng_seed read three ways (7.0 -> 7 / 7 / 42), optimizer keys in two lists. · Scope: project · Status: validated

### L-15 · A green test must be shown to fail, for every branch it claims
- Adjustment: break the guarded property in the code and see the test fail, once per branch its docstring names, with a fixture that exercises it; assert work skipped (call counts), not speed ratios; compare against an inlined old formula, never the function's wrapper; assert the fixture's premise in the test. Mutate how a new function computes, not only whether it runs: each constant, norm, guard and per-mode branch.
- Evidence: 2026-11-17 · round 10 A-4 / A-5 (a residual-only file passing, a sampler compared with itself); +2026-11-19 (a rescale test that widened, not tightened). +2026-11-21 (T-20: each new branch removed in turn failed exactly its own test). +T-27 (leaky physical_basis monkeypatch: the old assertion passed, the new one fails). +T-29 (a removed reset passed its test until the test ran a second run(); every mutation caught after). · Scope: project · Status: validated +T-31 (a surviving mutant showed a guard was dead code; the verifier found the one note branch no assertion covered, the explanation sentence with no figure). +T-36 (three hand-off mutations survived the first tests -- two paths the fixture never took, a Jacobian test too loose on a planar fixture; the target-prior test first ran with the target prior off; each found by mutating and fixed). +2026-11-22 (T-46: a probe, not a test -- model_error.auto looked bit-identical because the fixture already stated a model error; checking that the widths moved exposed it; rerun: 0.20 -> 1.76 mA) -2026-11-22 (T-49: my 4 mutations of split_stall were all caught, but the verifier's 10 left 6 uncaught -- I mutated whether the check runs, not how it computes: its norm, its limit, its guard, its mode branch)

### L-16 · Break an aggregate diagnostic down by row before naming its cause
- Adjustment: before attributing a failed check or ratio to a cause, print it per species and component and check whether the outliers share a property (precision, source, site).
- Evidence: 2026-11-19 · SE100 vinyl fluoride's 0.67% "miss" was five A rows of bad input; the defect probe's 1.66 was the round-number-A species. · Scope: project · Status: provisional

### L-17 · Run a remedy on every molecule it changes before calling it a fix
- Adjustment: list every molecule a change can move and run all of them on every yardstick before reporting a gain; say n when it is small.
- Evidence: 2026-11-20 · per-bond widths: CH2ClF 17.64 -> 2.44 mA, HNCO angle 1.49 -> 16.47 deg. +T-07 (ozone vs r_e, trans H2O2) +T-10 (HNCO moved by a C-Cl member). +T-17 (a per-molecule diagnostic over all nine found the one molecule a Hessian repair moves -- CH2ClF -- before any rerun; base + offsets measured on all nine before adoption). +T-18 (all 12 B3LYP geometries optimised tight before the fix; it moved 4 by 0.15-0.25 mA and 8 by <= 0.01, and every B3LYP row was rerun). · Scope: project · Status: provisional

### L-18 · In results prose, name the row and file each number comes from
- Adjustment: quote each figure with its config, leg (theory / ME / hybrid) and yardstick (best-of / offsets row; vs primary / vs r_e) and its file; when calling a move small, give the largest over every leg, and say against whom a method "wins"; generate before/after pairs from the JSON with a script, derived figures too; have the prose fact-checked against the data.
- Evidence: 2026-11-20 · a fact-check of 622b573 found 4 mixed-row errors. +T-10; -T-11 (README vs docstring mixed by a two-file grep); -T-08 (a hand-computed estimate was wrong). +T-15 (fact-check of ebb41e0 by own code: 0 wrong numbers of ~70, 5 misleading -- a 0.28 that was the ME leg beside a hybrid figure, "wins" without naming the loser; my own progress note had called AF's move 1e-4 from the hybrid leg when its ME leg moved 0.062). -T-17 (fact-check: I quoted CH2ClF's effective-reference move, 0.32, as its move against r_e, 0.29 -- the yardstick has to travel with the number). -T-18 (fact-check of 31f9c84: a tight run's 0.047 quoted for the regenerated geometry's 0.052, an unmeasured cost claim, a stale 5.7; the run behind a number has to travel with it too). -T-19 (a cost read off a verifier's instrumented runs -- its wrapper computed a second gradient per call -- reached a gate question as 7-14%; the change itself costs 0-4.5%; and a 1.75e-4 rounded twice to 1.8e-4). -T-24 (four 'before' figures written from memory or another data set; ranges of per-molecule medians left unlabelled; the report's warning text missed). -T-56 ("the note fires for acetyl fluoride, fluoroethane and that dimer only": my survey imported four of the reference module's lists, and the verifier found acetone and fluorobenzene in the others; a claim of "only" needs the whole population enumerated, not the lists at hand). -T-58 ("0.003 widths by the exact Hessian" was one direction's component, quoted as the fit's distance, a maximum over all directions; T-60's breakdown gave 0.054). -T-70 (page item 9, T-68/T-69 rows: a 'noise band' taken from a run made later than the two being compared (adjacent runs: -9% to +2%); 'internal mode' over a sentence whose +45% was the Cartesian dimer; RMS per component quoted without saying so, beside per-atom figures elsewhere; 'agree better' while 8 of 42 pairs moved apart. Adjustment added: a noise band comes from the runs being compared; every RMS states per atom or per component; a mode or seed label covers only the figures it names). · Scope: project · Status: provisional

### L-19 · A negative result through the engine is conditional on its configuration
- Adjustment: when an engine experiment finds "no effect", name the settings it depends on (weighting, prior, width) and say whether another setting could change it.
- Evidence: 2026-11-20 · T-09: a 13C species 6.00 -> 5.97 mA as configured, 6.63 -> 6.15 with correlated weights. +T-10 (D-43 names its width-rule dependence). · Scope: project · Status: provisional

### L-20 · Screen against the whole decision rule, not the side the last finding was on
- Adjustment: when a run list is cut by a screen on a test's statistic, write the test's full rule (both bounds, every pass, every parameter value compared) and screen each check against all of it before narrowing.
- Evidence: 2026-11-20 · T-13: "rerun B3LYP only where first-pass chi2 > 2.5" missed the lower bound 1/T; scanning every check on both bands found 8 rows (water 0.217, ozone 0.226) that tighten at 3 and 4 but not 5. They moved < 1e-5 mA, so the count held, but it would have been unmeasured. · Scope: project · Status: provisional

### L-21 · Compare two runs only under the same cache state
- Adjustment: before putting a difference between two runs down to code, rerun one side under the other's cache state (warm vs cold) or both cold; record the cache state beside the code vintage.
- Evidence: 2026-11-21 · T-15: 57/117 RHF rows differed between prototype and commit with docstring-only code changes; the prototype on a warm cache reproduced the commit to every digit (fresh vs cache-read Hessians through the 1e-9 A key, amplified to 0.04 mA by fluoroacetylene's tracked fits). Worktrees sharing one exact cache gave bit-identical duplicates (15 rows). +T-17 (engine vs prototype on the same cache: 8.6e-9 mA over 18 rows). · Scope: project · Status: provisional

### L-22 · A per-pass movement threshold is not a stationarity certificate
- Observation: restart chains stopped at the first pass under 1e-4 A; further restarts kept drifting fluorobenzene's split fit 7e-5 to 1.5e-4 A and +0.03 mA per pass, one-signed.
- Evidence: 2026-11-21 · T-21 fact-check (t21/factcheck/extend_split*.py): 10.78 -> 10.99 and 10.63 -> 10.76 over 4-5 more restarts.
- Adjustment: certify stationarity by the trend of the score over three or more restarts, not one pass under a threshold; report a drifting fit as unsettled, with its range. Before claiming a refit (or restart) is a no-op, run it from a start that is not already a minimum, and check the stop reason.
- Scope: project · Status: provisional · Outcomes: -2026-11-22 (T-30: not applied; I called a refit on identical weights a no-op because the fixture starts at an exact minimum; the verifier's shifted start stopped on plateau and the refits kept moving chi2 201.96614 -> 201.96874; D-76 reverted)

### L-23 · Split a step pathology into rigid and internal motion before naming its cause
- Observation: T-35 named "the rotational rows' curvature" from a nonlinearity statistic; the steps were rigid rotations driven by an orientation-dependent prediction (fact-check fc35), and the fix (D-88) removed the crawl.
- Evidence: 2026-11-22 · rigid-body share of the steps >= 0.9998; a 0.01 rad turn moved the distortion prediction 0.5%; after the fix all 18 fits stop in 11-12 iterations.
- Adjustment: before attributing a refused or crawling step to curvature, project it onto the rigid-body directions and check that every row is invariant under rigid motion.
- Scope: project · Status: provisional · Outcomes: +2026-11-22 (T-38: splitting fluoroacetylene's steps found them 95% translation + rotation at once, which pointed at J^T r's rigid part and the finite-difference fallback; my first hypothesis, a wrong rigid-mode count, was ruled out by the same numbers)

### L-24 · Decide a branch outside any loop a derivative can cross, and test at the boundary
- Observation: T-37's per-geometry linear test switched branches between a prediction and its finite-difference steps; NaN filled the Jacobian and a CLI fit crashed. My tests sat away from the threshold; the verifier probed it.
- Evidence: 2026-11-22 · 5731f0c -> 86c99da -> 8ef6bd8: CLI HCN+DCN abort fixed by pinning once per isotopologue; then HNCO and diatomics mis-pinned, fixed by B-alone + 0.4 A off-axis.
- Adjustment: pin a discrete choice once per fit (as the reduction is), never per evaluation; put one test exactly at each threshold and one for each class the rule must reject (here: B and C observed, an off-axis atom, a diatomic).
- Scope: project · Status: provisional · Outcomes: -

### L-25 · Before calling a projection exact, check every row family that reaches it
- Observation: T-38 claimed J = J P for every Cartesian row; I checked rotational constants, distortion rows and the internal prior, not the conformer mixture, whose lab-frame offsets make its rows orientation-dependent. The verifier's mixture fit stopped at chi-square 718 instead of 58.
- Evidence: 2026-11-22 · 7d8c068 -> fixed by skipping the projection for a conformer mixture (test with a premise).
- Adjustment: list every block stacked before the change (grep the call site's builders) and measure each one's invariance; any I skip goes into the docstring as unchecked.
- Scope: project · Status: provisional · Outcomes: -2026-11-22 (T-41: not applied to consumers -- I checked the coordinate set, not every reader of it; the verifier found the acceptance test, the bootstrap reference and multistart each mishandling it) +2026-11-22 (T-45: grepping every reader of the fallback's split fields before committing found the convergence check and the iteration print crash on None, and the bootstrap's trust-capped test silently reading 0) -2026-11-22 (T-49: I 'corrected' the README's sv_min_abs default from the API's constructor; the config runner, which the README's table names, passes 0. Check every entry path -- API, config runner, CLI -- before writing a default)

### L-26 · Check a backlog item is still open before starting it
- Adjustment: before starting a backlog item, check the log and the code for work that already did it (git log --grep, a linter, a grep); when a commit acts on an item, update its row in the same commit.
- Evidence: 2026-11-23 · T-52 started F3's "79 unused imports" eleven days after efabab3 had handled them; the row still listed them. +2026-11-23 (T-54: checked first; three 2026-11-19 commits had done most of the rest). Scope: project · Status: provisional

### L-27 · Do not name a cause in a diagnostic that it does not measure
- Adjustment: a warning states what it measured; where it names a cause, measure that cause's share (decompose the quantity) first, or leave the cause out.
- Evidence: 2026-11-23 · T-53: the new off-mode wording blamed the range-null coupling; decomposed, the coupling explained 1 of 11 cases (a prior target's pull, an unfinished range the others). Scope: project · Status: provisional -T-56 (I called seed 23's 23.5 mA between two fits 'a second minimum' from their distance_to_mode, which reads each fit's own quadratic model and so cannot tell two minima from one point on a stale Hessian; T-57 measured the stale Hessian on one objective with real energies).


### L-28 · Check a prototype patch computes the change it describes before measuring its effect
- Observation: T-62 to T-64's step patch added the curvature correction to the Hessian before the step floored it, so it also undid the floor and inflated the damping; the internal-mode gains it measured were partly that side effect.
- Evidence: 2026-11-23 · T-65 verifier: internal dimer 0.0006 (patch) against 0.0107 (correction added exactly); formyl fluoride 'reaches the restart answer' did not survive.
- Adjustment: before measuring outcomes, compare the patched step against the intended formula at a few iterations (the step matrix with and without, and their difference against the intended term), and say where the patch enters relative to floors, damping and caps.
- Scope: project
- Status: provisional
- Outcomes:
