# Project lessons

Recreated 2026-11-10 after a container swap wiped this directory (it is
gitignored, so a fresh clone does not carry it). The full reasoning for
everything before this point is in the commit messages on
claude/repo-audit-calculations-gtotzw, which is why nothing important lived
only here.

### L-P01 · A test that passes can be depending on the bug
- Observation: test_geometry_uncertainty had two tests that did not merely miss
  a defect, they REQUIRED it. geometry_uncertainty returned sqrt(max(var, 0))
  and so reported 0.0 when the propagated variance came out at -2.3e13. One
  test asserted "removing the sigma floor gives a tighter bond" (true only
  against 0.0); the other used that same 0.0 as the tight end of a
  looser-is-wider check. Both green, for the one reason that should have
  failed them.
- Evidence: 2026-11-10 - CI red on four consecutive heads of PR #6. Probing
  main and the branch in ONE environment gave identical values, which is what
  proved it pre-existing rather than mine. cond = 1.2e32, min_eig = -2.0e13.
- Adjustment: When a test's premise is an inequality between two computed
  numbers, print both before trusting it. If either is exactly 0.0, 1.0 or
  nan, suspect the comparison is passing degenerately. When a long-standing
  test fails only on another platform, the usual cause is that neither value
  was meaningful, not that one platform is wrong.
- Scope: project
- Status: provisional

### L-P02 · Cross-check on the CI interpreter before pushing a numerics fix
- Observation: My first fix asserted nan specifically. It passed on 3.11 and
  FAILED on 3.12, where the covariance is rejected a layer earlier and
  geometry_uncertainty returns None for everything. I would have shipped a
  second platform-dependent test into the same spot.
- Evidence: 2026-11-10 - /usr/bin/python3.12 existed locally; a venv with
  numpy 2.5.4 reproduced CI's stack and caught it in one run. Final state:
  1176 passed on both 3.11.18/numpy 2.4.6 and 3.12/numpy 2.5.4.
- Adjustment: For any change touching linear algebra, conditioning or
  tolerances, run the affected tests under the CI interpreter version before
  pushing. A venv costs a minute; a red CI round trip costs ten and teaches
  less.
- Scope: project
- Status: provisional

### L-P03 · Long background runs get reaped; bank the expensive part to disk
- Observation: The MP2 benchmark has been killed four times - twice by
  container swaps, twice by process reaping with no status file written, which
  means the wrapper itself was killed rather than Python crashing. Each loss
  was about an hour of MP2/cc-pVTZ gradients, because the only cache was in
  memory and the results file is written only after a whole molecule finishes.
- Evidence: 2026-11-10 - exit-code capture added to the wrapper; after the
  fourth death there was no status file and the log ended mid-Hessian on
  ordinary pyscf Krylov warnings. 24 Hessians (108 KB) were on disk by then,
  so the relaunch resumed instead of restarting.
- Adjustment: Scope a long run to what finishes inside one container lifetime,
  and make the expensive intermediate durable (hessian_cache) BEFORE launching
  it rather than after losing it. Capture the exit code in a wrapper so a
  silent death is distinguishable from a crash.
- Scope: project
- Status: validated

### L-P04 · Verify a computed quantity against a value already in the repo
- Observation: Tried to compute a per-molecule r_s floor by propagating the
  Costain rule through each reference's bonds. It gave 852 mA for ozone where
  the benchmark docstring quotes 4.1 mA from the literature - two orders of
  magnitude, in the direction that would have made every row look meaningless.
  The rule diverges for an atom on a principal axis and nearly every atom in
  these small planar molecules is on one.
- Evidence: 2026-11-10 - every bond in all nine references hit the 1 Ang cap.
  Checking against the 4.1 mA already in the repo is the only reason it did
  not ship.
- Adjustment: Before adding a derived quantity to the data model, compute it
  for a case whose answer is already written down somewhere in the repo and
  compare. If there is no such case, that absence is itself a reason to be
  suspicious of the quantity.
- Scope: project
- Status: validated

### L-P05 · Instrument the failure before hypothesising about it
- Observation: CCSD(T) died inside optimise() with "CCSD did not converge". I
  guessed three causes in turn -- too few CCSD cycles, too small a DIIS space,
  a bad starting geometry -- and tested each. All three were wrong: CCSD
  converges fine at the coarse guess, at the HF geometry, and at a distorted
  one, with pyscf's defaults. Logging the geometry at every energy evaluation
  found the real cause in a single run: L-BFGS-B's unbounded first step jumped
  the bonds from 1.233 to 2.215 Ang.
- Evidence: 2026-11-10 - three measured dead ends, then one log that showed
  step 0 ok / step 1 RuntimeError with the geometry printed.
- Adjustment: When a failure surfaces from inside a library call, log the
  INPUTS to that call across the failing sequence before forming a theory
  about the library's settings. The cheapest instrument usually beats the
  most plausible hypothesis.
- Scope: workflow
- Status: provisional

### L-P06 · A library module with only a script consumer is misplaced
- Observation: I put force_field_diagnostics in .github/backend/ and wired it
  only to scripts/accuracy_upgrades_benchmark.py. The user asked why, and the
  question exposed a real incoherence: if a bad force field corrupts a
  correction, every library caller should be warned, not just the benchmark.
- Evidence: 2026-11-10 - grep for importers returned the benchmark and its
  test, nothing in the engine.
- Adjustment: Before committing a new module under backend/, list its
  importers. If none of them are in backend/, either wire it into the library
  path or put it next to the consumer that exists.
- Scope: project
- Status: provisional

### L-07 · A finite uncertainty is not a checked uncertainty
- Observation: the v2 Laplace report printed sigma = 112 Angstrom on a 0.958
  Angstrom bond through a fully green test suite. Every existing assertion
  checked that widths were finite, non-negative, and not nan. None checked
  that a width was smaller than the thing it was a width on.
- Evidence: 2026-11-10 · `max std err 1.118297e+02` for water; physical
  widths 8.3e-3. The regularisation ridge made `matrix_rank` report full rank
  and put cond at 2.4e11, under the 1e12 warning threshold, so both guard
  rails read clean.
- Adjustment: for any reported physical quantity, assert a bound in the units
  of the thing itself (a bond sigma below the bond length), not just
  finiteness. Where two code paths compute from one intermediate, assert they
  agree in scale -- that cross-check is what caught this.
- Scope: project
- Status: provisional
- Outcomes: +2026-11-10 found a four-order-of-magnitude shipped defect

### L-08 · Deduplicate to create a cross-check, not just to tidy
- Observation: F2 looked like housekeeping (same fifteen lines in two places,
  agreeing to 5.8e-6). The value turned out not to be the tidying: once both
  paths ran off one covariance, they could be diffed, and they disagreed by
  2871x for a different reason entirely.
- Evidence: 2026-11-10 · the 112 Angstrom defect was invisible until v2
  delegated to the method.
- Adjustment: when two paths compute the same physical quantity, make them
  share the source and then assert they agree. The assertion is the deliverable
  as much as the shared code is.
- Scope: project
- Status: provisional
- Outcomes: +2026-11-10

### L-09 · Bound tests by a measured scale, never by a round number
- Observation: recurring. Asserted projector idempotency with atol=1e-18 on a
  matrix of Frobenius norm 3.1e4 -- fourteen orders below its round-off floor.
  Failed on noise in structurally-zero entries.
- Evidence: 2026-11-10 · `test_physical_covariance_is_idempotent`. Same
  mistake as the Coriolis threshold tests earlier in this session.
- Adjustment: before writing a numerical tolerance, print the quantity and the
  scale it should be compared against; set the bound from the measurement with
  stated headroom, and say the measured value in a comment.
- Scope: project
- Status: validated (twice in one session)
- Outcomes: +2026-11-10 (Coriolis) · +2026-11-10 (idempotency)
### L-10 · An "unstated means default" rule must survive every config entry path
- Observation: run_generic's D-09 and step-3 rules read "key absent", but
  runner/config_schema.normalize_config (load_config, the CLI path) filled in
  prior_weight 1 and widths 0.04/2.0 for every config, so neither rule ever
  reached a CLI run; scripts using yaml.safe_load + validate_config did see it.
- Evidence: 2026-11-13 · test_acceptance_test_row_consistency fixture still had
  the library prior on after the change; config_schema.py line 37.
- Adjustment: when a default depends on a key being absent, grep the schema /
  loader for setdefault on that key and test through load_config.
- Scope: project
- Status: provisional
### L-11 · A fix in one objective mode can break another
- Observation: holding the quantum reference fixed the joint bootstrap and
  broke the split one (12/12 -> 8/12 usable); nothing in my own measurements
  covered split.
- Evidence: 2026-11-13 · verify8/regress/split_rto.py re-run on b71a261 and
  on the fix-1 worktree.
- Adjustment: when a fix touches the step, the certificate or the bootstrap,
  re-run every reproduction script the verifiers left on both versions, for
  every objective mode they cover.
- Scope: project
- Status: provisional

