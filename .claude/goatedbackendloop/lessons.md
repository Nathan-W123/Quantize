# Project lessons
Last consolidated: 2026-11-17 (219 -> ~110 lines: L-10 merged into L-14;
L-P01 and L-13 merged into L-15; validated lessons compressed; L-P05 moved to
learnings.md as L-06, since it applies to all projects).

Recreated 2026-11-10 after a container swap wiped this directory (it is
gitignored, so a fresh clone does not carry it). The full reasoning for
everything before this point is in the commit messages on
claude/repo-audit-calculations-gtotzw, which is why nothing important lived
only here.

### L-P02 · Cross-check on the CI interpreter before pushing a numerics fix
- Adjustment: for any change touching linear algebra, conditioning or
  tolerances, run the affected tests under the CI interpreter version before
  pushing (a venv costs a minute; a red CI round trip ten).
- Evidence: 2026-11-10 · a nan-asserting fix passed on 3.11 and failed on 3.12.
- Scope: project · Status: provisional

### L-P03 · Long background runs get reaped; bank the expensive part to disk
- Adjustment: scope a long run to one container lifetime, make the expensive
  intermediate durable (hessian_cache) before launching, and capture the exit
  code in a wrapper so a silent death is distinguishable from a crash.
- Evidence: 2026-11-10 · the MP2 benchmark died four times; 24 cached Hessians
  let the last relaunch resume.
- Scope: project · Status: validated

### L-P04 · Verify a computed quantity against a value already in the repo
- Adjustment: before adding a derived quantity, compute it for a case whose
  answer is written down in the repo and compare; no such case is itself a
  reason for suspicion.
- Evidence: 2026-11-10 · a Costain-rule r_s floor gave 852 mA for ozone
  against the docstring's 4.1.
- Scope: project · Status: validated · Outcomes: +2026-11-19 (Vogt's CH2ClF r_e as the
  control row is what made the SE100 HNCO / vinyl fluoride check readable; reproducing the archived
  base rows to the digit made the A-row pull numbers the benchmark's own)

### L-P06 · A library module with only a script consumer is misplaced
- Adjustment: before committing a module under backend/, list its importers;
  if none are in backend/, wire it into the library path or move it.
- Evidence: 2026-11-10 · force_field_diagnostics had only the benchmark.
- Scope: project · Status: provisional

### L-07 · A finite uncertainty is not a checked uncertainty
- Adjustment: for any reported physical quantity, assert a bound in the units
  of the thing itself (a bond sigma below the bond length), not finiteness;
  where two paths compute from one intermediate, assert they agree in scale.
- Evidence: 2026-11-10 · sigma 112 A on a 0.958 A bond through a green suite.
- Scope: project · Status: provisional · Outcomes: +2026-11-10

### L-08 · Deduplicate to create a cross-check, not just to tidy
- Adjustment: when two paths compute the same quantity, make them share the
  source and assert they agree; the assertion is the deliverable.
- Evidence: 2026-11-10 · the 112 A defect showed only once v2 delegated to the
  method (2871x disagreement).
- Scope: project · Status: provisional · Outcomes: +2026-11-10

### L-09 · Bound tests by a measured scale, never by a round number
- Adjustment: print the quantity and the scale it compares against before
  writing a tolerance; set the bound from the measurement with stated headroom.
- Evidence: 2026-11-10 · atol=1e-18 on a norm-3.1e4 projector; Coriolis
  thresholds the same way.
- Scope: project · Status: validated

### L-11 · A fix in one objective mode can break another
- Adjustment: when a fix touches the step, the certificate or the bootstrap,
  re-run every reproduction script the verifiers left, on both versions, for
  every objective mode they cover.
- Evidence: 2026-11-13 · holding the quantum reference fixed joint and broke
  split (12/12 -> 8/12 usable).
- Scope: project · Status: provisional

### L-12 · Check what a "far from the truth" verdict is measured against
- Adjustment: before fixing to a reference, ask what the reference procedure
  does from the point being judged; if it does not move, the point is valid
  and the reference is the wrong comparator.
- Evidence: 2026-11-16 · split termini 17-25 widths apart, both stationary;
  two fixes tuned to the wrong comparator failed.
- Scope: project · Status: provisional

### L-14 · One value, one reader: validator, builder and CLI share the rule
- Observation: a config value read by two or three paths with different rules
  is this project's commonest config defect.
- Evidence: 2026-11-13 · (was L-10) normalize_config's setdefault hid
  "key absent" defaults from every CLI run. 2026-11-17 · round 10: rng_seed
  read three ways (7.0 -> 7 / 7 / 42), the optimizer keys in two lists, the
  workdir name default twice, the seed label read after normalization filled
  42, sigma fractions through an `or` chain (6bf3fe2, 0d1db72, 2b1e247).
- Adjustment: when a fix adds a check for a key, grep every reader of it and
  route them through one helper or one tuple; read provenance (stated or
  default) from the raw file, never after normalization; read optional numbers
  with `is not None`, never `or`; test through load_config.
- Scope: project · Status: validated (two sessions)

### L-15 · A green test must be shown to fail, for every branch it claims
- Observation: tests here have passed for the wrong reason four ways: by
  requiring the bug (a 0.0 width as the tight end of a comparison), by timing
  a ratio a later speed-up moved onto its line, by a fixture that never
  exercised one of the blocks the test's docstring claimed (no distortion
  rows), and by comparing a function with the wrapper that now calls it.
- Evidence: 2026-11-10 · (was L-P01) two geometry_uncertainty tests needed the
  -2.3e13 variance's 0.0. 2026-11-17 · (was L-13) the 2x timing test after
  6952211 (ratio 1.3-3.4x); round 10 A-4 (always building _cd_jacobian left the
  residual-only file passing) and A-5 (the rto sampler compared with itself);
  fixed in 4c9a2dd, 899e09f.
- Adjustment: for each new test, break the property it guards in the code
  (a mutation) and see it fail -- once per branch or block its docstring
  names, with a fixture that exercises that branch; assert the work skipped
  (call counts), not a speed ratio; compare against an inlined copy of the
  old formula, never the function's own wrapper; print both sides of an
  inequality before trusting it, suspecting exact 0.0, 1.0 or nan; assert
  the fixture's premise in the test itself (a rescale test passed while
  widening, not tightening -- 2026-11-19).
- Scope: project · Status: validated (two sessions) · Outcomes: +2026-11-19

### L-16 · Break an aggregate diagnostic down by row before naming its cause
- Observation: an aggregate mismatch pointed at the wrong culprit twice in one
  session; the per-row view found the real one both times.
- Evidence: 2026-11-19 · SE100's vinyl fluoride r_e "missed" by 0.67% RMS,
  but every B and C agreed to 0.04% and five A rows carried it all (bad input
  data, not a bad reference); the defect probe's 1.66 for vinyl fluoride had
  been read as correction bias, but the parent's ratio is 0.01 and the
  round-number-A species carry 1.9-2.0.
- Adjustment: before attributing a failed consistency check or diagnostic
  ratio to a cause, print it per species and per component; check whether the
  outliers share a property (precision, source, substitution site).
- Scope: project · Status: provisional
