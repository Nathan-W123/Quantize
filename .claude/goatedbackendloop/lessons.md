# Project lessons
Last consolidated: 2026-11-20 (161 -> ~80 lines: L-08 merged into L-07;
validated lessons compressed to adjustment + one line of evidence; outcome
lists shortened). Earlier: 2026-11-17 (219 -> ~110). Reasoning before
2026-11-10 lives in the branch's commit messages (this directory was wiped
once by a container swap).

### L-P02 · Cross-check on the CI interpreter before pushing a numerics fix
- Adjustment: for a change touching linear algebra, conditioning or tolerances, run the affected tests under CI's Python before pushing.
- Evidence: 2026-11-10 · a nan-asserting fix passed on 3.11, failed on 3.12. · Scope: project · Status: provisional

### L-P03 · Long background runs get reaped; bank the expensive part to disk
- Adjustment: scope a run to one container lifetime, make the expensive intermediate durable (hessian_cache) first, and make the run resumable (saved rows, a part-N log per resume).
- Evidence: 2026-11-10 · the MP2 benchmark died four times; cached Hessians let it resume. +2026-11-20 (T-13: a container restart and a 2 h timeout cost only the row in progress). · Scope: project · Status: validated

### L-P04 · Verify a computed quantity against a value already in the repo
- Adjustment: before adding a derived quantity -- including a hand-built stand-in for a library function's output -- compute it for a case whose answer is in the repo and compare values, shape and multiplicity; no such case is itself a reason for suspicion.
- Evidence: 2026-11-10 · a Costain r_s floor gave 852 mA for ozone against the docstring's 4.1. -2026-11-20 (T-10: candidate offsets one per atom vs measure_offsets' one per bond; a verifier caught it). · Scope: project · Status: validated

### L-P06 · A library module with only a script consumer is misplaced
- Adjustment: before committing a module under backend/, list its importers; if none are in backend/, wire it in or move it.
- Evidence: 2026-11-10 · force_field_diagnostics had only the benchmark. · Scope: project · Status: provisional

### L-07 · A finite uncertainty is not a checked uncertainty; two paths must agree
- Adjustment: assert a bound in the quantity's own units (a bond sigma below the bond length), not finiteness; where two paths compute one quantity, make them share the source and assert they agree in scale -- the assertion is the deliverable. (Merged L-08, 2026-11-20.)
- Evidence: 2026-11-10 · sigma 112 A on a 0.958 A bond through a green suite; it showed only once v2 delegated to the method (2871x disagreement). · Scope: project · Status: provisional

### L-09 · Bound tests by a measured scale, never by a round number
- Adjustment: print the quantity and its scale before writing a tolerance; set the bound from the measurement with stated headroom.
- Evidence: 2026-11-10 · atol=1e-18 on a norm-3.1e4 projector. · Scope: project · Status: validated

### L-11 · A fix in one objective mode can break another
- Adjustment: when a fix touches the step, the certificate or the bootstrap, rerun every reproduction script the verifiers left, on both versions, for every objective mode they cover.
- Evidence: 2026-11-13 · holding the quantum reference fixed joint and broke split (12/12 -> 8/12 usable). · Scope: project · Status: provisional

### L-12 · Check what a "far from the truth" verdict is measured against
- Adjustment: before fixing to a reference, ask what the reference procedure does from the point being judged; if it does not move, the point is valid and the comparator is wrong.
- Evidence: 2026-11-16 · split termini 17-25 widths apart, both stationary; two fixes tuned to the wrong comparator failed. · Scope: project · Status: provisional

### L-14 · One value, one reader: validator, builder and CLI share the rule
- Adjustment: when a fix adds a check for a config key, grep every reader and route them through one helper; read provenance from the raw file, never after normalization; read optional numbers with `is not None`, never `or`; test through load_config.
- Evidence: 2026-11-17 · round 10: rng_seed read three ways (7.0 -> 7 / 7 / 42), optimizer keys in two lists. · Scope: project · Status: validated

### L-15 · A green test must be shown to fail, for every branch it claims
- Adjustment: break the guarded property in the code and see the test fail, once per branch its docstring names, with a fixture that exercises it; assert work skipped (call counts), not speed ratios; compare against an inlined old formula, never the function's wrapper; assert the fixture's premise in the test.
- Evidence: 2026-11-17 · round 10 A-4 / A-5 (a residual-only file passing, a sampler compared with itself); +2026-11-19 (a rescale test that widened, not tightened). · Scope: project · Status: validated

### L-16 · Break an aggregate diagnostic down by row before naming its cause
- Adjustment: before attributing a failed check or ratio to a cause, print it per species and component and check whether the outliers share a property (precision, source, site).
- Evidence: 2026-11-19 · SE100 vinyl fluoride's 0.67% "miss" was five A rows of bad input; the defect probe's 1.66 was the round-number-A species. · Scope: project · Status: provisional

### L-17 · Run a remedy on every molecule it changes before calling it a fix
- Adjustment: list every molecule a change can move and run all of them on every yardstick before reporting a gain; say n when it is small.
- Evidence: 2026-11-20 · per-bond widths: CH2ClF 17.64 -> 2.44 mA, HNCO angle 1.49 -> 16.47 deg. +T-07 (ozone vs r_e, trans H2O2) +T-10 (HNCO moved by a C-Cl member). · Scope: project · Status: provisional

### L-18 · In results prose, name the row and file each number comes from
- Adjustment: quote each figure with its config and yardstick (best-of / offsets row; vs primary / vs r_e) and its file; generate before/after pairs from the JSON with a script, derived figures too; have the prose fact-checked against the data.
- Evidence: 2026-11-20 · a fact-check of 622b573 found 4 mixed-row errors. +T-10; -T-11 (README vs docstring mixed by a two-file grep); -T-08 (a hand-computed estimate was wrong). · Scope: project · Status: provisional

### L-19 · A negative result through the engine is conditional on its configuration
- Adjustment: when an engine experiment finds "no effect", name the settings it depends on (weighting, prior, width) and say whether another setting could change it.
- Evidence: 2026-11-20 · T-09: a 13C species 6.00 -> 5.97 mA as configured, 6.63 -> 6.15 with correlated weights. +T-10 (D-43 names its width-rule dependence). · Scope: project · Status: provisional
