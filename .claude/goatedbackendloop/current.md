# Current task
Status: done (search follow-ups: defect fix + small fixes)
Request: N4 -- config runs carry no model-error term (no config key reaches
sigma_systematic_constants).

Earlier tasks: steps 1-8, rounds 1-9, N1, N6 -- their progress is in this
file's git history up to f5b6c36; decisions D-01..D-25, lessons L-01..L-12.

## Brief
Finding: partly true. Rows with a B0 -> Be correction of stated sigma carry
the model error (Hessian-built: 2-4% of the constant; alpha_mhz: 0.1-0.4%);
raw-B0 configs carry none (water: 0.28 mA vs 6.9). Every config run printed
"measurement precision ONLY", corrected or not. User chose: keys plus an
honest label, default unchanged (D-26).

## Progress
- DONE 61c0cff: sigma_systematic_mhz / sigma_includes_systematic per
  isotopologue; model_error {auto, isotopologue_correlation};
  model_error_sources() + model_error_warnings() from the fitted rows
  (replacing the constructor warning); bundle model_error; per-component
  lists follow the internal_rotor_bc filter; validation. Shipped water
  configs bit-identical. Suite with the change: 1586 passed, 23 skipped.

- Next request (user, free text): "search for anything that will improve
  accuracy or code efficiency". Two background agents: efficiency profiling
  of the user paths (benchmark fit, quantize uncertainty bootstrap,
  posterior, slow tests), and accuracy levers on the 9-molecule HF/6-31G
  benchmark with reference-independent evidence. Trades flagged (D-03).

- Search results (two agents): efficiency -- engine time dominated by the
  Python-loop rotational Jacobian; CD operator algebra rebuilt per call;
  benchmark fits recompute the table's Hessian; trial Jacobian discarded;
  rto shift root per draw. Accuracy -- only free change a one-line
  defect-probe fix (A1); GLS weighting (F14a) -3.8 mA on r_s references but
  worse on r_e bonds; offsets -2.9 mA at 7-37 Hessians per molecule;
  nothing else above reference resolution.
- User chose the free efficiency wins only. DONE: 6952211 closed-form
  Jacobian (2.4e-14 rel; fit engine 1.9-5.5x), ff836fb CD cache
  (bit-identical; CD run 25.0 -> 15.3 s), fb7ea3e benchmark Hessian reuse
  (fits 75-79% faster; <= 2.8e-11 A), c7ea83f trial Jacobian skipped
  without CD rows (bit-identical; kept with CD rows, where skipping moved
  fits within tolerance), e84d1e7 rto shift sampler (bit-identical; test
  11.6 -> 0.25 s). Clean-checkout suite at e84d1e7: 1590 passed, 23
  skipped, 127 s (was 137-151).

- User then chose "Defect fix + small fixes". DONE: 024caec defect probe
  read at the prior target (CFM restart 5.85 -> 0.000 mA, AcF 1.74 ->
  0.0005; first fits identical), 14125f3 the config's rng_seed seeds the
  draws without --seed (bundle records seed and source), 9aa7c28 docs commit
  dating the README table (4f0ece6, 2026-10-05) and correcting the
  water/B3LYP claim (re-measured 0.050 mA at sigma_x 0.007; 0.403 / 0.753
  wider). Clean-checkout suite: 1591 passed, 1 failed: the timing test
  test_residual_only_is_actually_cheaper (ratio now 1.3-3.4x since the
  closed-form Jacobian made the full path cheap), rewritten to count
  Jacobian builds (mutation-checked), 4c9a2dd; clean suite 1592 passed.

## Observations
- 2026-11-17: the clean-checkout suite caught a timing test that a speed-up
  (6952211) had pushed onto its 2x line: it passed at 1590 earlier only by
  chance. Counting the skipped work is the exact check; timing ratios move
  whenever either path gets faster.
- E4: the agent's "bit-identical for Cartesian fits" held, but the CD-path
  difference it flagged as small was a trajectory change (multistart spread
  1e-5 relative), not rounding; checking the whole payload, not just the
  best geometry, showed it. Guarded instead of accepted.
- The step-7 restart changed two things at once (a fresh optimizer is also a
  fresh Hessian), and the backlog item read the result as a stopping-rule
  defect. Splitting the restart into same-object / fresh / refresh took one
  9-molecule run (~4 min) and named the real cause. Separate the variables a
  "restart" changes before diagnosing it.

- Round 9: my 563b1a1 narrowed a catch correctly for runtime errors and
  in doing so re-exposed config errors that only the constructor caught.
  When narrowing an exception handler, list what it used to catch that is
  still the caller's fault, and move those checks earlier.

## Handoff
