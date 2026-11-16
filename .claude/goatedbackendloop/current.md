# Current task
Status: done (N4)
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

## Observations
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
