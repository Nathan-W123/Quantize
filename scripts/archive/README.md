# Archived investigation scripts

One-off studies, each written to answer a single question; the answer is in
the script's own docstring and in the commit that added it. None is imported
by the engine, the tests or another script, and none is mentioned in the
documentation, so they are kept here as a record rather than maintained.

They are not covered by the test suite and may have drifted from the current
API. They still find the repository root from this directory and run from the
repository root, e.g. `python scripts/archive/run_water_endtoend.py`.

Moved here on 2026-11-19 by the bloat audit; later one-off studies are added
here directly.

| Script | Question it answered | Last changed |
|---|---|---|
| `alpha_level_ladder.py` | Is the alpha correction's error set by the expansion geometry or the force field? | 3e3a1e0 2026-09-29 |
| `bob_validation.py` | Does the data support the BOB u-parameters the engine assumes? | 74373d2 2026-10-05 |
| `build_monofluoro_full_report.py` | Build the full monofluorinated benchmark report as a PDF. | 7216d87 2026-08-14 |
| `correction_scale_detector.py` | Can the data tell us the correction is biased, with no reference structure? | 45e94b5 2026-11-12 |
| `correction_selfconsistency.py` | Does the correction need to follow the geometry the optimiser moves to? | fae16b7 2026-09-29 |
| `fluorobenzene_three_way.py` | Fluorobenzene: theory alone vs experiment alone vs hybrid. | 7216d87 2026-08-14 |
| `g_tensor_validation.py` | Validate the rotational g-tensor against every measured value we could verify. | 2ba7b32 2026-09-25 |
| `hybrid_sigma_diagnostic.py` | Is the hybrid being dragged off a good prior by a sigma it has not earned? | dcbe358 2026-09-19 |
| `perbond_prior_widths.py` | Would widening the prior only on uncalibrated bond classes fix the underdetermined split? (No: CH2ClF better, HNCO worse; D-33.) | 2026-11-20 |
| `reference_structure_sensitivity.py` | How much of the reported error is the engine, and how much is the ruler? | ffb2160 2026-10-07 |
| `run_water_endtoend.py` | Drive a real end-to-end water run on the analytic PES. | 7216d87 2026-08-14 |
