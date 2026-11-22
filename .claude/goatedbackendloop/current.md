# Current task
Task: T-28
Status: building
Request: (D-72, auto-picked at Q-39) F3 test-suite hygiene: tests writing run directories into the repo instead of tmp_path; force-field multireference tests KeyError on a partial gitignored output/theory_geometries_hf_6-31g.json.

## Brief
- Outcome: running the CI selection leaves the working tree's files (tracked
  and ignored) unchanged; the force-field tests skip with a reason when the
  cached theory file lacks their molecule.
- How: a pytest plugin (scratchpad) snapshots the repo around every test and
  lists what each one creates or modifies; fix each writer to use tmp_path
  (config output root / run_dir override); skip on a missing key.
- Success: plugin run shows no writes; the force-field tests skip on a
  partial file and run on a full one; CI selection green.

## Graph
- N1 · diagnose · per-test write log · after: none · running · handoff: -
- N2 · build · redirect writers, skip on missing key · after: N1 · pending · handoff: -
- N3 · verify · plugin rerun clean; partial-file fixture; CI selection · after: N2 · pending · handoff: -

## Progress

## Observations

## Pending question
