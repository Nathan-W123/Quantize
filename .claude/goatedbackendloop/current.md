# Current task
Task: T-28
Status: done
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
- N1 · diagnose · per-test write log · after: none · done: 3 tests in test_endtoend_water_pipeline.py (script functions tve._pipeline / tune.run, cwd-relative output/runs), 4 dirs per suite run; nothing else · handoff: -
- N2 · build · run_root on the two script functions, tests pass tmp_path (8cc9c59); _geom skips on a missing file or molecule (c507742) · after: N1 · done · handoff: -
- N3 · verify · full suite with per-test and whole-session snapshots: no writes but my own backlog edit; caches none / 11-17 partial / full: skip / skip (KeyError before) / pass 2.2 s; CI selection 1631 passed, 15 skipped · after: N2 · done · handoff: -

## Progress

## Observations
- Measuring the writers per test (plugin) found 1 file of 3 suspected; no guessing at callers.
- The multireference tests always skip in a fresh clone (gitignored cache): noted in F3, not widened.

## Pending question
- Q-40 (asked 00:21 UTC, deadline 00:36): what's next. 1 N19 the chi-square rescale switch: measure how often fits sit near it, then propose (recommended) · 2 embed the two 3-atom geometries so the multireference tests run in a fresh clone · 3 a user's-call item: model_error.auto as default, N22, N23 (never auto-picked) · 4 done for now.
