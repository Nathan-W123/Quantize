# Current task
Task: T-20
Status: building
Request: (D-60, auto-picked at Q-27) fix the validator's inertial-defect false positive on non-planar molecules (backlog F3 part).

## Brief
- Outcome: no warning for a real non-planar molecule; still a warning for
  impossible or mislabelled constants.
- How: compare the observed defect with the config structure's (coords
  geometry, the isotopologue's masses), tolerance 1.0 + 10% of the
  structure's; without coordinates, one-sided (> +1.0); separate A >= B >= C
  ordering warning.
- Success: tests that fail on the old code (non-planar quiet, ordering
  message) and one that fails on a one-sided-only variant; shipped configs
  and the suite unchanged otherwise.

## Graph
- N1 · diagnose · measure observed vs structure defects · after: none · done (t20/defects.py, startgeo.py) · handoff: -
- N2 · build · the check + tests · after: N1 · done (runner/usability.py; 6 new tests) · handoff: -
- N3 · verify · tests fail on the old code / one-sided variant; shipped configs; CI selection · after: N2 · running (CI selection) · handoff: -

## Progress
- N1: observed - reference-structure defect, 37 benchmark isotopologues:
  -0.070 to +0.113 amu*A^2 (non-planar AF, FE, CH2ClF match to <= 0.08);
  against the benchmark start geometries <= 0.26 (CH2ClF, 7.5%). Shipped
  coords configs: water gaps <= 0.07, fluorobenzene +5.6 (its known bad
  constants).

- N2/N3: new tests on the old code: 4 of 6 fail (non-planar with and without
  coordinates, the ordering warning, the structure helper); each of three
  branch-removed variants of the new code fails exactly its own test (no
  structure comparison -> the planar-geometry test; no positive fallback ->
  impossible-without-coordinates; no ordering check -> swapped labels).
  Shipped configs, old -> new: example_methanol_lam.yaml flagged -> quiet
  (defect -3.16 to -3.24: its methyl H's, SMILES geometry, one-sided rule);
  fluorobenzene flagged in both; the rest quiet in both; example_water.yaml
  errors in both (legacy format, before the check).

## Observations

## Pending question

## Handoff
