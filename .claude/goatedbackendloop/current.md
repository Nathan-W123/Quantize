# Current task
Task: T-27
Status: verifying
Request: (D-71, auto-picked at Q-38) F3 hygiene: the UTF-8 BOMs, `scripts/` imported as a package without `__init__.py`, and the imprecise rigid-mode test assertion -- confirm each, fix the confirmed ones, CI selection.

## Brief
- Outcome: source files parse as plain UTF-8; `import scripts.X` cannot be
  shadowed by another `scripts` package; the rigid-mode test fails when rigid
  modes leak.
- How: strip BOM bytes (first line only); add `scripts/__init__.py`; test the
  covariance's share along a rigid basis built in the test, plus a measured
  width bound.
- Success: 0 BOM files, all tracked .py parse; shadow check imports ours;
  new test fails under a leaky physical_basis; CI selection green.

## Graph
- N1 · diagnose · confirm each item · after: none · done: 57 BOM files (not 2), all fail ast.parse; namespace `scripts` loses to a later regular one (ImportError); old test passes with rigid leak (widths 0.0119 A < 1.0) · handoff: -
- N2 · build · strip, __init__, test · after: N1 · done · handoff: -
- N3 · verify · leaky-basis monkeypatch fails the new test (share 1.00); shadow check imports ours; CI selection · after: N2 · running · handoff: -

## Progress
- New test: share 4e-17, widths 2.8e-4 A; under leak: share 1.0, 1.2e-2 A.

## Observations
- F3's BOM item named 2 files; the real count was 57 (L-09: measure before writing).

## Pending question
