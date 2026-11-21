# Current task
Task: T-21
Status: building
Request: (D-61, auto-picked at Q-28) re-measure the README's sparse-data sigma rows that predate the Cartesian-alpha fix (backlog N2) -- widened to the whole "Which objective" table once the scripts' "split" legs turned out to run joint.

## Brief
- Outcome: the README table and prose state, per case, which objective and
  width actually wins at HF/6-31G on the current engine, with numbers.
- How: t21/scan.py runs each case with each script's own data, start and
  settings, passing objective_mode explicitly (split; joint at the
  calibrated default 0.020; joint 0.003-0.050), and records what each
  script's own "split" leg resolves to.
- Tradeoff: none in the code; the guidance may change.
- Success: the table; the scripts' split legs fixed to pass split; README
  rows and prose updated and fact-checked.

## Graph
- N1 · diagnose · scans: mono (3 molecules x 2 data levels), fb1, fb8 · after: none · running (t21/scan_*.log) · handoff: -
- N2 · build · scripts: pass objective_mode="split" where a leg is labelled split · after: N1 · todo · handoff: -
- N3 · build · README table + prose · after: N1 · todo · handoff: -
- N4 · verify · fresh agent fact-check against t21 JSON · after: N3 · todo · handoff: -

## Progress
- Found by reading: MolecularOptimizer's default objective_mode became
  "joint" at 7216d87 (2026-08-14; "split" at c1f031a). Legs labelled split
  that pass no objective_mode: fluorobenzene_full_data.py ("split", dict()),
  monofluoro_benchmark.py ("hybrid, split", {}), monofluoro_tune.py's
  sv_threshold scan. The scan records what each resolves to.
- monofluoro_benchmark.py itself already uses sigma 0.030 (its docstring:
  0.005 overstated RHF/6-31G five-fold), against the README's 0.005.

## Observations

## Pending question

## Handoff
