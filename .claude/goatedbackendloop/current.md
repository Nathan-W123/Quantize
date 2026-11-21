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
- N1 · diagnose · scans: mono (3 molecules x 2 data levels), fb1, fb8 · after: none · done (t21/scan_*.json) · handoff: -
- N1b · verify · are the plateau stops optima? refit + restart + tight (conv_freq 0.01, 200 it) · after: N1 · running (t21/restart_*.log) · handoff: -
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

- N1 (HF/6-31G, objective explicit; every script's as-written "split" leg
  resolves to joint at 0.020). RMS bond mA (theory):
  VF parent 18.24 split / 18.31 joint-default / 18.61 j0.005 (19.05);
  VF all 15.87 / 17.41 / 18.38, j0.050 15.38; AF parent 14.78 / 15.20 /
  15.61 (15.75); AF all 14.45 / 14.58 / 15.06, j0.050 14.38; FE parent
  11.55 / 13.14 / 14.27 (14.99); FE all 12.12 / 12.49 / 13.53, j0.050 11.32.
  Fluorobenzene 1 iso: theory 11.42, expt-alone 13.63, split 10.70, joint
  default 9.93, j0.005 9.80 (best of the grid by 0.04). 8 iso: expt-alone
  12.92, split 10.38, joint default 9.74, j0.050 9.16 -- far from the
  README's 4.0 (split) / 8.2 (expt alone) / 8.9 (j0.005).
  Every hybrid fit ends on the plateau rule (fb8: all at iteration 11, the
  earliest the 10-iteration window allows), so N1b checks they are optima.

## Observations

## Pending question

## Handoff
