# Current task
Task: T-13
Status: building
Request: the T-12 chi2-threshold check on the benchmark's other
configurations (D-47, auto-picked at Q-15); measurement only.

## Brief
- Outcome: per (molecule, configuration, level) row, the first-pass
  reduced chi2 of every hybrid fit (tracking rows run several), and
  hybrid_rms_ma at thresholds 3 / 4 / 5; the list of rows that move by
  more than 0.01 mA, and whether any of them is a best-of entry in the
  README table.
- How: t13/chi2run2.py (t12's logger plus THRESH= and a CONFIGS lookup tag
  for the configuration), in three copies of the t12 worktree (wt3, wt4,
  wt5; sources identical to HEAD, merged Hessian cache). HF: 11 configs;
  B3LYP: 9 (no cd rows were ever run there). The T=4 run must reproduce
  the archived rows. Torsion / offsets+torsion need relaxed scans (not
  cached; B3LYP ones outlast 2 h) and come last, HF only if at all.
- Success: the count over 99 + 81 rows, the archive reproduced, and an
  independent check of the flips.

## Graph
- N1 · build · T=4 run, both levels · after: none · running · handoff: -
- N2 · build · T=3 and T=5 runs · after: N1 · pending · handoff: -
- N3 · verify · fresh agent recomputes the flips and checks best-of impact · after: N2 · pending · handoff: -

## Progress
- 14:52 UTC: t13/wt3, wt4, wt5 copied from t12/wt (code = HEAD; only
  bytecode differs); T=4 run started (t4_{hf,b3lyp}.log / .json).

## Observations

## Pending question

## Handoff
