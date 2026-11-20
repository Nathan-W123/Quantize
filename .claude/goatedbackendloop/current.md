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
- N1 · build · T=4 run, both levels · after: none · HF done, B3LYP resumed · handoff: HF 99/99 rows reproduce
- N2 · build · T=3 and T=5 runs · after: N1 · HF running; B3LYP only rows with first-pass chi2 > 2.5 · handoff: -
- N3 · verify · fresh agent recomputes the flips and checks best-of impact · after: N2 · pending · handoff: -

## Progress
- 14:52 UTC: t13/wt3, wt4, wt5 copied from t12/wt (code = HEAD; only
  bytecode differs); T=4 run started (t4_{hf,b3lyp}.log / .json).
- Container restarted ~15:35 UTC; scratch survived. HF T=4 had finished
  (99 rows, 151 hybrid fits, 31 min); B3LYP had 16 / 81 rows saved (46
  min -- 102 B3LYP and 143 HF Hessians were not in the cache: the
  5cfc5d4-vintage rows differ in the 9th decimal, so their tracked
  geometries miss). Stale lock (pid 198, no logger running) removed;
  B3LYP T=4 resumed at 15:40 (log part 2: t4_b3lyp_p2.json; part 1 kept),
  HF T=3 / T=5 started alongside with the new Hessians shared.
- HF T=4: 79 / 99 rows bit-exact vs archive, the rest <= 4.6e-4 mA (all
  5cfc5d4 vintage). First-pass chi2 > 2.5 in only: isocyanic_acid
  offsets+bob / offsets+defect / offsets+cd / offsets+track, all 3.468 (the
  same fit as its offsets row); and fluoroacetylene track /
  offsets+track / track_quadratic, whose 2nd and 3rd tracked fits start at
  16.00 (dof 6.0 = n_obs) and land on 1.000 after a x4 widening -- far
  from 4, so not threshold-fragile, but odd (see Observations).

## Observations
- 2026-11-20 · fluoroacetylene's tracked hybrid fits (HF, track and offsets+track alike) start at reduced chi2 16.00 with dof 6.0 = n_obs (no data leverage), then 1.000 after the x4 widening; the first fit in the same row has 0.05 / dof 4.3. The rows get worse with tracking (13.59 -> 19.16, 4.36 -> 4.96). I guessed a sigma state leak between tracked fits; the engine copies each isotopologue dict, so that guess is wrong. Cause unknown -- backlog, not this task.

## Pending question

## Handoff
