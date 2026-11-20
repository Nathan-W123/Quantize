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
- N1 · build · T=4 run, both levels · after: none · done · handoff: HF 99/99, B3LYP 81/81 rows reproduce
- N2 · build · T=3 and T=5 runs · after: N1 · HF done; B3LYP 8 low-band rows at T=3 / T=5 running · handoff: HF: 4 rows flip at T=3, none at T=5
- N3 · verify · fresh agent recomputes the flips and checks best-of impact · after: N2 · HF half running (own HEAD copy, threshold set by source edit); B3LYP half after N1 · handoff: -

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
- 17:34 UTC: part 2 hit its 7000 s timeout at 41 / 81 B3LYP rows (25 rows in
  ~2 h while the HF runs shared the 4 CPUs). HF T=3 / T=5 finished.
- HF T=3 / T=5 vs T=4 over all 99 rows (hybrid_rms_ma, _deg, _ma_vs_re,
  tolerance 1e-5): only isocyanic_acid offsets+bob / offsets+defect /
  offsets+cd / offsets+track move, all at T=3, all 8.1373 -> 13.4838 mA
  (deg 1.835 -> 2.919) -- the offsets row's T-12 flip. T=5: nothing moves.
- Best-of impact (RHF): HNCO's best hybrid is 8.14, shared by all six
  offsets-family rows (offsets+torsion too: torsion_applied False, so the
  same fit; not rerun, inferred). At T=3 it would read 13.48, moving the
  README RHF best-of hybrid mean 5.34 -> ~5.93; still below theory (22.10)
  and ME (48.17) for HNCO, so no win count changes. Nothing at T=5.
- B3LYP: first-pass chi2 of the 42 rows logged so far is <= 0.8 (scan.py).
- 17:36 UTC: chi2run2.py takes molecule keys; B3LYP T=4 resumed
  as two processes (OMP_NUM_THREADS=2 each): wt4 fluoroacetylene +
  chlorofluoromethane -> t4_b3lyp_p3.json; wt5 (wt4's cache merged in)
  water + ozone + isocyanic_acid -> t4_b3lyp_p3b.json. Task b8ihmpgy9.
- 17:52 UTC: both halves done in ~16 min (cache hits). B3LYP T=4: 81 / 81
  rows reproduce the archive (42 bit-exact, worst 1.4e-4 mA,
  fluoroacetylene track_quadratic); merged copy t13/b3lyp_t4_merged.json.
- B3LYP checks (every pass, not only the first): max 0.763; none in [3, 5].
  But 8 rows have checks in [0.2, 1/3]: water lam / bob / defect / track /
  track_direct / track_quadratic (0.2174) and ozone offsets+bob /
  offsets+defect (0.2260). They tighten at T=4 (lo 0.25) and T=3 (lo 0.333)
  and not at T=5 (lo 0.2), and a second pass ran at T=4 (the floor did not
  hold every row). The scale factor is clip(sqrt(chi2), 1/4, 4), not a
  function of T, and no B3LYP check lies in [0.25, 1/3] or [3, 4], so T=3
  takes T=4's path exactly. HF had no check in either band.
- My D-47 cut (B3LYP T=3/5 only for first-pass chi2 > 2.5) missed the low
  side; corrected: those 8 rows rerun at T=3 and T=5 (wt3, wt5; wt5's T=4
  file moved to output/_aside), task bsp1pbh2a.

## Observations
- 2026-11-20 · fluoroacetylene's tracked hybrid fits (HF, track and offsets+track alike) start at reduced chi2 16.00 with dof 6.0 = n_obs (no data leverage), then 1.000 after the x4 widening; the first fit in the same row has 0.05 / dof 4.3. The rows get worse with tracking (13.59 -> 19.16, 4.36 -> 4.96). I guessed a sigma state leak between tracked fits; the engine copies each isotopologue dict, so that guess is wrong. Cause unknown -- backlog, not this task.
- 2026-11-20 · my plan's B3LYP cut (rerun only rows with first-pass chi2 > 2.5) looked at one side of a two-sided test; scanning every check against both bands (lo = 1/T) found 8 rows the cut would have skipped. Scan the whole decision rule, not the side the last finding was on.

## Pending question

## Handoff
