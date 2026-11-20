# Current task
Task: T-12
Status: done
Request: count how many base / offsets benchmark rows sit near the chi2
rescale switch (N19; D-46, auto-picked at Q-14); measurement only.

## Brief
- Outcome: for each of the 36 rows (9 molecules x base/offsets x HF/B3LYP),
  the first-pass reduced chi2, the effective dof, and how many passes ran;
  a count of rows within ~25% of a threshold (4.0 above, 0.25 below) where
  the gate is open (dof >= CHI2_RESCALE_MIN_DOF = 1). A tightening pass
  stops at each row's model-error floor, so a low-chi2 row that already
  sits on the floor is not fragile -- the passes / fit change shows it.
- How: t12/chi2run.py runs the benchmark's own main() from a fresh
  worktree (HEAD 04ef980, engine = c8904da) with the five earlier Hessian
  caches merged (no overwrite), patching only MolecularOptimizer.run /
  _optimise_once / reduced_chi_square to log, and hybrid_fit to tag each
  call with its molecule and score. The run's rows must reproduce the
  archived ones; that is the check the instrumentation changed nothing.
- Then: for rows near a threshold, the size of the jump (a sigma sweep as
  in T-11) is a follow-up, not part of this count.
- Success: 36 logged rows matching the archive, the count, and a fresh-agent
  check.

## Graph
- N1 · build · chi2run.py, both levels in the background · after: none · done · handoff: 36/36 rows reproduce; 1 row near a switch that matters (HF HNCO offsets)
- N2 · verify · fresh agent: rows reproduce the archive, counts recomputed, threshold moved to 3 / 5 · after: N1 · done · handoff: confirmed; only HF HNCO offsets changes (13.48 at T=3)
- N3 · gate · Q-15: what to do with the count · after: N2 · skipped · handoff: conventional default -- record it (backlog N19, one docstring clause); no engine change

## Progress
- 14:20 UTC: worktree t12/wt at 04ef980; 4367 cached Hessians merged from
  n13/wt, n13/wt_vf, n13/wt_vf2, t07/wt, t07/wt2; runs started (logs
  t12/chi2_{hf,b3lyp}.log, data chi2_{hf,b3lyp}.json).
- Done 14:18 UTC (2 min per level, all Hessians cached). 18 hybrid fits
  per level, matched in order (base, offsets) to the archived rows: offsets
  rows bit-exact; base rows within 4.3e-4 mA (HF acetyl fluoride; base rows
  are 5cfc5d4 vintage). Rescale-off rerun (NORESCALE=1; every run 1 pass)
  for the branch difference. Table: t12/table.txt / table.json.
- First-pass reduced chi2: 33 of 36 rows at 0.008-0.08 (HF) / 0.01-0.16
  (B3LYP), i.e. far below 0.25 -- each gets one tightening refit, which
  moves nothing (|off - on| <= 0.001 mA on every row: the rows sit on their
  model-error floor). Near a threshold (within 25%): HF HNCO offsets 3.468
  (dof 1.6) -- the one that matters, 1 pass now, ~12.8 if it crossed (T-11);
  B3LYP water base 0.217 and ozone offsets 0.226 -- harmless, since the
  tightening is a no-op. HF HNCO base 1.425 (in range). Highest elsewhere:
  0.22.
- Count: 1 of 36 base / offsets rows is fragile to the switch. Not
  measured: the other ten configurations (bob, defect, cd, track, torsion,
  lam, corr families) that feed the README's best-of table.
- N2 (fresh agent, 14:35 UTC): thresholds 3.0 / 4.0 / 5.0 on all 36 rows
  (three worktree copies, 108 fits). Only HF HNCO offsets changes: 8.1373
  -> 13.4838 at T=3 (first chi2 3.468 > 3, widened x1.862, second chi2
  2.017, 2 passes); flips only below T = 3.468. All others within 1e-5 mA.
  The tightening side is inert: first scale changes every sigma, chi2
  barely moves, the second scale is a no-op at the floor (max effect
  8.5e-4 mA). B3LYP water base / ozone offsets do change their decision at
  T=5 (1 pass instead of 2) but move 8.6e-10 / 9.6e-6 mA. The dof gate
  never binds (min dof 1.37). T=4 rerun bit-identical to mine; offsets rows
  18/18 bit-exact vs archive; base rows 8/18 bit-exact, rest <= 4.3e-4 mA
  (hybrid only; theory, ME, sigma_x exact).

## Observations

## Pending question
- Q-15 (asked 14:31 UTC, deadline 14:46 UTC; send_later trig_01Sa669HbR21Ng6fjEMiF4UQ): what's next?
  1 the same threshold check on the other ten configurations (Recommended;
  auto-pick) · 2 F14 correlated (GLS) weighting study · 3 N3 r_e^SE
  references · 4 Done for now.

## Handoff
