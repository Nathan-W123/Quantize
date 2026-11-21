# Current task
Task: T-22
Status: done
Request: (D-62, auto-picked at Q-29) measure why joint fits from a displaced start stop short of their optimum, on more molecules and starts (backlog N23); measurement first, any change proposed before it is made.

## Brief
- Outcome: per molecule and start, how far the plain joint fit (the CLI's
  objective, no target) lands from its settled answer, and what the
  engine's at-answer refresh (one Hessian) or a 1 mA refresh threshold
  recovers, at what cost (Hessians, gradients, seconds).
- How: t22/stopshort.py; settled = restart chain until three passes in a
  row move < 1e-5 A RMS (L-22: a trend, not one pass).
- Tradeoff: any remedy in the plain fit costs every run (D-03 preference:
  flag, the user decides).
- Success: the table; a proposal (or not).

## Graph
- N1 · diagnose · stop-short on 9 molecules x {warm, cold} · after: none · done (t22/stopshort.json) · handoff: -
- N2 · verify · fresh agent, own experiment through the CLI (HCOF, CH2ClF; theory-minimum and displaced starts) · after: N1 · done (t22/verify/results) · handoff: -
- N3 · gate · propose (or not) · after: N1, N2 · done: flagged, not changed (D-63, auto-picked at Q-30); README Running note + refresh_hessian_at_answer docstring · handoff: -

## Progress
- Code read: refresh_hessian_at_answer exists (one Hessian, re-converge,
  up to 3 passes); only `quantize uncertainty` (cli.py:654) calls it; its
  docstring: plain fit skips it to save a Hessian; measured at HF/6-31G on
  nine molecules (warm starts) under 0.8 mA of geometry.
- Water trial: plain fit 0.09 mA (warm) / 0.29 mA (cold, max atom) from its
  settled answer, bond RMS 1.98 / 1.97 vs settled 1.93; refresh closes it
  (3e-7 A) for one Hessian; orca_update_thresh 0.001 does not (0.08 / 0.28).

- N1 (t22/stopshort.json; HF/6-31G, all species, B0 as Be, calibrated
  joint, no target; settled = 3 passes in a row < 1e-5 A RMS, all 18
  settled). Plain fit vs settled, max atom / |bond RMS change|:
  warm (theory minimum) <= 7.5e-4 A / <= 0.25 mA (CH2ClF 18.99 vs 18.74;
  ozone 0.10); cold (0.03 A displaced) up to 8.4e-3 A (AF, methyl torsion,
  bonds equal) / up to 1.29 mA (HCOF 12.47 vs 13.76, CH2ClF 17.47 vs
  18.74 -- the unsettled answer nearer the reference, as on fluorobenzene).
  refresh_hessian_at_answer (+1 Hessian, 0-9 s; plain fits 0-24 s, 1-5
  Hessians) reaches the settled answer to <= 1.7e-5 A in 16/18; the two
  others (AF, FE cold) differ only along a methyl torsion (bonds within
  0.04). orca_update_thresh 0.001 does not fix it (HCOF cold still 3e-4,
  CH2ClF cold 5e-4; extra gradients). The miss tracks the Hessian's age at
  the answer: age < 5 mA (6 cases) |dbond| <= 0.055; age >= 8.5 mA (12
  cases, warm and cold) 0.005-1.29.

- N2 (CLI, `cli run`, pyscf_hf HF/6-31G, BALANCED 3 starts, internal
  coordinates, BOB on, no VPT2): plain answer vs its own optimum (Hessian
  at the answer; n_starts=1 restart chain and refresh_hessian_at_answer
  agree to 0.1-1.4 microA): HCOF warm 0.025 mA rms / 0.061 max, cold 0.587 /
  1.386; CH2ClF warm 0.109 / 0.298, cold 0.367 / 1.122 -- about 1-2% of the
  start-to-answer distance (the selected start's Hessian 2.9-32.8 mA from
  its answer). Bond RMS vs reference, plain -> optimum: HCOF 13.767 ->
  13.767 (warm), 13.662 -> 13.766 (cold); CH2ClF 18.582 -> 18.749,
  18.231 -> 18.750 (the short answers nearer the reference). Refresh: one
  pass, +1 Hessian, 0 gradients, +15-21% wall time (2.5-7.5 s on 11-35 s).
  Warm and cold refreshed answers agree to 1 microA; plain ones differ 0.3-0.6
  mA.
- N2 surprises: (1) the multistart ranks starts on a stale energy -- each
  start's energy at its last quantum point, not at its answer (reported
  energies off by up to 6.3e-4 Ha against real differences ~1e-5 Ha); HCOF
  picked the wrong start, and its BALANCED restart chain oscillates between
  two starts (period 2, 0.054 mA rms). (2) at OMP 4 threads HCOF cold's
  plain answer varies 0.54 mA rms run to run (bitwise identical at 1
  thread); the refreshed answer does not. (3) restarts from the answer print
  "Low acceptance (<20%)".

## Observations
- 2026-11-21 · told only to measure through the CLI, the verifier found the multistart's stale-energy ranking (N24) and a thread-count non-determinism -- neither visible in my direct-optimizer measurement. The CLI gap was also smaller than the optimizer-level one (internal coordinates, multistart).

## Pending question
- Q-31 (asked 20:18 UTC, deadline 20:34, send_later trig_01LSidfAH5MaKziYVtRLQxE9): what's next. 1 the multistart's stale-energy ranking (N24, recommended) · 2 auto_systematic_sigma magnitudes (F13) · 3 implement a flagged trade-off, N22 or N23 (user's call; never auto-picked) · 4 done for now.

## Handoff
