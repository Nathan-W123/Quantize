# Current task
Status: done
Request: T-42 (N30, D-102): measure, then run bootstrap refits without forking a multi-threaded process
## Brief
- Outcome: no bootstrap refit (or other pool) forked from a process whose
  quantum backend has started threads; same clouds; no time cost, or one
  stated and proposed before committing.
- Approach: list every process pool; measure fork vs inline (1 worker) and
  fork vs spawn (2 workers): identical results? wall time? Try to reproduce
  the deadlock with fork, show the replacement does not hang.
- Success: identical clouds bit for bit; times measured; the fork warning
  gone from the test run.
## Progress
- N1 · measure · pools, fork vs inline vs spawn · done (t42 pools.py; t40 spread_pool.py)
- N2 · build · one worker refits inline (bootstrap.py); test + mutation; README note · done; CI selection 1688 passed (37 warnings remain: they come from 2-worker tests)
- N3 · propose · Q-65 (multi-worker pools) · deadline: option 1 (D-103)
- N4 · build · fork_hang_warning (registry), runs_in_process on backends (ORCA False), printed by the bootstrap and multistart pools; 3 tests, 3 mutations each fail one; CI selection 1691 passed · done
## Observations
- N1: two pools fork (bootstrap; multistart with > 1 worker -- its 1-worker
  path already runs inline). Clouds bit-identical fork / inline / spawn /
  forkserver (water, formyl fluoride, 20 samples). Time, 1 worker: fork 1.0 /
  1.1 s, inline 0.8 / 1.3 s (noise); 2 workers: fork 0.7 / 0.8, spawn 1.3 /
  1.6, forkserver 1.3 (formyl fluoride). Deadlock: the T-40 configuration
  (water split internal, 50 samples, OMP 4) hung 3/3 with fork, finished
  3/3 inline (6-7 s); pools.py's simpler configuration did not hang in 4.
  spawn/forkserver re-import the caller's main module: a script without a
  __main__ guard breaks (my own harness did).

- 2026-11-22 · my spawn harness broke because the script had no __main__
  guard -- the same thing would break a user's script, which is why
  forkserver became the user's call rather than a fix.

## Pending question
## Handoff
Q-65 (asked 17:30 UTC, deadline 17:45 UTC): several workers still fork -- what then?
1. Warn: when workers > 1 and the backend runs in-process with
   OMP_NUM_THREADS != 1, say so before forking (no cost, no behaviour
   change) (Recommended)
2. Use a forkserver pool for several workers: no hang, +0.5 s per run
   (formyl fluoride 0.8 -> 1.3 s), and library scripts that call it need a
   __main__ guard (a cost and a usability change: your call)
3. Leave it at the README note
4. Done for now
