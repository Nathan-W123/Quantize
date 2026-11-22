# Current task
Status: blocked
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
- N3 · propose · Q-65 (multi-worker pools) · asked
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

## Pending question
## Handoff
