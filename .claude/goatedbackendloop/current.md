# Current task
Status: clarifying
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
- N1 · measure · pools, fork vs inline vs spawn · pending
## Observations
## Pending question
## Handoff
