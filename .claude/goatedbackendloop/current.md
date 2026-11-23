# Current task
Status: done
Request: T-52 (D-121): remove unused imports and dead locals (F3)
## Brief
- Outcome: a linter finds them; anything re-exported or used elsewhere stays.
## Progress
- N1 · lint (ruff 0.17, F401/F841 and all F rules) · nothing found in .github, runner, cli.py, scripts, dev; a planted unused import is found
- N2 · history · efabab3 (2026-11-19) already removed the 7 genuine ones and kept 71 on purpose; backlog row updated
## Observations
- 2026-11-23 · the backlog row was stale: the commit that acted on it did not update it (L-26)
## Pending question
## Handoff
