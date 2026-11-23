# Current task
Status: done
Request: T-51 (D-120): make the two multireference tests run in a fresh clone (F3)
## Brief
- Outcome: the tests read embedded geometries instead of the gitignored cache, still pass, and fail when the threshold is broken.
## Progress
- N1 · regenerate geometries (t51/geoms.py) · max T2 0.17935 / 0.04713 reproduce the tests' values
- N2 · embed, drop the cache reader · 17 passed; threshold 0.04 / 0.2 each fail one test; CI 1742 passed, 13 skipped
## Observations
## Pending question
## Handoff
