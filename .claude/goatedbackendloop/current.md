# Current task
Status: verifying
Request: Round 10 -- independent check of everything after the round-9 check
(f5b6c36): round-9 fixes, N4, efficiency wins E1-E4, A1, S1, S2, the timing
test rewrite.

## Brief
- Outcome: defects in the unreviewed commits found and fixed before the
  results table (N13) is regenerated on this code.
- Approach: two independent background agents given the code, not my
  conclusions. A: numerics (6952211 closed-form Jacobian, ff836fb CD cache,
  c7ea83f trial Jacobian skip, e84d1e7 RTO sampler, 024caec defect probe,
  fb7ea3e benchmark Hessian reuse, 9a0c9ce departure bounds, 84b609c docs,
  4c9a2dd test). B: config/CLI/IO (61c0cff model-error keys, 97d7497
  validation, c701a8a CLI bundle, 5e40ada strict JSON, 14125f3 seed,
  9aa7c28 README). Each finding needs a reproduction or is marked plausible;
  I re-measure every one before fixing (L-12 / round-8 practice).
- Tradeoff: two agents' tokens vs. a read not anchored on mine.
- Success: every finding reproduced or refuted by me; fixes one commit each
  with before/after and a test; clean-checkout suite green.

## Progress
- Previous task (defect fix + small fixes) done: 024caec, 14125f3, 9aa7c28,
  4c9a2dd; records 3a0152b. Its full progress log is in git history
  (current.md at 3a0152b).

## Observations

## Handoff
