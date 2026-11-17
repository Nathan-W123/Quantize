# Current task
Status: done
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

- Two background agents (numerics; config/CLI/IO) reported 5 + 14 findings
  and one pre-existing issue. Every one re-measured through its real entry
  point before fixing (before/after in each commit); one agent count
  corrected (14 -> 11 backend commits).
- Fixed, one commit each with tests that fail on the parent:
  e019e96 defect probe only where the target prior is active; flat target.
  6bf3fe2 seed_value: one reading for validator, builder, CLI; --seed < 0
  refused before the fit; source label from the raw file.
  0d1db72 failure bundle lands where its run does; earlier runs' files named
  in the bundle, not deleted (D-28).
  8075792 a correction replacing a stated model error is labelled and warned
  (D-27); sigma_includes_systematic alone refused; labels by component.
  2b1e247 unknown optimizer keys refused (one tuple for validator and
  builder); sv_min_abs nullable; a stated 0 fraction is 0.
  899e09f distortion-Jacobian skip and rto sampler tests that can fail.
  41c2b75 README vintage: recorded 41a4a02, copied 4f0ece6.
  dce5dd9 departure check: per-coordinate 50%-power bounds with the 2% bar.
- Pre-existing near-linear FD Jacobian -> backlog N14 (no trade).
- Clean-checkout suite: 1621 passed, 23 skipped (was 1592 + 29 new tests).
- Records: D-27, D-28; L-14 validated (absorbs L-10), L-15 validated
  (absorbs L-P01, L-13); lessons.md consolidated 219 -> 109 lines; L-P05
  moved to learnings.md as L-06; backlog 9 open (N9, N11 into F3; N12 into N7).

## Observations
- 2026-11-17: an independent read again paid: 19 defects in 15 commits I had
  measured myself, 5 of them one value read by several paths (L-14).
- 2026-11-17: B-7 (stated value replaced) did not show at construction --
  the Hessian corrections apply during the fit -- so the first reproduction
  read "stated" and only the CLI run showed it. Reproduce through the full
  entry point, not the constructor (L-04 outcome).

## Handoff
