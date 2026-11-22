# Current task
Status: done
Request: T-46 (D-111): one page of the decisions waiting for the user
## Brief
- Outcome: a page listing each open user's-call decision: what it changes, measured evidence, options, recommendation.
- Approach: collect from decisions.md and backlog.md (every "user's call" / "flagged"), re-check each figure against its source (L-18), publish as a private page.
- Success: every open flagged item appears once, every figure traced to a record or a rerun.
## Progress
- Page published (private): https://claude.ai/artifact/144eMMauzGEeS3WLmW51Db -- 10 items: yes (N29, dihedrals auto, cartesian fallback, N23 refresh, N22 grid gradient), your call (model_error.auto, offsets alpha geometry), leave (forkserver, rescale widening, multistart ranking)
- Rerun for item 6 (t46/me_auto.py): analytic water with no stated model error, O-H width 0.20 -> 1.76 mA, geometry bit-identical (spectral-only and joint)
- Fact-check fixes before publishing: ozone's 8.6x is against its r_e; item 4 is two CLI molecules; removed an unmeasured run-time estimate
## Observations
- 2026-11-22 · first me_auto probe was vacuous (fixture states sigma_systematic); caught by checking the widths moved (L-15)
- 2026-11-22 · items 2 and 3 both act on acetyl fluoride and were never measured together; the page asks for that run before either becomes a default
- 2026-11-22 · a record-writing script failed after git add and the commit went out without it; check script exit before committing
## Pending question
- Q-72 (asked 20:40 UTC, deadline 20:55 UTC): what next? 1 (Recommended) measure items 2 and 3 together on the nine molecules; 2 N7 split-objective widths; 3 N34 dimer in joint mode; 4 Done for now. Deadline takes 1. Answers to the page's items override.
## Handoff
