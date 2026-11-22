# Current task
Status: verifying
Request: T-47 (D-112): measure use_dihedrals: auto and cartesian_fallback together on the nine molecules
## Brief
- Outcome: for decision page item 3, the numbers for neither / dih / fb / both on the nine molecules (HF/6-31G, joint, internal mode, autoconfig off): distance to the fit's own optimum (posterior widths), bond and angle RMS, Hessians (counted calls to run_hessian), wall time.
- Then: bootstrap (50 refits, error-model, seed 7) and autoconfig-on fits only for molecules where a setting changes anything.
- No default changes; report.
- Harness: scratchpad t47/both.py (from T-43's fb.py), two lanes like T-43.
## Progress
- N1 · fits · 36 runs (t47/run.sh) · running since 20:59 UTC
## Observations
## Pending question
## Handoff
