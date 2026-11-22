# Current task
Status: verifying
Request: T-47 (D-112): measure use_dihedrals: auto and cartesian_fallback together on the nine molecules
## Brief
- Outcome: for decision page item 3, the numbers for neither / dih / fb / both on the nine molecules (HF/6-31G, joint, internal mode, autoconfig off): distance to the fit's own optimum (posterior widths), bond and angle RMS, Hessians (counted calls to run_hessian), wall time.
- Then: bootstrap (50 refits, error-model, seed 7) and autoconfig-on fits only for molecules where a setting changes anything.
- No default changes; report.
- Harness: scratchpad t47/both.py (from T-43's fb.py), two lanes like T-43.
## Progress
- N1 · fits · 36 runs (t47/run.sh) · done 21:02: 'both' is bit-identical to dih on AF and FE, to fb on FA; the other six bit-identical in all four settings. AF: dih and fb both reach their optimum (1.67e-6, 2.78e-6 widths) but 3.4e-3 A apart (max atom); bond RMS 14.584 vs 14.599. Hessians: AF 1 -> 2 in dih, fb and both; FE, FA unchanged.
- N2 · Cartesian reference fits, autoconfig on, bootstraps (t47/run2.sh) · running since 21:03 UTC
## Observations
- 2026-11-22 · AF's two remedies reach different optima (3.4 mA apart): the internal-mode objective depends on the coordinate set; a Cartesian fit decides which is the coordinate-free answer
- 2026-11-22 · ozone stops 5.8e-3 widths short in all four settings (same as T-43); neither remedy touches it
## Pending question
## Handoff
