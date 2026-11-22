# Current task
Status: done
Request: T-47 (D-112): measure use_dihedrals: auto and cartesian_fallback together on the nine molecules
## Brief
- Outcome: for decision page item 3, the numbers for neither / dih / fb / both on the nine molecules (HF/6-31G, joint, internal mode); no default changes.
## Progress
- N1 · 36 fits, autoconfig off · both = the acting setting bit for bit; six molecules bit-identical in all four
- N2 · autoconfig on (FA, AF, FE x4), Cartesian reference fits, bootstraps 50 (seed 7) · same pattern; bootstraps AF 0 -> 42, FE 4 -> 50, FA 27 -> 50, both = single
- N3 · cross distances (t47/cross.py) · internal vs Cartesian answers 0.3-4.7 mA apart (worst atom, <= 0.06 widths); refreshed Hessian at each answer: 0.01-0.15 mA (<= 1.2e-3 widths) -> expansion point, not coordinate set
- Decision page v2 (items 2-4), D-113
## Observations
- 2026-11-22 · AF's two remedies looked like different optima (3.4 mA); testing the Hessian-expansion-point hypothesis directly (refresh at the answer) settled it instead of reporting a coordinate-set dependence that was not there
- 2026-11-22 · ozone stops 5.8e-3 widths short in all four settings (same as T-43); neither remedy touches it
## Pending question
## Handoff
