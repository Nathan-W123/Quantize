# Current task
Status: clarifying
Request: T-57 (D-129): measure the water dimer's second minimum (seed 23: two answers at their mode, 23.5 mA apart)
## Brief
- Outcome: whether the joint objective has two minima for the C1 dimer, which is lower (real HF/6-31G energies, not each fit's own quadratic model), how often starts land in each, and what the report says. No code change before the measurement.
- Method: evaluate both answers on one objective; refit each mode from the other's answer; compare internal coordinates; survey starts.
- Success: a measured statement for each question, with the script and figures.
## Progress
- N1 · seed 23 (t57/minima.py, minima23.txt): not two minima. C is below A on both terms (E_HF by 2.7e-5 Eh, 1/2|r|^2 by 0.0045), so for any alpha; refits from A move to 1.2 (internal) and 4.1 mA (Cartesian) from C, refits from C stay. alpha differs per fit (3111 A, 3050 C, refits 3332/3045): calibrated alpha = 1/(1/2 mean positive curvature sigma_x^2) follows the Hessian, so each refresh changes the objective.
- N2 · refresh (t57/refresh.txt): Hessian age 4.5-40.6 mA, alpha 3035-3712 by start; one refresh_hessian_at_answer: A moved 0.6-14.4 mA, C 0.2-5.7; A-C 0.3-23.5 -> 0.7-2.6 mA; alpha 3036-3048; truth 13.1-13.6 mA for all
- N3 · report (t57/age_note.txt): the posterior's Hessian-age warning fires (40.61 vs 5 mA) but speaks of widths only; the Laplace summary has no age warning
## Observations
- 2026-11-23 · my T-56 note named a 'second minimum' from two fits' distance_to_mode (each fit's own quadratic model); one objective with real energies refuted it (L-27 revised)
## Pending question
- Q-86 (asked 04:20 UTC, deadline 04:35 UTC): 1 (Recommended) wording and coverage only: the Hessian-age warning says the answer itself was fitted on that curvature and weight, with the dimer's figures and refresh_hessian_at_answer / quantize uncertainty as the remedy, and the Laplace summary carries it too; no behaviour change; 2 refresh the Hessian at the answer in the plain fit, or whenever drift passes orca_update_thresh (user's call: changes answers, costs a Hessian per fit; never auto-picked); 3 report on the 13 never-passed settings; 4 Done for now. Deadline takes 1.
## Handoff
