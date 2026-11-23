# Current task
Status: building
Request: T-57 (D-129): measure the water dimer's second minimum (seed 23: two answers at their mode, 23.5 mA apart)
## Brief
- Outcome: whether the joint objective has two minima for the C1 dimer, which is lower (real HF/6-31G energies, not each fit's own quadratic model), how often starts land in each, and what the report says. No code change before the measurement.
- Method: evaluate both answers on one objective; refit each mode from the other's answer; compare internal coordinates; survey starts.
- Success: a measured statement for each question, with the script and figures.
## Progress
- N1 · seed 23 (t57/minima.py, minima23.txt): not two minima. C is below A on both terms (E_HF by 2.7e-5 Eh, 1/2|r|^2 by 0.0045), so for any alpha; refits from A move to 1.2 (internal) and 4.1 mA (Cartesian) from C, refits from C stay. alpha differs per fit (3111 A, 3050 C, refits 3332/3045): calibrated alpha = 1/(1/2 mean positive curvature sigma_x^2) follows the Hessian, so each refresh changes the objective.
- N2 · running: refresh_hessian_at_answer on A and C for 8 C1 starts (t57/refresh.py)
## Observations
## Pending question
## Handoff
