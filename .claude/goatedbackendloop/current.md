# Current task
Task: T-35
Status: building
Request: (D-85, auto-picked at Q-51) Why distortion-row steps beyond ~2 mA fail the ratio test; then propose.

## Brief
- Outcome: which part of the objective the Gauss-Newton model gets wrong on
  the refused steps (rotational rows, distortion rows, quantum term), how
  large the miss is against the predicted gain, and a proposal.
- How: wrap MolecularOptimizer._joint_step_ratio in the formyl fluoride x1.0
  and x0.7 distortion fits (HF/6-31G, autoconfig off, shipped trust rule and
  T-34's growth rule): per step |dp|, rho, and per block actual vs predicted
  decrease (rows split by position: the distortion rows are stacked last);
  the nonlinearity |r1 - (r0 - J dp)| / |J dp| per block.
- Success: the block that makes rho fail, with numbers; then a proposal.

## Graph
- N1 · measure · per-block step accounting · after: none · done · handoff: -
- N2 · gate · proposal (Q-52) · after: N1 · done (D-86, auto-picked 1) · handoff: -
- N3 · build · README note + backlog N8 (f), figures by script · after: N2 · done · handoff: -
- N4 · verify · independent fact-check · after: N3 · done: mechanism wrong · handoff: -
- N5 · verify · rotation invariance, my own check · after: N4 · done: confirmed defect · handoff: -
- N6 · build · README / backlog / D-87 corrected · after: N5 · done · handoff: -
- N7 · gate · fix proposal (Q-53) · after: N6 · asked · handoff: -

## Progress
- N1 (t35/acct.py, ff_{ship,grow}_{1.0,0.7}.json): formyl fluoride, 200
  iterations each, every step's ratio test split by block (3 distortion rows
  stacked last). On refused steps the ROTATIONAL rows miss: GN predicts a
  decrease (median +1.1e-2 ship x1.0, +1.7e-1 ship x0.7) and they rise;
  their nonlinearity |r1 - (r0 - J dp)| / |J dp| has median 1.0-3.1 (max
  5.4-8.9); on accepted crawl steps (1.56e-3 A) it is ~0.2 already. The
  distortion rows stay near-linear (median 0.2-1.1) and are 1e2-1e3 smaller
  (pred 3e-6..1e-2). Quantum term: exact here (no target), cancels in rho.
  T-33's history: the steps are almost all null-space (dx_range 4e-6, dx_null
  1.56e-3 A). So: the distortion rows pull along directions the rotational
  constants do not see to first order; there their second derivative, which
  Gauss-Newton drops, outweighs the linear change beyond ~2 mA, and every
  longer step is refused.

- N4 (fc35): steps are rigid rotations (share >= 0.9998 after iteration 20,
  median 0.99999); structural part linear (<= 0.002 on refused steps); the
  distortion prediction moves under an exact rotation (1 mrad: up to 6e-4);
  along rotations only lambda damping holds the step (eigenvalues 1e-4-1e-3
  vs 4.8e5), raw step 0.2-1.1 A, clipped by the radius; the "curvature" is a
  straight-line rotation stretching the molecule (exactly 0.250 at half
  step); the molecule turns 21 deg (x1.0) / 34 deg (x0.7) over the fit; with
  the rows' Jacobian projected off rigid modes the step falls to 1.7-2.7 mA
  and is linear (<= 0.001). Also: dx_null_norm counts directions A/B/C see
  (rank 2 by sv_min_abs = 1/sigma_x); 15-187 is over all steps (refused:
  15-168).
- N5 (t35/rotcheck.py): 1e-3 / 1e-2 / 0.1 rad about x, y, z: distortion
  constants change up to 5.0e-4 / 5.0e-3 / 5.5e-2 relative; A, B, C <= 7e-15.

## Observations
- 2026-11-22 · I named a mechanism ("rotational rows' curvature") from a
  nonlinearity statistic without decomposing the step into rigid and
  internal parts; the verifier did, and the cause was a rotation. Before
  naming a mechanism for a step pathology, split the step into rigid-body
  and internal motion.

## Pending question
Q-53 · asked 2026-11-22T09:17Z · deadline 09:32Z · auto-pick 1
1. (Recommended) Fix the orientation defect: predict the distortion constants
   in the Hessian's own frame (align the trial geometry onto the geometry
   the Hessian was computed at before predicting); test rotation invariance,
   water's validated constants at its own Hessian geometry, and rerun the
   T-32 distortion cases (convergence, moves). Changes opt-in distortion
   fits only; a correctness fix, not a trade-off.
2. Project rigid modes out of the distortion rows' Jacobian instead (keeps
   the prediction orientation-dependent; the fit stops turning the molecule).
3. Leave it documented as a known defect.
4. Done for now.
