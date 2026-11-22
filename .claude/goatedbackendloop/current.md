# Current task
Status: building
Request: T-40 (N7, D-97): measure the split estimator's reported widths against its actual spread, then propose
## Brief
- Outcome: per molecule and coordinate, reported width / Monte Carlo spread
  for split, with joint as the control, under the noise model the posterior
  assumes (independent sigma, no systematic part, so both sides mean the same
  thing).
- Approach: converged base fit; geometry_uncertainty(); K refits with the
  observed constants redrawn from N(obs, sigma), each from the base's
  geometry; spread of internal coordinates; ratio.
- Success: ratios with their sampling error named (K refits); a cause for any
  ratio far from 1 shown by a second mechanism; a fix proposed, not
  implemented, until agreed.
## Progress
- N1 · measure · water + 2-3 molecules, split and joint, HF/6-31G, via the engine's own BootstrapEngine error-model check (analytic_reference: cloud sd vs target sd = reported covariance less its prior part) · running (t40 spread.py; water split internal 50 refits > 10 min)
## Observations
## Pending question
## Handoff
