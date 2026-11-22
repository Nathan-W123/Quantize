# Current task
Status: done
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
- N1 · measure · 9 molecules x split/joint x internal/cartesian, 50 refits, engine's error-model check · done (t40 out/, refused.py)
## Observations
- 2026-11-22 · a 50-refit bootstrap hung: the pool forks after pyscf's OpenMP threads (OMP 4) and a child deadlocked (N30). Single-threaded runs are safe; I lost ~30 min before dumping the stack. Next time dump the stack (faulthandler) at the first unexplained stall.
- N1 results (HF/6-31G, benchmark data and starts, autoconfig off, 50
  refits, cloud sd / target sd = reported covariance less its prior part):
  joint Cartesian: all 9 usable 50/50, ratios 0.53-1.55, no departure flagged.
  split (both modes): water and ozone (null response 0) 0.86-1.00, no
  departure; the other 7 depart on 1-8 coordinates, ratios 0.05-58.7,
  worst where the null response is large (HNCO |K| 26-27: 0.05-0.65; FA
  Cartesian angle 58.7); acetyl fluoride's split runs did not finish in
  500 s. N7 confirmed on 7 of 9.
  NEW, default path: joint INTERNAL keeps few refits for acetyl fluoride
  (0/50), fluoroethane (4/50), fluoroacetylene (27/50); Cartesian 50/50.
  Cause measured on AF: the internal-mode base fit stops 0.19 posterior
  widths from its own optimum (Cartesian: 9e-7), refits 0.03-1.0 (Cartesian
  2e-5 - 1e-3), against a gate scaled to the data cloud (0.028). With
  autoconfig on: AF base 0.013, 1/6 usable; FE base 0.16, 0/6.
  The CLI default is joint + internal, so its bootstrap returns little or
  nothing for these molecules. Related to N23 (stop short), but there the
  cause was a stale Hessian; refits hold a fixed quadratic, so this one is
  the internal-mode optimiser itself.

## Pending question
## Handoff
Q-61 (asked 15:26 UTC, deadline 15:41 UTC): what next after T-40?
1. N31: diagnose why joint internal-mode fits (the CLI default) stop
   0.03-1.0 posterior widths short of their optimum, then propose (Recommended)
2. N7: compute the split estimator's covariance from its linear response,
   measure it against these clouds, then propose
3. N30: run the bootstrap's refits in spawned, not forked, workers
4. Done for now
