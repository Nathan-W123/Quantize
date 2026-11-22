# Current task
Status: done
Request: T-39 (N29, D-95): measure how much the internal prior's 57.3x angle-row Jacobian moves the fits that use it, then propose the unit fix
## Brief
- Outcome: who reaches InternalPriorEngine (runner presets, library flags),
  how much its unit error moves their fits (geometry, iterations, the ratio
  test), and a proposed fix measured before it is written.
- Approach: find every path that stacks it; run fits with the shipped rows and
  with the Jacobian put in degrees (a monkeypatch), same starts.
- Success: numbers per path, the population named; a fix proposed, not
  implemented, until agreed.
## Progress
- N1 · reach · which configs stack the Cartesian internal prior · done: Cartesian mode only (internal mode reads it for a diagnostic); CLI: quantum.backend none + coordinate_mode cartesian + BALANCED (default) or STRICT preset (weight 1, 0.04 A / 2 deg, start-anchored, no dihedrals); library: use_internal_priors=True in Cartesian mode. Consumers of its J: the step, the posterior's structural-prior block, the objective gradient, the split null projector. The default internal-mode q-space prior converts degrees correctly (internal_fit.py).
- N2 · measure · shipped vs unit-consistent rows · after: N1 · done (t39 units.py, units2.py, compare2.py)
- N3 · propose · Q-59 · deadline: option 2 (D-96): README note + code comment; option 1 left for the user
- N4 · document · README figures checked by script (58/67 rejections, 3 no-width, 38-64 deg) · done
## Observations
- N2 (9 benchmark molecules, spectral only, Cartesian, the CLI's prior, max 500
  iterations): from the noisy benchmark starts (autoconfig off) ratio-test
  rejections 58 -> 4 in total; 3/9 fits got no uncertainty at all (the
  Gauss-Newton check found a saddle: J and residuals disagree) and 3 more
  reported angle widths of 38-64 deg (VF 5-7) where 2 deg was configured [corrected: 13-63 came from mismatched labels]; fixed: all 9
  report, angle widths 0.3-2.1 deg. Accuracy both ways: mean bond RMS 24.16 ->
  23.00 mA, mean angle RMS 1.578 -> 1.922 deg (FA 1.12 -> 3.86: the prior now
  holds the noisy start); autoconfig on 24.61 -> 24.04, 1.636 -> 1.977. From
  the reference geometry: 2.92 -> 3.27 mA, 0.217 -> 0.193 deg, rejections
  52 -> 5. Wider sigma (fixed, noisy): 3 deg 23.24 / 1.882, 5 deg 23.65 / 1.825.
- Side: InternalPriorEngine dihedral residuals are not wrapped (+-180 jump);
  the CLI does not use dihedrals.

## Pending question
## Handoff
Q-59 (asked 13:16 UTC, deadline 13:31 UTC): the internal prior's angle units
1. Fix the units: the prior becomes the 0.04 A / 2 deg one the config states;
   rejections 58 -> 4, uncertainty reported for all 9, angle widths 0.3-2.1
   deg (shipped: 38-64 deg for 3, none for 3); accuracy moves both ways (noisy starts: bonds 24.16 -> 23.00 mA,
   angles 1.58 -> 1.92 deg; reference starts: 2.92 -> 3.27 mA, 0.217 -> 0.193
   deg) (Recommended; NOT auto-picked: accuracy moves both ways)
2. Leave the code; document the defect (README spectral-only note, code
   comment, backlog) -- the deadline's pick
3. Move on to N7 (split widths), leaving N29 open
4. Done for now
Q-60 (asked 13:35 UTC, deadline 13:50 UTC): what's next after T-39?
1. N7: measure the split estimator's reported widths against its actual
   spread, then propose (Recommended)
2. N29 option 1: fix the internal prior's angle units (accuracy moves both
   ways: your call, never auto-picked)
3. N3: search again for semi-experimental references
4. Done for now
