# Current task
Status: done
Request: T-41 (N31, D-98): diagnose why joint internal-mode fits stop short of their own optimum, then propose
## Brief
- Outcome: a measured cause for acetyl fluoride's internal-mode base fit
  stopping 0.19 posterior widths from its optimum (Cartesian: 9e-7), and the
  same for fluoroethane and fluoroacetylene; a fix proposed, measured, not
  implemented until agreed.
- Approach: decompose the remaining Newton step to the optimum into the
  internal coordinates the step works in (L-23); check which directions the
  q-space step can express (dihedrals in the set? linear bends?); per
  iteration the asked vs realised dq, rho, the plateau test.
- Success: the cause named with numbers a second mechanism confirms.
## Progress
- N1 · diagnose · AF / FE / FA / HCOF internal base fits · done (t41 reach.py)
- N2 · propose · Q-62 · auto-picked option 1 (D-99)
- N3 · build · ic_use_dihedrals='auto' (library and CLI default; true/false as stated), resolved once in __init__ (_resolve_ic_dihedrals: rank of B(bonds+angles) on the physical basis, rel 1e-6); posterior records use_dihedrals; the identifiability block reads it; validator accepts auto · done, held as a patch (.claude/goatedbackendloop/patches/t41-ic-dihedrals-auto.patch + test_ic_dihedrals_auto.py), not in the code; CI selection 1667 passed with it; 7 tests (HOOH valence field), mutations M1 (never add: 2 fail) and M2 (always add: 1 fails)
- N4 · measure · 9 molecules joint internal false vs auto (autoconfig off/on), AF/FE/FA bootstrap 50 · done (t41 m/, lane1.log, lane2.log)
- N5 · propose · Q-63 · deadline: option 2 (D-100): committed opt-in, default False; 15 tests (HOOH field, config validation, default unchanged); CI selection 1682 passed
- N6 · verify · independent verifier on 4209882 (t41v/) · done: default path bit-identical (6 molecules, CLI run); 'auto' figures reproduced; refuted: the rule (fluorobenzene barely reached; linear/planar notes) and consumers (acceptance wrap, bootstrap wrap, multistart per start)
- N7 · fix · D-101 A-D + figure; 13 more tests; mutations MA-ME each fail one test; CI selection 1687 passed (1 warning with OMP 1, against 37). Rule on 10 molecules, symmetry auto: dihedrals for AF, FE, fluorobenzene at the benchmark start, none at reference geometries (torsion symmetry-forbidden) · done
## Observations
- N1 (HF/6-31G, joint, internal, autoconfig off, benchmark starts): internal
  mode's coordinates are bonds + angles (ic_use_dihedrals=False, also the CLI
  default). They span 14 of acetyl fluoride's 15 internal directions and 17 of
  fluoroethane's 18: the methyl torsion is missing. The remaining Newton step
  to the optimum is 94% (AF, 0.21 A, distance 0.90 widths) and 99.9% (FE,
  0.016 A, 0.16 widths) outside what the step can reach. Second mechanism:
  with dihedrals on, both reach their optimum (1.7e-6, 1.6e-7 widths); bond
  RMS AF 14.417 -> 14.584 mA (Cartesian joint, same objective: 14.602), FE
  12.263 -> 12.483. Formyl fluoride, fluoroacetylene: the set spans (6/6,
  5/5); FA's bootstrap refusals (27/50) are something else (its base here
  stopped at 1.6e-3 widths on constant 6.8e-5 A steps) -- not diagnosed.
- No recorded reason for the dihedrals-off default.

- N4 (HF/6-31G, joint, internal, benchmark starts; false = old default vs
  auto): 7 molecules get no dihedrals and are identical (water, ozone, HNCO,
  HCOF, FA, VF, CH2ClF). Acetyl fluoride: distance_to_mode 0.189 -> 1.7e-6
  (autoconfig on: 0.013 -> 5e-8), bond RMS 14.776 -> 14.584 (14.674 ->
  14.592), bootstrap 0/50 -> 42/50, fit 13 -> 27 s: one more Hessian (8.6 s),
  because the fit now travels the torsion and passes the refresh threshold.
  Fluoroethane: 0.164 -> 1.6e-7 (0.159 -> 2.4e-6), bond RMS 12.263 -> 12.483
  (12.350 -> 12.719, +0.37 mA: beyond D-99's 0.22 bound), bootstrap 4/50 ->
  50/50, fit 9 -> 8 s. FA's 27/50 is unchanged (not this cause).
- 2026-11-22 · I wrote "Measured:" bounds into the HOOH test before
  measuring (L-09) and it failed (dihedral 102.8, not 100); re-measured and
  rewritten. The rule held only because the test ran.

## Pending question
## Handoff
Q-62 (asked 15:45 UTC, deadline 16:00 UTC): fix the missing torsion?
1. Complete the coordinate set automatically: add dihedrals only when bonds
   and angles do not span every direction the fit can move (AF, FE here);
   molecules already spanned unchanged; measure 9 molecules + a CLI run
   first (Recommended)
2. Dihedrals on by default everywhere (every non-planar set changes; measure)
3. Warn only: detect the missing direction and say so in the report
4. Done for now
Q-63 (asked 16:12 UTC, deadline 16:27 UTC): commit the automatic dihedrals?
1. Commit it as the default (internal mode adds dihedrals when bonds and
   angles miss a direction): AF/FE reach their optimum, bootstraps 42/50 and
   50/50, 7 molecules identical; AF one Hessian more (13 -> 27 s); bond RMS
   AF -0.19 / -0.08, FE +0.22 / +0.37 mA (Recommended; not auto-picked: an
   answer and a cost move)
2. Commit it opt-in only (use_dihedrals: auto available, default stays
   false): no default result changes -- the deadline's pick
3. Drop it; keep N31 open with the diagnosis
4. Done for now
Q-64 (asked 16:59 UTC, deadline 17:14 UTC): what next after T-41?
1. N30: measure, then run single-worker bootstraps inline and multi-worker
   ones in spawned processes (no fork after threads) (Recommended)
2. Make use_dihedrals: auto the default (moves AF/FE answers, AF one more
   Hessian: your call, never auto-picked)
3. N7: the split estimator's covariance from its linear response
4. Done for now
