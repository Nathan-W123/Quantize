# Current task
Status: blocked
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
- N2 · propose · Q-62 · asked
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
