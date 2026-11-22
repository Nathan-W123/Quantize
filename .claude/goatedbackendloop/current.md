# Current task
Status: done
Request: T-43 (N31 rest, D-104): why fluoroacetylene's joint internal-mode bootstrap keeps 27/50 refits, then propose
## Brief
- Outcome: a measured cause for the refused refits (where they stop, why the
  step stops there, how the gate is scaled), and a proposal.
- Approach: per refit mode_distance, termination, last steps; the base fit's
  own distance; decompose the remaining Newton step into the q-space's
  directions (bends of a near-linear chain are near-degenerate in angle
  coordinates) (L-23).
- Success: the cause named with numbers a second mechanism confirms.
## Progress
- N1 · diagnose · FA refits · done (t43 decomp.py, cart_from.py; t40 refused.py)
- N2 · propose · Q-67 · auto-picked option 1 (D-105)
- N3 · build · Cartesian fallback (opt-in), floor, switch, history flag, CLI key, README · done; 10 tests on an H-C-C-F field; mutations (no switch / never accept / no floor) each fail; CI selection 1701 passed
- N4 · measure · 9 molecules off vs on (t43 fb.py, run2.sh, m_*_floor.json): FA fixed (1.9e-6, 50/50); AF base 2.8e-6, bond -0.18, 13 -> 24 s, refits 9/50; 7 bit-identical; without the floor FF/CFM moved 4e-7 A and AF/FE refits were rescued only by round-off triggers
## Observations
- N1 (HF/6-31G, joint, internal, autoconfig off): the base stops on plateau
  at 0.008 widths, refits at 0.001-0.043 against a cloud-scaled gate of
  0.010 (9/20 kept here). At the stop the angles are 179.995 and 179.998 deg;
  the remaining Newton step (4.0e-4 A) is along the two bends; the q-space
  step back-transforms with an error of 1.6e-3 (accepted steps: 7e-5), the
  realised step raises the objective and is refused, and the same 1.39e-4 A
  step is refused 5 times while the trust radius halves 5e-2 -> 1.6e-3 (the
  step is inside it), then the plateau rule stops the fit. Cause: a valence
  angle has its maximum (and a kink) at 180 deg, and this optimum is
  exactly linear -- a Cartesian fit from the internal stop reaches 180.0000
  deg and 7e-11 widths in 11 iterations, moving atoms 5e-5 A. The answer
  barely moves; the bootstrap's gate is what notices.

- 2026-11-22 · water from r 1.25 A / 80 deg in internal mode (analytic
  backend) refused 10 zero-length steps and ended 820 widths from its mode --
  not chased (noted N33).

## Pending question
## Handoff
Q-67 (asked 18:12 UTC, deadline 18:27 UTC): fix the near-linear stall?
1. Cartesian fallback: in internal mode, a step refused with a large
   back-transform error is retaken in Cartesian coordinates on the same
   objective; measure on FA and the 8 others, propose before committing if
   anything else moves (Recommended)
2. Linear-bend coordinate pairs for angles near 180 deg (the textbook
   remedy; changes the reported coordinate names for those molecules)
3. Document it (FA's bootstrap keeps about half its refits)
4. Done for now
