# Current task
Status: done
Request: T-44 (N33, D-107): why internal-mode water from a far start refuses zero-length steps, then propose
## Brief
- Outcome: a measured cause and a proposal.
- Approach: per-iteration step, dq, back-transform, guardrails, ratio test;
  the same start in Cartesian mode; other far starts and molecules.
## Progress
- N1 · diagnose · done (t44 connect.py)
- N2 · propose · Q-69 · deadline: option 1 (D-108)
- N3 · build · quantum._join_fragments, used by InternalCoordinateSet only (guardrails, conformer generation, Cartesian Wilson B keep the plain rule); 5 tests, mutation fails 3; 9 internal-mode benchmark fits bit-identical; CI selection 1706 passed · done
## Observations
- N1: bonds are detected at distance < 1.2 x (sum of covalent radii) --
  O-H 1.164 A -- at every iteration (backend/quantum.py _detect_bonds). Water
  started at 1.25 A has no bond and so no angle: the internal set is empty,
  every step is zero (3.5e-17, then 0), and the plateau rule ends the fit at
  its start, 820 widths from its mode (1.4 A / 120 deg: 1090), silently.
  Cartesian mode from the same start reaches its mode (1.3e-10). Prototype:
  join disconnected fragments by their shortest inter-fragment distance
  (the usual redundant-coordinate rule): 1.25/80 -> 6.5e-6 widths, 1.4/120
  -> 1.8e-7; the normal start unchanged. A weakly bound complex (two
  molecules) is disconnected by the same rule, so its intermolecular
  motion is unreachable in internal mode today; joining reports an
  intermolecular "bond" coordinate.

## Pending question
## Handoff
Q-69 (asked 19:04 UTC, deadline 19:19 UTC): fix the empty coordinate set?
1. Join disconnected fragments by their shortest inter-fragment distance
   (connected molecules unchanged, checked on the 9; a complex gains an
   intermolecular distance coordinate in its report) (Recommended)
2. Warn only: when the internal coordinates cannot move every direction at
   the start, say so and suggest coordinate_mode: cartesian
3. Both: join, and warn when the set still does not span
4. Done for now
