# Current task
Task: T-15
Status: building
Request: measure whether projecting translations and rotations out of the
Hessian before the alpha normal-mode analysis moves any benchmark row
(backlog N19 b-i; D-51, auto-picked at Q-18); measurement only, any engine
change proposed before it is made.

## Brief
- Outcome: per row (both levels, non-torsion configs), hybrid_rms_ma with
  and without the projection, and whether any README figure moves.
- How: prototype in a git-archive copy of HEAD (t15/proj): a mass-weighted
  rigid-mode projector applied to the Hessian at both _normal_modes call
  sites in harmonic_alpha.py (compute_harmonic_alpha and the cubic
  normal-mode derivative). Reference: HEAD rows from t14/fix (HF, all 117,
  FA from the corrected run) and t13/b3lyp_t4_merged + t12/run1 (B3LYP).
- Tradeoff to watch: at a stationary point the projection should be a
  no-op; any change there is noise or a defect in the projector.
- Success: the per-row table, a sanity check at stationary points, and an
  independent check of anything that moves.

## Graph
- N1 · build · prototype projector + stationary-point sanity check · after: none · done · handoff: corrected once (see Progress)
- N2 · build · all rows at both levels · after: N1 · running (task b0ok7z9ar) · handoff: -
- N3 · verify · fresh agent on whatever moves · after: N2 · pending · handoff: -
- N4 · gate · propose (or not) an engine change · after: N3 · pending · handoff: -

## Progress
- 19:52 UTC: t15/proj from HEAD (129ca37 code = 2eb9b6c engine), caches
  from t13/wt3 + t14/fix. Projector: translations sqrt(m) e_a, rotations
  sqrt(m) (w x r), P F_mw P, back to Cartesian.
- Sanity check 1 (HF, theory geometries): water alpha moves 0.1-0.6 MHz of
  1e4 (noise at a stationary point); fluoroacetylene LOST a mode (7 -> 6
  frequencies, a 440.9 cm^-1 bend component became 765.6) and alpha_B went
  41.26 -> 64.82. Cause: my projector kept the rotation about the molecular
  axis (norm ~1e-5 of the others at I_a 6e-9) as a sixth rigid column --
  the very case quantize.py's _rigid_mode_projector docstring warns about.
- Corrected: rotations about the principal axes, the smallest-moment axis
  dropped when _rigid_mode_count says linear. Re-check: FA 7 modes, alpha_B
  41.2591 vs 41.2584 at the theory geometry; 43.955 at the slope's minus
  point (verifier, unprojected: 43.966); FA track fits 13.5909 / 13.6366 /
  13.6399 (HEAD 13.6402). Water unchanged (0.5593).
- 19:55 UTC: full run started: HF 9 x 13 configs, B3LYP 9 x 11 configs.

## Observations
- 2026-11-20 · the stationary-point sanity check caught my projector removing a real bend on a near-linear molecule before any long run -- the same linearity trap as T-14, a second time in one day (L-P04 applied; candidate lesson: every new rigid/linear treatment goes through rigid_mode_count, never its own threshold).

## Pending question

## Handoff
