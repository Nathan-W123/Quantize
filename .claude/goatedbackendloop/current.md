# Current task
Task: T-25
Status: done
Request: (D-68, auto-picked at Q-35) check defect_bias_floor and defect_bias_scale for the per-component split D-67 removed.

## Brief
- By reading: what each does per component and where it is reachable.

## Graph
- N1 · diagnose · read both and their callers · after: none · done (D-68) · handoff: -
- N2 · build · docstring note on _apply_defect_bias_floor; backlog F13 updated · after: N1 · done · handoff: -

## Progress
- defect_bias_scale (benchmark `defect` configuration): sigma x max(1,
  |defect after| / |defect before|), the same factor on A, B and C -- no
  split. defect_bias_floor: per-component floor from defect_model_sigma
  (the split), off by default, not passed by MolecularOptimizer, not in the
  CLI; tests call resolve_corrections(defect_bias_floor=True) directly.

## Observations
- 2026-11-21 · reading the mechanism first showed the scale could not have the flaw, so no measurement was run -- the auto-pick's "measure" was not needed.

## Pending question

## Handoff
