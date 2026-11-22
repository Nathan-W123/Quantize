# Current task
Task: T-32
Status: building
Request: (D-80, auto-picked at Q-46) N8 (b): measure whether the 0.01 MHz distortion sigma floor swamps heavy molecules' kHz-size DJ, then propose.

## Brief
- Outcome: per reference molecule and key (DJ, DJK, DK), the predicted
  constant, the 40% model sigma, whether the 0.01 MHz floor replaces it, and
  the weight lost (floor / model)^2; then a proposal with its cost.
- Code: SpectralEngine._cd_rows: sigma = max(hypot(cd_sigma, model), 0.01),
  model = cd_model_sigma or _default_cd_sigma = max(0.4 |observed|, 0.01).
  Units MHz. Water's DJ is ~0.04 MHz, so 40% clears the floor there.
- Data gate: no reference molecule has observed distortion constants; the
  predicted constants (compute_cd_constants at the reference geometry, HF/
  6-31G Hessian, the reduction the engine would pick) stand in.
- Success: the count of floored rows and their weight factors; whether any
  geometry effect is worth measuring next.

## Graph
- N1 · measure · predicted CD constants vs the floor · after: none · pending · handoff: -
- N2 · gate · proposal (Q-47) · after: N1 · pending · handoff: -

## Progress

## Observations

## Pending question
