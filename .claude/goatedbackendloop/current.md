# Current task
Task: T-08
Status: verifying
Request: why chlorofluoromethane's offsets hybrid is worse at B3LYP than at
RHF (6.00 vs 2.91 mA vs Vogt r_e^SE); diagnostic only, no engine change
(D-39, auto-picked at Q-07).

## Brief
- Outcome: the gap attributed, with numbers, to (a) the prior centre (how
  well each class's leave-one-out offset transfers to CH2ClF at each level),
  (b) the fit (what two Cl isotopologues, rank 5/9, can and cannot place,
  and what the vibrational corrections do), or both.
- Approach: reproduce the offsets row at both levels from the frozen caches
  (t07/wt2, c8904da), then per bond vs Vogt r_e^SE at each stage: raw
  theory, offset-corrected centre, mixed estimation, hybrid. Then two
  counterfactual fits: centre with CH2ClF's own C-Cl error removed, and
  centre = r_e itself (how far the data + corrections pull the fit off the
  truth at each level).
- Hypothesis to test, not assume: B3LYP's C-Cl error transfers worse from
  CH3Cl (+25.4 mA) to CH2ClF than HF's does (+85.6), and the 2-isotopologue
  data spread the residual into C-F (N16 mechanism).
- Success: each stage reproduces the archived RMS (2.91 / 6.00 vs r_e,
  5.42 / 8.18 vs primary) and the attribution sums to the gap.
- Trade-off: none in code; the cost is one fit per counterfactual, Hessians
  cached.

## Graph
- N1 · explore · per-bond breakdown at each stage, both levels · after: none · running · handoff: done; visibility (t08/vis.py) both levels done
- N2 · explore · transferability: pool offset vs CH2ClF's own error per class · after: none · done · handoff: C-Cl misses 12.4 (HF, over) / 12.8 (B3LYP, under); C-F 2.2 / 5.7
- N3 · explore · counterfactual fits (C-Cl oracle centre, r_e centre) · after: N1 · done · handoff: B3LYP oracle 6.00 -> 2.95, HF oracle 2.91 -> 4.16; r_e centre stays <= 0.44 at both
- N4 · verify · fresh agent re-derives the attribution from the script and data · after: N1, N2, N3 · running · handoff: -
- N5 · build · README sentence + backlog N16 note · after: N4 · pending · handoff: -

## Progress
- Script: scratchpad t08/diag.py <hf|b3lyp> <out.json> (frozen caches t07/wt2,
  repo code; engine unchanged since c8904da).
- RHF (t08/hf.json): reproduces the archived offsets row exactly (hybrid
  5.4233 vs primary, 2.9080 vs r_e; ME 4.2636 vs r_e). Per bond vs Vogt
  r_e^SE (mA), C-Cl / C-F / C-H:
  theory +73.2 / +25.9 / -15.0; pool offsets +85.6 (CH3Cl) / +28.1 / -10.6;
  centre -12.4 / -2.2 / -4.4 (C-Cl over-corrected: CH3Cl's HF C-Cl error is
  12 mA larger than CH2ClF's); ME -5.4 / +1.5 / -4.8; hybrid +1.0 / +2.9 /
  -4.0. Counterfactuals: C-Cl-oracle centre (2.84) -> hybrid 4.16 (C-Cl
  +5.4, C-F -1.1) -- worse than the real centre; r_e centre -> hybrid 0.44,
  ME 0.35 (data + HF corrections are consistent with r_e to <= 0.7 mA per
  bond). So RHF's 2.91 is partly a cancellation: the fit's pull on C-Cl
  (about +5 to +13 mA from a centre off in C-F / C-H) meets a centre 12 mA
  short in C-Cl.
- B3LYP (t08/b3lyp.json): reproduces the archived row exactly (8.1777 vs
  primary, 5.9958 vs r_e; ME 6.7127). C-Cl / C-F / C-H vs r_e (mA): theory
  +38.2 / -1.2 / +6.7; pool offsets +25.4 (CH3Cl) / +4.5 / +5.2; centre
  +12.8 / -5.7 / +1.4 (C-Cl under-corrected by 12.8 -- the same size of miss
  as HF's 12.4 over-correction, opposite sign; the B3LYP C-F pool spreads
  -5.4 to +13.5 against HF's 24.6-30.7); ME +8.3 / -7.9 / +1.8; hybrid
  +7.4 / -7.2 / +1.4. Counterfactuals: C-Cl-oracle centre (3.40) -> hybrid
  2.95 (C-Cl +3.0, C-F -3.7); r_e centre -> hybrid 0.39, ME 0.22.
  So the hypothesis "B3LYP's C-Cl transfers worse" is wrong: both levels
  miss by ~12-13 mA. Removing the C-Cl centre error halves B3LYP's error
  (6.00 -> 2.95); most of the rest is C-F's offset (centre -5.7).
- Visibility, RHF (t08/vis_hf.json; data chi2 over the 6 corrected
  targets, 2 isotopologues x A,B,C, the fit's sigmas): r_e 3.70, theory
  10121, centre 1598, ME 0.02, hybrid 0.29. Line r_e -> centre: 3.7, 128,
  433, 921, 1598. Single-class +10 / -10 mA from r_e: C-Cl 139 / 248, C-F
  63 / 133, C-H 7 / 12. Centre's bond errors alone on r_e: 464.
- Visibility, B3LYP (t08/vis_b3lyp.json): r_e 2.39, theory 2688, centre
  232, ME 0.06, hybrid 0.11. Line r_e -> centre: 2.4, 7.1, 47, 122, 232.
  Single-class +10 / -10 mA: C-Cl 136 / 220, C-F 63 / 115, C-H 7 / 9 (the
  same per-bond sensitivity as RHF). Centre's bond errors alone on r_e: 110.
- Reading: both fits end below r_e's own chi2 (0.29 / 0.11 vs 3.70 / 2.39),
  so the 2-isotopologue data have a valley of structures they fit equally
  well, several mA wide, and the fit keeps whatever part of the centre's
  error lies along it. Quadratic check from the single-class costs: RHF's
  centre bond error (all three short) predicts ~390 if independent and
  scores 464 (the errors add in the moments: across the valley, removed);
  B3LYP's (C-Cl long, C-F short) predicts ~260 and scores 110 (they cancel:
  along the valley, kept). Not a worse level: B3LYP's raw structure is 22.4
  mA from r_e against HF's 45.7. RHF's 2.91 is partly luck -- from a better
  centre (C-Cl oracle) it gets 4.16.
- Remedies (flag, not implement): (a) more data -- Vogt's r_e^SE used more
  isotopologues than the benchmark's two Cl species (unchecked how many,
  and whether their constants are published); (b) a second C-Cl
  calibration molecule so the C-Cl offset is a mean, not one molecule's.

## Observations

## Pending question

## Handoff
