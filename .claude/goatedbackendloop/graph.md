# goatedbackendloop context graph
Started 2026-11-20. Earlier tasks are folded in from current.md history and
the decisions record; tasks before T-01 live in git history only.

## Components
- C:joint-optimizer · hybrid joint fit (spectral residuals + quantum prior), steps, termination · .github/backend/quantize.py (MolecularOptimizer), .github/backend/spectral/SVD.py (SubspaceOptimizer)
- C:quantum-prior · calibrated alpha_q = 1/(0.5 lam_bar sigma_x^2) on the rigid-projected Hessian, prior target · quantize.py (_project_quantum_terms, _projected_reference_terms, quantum_prior_weight), SVD.py (_calibrated_alpha_q)
- C:chi2-rescale · sigma rescaling by reduced chi2, floored at the model-error part · quantize.py run(), SpectralEngine.scale_sigma
- C:bond-offsets · leave-one-out bond-class offsets and prior widths · .github/backend/spectral/bond_offsets.py, prior_widths.py
- C:correction-tables · rovibrational corrections (YAML and CSV) · .github/backend/spectral/correction_models.py, rovib_tables.py, harmonic_alpha.py
- C:references · benchmark molecules, constants, r_e / r_e^SE references · dev/monofluoro_references.py
- C:benchmark · accuracy-upgrades benchmark and archived rows · scripts/accuracy_upgrades_benchmark.py, reports/accuracy_upgrades/, scripts/archive_benchmark_results.py
- C:cli-config · CLI, config validation and builder · .github/runner/
- C:readme · results tables and claims · README.md

## Tasks
- T-01 · N13: regenerate the benchmark results tables (best-of and base) · 2026-11-19 · done
- T-02 · vinyl fluoride: r_e^SE reference and Hayashi & Inagusa constants · 2026-11-19 · done
- T-03 · chi2 rescale stops when the model-error floor makes it a no-op · 2026-11-19 · done
- T-04 · bloat audit and cleanup (dead code, five modules, ten scripts archived) · 2026-11-19 · done
- T-05 · CSV correction tables reach the engine · 2026-11-19 · done
- T-06 · per-bond prior prototype (diagnostic): rejected, CH2ClF +, HNCO - · 2026-11-20 · done
- T-08 · why CH2ClF is worse at B3LYP: C-Cl and C-F offset misses along a direction two isotopologues cannot see · 2026-11-20 · done
- T-07 · calibration molecules for uncovered bond classes: CH3Cl, HCN, H2O2 as calibration-only, default pool, offsets rows rerun · 2026-11-20 · done

## Edges
- C:quantum-prior --depends-on--> C:joint-optimizer · 2026-11-20
- C:bond-offsets --depends-on--> C:references · 2026-11-20
- C:benchmark --depends-on--> C:bond-offsets · 2026-11-20
- C:benchmark --depends-on--> C:correction-tables · 2026-11-20
- C:cli-config --depends-on--> C:correction-tables · 2026-11-20
- T-01 --touched--> C:benchmark · 2026-11-19
- T-01 --touched--> C:readme · 2026-11-19
- T-01 --decided--> D-29 · 2026-11-17
- D-29 --governs--> C:benchmark · 2026-11-17
- T-02 --touched--> C:references · 2026-11-19
- T-02 --decided--> D-32 · 2026-11-19
- D-32 --supersedes--> D-31 · 2026-11-19
- D-32 --governs--> C:references · 2026-11-19
- T-02 --taught--> L-16 · 2026-11-19
- T-03 --touched--> C:chi2-rescale · 2026-11-19
- T-03 --taught--> L-15 · 2026-11-19
- T-04 --touched--> C:joint-optimizer · 2026-11-19
- T-04 --taught--> L-07 · 2026-11-19
- T-05 --fixed--> C:correction-tables · 2026-11-19
- T-05 --touched--> C:cli-config · 2026-11-19
- T-06 --touched--> C:quantum-prior · 2026-11-20
- T-06 --touched--> C:bond-offsets · 2026-11-20
- T-06 --taught--> L-06 · 2026-11-20
- T-06 --taught--> L-P04 · 2026-11-20
- T-06 --decided--> D-33 · 2026-11-20
- D-33 --governs--> C:quantum-prior · 2026-11-20
- D-33 --governs--> C:bond-offsets · 2026-11-20
- T-06 --taught--> L-17 · 2026-11-20
- T-07 --decided--> D-34 · 2026-11-20
- T-07 --touched--> C:bond-offsets · 2026-11-20
- T-07 --touched--> C:references · 2026-11-20
- T-07 --decided--> D-35 · 2026-11-20
- D-35 --governs--> C:bond-offsets · 2026-11-20
- D-35 --governs--> C:references · 2026-11-20
- T-07 --decided--> D-36 · 2026-11-20
- D-36 --governs--> C:benchmark · 2026-11-20
- T-07 --touched--> C:benchmark · 2026-11-20
- T-07 --decided--> D-37 · 2026-11-20
- D-37 --governs--> C:benchmark · 2026-11-20
- T-07 --decided--> D-38 · 2026-11-20
- D-38 --governs--> C:benchmark · 2026-11-20
- D-38 --governs--> C:bond-offsets · 2026-11-20
- T-07 --touched--> C:readme · 2026-11-20
- T-07 --taught--> L-17 · 2026-11-20
- T-07 --taught--> L-18 · 2026-11-20
- T-08 --decided--> D-39 · 2026-11-20
- T-08 --touched--> C:bond-offsets · 2026-11-20
- T-08 --touched--> C:readme · 2026-11-20
- T-08 --taught--> L-18 · 2026-11-20
