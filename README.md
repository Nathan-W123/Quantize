# Quantize

Hybrid molecular geometry inversion from rotational spectroscopy and quantum chemistry.

Quantize optimizes molecular structures (bond lengths and angles) for spectroscopically underdetermined systems, where the isotopologue rotational constants alone do not fix the geometry. It combines the spectroscopic data with quantum mechanics by minimizing a single joint objective function.

## Core idea

- Use observed rotational constants (`A`, `B`, `C`) from one or more isotopologues.
- Map ground-state constants onto equilibrium targets via $B_e = B_0 + \Delta_\mathrm{vib} + \Delta_\mathrm{elec} + \Delta_\mathrm{BOB}$ (see [Ground state to equilibrium](#ground-state-to-equilibrium-b_0-to-b_e)).
- Stack the σ-weighted spectral Jacobian $J$ and residuals $r$ across all isotopologues, and take the electronic-energy **gradient** $g$ and **Hessian** $H$ from Psi4 or ORCA.
- Minimize one **joint objective**, the spectral $\tfrac12\chi^2$ plus $\alpha_q$ times the electronic energy, with damped Newton steps $`(J^TJ + \alpha_q H + \lambda I)\,\Delta p = J^T r - \alpha_q g`$. No direction is handed wholesale to either source: each carries weight in proportion to how well it determines that direction, so theory holds the structure where the data are silent and yields where they are informative.
- Set $\alpha_q$ through `quantum_prior_sigma_ang`, the displacement over which the quantum surface is trusted (roughly the method's geometry error; `MolecularOptimizer` defaults to 0.020 Å).

An SVD-based `split` objective is also available; see [How data and theory share authority](#how-data-and-theory-share-authority).

Key formulas are documented in module docstrings (see `.github/backend/spectral/`, `.github/backend/kraitchman.py`, and `.github/backend/torsion/`); primary literature references are cited inline (Gordy & Cook 1984 for spectroscopic relations).

## Results

Against published structures for nine molecules
([`scripts/accuracy_upgrades_benchmark.py`](scripts/accuracy_upgrades_benchmark.py),
`--table results=reports/accuracy_upgrades`), mean all-bond RMS error in mÅ:

| Level of theory | Theory alone | Mixed estimation | Quantize hybrid |
|-----------------|-------------:|-----------------:|----------------:|
| RHF/6-31G, best configuration per method | 10.39 | 9.84 | **6.43** |
| RHF/6-31G, base configuration            | 21.47 | 17.35 | **11.03** |
| B3LYP/6-31G(d), best configuration per method | 5.30 | 4.02 | **3.75** |
| B3LYP/6-31G(d), base configuration            | 9.20 | 5.45 | **4.58** |

"Best configuration" gives each method its own best of the 15 (RHF) or 13
(B3LYP) settings per molecule, which flatters all three; "base" is one setting
for everything. On the best-of rows the hybrid is the best of the three on 4 of 9
molecules at RHF and 5 of 9 at B3LYP, beats mixed estimation on 6/9 and 5/9 and
theory alone on 7/9 and 8/9, and improves from RHF to B3LYP on eight of nine.
The exception, chlorofluoromethane (5.43 -> 8.14), is its offsets, not the
level: at B3LYP methyl chloride's C–Cl offset leaves the prior's C–Cl 13 mÅ
long and the fluorides' C–F offset leaves its C–F 6 mÅ short, a combination its
two chlorine isotopologues barely see, so the fit keeps most of it (against r_e,
C–Cl +7.4 and C–F −7.2 mÅ); started from r_e itself, the fit stays within
0.44 mÅ RMS at both levels (measured at c8904da, before D-52 and D-55)
([`scripts/archive/chlorofluoromethane_level_gap.py`](scripts/archive/chlorofluoromethane_level_gap.py)).

The best-of rows owe most of their margin over the base rows to the bond-class
offsets: theory corrected by the error each bond class shows in the other
molecules (leave-one-out). That correction helps all three methods, so at RHF
mixed estimation now beats the hybrid on chlorofluoromethane (4.26 against
5.43) and formyl fluoride (by 0.07 mÅ), as it already did on water. A class seen in
only one molecule cannot be calibrated that way, so the pool also takes three
equilibrium structures that are measured for their offsets and never fitted or
scored -- methyl chloride, hydrogen cyanide and hydrogen peroxide
(`CALIBRATION_ONLY` in [`dev/monofluoro_references.py`](dev/monofluoro_references.py))
-- which give C–Cl, C–N, O–H and O–O a second member. N–H still has none.
HF/6-31G puts hydrogen peroxide trans-planar, another conformer, so it is left
out at RHF. One known cost: its O–O is a single bond and ozone's is not, so at
B3LYP the `offsets` fit of ozone moves toward its r_s reference (0.03 mÅ in
the best-of table) but away from its r_e (1.03 mÅ; 0.37 with the earlier pool,
at an older code vintage). `calibration=scored` reproduces the earlier pool.

Most of these references are r_s or effective structures, which sit several mÅ
from equilibrium (vinyl fluoride's is 5.7 mÅ from its r_e), so they cannot rank
differences below that. Against the four molecules with a published r_e
(base configuration, mÅ):

| Molecule (r_e source) | RHF theory | ME | hybrid | B3LYP theory | ME | hybrid |
|---|---:|---:|---:|---:|---:|---:|
| water (Császár 2005) | 8.14 | **0.17** | 0.56 | 10.89 | 0.92 | **0.05** |
| ozone (Tyuterev 1999) | 22.09 | **2.88** | 3.13 | 8.64 | 0.21 | **0.07** |
| vinyl fluoride (Piccardo 2015, r_e^SE) | 17.98 | **10.70** | 11.39 | 5.83 | 3.92 | **3.34** |
| chlorofluoromethane (Vogt 2014, r_e^SE) | 45.68 | 28.16 | **10.80** | 22.38 | 14.54 | **8.94** |

The hybrid beats theory alone on all four at both levels and is the best of the
three on all four at B3LYP; at RHF mixed estimation is ahead on three.

> **Provenance.** The B3LYP rows are at code vintage 4614cd0, the RHF rows at
> ba67cfe (both 2026-11-21; 4614cd0 changes only the DFT geometry
> optimiser, so the RHF rows are the same). The B3LYP rows carry three
> changes, the RHF rows the first two: two to the Hessian, one behind the
> harmonic alpha term (D-52) and one behind both that and the quantum prior's
> curvature (D-55), and one to the B3LYP theory geometries (D-57).
>
> *Rigid motions projected out* (b29cec3, decisions D-52). Off a stationary
> point the raw Hessian's rotational block is not zero, and in isocyanic
> acid's RHF `offsets` fit a 0.7% rotation leaking into the H–N–C bend had put
> alpha_B off by about 100 MHz: that fit (and its `+bob`, `+defect`, `+cd`,
> `+track` and `+torsion` rows) went from 8.14 to 28.56 mÅ, its `track_direct`
> from 22.13 to 18.04, now its best RHF row, taking the RHF best-of hybrid mean
> from 5.34 to 6.43 with win counts unchanged. Off a stationary point the
> harmonic force field is not unique, so that row depends on the recipe
> (23.2–28.6 mÅ). Every other RHF row moved by 0.09 mÅ or less, every B3LYP
> row by 0.28 or less.
>
> *Translational sum rule imposed* (ba67cfe, D-55). pyscf's analytic DFT
> Hessian has no grid-weight response, so at its default grid every B3LYP
> Hessian here breaks the rule that moving the whole molecule changes nothing
> (row sums 2e-4 to 1.9e-2 Ha/bohr²; at the theory geometries each molecule's
> largest rigid-mode eigenvalue is 23–117 cm⁻¹), and
> projection cannot reach the part of that error inside the vibrations.
> Imposing the rule brought the harmonic alpha of formyl fluoride, ozone and
> chlorofluoromethane to within 0.006–0.032 MHz of a converged grid (0.4–0.8
> MHz before) and chlorofluoromethane's vibrations to within 0.1 cm⁻¹ (5 before).
> The B3LYP rows move by 0.05 mÅ or less, except chlorofluoromethane's (up to
> 0.32 against its effective reference), whose two-isotopologue fit leans on
> the prior's curvature: against its r_e they moved 9.24 -> 8.95 mÅ (base) and
> 5.99 -> 5.91 (`offsets`) at ba67cfe. RHF Hessians
> satisfy the rule to ~1e-9; 131 of the 135 RHF rows move by less than 1e-4
> mÅ, ozone's `track_direct` by 1.3e-4, and fluoroacetylene's three tracked
> rows, whose fits have two nearby solutions, by up to 0.04 (none of the four
> is a best-of row).
>
> *B3LYP theory geometries re-optimised on the energy's own gradient*
> (4614cd0, D-57). pyscf's DFT gradient also leaves out the grid-weight
> response, so it is not the derivative of the energy the optimiser
> minimises: L-BFGS-B stopped where the energy stopped falling, up to 16 times
> above its gradient tolerance, and returned without saying so. The B3LYP
> theory geometries -- the "theory alone" answers and the prior's centres --
> sat up to 0.24 mÅ (in the worst bond) from their energy's minimum.
> Optimised on the grid-response gradient, all twelve converge, and formyl
> fluoride's bonds land nearer a converged grid's minimum (0.093 -> 0.052 mÅ
> RMS against level 9). On a row, the theory leg moves by up to 0.12 mÅ
> (formyl fluoride's base row), mixed estimation by up to 0.35
> (chlorofluoromethane, whose own geometry moved less than 0.01 -- that leg
> is sensitive there), the hybrid by 0.07 or less against either reference;
> fluoroethane's best-of hybrid now beats theory alone, narrowly (6.58
> against 6.62; mixed estimation's 6.16 beats both).
> The RHF geometries were already converged (bonds within 0.008 mÅ).
>
> *Flagged, not changed: the fit's own gradient* (D-59). The gradient a fit
> takes from the `pyscf_hf` backend -- `run_hessian`'s at each Hessian,
> `run_gradient`'s between them -- is still pyscf's default, without the grid
> response: up to 4.4e-4 Ha/Å off at the B3LYP theory geometries. The
> benchmark cannot see it, because its target prior uses H(x - target) in
> place of the gradient (a row came out bit-identical with the gradient
> replaced or corrupted). A fit without a target -- the CLI's -- centres its
> quantum prior at x - H⁻¹g, which the missing term moves by up to 0.27 mÅ in
> a bond. Run through the CLI (B3LYP/6-31G(d), the benchmark's base
> corrections) with the grid response on both, formyl fluoride's answer moves
> 1.5e-4 Å (bond RMS 2.53 -> 2.49 mÅ against its reference) and
> fluoroethane's 1.8e-4 Å (6.68 -> 6.61), for 7-14% more wall time. That
> trade is left open (backlog N22).
>
> Not run: the `elec` and `all` configurations (they need `pyscf.prop`, not
> installed here), and at B3LYP the `torsion` and `offsets+torsion` rows of
> fluoroethane and acetyl fluoride (a single scan outlasts the 2-hour job
> limit), so those two molecules' B3LYP best-of is over 11 configurations.
> Results are archived in
> [`reports/accuracy_upgrades/`](reports/accuracy_upgrades/).

Held-out prediction needs no reference structure at all. Fitted to every
isotopologue but one, the hybrid predicts the withheld species' rotational
constants better than theory alone on 5 of 5 molecules at B3LYP/6-31G(d), by a
median factor of 17 ([`scripts/holdout_validation.py`](scripts/holdout_validation.py)).

## Main modules (`.github/backend/`)

The library lives under `.github/backend/`; `paths.ensure_repo_paths` puts
`.github` on `sys.path` so it imports as `backend.*`.

| Module | Role |
|--------|------|
| [`backend/quantize.py`](.github/backend/quantize.py) | `MolecularOptimizer`: spectral + quantum hybrid loop (joint objective by default) |
| [`backend/spectral/spectral.py`](.github/backend/spectral/spectral.py) | Inertia tensor, $A,B,C$, Jacobians, residuals, weighting, optional conformer mixtures |
| [`backend/spectral/SVD.py`](.github/backend/spectral/SVD.py) | `SubspaceOptimizer`: joint-objective step, plus the optional SVD range/null split |
| [`backend/spectral/harmonic_alpha.py`](.github/backend/spectral/harmonic_alpha.py) | Vibration-rotation $\alpha_r$: harmonic, Coriolis, and cubic anharmonic terms |
| [`backend/spectral/centrifugal_distortion.py`](.github/backend/spectral/centrifugal_distortion.py) | Normal modes, $\partial B/\partial Q$, $\tau'$, Watson CD constants |
| [`backend/spectral/correction_models.py`](.github/backend/spectral/correction_models.py) | Vibrational, electronic (g-tensor), and BOB corrections |
| [`backend/spectral/rovib_corrections.py`](.github/backend/spectral/rovib_corrections.py) | Resolves all corrections into $B_e^{SE}$ targets with provenance |
| [`backend/quantum.py`](.github/backend/quantum.py) | ORCA parsers; Wilson **B**-matrix; primitive internal-coordinate derivatives |
| [`backend/psi4/Psi4.py`](.github/backend/psi4/Psi4.py) | Psi4 energy / gradient / Hessian with unit conversion to Å |
| [`backend/internal/internal_prior.py`](.github/backend/internal/internal_prior.py) | Optional internal-coordinate priors stacked with the spectral block |
| [`backend/conformers/geometryguess.py`](.github/backend/conformers/geometryguess.py) | Template guesses and spring-style relaxation |
| [`backend/multistart.py`](.github/backend/multistart.py) | Parallel multi-start runs and best-run selection |
| [`backend/priors/bayes_tune.py`](.github/backend/priors/bayes_tune.py) | Optional Bayesian hyperparameter search (`scikit-optimize`) |
| [`backend/symmetry.py`](.github/backend/symmetry.py) | Optional point-group projection of steps and coordinates |
| [`backend/autoconfig.py`](.github/backend/autoconfig.py) | Adaptive trust region / damping / weight policy from diagnostics |
| [`backend/kraitchman.py`](.github/backend/kraitchman.py) | Kraitchman single-substitution rs coordinates, planar moments, inertial defects (auto-run; see `exports/kraitchman_rs.csv` and the report section) |

## Ground state to equilibrium ($B_0 \to B_e$)

Fitting a geometry to observed $B_0$ directly conflates structure with
zero-point motion. The correction chain maps the observed constants onto
semi-experimental equilibrium targets:

$$B_e = B_0 + \Delta_\mathrm{vib} + \Delta_\mathrm{elec} + \Delta_\mathrm{BOB}$$

**$\Delta_\mathrm{vib} = \tfrac12 \sum_r \alpha_r$**, with $\alpha_r$ built from three terms
(Mills 1972; Papoušek & Aliev 1982):

| Term | Source | Enabled by |
|------|--------|-----------|
| Harmonic $`\langle Q_r^2\rangle\,\partial^2 B/\partial Q_r^2`$ | Cartesian Hessian | `harmonic_from_hessian` |
| Coriolis $\zeta^{(\xi)}_{rs}$ coupling | Normal-mode eigenvectors | `harmonic_from_hessian` |
| Anharmonic $\phi_{rrs}$ (cubic) | Finite-difference cubic force field | `anharmonic_from_hessian` |

The anharmonic term is **not** a small refinement — it is usually the largest of
the three and carries the opposite sign to the harmonic term. For CO the
harmonic term alone gives $\alpha = -0.0103\ \mathrm{cm^{-1}}$ against an observed
$+0.0175$; adding the cubic term reproduces the Dunham/Pekeris value to 0.1%.
It costs $6N$ extra Hessian evaluations, so it is opt-in; when it is off, the
reported $\alpha$ uncertainty is widened to 100%.

**The corrections' level can differ from the fit's.** By default they come from
the fit's own Hessian, re-expanded at the current geometry on each Hessian
refresh. `correction_method` / `correction_basis` in `rovibrational_corrections`
(with `harmonic_from_hessian: true`; the `pyscf_hf` backend) take them from
another level instead: that level is optimised to its own minimum, the table is
built there once, and it is kept for the whole fit, since $\alpha$ describes a
force field at its stationary point, not wherever the fit has got to. On water
(CCSD/cc-pVDZ fit and prior, H$_2$O + D$_2$O, against $r_e$) the bond went from
$-0.58$ mÅ with the fit level's own corrections to $-0.03$ with them at
MP2/cc-pVTZ (seven Hessians, 3.7 min) and $-0.08$ at CCSD/cc-pVTZ (33 min); the
angle from $+0.16°$ to $+0.14°$ and $+0.10°$. Off by default, because it adds
the correction level's optimisation and Hessians to every run.

**$`\Delta_\mathrm{elec} = -(m_e/m_p)\, g_\alpha B_0`$** requires the rotational
g-tensor via `g_tensor`. Without it the code falls back to a crude $1/M_\mathrm{total}$
estimate that is roughly an order of magnitude too small and has the wrong sign
whenever $g < 0$ (as for OCS), so that path reports 100% uncertainty.

**$\Delta_\mathrm{BOB}$** uses per-element Watson u-parameters. The built-ins are
order-of-magnitude estimates; supply `bob_params` for sub-milliångström work.

See [`configs/example_water_semi_experimental.yaml`](configs/example_water_semi_experimental.yaml)
for a complete worked example, and `dev/tests/test_alpha_against_experiment.py`
for the validation suite (closed-form Dunham/Pekeris results, C₂ᵥ symmetry
constraints, and published CO/H₂O constants — no Psi4 or ORCA needed).

> **Centrifugal distortion.** With `fit_cd_constants: true` and `cd_weight > 0`,
> observed $\Delta_J$, $\Delta_{JK}$ and $\Delta_K$ (`cd_observed`, MHz) become fit rows
> against the harmonic prediction from the quantum Hessian at each trial geometry.
> The $\tau' \to$ Watson mapping is validated on water only (DJ +1.6%, DJK −36%,
> DK −7.6% against experiment), so each row's sigma is its measurement sigma
> (`cd_sigma`) and a 40% model error in quadrature (the larger `cd_sigma_fraction`
> if one is set; `cd_model_sigma` states another per constant, in MHz). `cd_weight` scales every row by $\sqrt{\texttt{cd\_weight}}$: a row's noise
> is its sigma $/\sqrt{\texttt{cd\_weight}}$, so only 1 keeps the stated model error.
> The three constants differ between the A and S reductions (water: 3.8%, 3.7%,
> 0.8%); state the observed block's with `cd_reduction: A` or `S`, otherwise the
> fit picks one from the asymmetry of the observed A, B, C (S when $|\kappa| \ge 0.9$;
> the start geometry only when no isotopologue observes all three) and keeps it. $\delta_J$ and $\delta_K$ are not fitted. `fit_cd_constants` defaults to off.

> **Model error.** The reported widths include the $B_0$-versus-$B_e$ model
> error, propagated as shared across isotopologues, only where a row carries it:
> a $B_0 \to B_e$ correction with a stated sigma (the Hessian-built corrections
> and `alpha_mhz` both have one), or `sigma_systematic_mhz` on the isotopologue
> (MHz per component; add `sigma_includes_systematic: true` when `sigma_mhz`
> already holds it, so it is counted once). `model_error: {auto: true}` fills it
> in from the inertial defect where neither is given (that estimate runs 3-10x
> large on A and B), and `model_error.isotopologue_correlation` (default 0.99)
> sets how much of it isotopologues share. A run whose rows carry none -- raw
> $B_0$ fitted as $B_e$ -- reports widths of measurement precision only and
> says so in its report and uncertainty bundle (`model_error`: null per
> isotopologue); on analytic water those are 0.28 mÅ against 6.9 with the
> corrections' sigma.

## How data and theory share authority

`MolecularOptimizer` uses the **joint** objective by default
(`objective_mode="joint"`, `quantum_prior_sigma_ang=0.020`): every direction
stays contested, weighted by how well the data and the quantum surface each
determine it.

The alternative `split` objective partitions the parameter space hard: whatever
survives the SVD rank cutoff is handed **entirely** to the spectral data, and the
quantum surface governs only the null space. That works when the retained
directions are well determined — but the cutoff is *relative*
(`sv_threshold × s_max`, with `sv_min_abs` defaulting to 0), so a direction the
data resolves only loosely is still treated as fully constrained and theory gets
no vote in it. Water's bond angle is such a direction, and under `split` it comes
out worse than either theory or experiment alone would give.

| Control | Effect |
|---------|--------|
| `optimizer.objective_mode: joint` with `optimizer.quantum_prior_sigma_ang` | The default. Solves $`(J^TJ + \alpha_q H + \lambda I)\,\Delta p = J^T r - \alpha_q g`$, leaving every direction contested and weighted by how well each source knows it. `quantum_prior_sigma_ang` is the displacement over which the quantum surface is trusted, roughly the geometry error of the method, which is what makes $\alpha_q$ interpretable rather than an arbitrary knob. |
| `optimizer.objective_mode: split` with `optimizer.sv_min_abs` | The SVD partition. `sv_min_abs` is an absolute floor on the singular value. The Jacobian is σ-weighted, so $1/s$ is the parameter uncertainty along a direction — the floor means "only trust what the data resolves this well". All-or-nothing per direction. |

> **Note.** The config-driven runner (`python -m cli run`) fits the same
> objective as the API and the benchmark: `joint`, with `quantum_prior_sigma_ang`
> calibrated to the level of theory unless the config states one, the detected
> point group imposed unless the config has a `symmetry` key (an explicit `null`
> means unconstrained), no rule-library internal prior while the quantum prior is
> on unless `internal_coordinates.prior_weight` is stated, and the engine's
> convergence scale. It used to pass `split` with an inert legacy quantum weight;
> an explicit `optimizer.objective_mode: split` still gets that legacy set-up.
> On 9 molecules at HF/6-31G the change took config runs from 2 of 9 to 8 of 9
> restart-stationary and the mean angle error from 4.05 to 0.97 degrees.

### Which objective to use depends on how much data you have

This is the single most consequential setting, and it flips with the size of the
dataset. Two measured cases:

| Case | Observables | Internal DOF | Best objective |
|------|-------------|--------------|----------------|
| Water, one isotopologue (`scripts/theory_vs_experiment_vs_hybrid.py`) | 3 | 3 | `joint`, `quantum_prior_sigma_ang: 0.005` |
| Fluorobenzene, one isotopologue (`scripts/fluorobenzene_vs_published.py`) | 3 | 30 | `joint`, `quantum_prior_sigma_ang: 0.005` |
| Fluorobenzene, eight isotopologues (`scripts/fluorobenzene_full_data.py`) | 24 | 30 | `split` |
| Vinyl / acetyl fluoride, fluoroethane — parent only (`scripts/monofluoro_benchmark.py`) | 3 | 12–18 | `joint`, `quantum_prior_sigma_ang: 0.005` |
| Vinyl / acetyl fluoride, fluoroethane — all species (same script) | 18 | 12–18 | `split` |

With few observables the split partition hands whole directions to data that
barely resolves them, drives the residual below what the physics justifies, and
distorts the structure doing it — on water the angle error trebles, on
fluorobenzene the C–H angles do. A calibrated prior keeps those directions
contested and fixes most of it.

With a full isotopologue set the position reverses. Against the published
fluorobenzene structure, 24 observables give RMS bond errors of 11.4 mÅ from
theory alone, 8.2 mÅ from spectroscopy alone, and **4.0 mÅ** from the split
hybrid — better than either input on bond lengths, angles and the C–F distance
at once. Forcing `quantum_prior_sigma_ang: 0.005` there costs more than half the
gain (8.9 mÅ), because the prior now over-constrains directions the data
determines perfectly well.

Rule of thumb: keep the joint objective's calibrated prior when the fit is
undersaturated, and try `split` when it is not. `python scripts/tune_quantum_prior.py`
scans the value; 0.005 Å is a sparse-data result, not a validated default.

> **The 0.005 Å rows above predate a units fix.** Internal-coordinate runs
> used to take $\alpha_q$'s typical curvature from the q-space Hessian, which
> mixes Hartree/Å² with Hartree/rad²; on water that made the prior 1.5–2.1×
> stronger at a given σ than in Cartesian mode. It now comes from the
> Cartesian Hessian in both modes, so σ means the same thing in each, and a
> 0.005 measured then corresponds to roughly 0.0035 Å now. Re-scanned on the
> one-isotopologue water case, the corrected engine's trade is monotone and
> 0.003 Å gives the old combined score (`dev/tests/test_endtoend_water_pipeline.py`).
> The fluorobenzene and monofluoro rows have not been re-scanned.
[`backend/spectral/prior_widths.py`](.github/backend/spectral/prior_widths.py)
carries a per-level table (HF/6-31G 0.020 Å, B3LYP/6-31G(d) 0.007, B3LYP/cc-pVTZ
0.004) and `MolecularOptimizer` uses it when `quantum_prior_sigma_ang` is left
unset, so the width follows the level of theory rather than a single default.

### Monofluorinated benchmark — and a caveat on the two rows above

`scripts/monofluoro_benchmark.py` runs theory, spectroscopy-only, and both
hybrids over three published structures (vinyl fluoride, acetyl fluoride,
fluoroethane) at RHF/6-31G, at two data levels each. Every rotational constant
is a **measured** literature value transcribed from *NBS Monograph 70,
Microwave Spectral Tables* — nothing is back-calculated from a geometry.
`scripts/check_monofluoro_references.py` validates each species before use, and
`scripts/build_monofluoro_report.py` turns the run's JSON into
[`reports/monofluoro_benchmark_report.pdf`](reports/monofluoro_benchmark_report.pdf).
This benchmark predates the vibration–rotation corrections it calls for below;
the [Results](#results) table reflects the current engine.

**The objective-choice rule above does not survive measured data.** The water
and fluorobenzene rows use isotopologue constants *derived* from their reference
structures, which are mutually consistent by construction. That removes the
systematic r_s-vs-r_0 offset which turns out to drive the behaviour. Re-run on
measured constants, `split` wins at rank deficits of 15, 3 and 1 while `joint`
wins at 12, 10 and 3 — no ordering at all. Treat those rows as describing
self-consistent synthetic data, not experiment.

What does hold up:

- **RHF/6-31G has a systematic C–F bias**, +27 to +31 mÅ, same sign in all three
  molecules. A method bias, not scatter, and what the spectral data is good at
  removing: some combination of data and theory beats theory alone on C–F in 6
  of 6 cases.
- **Everything is undersaturated.** Counting constants overstates the
  information badly. Vinyl fluoride's 22 measured constants carry rank **9**
  against 12 internal DOF; acetyl fluoride 30 → rank 14 of 15; fluoroethane
  18 → rank 15 of 18. All published data still leaves every molecule
  underdetermined, so `sv_min_abs` / `objective_mode` matter in every real case.
- **On overall bond error the hybrid is not dependably better than theory** —
  5 of 6 even choosing the objective with hindsight. The binding constraint is
  no longer the objective but the missing vibration-rotation correction: fitting
  r_0 constants uncorrected against an r_s reference leaves a one-signed ~0.5%
  residual that the fit removes by distorting the structure. In vinyl fluoride
  that makes the spectroscopy-only fit *worse* with 22 constants (31.1 mÅ) than
  with 3 (21.0 mÅ). Applying `anharmonic_from_hessian` here is the obvious next
  step. *(2026-11-19, code 5cfc5d4: with the corrections applied, the
  spectroscopy-only fit to all 22 constants is still 25.6 mÅ at RHF/6-31G. Most
  of that comes from five low-precision deuterated A₀ values in the
  compilation (784 and 788 have negative inertial defects; see
  `dev/monofluoro_references.py`). Without those five, the corrected
  spectroscopy-only fit is 5.4 mÅ from the reference and 0.9 mÅ from SE100's
  r_e^SE, which was itself derived from much the same B and C data. The
  constants have since been replaced by Hayashi & Inagusa's 1989
  measurements, with which every species' inertial defect is normal.)*

## Torsion / Large-Amplitude Motion (LAM) pipeline

A self-contained torsion-rotation pipeline handles molecules with an internal methyl (or other Cn) rotor. It uses a RAM-lite (rho-axis method) Hamiltonian in a Fourier basis |m⟩ and is independent of the geometry-inversion loop above.

### Torsion backend modules

| Module | Role |
|--------|------|
| [`backend/torsion/torsion_hamiltonian.py`](.github/backend/torsion/torsion_hamiltonian.py) | `TorsionHamiltonianSpec`, Fourier potential matrix, `solve_ram_lite_levels`, `torsion_probability_density` |
| [`backend/torsion/torsion_rot_hamiltonian.py`](.github/backend/torsion/torsion_rot_hamiltonian.py) | Full J-block torsion-rotation Hamiltonian with centrifugal distortion |
| [`backend/torsion/torsion_symmetry.py`](.github/backend/torsion/torsion_symmetry.py) | C3 block decomposition (A/E1/E2), tunneling splittings, selection rules, nuclear-spin weights |
| [`backend/torsion/torsion_average.py`](.github/backend/torsion/torsion_average.py) | Quantum and Boltzmann torsion-scan averaging of A/B/C constants; rigorous uncertainty propagation |
| [`backend/torsion/torsion_intensities.py`](.github/backend/torsion/torsion_intensities.py) | `⟨ψ\|cos(α)\|ψ⟩` matrix elements, Hönl-London factors, complete line-list generation |
| [`backend/torsion/torsion_fitter.py`](.github/backend/torsion/torsion_fitter.py) | Damped Gauss-Newton fitting to levels, transitions, or joint levels + rotational constants |
| [`backend/torsion/torsion_uncertainty.py`](.github/backend/torsion/torsion_uncertainty.py) | Jacobian, covariance, Fisher information, identifiability |
| [`backend/torsion/torsion_lam_integration.py`](.github/backend/torsion/torsion_lam_integration.py) | LAM correction report with uncertainty propagation into the main spectral fit |
| [`backend/torsion/hindered_rotor.py`](.github/backend/torsion/hindered_rotor.py) | Independent 1D hindered-rotor solver (legacy; used for Boltzmann weight helper only) |

### What the torsion pipeline provides

**Energy levels and tunneling** — `solve_ram_lite_levels` diagonalises the RAM-lite Hamiltonian for any (J, K) block. `torsion_symmetry` decomposes levels into A/E1/E2 species and reports A–E tunneling splittings (validated against methanol literature values).

**Scan averaging with uncertainty propagation** — `torsion_average` weights torsion-scan geometries by the quantum probability density |ψ(α)|² of any eigenstate (or a thermal mixture), producing effective A/B/C constants with rigorously propagated uncertainties from both grid-point measurement errors (Hessian-diagonal σ) and representational scatter.

**Line intensities** — `torsion_intensities` computes |⟨ψ_hi|cos(α)|ψ_lo⟩|² transition dipole matrix elements, applies Hönl-London factors (symmetric-top approximation), applies C3 nuclear-spin statistical weights (A:E = 1:2 for CH₃), and exports a complete sorted line list as CSV. Selection rules (A↔A, E↔E allowed; A↔E forbidden) are enforced automatically.

**Parameter fitting** — `torsion_fitter` provides three fitting modes:
- `fit_torsion_to_levels` — fit to observed torsional level energies
- `fit_torsion_to_transitions` — fit to observed transition frequencies
- `fit_torsion_joint` — joint fit to torsional levels **and** torsion-averaged rotational constants (A/B/C) simultaneously via a unified Gauss-Newton loop

### Torsion config keys

Add a `torsion_hamiltonian` block to a run config to activate the pipeline:

```yaml
torsion_hamiltonian:
  enabled: true
  F: 5.1753          # internal rotation constant (cm-1)
  rho: 0.0812        # coupling between internal and overall rotation
  n_basis: 12        # Fourier basis truncation (|m| ≤ n_basis)
  potential:
    v0: 185.5        # potential offset (cm-1)
    vcos: {3: -185.5}   # cos(3α) term
  line_list:
    enabled: true
    max_freq_mhz: 50000
    min_line_strength: 1e-6
  fitting:
    enabled: true
    params: [Vcos_3, F]
    targets:         # observed torsional levels
      - {J: 0, K: 0, level_index: 0, energy_cm-1: 0.0}
    targets_rotational:   # joint fit: observed B0 constants
      - {component: B, obs_cm1: 0.8220, sigma_cm1: 0.002}
```

See [`configs/example_methanol_lam.yaml`](configs/example_methanol_lam.yaml) for a complete working example.

## Repository layout

- **`.github/backend/`** — core library (spectral, quantum, optimizer, priors, symmetry, torsion).
- **`cli.py`** — main entry point (`python -m cli <command>`).
- **`runner/`** — config loading/validation (`usability.py`), the generic run driver
  (`run_generic.py`), report and export generation (`reporting.py`), presets (`run_settings.py`).
- **`configs/`** — example run configs.
- **`molecule_runners/`** — per-molecule driver scripts.
- **`dev/tests/`** — test suite; **`dev/benchmarks/`** — LAM and conformer benchmark suites.
- **`output/`** — all generated artifacts (gitignored); runs land in `output/runs/`.
- **`requirements.txt`** — Python dependencies.

## Requirements

- Python 3.10+
- **NumPy**, **SciPy** (see `requirements.txt`)
- **Psi4** — if using `quantum_backend="psi4"` (often via Conda).
- **ORCA** — optional, if using `quantum_backend="orca"`.
- **`scikit-optimize`** — optional, for `backend/bayes_tune.py` (`pip install scikit-optimize`).

Example environment (Unix-like):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` or `Activate.ps1`.

## Running

From the project root, the config-first interface is:

```bash
python -m cli validate configs/example_water_semi_experimental.yaml
python -m cli run      configs/example_water_semi_experimental.yaml
python -m cli report   output/runs/<timestamped_run_dir>
```

`run` creates a timestamped directory under `output/runs/` by default, copies the
input config, and writes `report.md`, `exports/residuals.csv`,
`exports/final_geometry.csv`, and diagnostic plots under `plots/`.
`report` rebuilds `report.md`, `report.html`, and plots from an existing run.

Other commands: `lam-scan`, `lam-fit`, `lam-diagnose`, `uncertainty`, `benchmark`.
Run `python -m cli --help` for the full list.

Ready-to-run config examples:

- `configs/example_water_semi_experimental.yaml` — full $B_0 \to B_e$ correction chain
- `configs/example_water.yaml`
- `configs/example_OCS.yaml`
- `configs/example_CO2.yaml`
- `configs/example_SO2.yaml`
- `configs/example_formaldehyde.yaml`
- `configs/example_propyne.yaml`
- `configs/example_acetaldehyde.yaml`
- `configs/example_methanol_lam.yaml`
- `configs/example_acetaldehyde_lam.yaml`
- `configs/benzene.yaml`, `configs/fluorobenzene.yaml`

### Bayesian / Bootstrap Uncertainty (v2)

Run uncertainty with the greenfield v2 engine:

```bash
python -m cli uncertainty configs/example_water_semi_experimental.yaml \
  --mode both \
  --samples 20 \
  --mcmc-steps 5000 \
  --burn-in 1000 \
  --chains 4 \
  --seed 42
```

The reported uncertainty is the analytic posterior (the Laplace covariance
plus the model error, shared across isotopologues). Before any width,
`quantize uncertainty` recomputes the quantum Hessian at the converged
geometry and re-converges on it: the fit computes its Hessian on its own
schedule, usually at the start (4-20 mA from the answer on nine HF/6-31G
molecules, widths up to 7% off). That costs one Hessian (up to three where the
corrections are built from the Hessian and so move with it; the summary records
whether the answer settled); `--no-hessian-refresh` skips it. A fit that stops
at `max_iter` is warned about: its widths describe where it stopped. `quantize run` keeps the fit's own Hessian and its report says how
far away it was. The interval table, the
predictive check and the plots describe a sample of it: the MCMC chains when
they pass their convergence checks, an `rto` bootstrap when one was run,
otherwise draws from the reported posterior itself; clouds are never pooled.
`--bootstrap-strategy` says what each refit draws: `error-model` (default)
the constants from the engine's error model, a check of the posterior less
its prior part; `rto` the quantum prior's centre as well, a sample of the
posterior; `isotopologue-gaussian` the legacy independent draw. Each run
prints its cloud beside the analytic value it should match, and flags a
coordinate where the two differ beyond their Monte Carlo error (taken from
the cloud's own kurtosis, at a family-wise 3-sigma rate over the
coordinates) by enough to move the reported width 2% or more. It also says
where its bar sits at that many refits -- at the default 20 and nine
coordinates a ratio of 0.56 or 1.80 is flagged only about half the time, so a
clean check there is weak evidence -- and that a cloud flagged as narrower may
instead be heavy-tailed with its tails unsampled.
Exit status: 0 interval reported; 1 no interval; 2 usage error or a strategy
refused before anything past the base fit ran; 3 interval reported but a
requested part produced nothing usable (a bootstrap refused after the fit or
left with too few usable refits, or refused MCMC chains; the bundle says which).

Outputs are written under the run workdir, including:

- `uncertainty_v2/posterior_laplace_summary.json`
- `bootstrap_v2/bootstrap_summary.json` (unless `--no-bootstrap-persist`)
- `mcmc_v2/mcmc_summary.json` (unless disabled)
- `uncertainty_v2/uncertainty_run_summary.json` (consolidated run bundle)
- `plots/uncertainty/*.png` diagnostic plots

The lower-level runner remains available for compatibility:

```bash
python runner/run_from_config.py configs/example_water_semi_experimental.yaml
python runner/run_from_config.py configs/example_water.yaml --no-run-dir
```

You can also call a molecule driver module directly:

```bash
python -m molecule_runners.run_water
```

Drivers build isotopologue inputs, generate a starting geometry, run multistart
optimization, and print a summary. Presets live in `runner/run_settings.py`.

### ORCA and `runner/run_settings.py`

Drivers read `runner/run_settings.py`, which defaults to `quantum_backend="orca"` and `orca_exe=None`. The optimizer then searches for ORCA in this order: **`orca` on your PATH**, a **full path** if you set one, then an **`orca` or `orca.exe` file in the current working directory** (so you can drop or symlink the binary into the project folder). You can also set the path before running:

```bash
export ORCA_EXE="/full/path/to/orca"
python -m cli run configs/example_water.yaml
```

On Windows, you can instead set `orca_exe` in `BASE_SETTINGS` to your `orca.exe` path. If you do not have ORCA, either install it, point `ORCA_EXE` at it, or switch to **Psi4** (`quantum.backend: psi4` in the config and a Conda environment with `psi4` installed), or set `quantum.backend: none` for spectral-only mode.

**Parallel multistart + ORCA:** many licenses allow only one ORCA job at a time. `run_multistart` therefore defaults to **`max_workers=1`** when `quantum_backend="orca"`. If your license allows multiple processes, set **`QUANTIZE_ALLOW_PARALLEL_ORCA=1`** before running to use the preset worker count.

**Paths with spaces:** older ORCA builds can mishandle absolute paths that contain spaces; the driver runs ORCA with `cwd` set to the job directory and passes **`quantize_orca.inp`** by filename only. If you still see startup errors, move the project to a directory whose full path has no spaces.

## Interpreting output

- **Rank** — number of directions retained above the relative singular-value cutoff in the stacked Jacobian SVD (a diagnostic under `joint`; under `split` it also decides which directions the data own).
- **RMS MHz** — root-mean-square residual of rotational constants in MHz (unweighted block).
- **$`\|\Delta x_r\|`$** — norm of the step projected onto the spectral range space.
- **$`\|\Delta x_n\|`$** — norm of the step projected onto the null space.
- **$`\|g_n\|`$** — norm of the gradient projected onto the null space (hybrid mode).
- **$`\|\Delta E\|`$** — magnitude of energy change between iterations (Hartree).

If rank stays low and residuals plateau, add more informative isotopologues and/or check consistency of $B_0$, $\alpha$, and uncertainties $\sigma$.

## Notes

- Run scripts use `if __name__ == "__main__":` and `freeze_support()` for multiprocessing on Windows.
- Equations and weighting rules are documented in the docstrings of the modules that implement them.
