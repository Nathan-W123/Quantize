# Quantize

Hybrid molecular geometry inversion from rotational spectroscopy and quantum chemistry.

Quantize optimizes molecular structures (bond lengths and angles) for spectroscopically underdetermined systems, where the isotopologue rotational constants alone do not fix the geometry. It combines the spectroscopic data with quantum mechanics by minimizing a single joint objective function.

## Core idea

- Use observed rotational constants (`A`, `B`, `C`) from one or more isotopologues.
- Map ground-state constants onto equilibrium targets via $B_e = B_0 + \Delta_\mathrm{vib} + \Delta_\mathrm{elec} + \Delta_\mathrm{BOB}$ (see [Ground state to equilibrium](#ground-state-to-equilibrium-b_0-to-b_e)).
- Stack the σ-weighted spectral Jacobian $J$ and residuals $r$ across all isotopologues, and take the electronic-energy **gradient** $g$ and **Hessian** $H$ from Psi4 or ORCA.
- Minimize one **joint objective**, the spectral $\tfrac12\chi^2$ plus $\alpha_q$ times the electronic energy, with damped Newton steps $`(J^TJ + \alpha_q H + \lambda I)\,\Delta p = J^T r - \alpha_q g`$. No direction is handed wholesale to either source: each carries weight in proportion to how well it determines that direction, so theory holds the structure where the data are silent and yields where they are informative.
- Set $\alpha_q$ through `quantum_prior_sigma_ang`, the displacement over which the quantum surface is trusted (roughly the method's geometry error; `MolecularOptimizer` defaults to 0.020 Å). $\alpha_q$ is chosen so the prior's RMS bond-stretch width equals it, from each molecule's own Hessian (D-170); `optimizer.match_prior_bond_widths: false` restores $\alpha_q = 1/(\tfrac12\bar\lambda\sigma_x^2)$, whose bond widths came out 0.65–0.77 $\sigma_x$ on the nine benchmark molecules.

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
mixed estimation beats the hybrid on formyl fluoride (by 0.07 mÅ), as it
already did on water (an earlier claim that it also won on chlorofluoromethane
is withdrawn: see the correction below). A class seen in
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

> **Correction (2026-11-26): the mixed-estimation baseline broke symmetry.**
> Its predicates held only the average of each set of equivalent bonds (and
> angles), so where the isotopologues cannot see an antisymmetric distortion
> the fit used it, and the symmetry-averaged score hid it. Chlorofluoromethane's
> two C–H bonds came out 0.96 Å apart at RHF/6-31G; that structure was the
> 4.26 mÅ that had mixed estimation ahead of the hybrid's 5.43 there. Mixed
> estimation now keeps the molecule's symmetry, as fitting symmetry-unique
> parameters does (decisions D-161). Measured at RHF/6-31G against the same
> references: chlorofluoromethane base 31.38 -> 26.00 mÅ, `offsets`
> 4.26 -> 6.03 (the hybrid ahead); fluoroethane base 12.66 -> 11.28, `offsets`
> 8.24 -> 9.07; acetyl fluoride within 0.02; the other six unchanged, water
> and ozone included. The mixed-estimation entries in the tables here
> (best-of means, B3LYP rows, chlorofluoromethane's r_e row) still carry the
> old baseline until the tables are rebuilt.

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
> *The fit's own gradient, now with the grid response* (D-59 flagged it;
> D-164, 2026-11-26, the user's call). The gradient a fit
> takes from the `pyscf_hf` backend -- `run_hessian`'s at each Hessian,
> `run_gradient`'s between them -- was pyscf's default, without the grid
> response: up to 4.4e-4 Ha/Å off at the B3LYP theory geometries (3.3e-4 once
> the rigid motions the fit projects out are removed). The benchmark cannot
> see it, because its target prior (Cartesian mode) uses H(x - target) in
> place of the gradient (a row came out bit-identical with the gradient
> replaced or corrupted). A fit without a target -- the CLI's -- centres its
> quantum prior at x - H⁻¹g, which the missing term moves by up to 0.27 mÅ in
> a bond. Run through the CLI (B3LYP/6-31G(d), the benchmark's base
> corrections plus the CLI's default BOB terms) with the grid response on
> both, the selected answers move 1.5e-4 Å (formyl fluoride; bond RMS 2.53 ->
> 2.49 mÅ against its effective reference) and 1.7e-4 Å (fluoroethane; 6.68 ->
> 6.61), all of it through `run_hessian`'s gradient: the selected start never
> calls `run_gradient`. With the response, formyl fluoride's CLI answer lands
> 6e-6 Å from the benchmark's (1.6e-4 without). Timed through the CLI (two
> runs each, alternating), the change costs formyl fluoride nothing (55.5 /
> 55.4 s without, 53.5 / 53.1 with) and fluoroethane 4.5% on the means (122.8
> / 128.2 s against 126.4 / 135.8), about the spread between repeats. Both
> now take the response: on displaced water (B3LYP/sto-3g) the gradient
> matches a finite difference of the energy to 3.7e-7 Ha/bohr against 4.8e-6
> without (`dev/tests/test_dft_fit_gradient.py`). The benchmark rows above do
> not move (their target prior does not use the gradient).
>
> *The prior's bond widths match σx* (D-170, 2026-11-27, the user's call,
> decision page item 13). The tables in this README predate it and have not
> been regenerated since. The weight 1/(½ λ̄ σx²) set a prior whose RMS bond width
> was 0.77 σx at RHF/6-31G and 0.65 σx at B3LYP/6-31G(d) on the nine
> molecules; the weight now makes it σx (nothing fitted: each molecule's
> factor comes from its own Hessian). Measured at cd0365f (scratch rows),
> hybrid mean bond RMS in mÅ, base / offsets configurations: the nine at
> RHF 11.05 / 7.69 → 10.52 / 7.42, at B3LYP 4.58 / 4.10 → 4.55 / 4.13; the
> six validation molecules against r_e^SE at RHF 4.59 / 3.39 → 4.19 / 3.28
> (9 of 12 rows better), at B3LYP 2.71 / 1.96 → 2.41 / 1.82 (12 of 12).
> Reported widths widen; the worst validation error in reported widths goes
> 5.9 → 4.6 at RHF and 2.2 → 1.9 at B3LYP.
>
> Not run: the `elec` and `all` configurations (they need `pyscf.prop`, not
> installed here), and at B3LYP the `torsion` and `offsets+torsion` rows of
> fluoroethane and acetyl fluoride (a single scan outlasts the 2-hour job
> limit), so those two molecules' B3LYP best-of is over 11 configurations.
> Results are archived in
> [`reports/accuracy_upgrades/`](reports/accuracy_upgrades/).

### Validation set: six molecules nothing was tuned on

Every default above was tuned on the nine benchmark molecules, so their rows
partly score the tuning. Six more molecules, each with a semi-experimental
equilibrium structure (r_e^SE, bond uncertainty at most 0.35 mÅ at 1σ), are
held out of every calibration and of the offsets pool
([`dev/validation_references.py`](dev/validation_references.py); sources and
transcription checks in [`dev/validation_molecules/`](dev/validation_molecules/)).
They are scored in a results file of their own
(`scripts/accuracy_upgrades_benchmark.py scored=validation`). Bond RMS error
against r_e^SE in mÅ, code vintage d711d99; the best of the three in bold:

| Molecule (species) | RHF base: theory | ME | hybrid | RHF offsets: theory | ME | hybrid |
|---|---:|---:|---:|---:|---:|---:|
| formaldehyde (6) | 14.15 | **0.27** | 0.75 | 5.88 | **0.45** | 1.48 |
| thioformaldehyde (8) | 24.67 | 0.67 | **0.22** | 23.21 | 0.39 | **0.21** |
| ketene (16) | 7.36 | 3.97 | **3.80** | 3.12 | 1.74 | **1.13** |
| sulfur dioxide (3) † | 123.84 | **0.59** | 0.86 | 123.84 | **0.49** | 0.59 |
| pyrimidine (16) | 8.84 | 8.15 | **6.79** | 12.88 | 5.25 | **4.68** |
| thiophene (24) | 38.85 | 20.83 | **15.09** | 40.10 | 22.15 | **12.27** |

| Molecule | B3LYP base: theory | ME | hybrid | B3LYP offsets: theory | ME | hybrid |
|---|---:|---:|---:|---:|---:|---:|
| formaldehyde | 7.02 | 3.70 | **2.57** | 5.42 | 4.70 | **3.60** |
| thioformaldehyde | 8.34 | 2.85 | **2.47** | 6.67 | 2.25 | **1.17** |
| ketene | 7.70 | 4.71 | **4.68** | 2.15 | 2.88 | **1.57** |
| sulfur dioxide | 32.85 | **0.42** | 0.52 | 32.85 | **0.38** | 0.41 |
| pyrimidine | 6.00 | **2.83** | 2.98 | 2.10 | **1.51** | 1.57 |
| thiophene | 12.25 | 5.94 | **3.01** | 11.25 | 6.12 | **3.47** |

† RHF's force field is flagged for SO2 (the multireference diagnostic fires).
Thiophene's B3LYP rows were run at e2b4903 and are identical at d711d99.
Both tables predate D-170 (the prior's bond widths now match σx; its measured
effect on these rows is in the provenance note under Results).

The hybrid beats theory alone on all 24 rows and mixed estimation on 16:
mixed estimation is ahead on formaldehyde at RHF, on sulfur dioxide at both
levels, and narrowly (0.06–0.15 mÅ) on pyrimidine at B3LYP. Two things to
know when reading the offsets columns. At RHF the pool's
C–N offset comes from isocyanic acid's and hydrogen cyanide's multiple bonds,
so it lengthens pyrimidine's aromatic C–N bonds by 23 mÅ and makes the theory
row worse (8.84 → 12.88); the offsets ignore bond order (backlog N41). Each
offsets row now records what its class's offset was pooled from, and the
benchmark warns when a bond borrows only from bonds more than 0.08 Å
different in theory length (D-171): it flags this C–N, and also vinyl
fluoride's C=C and fluoroacetylene's C≡C, whose offsets help them. And
thiophene's S–C bond has no offsets class at all: at RHF/6-31G (no d functions
on sulfur) it starts 80 mÅ long. At B3LYP the offsets help pyrimidine's hybrid
(2.98 → 1.57) but not thiophene's (3.01 → 3.47).

**When the data cannot pin the structure down.** Cut to one to three
isotopologues each (fewer independent constants than structural parameters,
as chlorofluoromethane's two chlorine species are), the hybrid still moves the
bonds toward r_e^SE on 9 of 10 RHF rows (formaldehyde 14.15 → 8.57 mÅ from the
parent alone; ketene's offsets row, 3.12 → 3.71, is the exception), but with
one species part of the correction goes into the angles (formaldehyde
0.15 → 1.48°). Its reported widths grow (median 3.6–3.9× for formaldehyde and
thioformaldehyde from the parent alone, 1.1–1.8× for the others) and the
labels move from "data-constrained" to "mixed information"; on the three
small molecules the errors stay within 1.9 of the reported 1σ widths (within
0.9 with every species, sulfur dioxide included). On the two rings the widths are too narrow with every species as
with three, so the cause is not the data: pyrimidine's ring angles miss by
1.5–2.0° at 0.25–0.34° widths in the offsets configuration, whose narrowed
prior covers angles the offsets never correct, and thiophene's S–C and C–C
bonds end 17–31 mÅ off at 5–8 mÅ widths (backlog N42)
([`scripts/archive/validation_thin_data.py`](scripts/archive/validation_thin_data.py),
code vintage 6679d00).

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
> the start geometry only when no isotopologue observes all three) and keeps it. $\delta_J$ and $\delta_K$ are not fitted. A linear molecule has one quartic constant, $D$: give it as `DJ`; its prediction comes from the two perpendicular axes, so it stays smooth as the geometry bends (fluoroacetylene and HCN, HF/6-31G: under 0.4% at 5°), and `DJK` or `DK` stated for it are not fitted. It is recognised once per isotopologue -- every isotopologue observes B alone and every atom of the start geometry lies within 0.4 Å of the molecule's axis -- or stated with `cd_reduction: linear`. `fit_cd_constants` defaults to off.
>
> Measured 2026-11-22 (HF/6-31G; formyl fluoride, vinyl fluoride and chlorofluoromethane;
> synthetic `cd_observed` at 0.7, 1.0 and 1.3 times the harmonic prediction at the reference
> structure; `cd_weight` 1). The prediction is made in the frame of the Hessian it uses: the trial
> geometry is turned onto the geometry that Hessian was computed at first. Before that fix (D-88) a
> Cartesian Hessian was used in whatever orientation the molecule had, so turning the molecule
> changed the predicted constants (formyl fluoride: up to 0.5% for 0.01 rad, while A, B and C did
> not move); the fit turned the molecule to fit the rows, the steps were refused, and most fits had
> not settled after 200 iterations. Now all 18 fits (autoconfig on and off) stop after 11–12
> iterations, as they do without the rows, with last steps of at most 6.1e-4 Å. Observed constants
> 30% off the prediction moved the bonds 0.08–3.9 mÅ and the angles at most 0.14° from the fit
> without rows; matching data moved them at most 0.08 mÅ; the two autoconfig settings agree as
> they do without rows. For these molecules $\Delta_J$ is 3–8 kHz, so 40% of it is below the rows'
> 0.01 MHz sigma floor, which then sets the sigma 3–8 times above 40% of the predicted constant.
> Removing the floor changed the bonds by at most 0.08 mÅ, except formyl fluoride at 0.7 times
> (2.9 mÅ, its bond RMS against the reference 1.1 mÅ worse), so it stays.

> **Model error.** The reported widths include the $B_0$-versus-$B_e$ model
> error, propagated as shared across isotopologues, only where a row carries it:
> a $B_0 \to B_e$ correction with a stated sigma (the Hessian-built corrections
> and `alpha_mhz` both have one), or `sigma_systematic_mhz` on the isotopologue
> (MHz per component; add `sigma_includes_systematic: true` when `sigma_mhz`
> already holds it, so it is counted once). Where neither is given the engine
> fills it in with 1% of A and 0.5% of B and C (`model_error.auto`, on by
> default since 2026-11-26, D-165) -- within 0.2-2.6x
> of the B₀-B_e gaps the benchmark's B3LYP VPT2 corrections give on seven
> molecules (0.2-2.5x as per-molecule medians), though 8.6x on ozone's A against
> its r_e; the inertial-defect estimate it replaced ran 0.007-9.1x (medians
> 0.04-8.9x), too small on B and C everywhere but water -- and `model_error.isotopologue_correlation` (default 0.99)
> sets how much of it isotopologues share. It changes the widths, not the
> fit: raw $B_0$ fitted as $B_e$ on analytic water gives an O–H width of
> 1.76 mÅ (angle 0.32°) with the estimate, 0.20 mÅ (0.03°) without, at a
> bit-identical structure, and so do the nine benchmark molecules and water
> with a correction table. The one difference measured is a fit with no
> quoted sigma whose χ² rescale would tighten it: the estimate is a floor the
> rescale cannot lower, and the structure differs by 1.5×10⁻⁵ Å. A species
> whose `sigma_mhz` already includes the model error
> (`sigma_includes_systematic`) must say how much (`sigma_systematic_mhz`)
> or ask for the estimate explicitly (`model_error: {auto: true}`). `model_error: {auto: false}` (`auto_systematic_sigma=False`) asks
> for measurement precision only; a run whose rows then carry none says so in
> its report and uncertainty bundle (`model_error`: null per isotopologue). A
> B₀ → B_e correction with its own sigma takes the estimate's place on its
> rows without a warning; a stated `sigma_systematic_mhz` it replaces is
> still named.

## How data and theory share authority

`MolecularOptimizer` uses the **joint** objective by default
(`objective_mode="joint"`, `quantum_prior_sigma_ang=0.020`): every direction
stays contested, weighted by how well the data and the quantum surface each
determine it.

The alternative `split` objective partitions the parameter space hard: whatever
survives the SVD rank cutoff is handed **entirely** to the spectral data, and the
quantum surface governs only the null space. That works when the retained
directions are well determined — but in the config runner's split set-up the
cutoff is *relative* (`sv_threshold × s_max`, with `sv_min_abs` 0), so a
direction the data resolves only loosely is still treated as fully constrained
and theory gets no vote in it. Water's bond angle is such a direction, and under
`split` it comes out worse than either theory or experiment alone would give.
The Python API instead defaults `sv_min_abs` to $1/\sigma_x$ when the quantum
prior is on, which hands a direction the data resolve less well than the prior
to theory.

| Control | Effect |
|---------|--------|
| `optimizer.objective_mode: joint` with `optimizer.quantum_prior_sigma_ang` | The default. Solves $`(J^TJ + \alpha_q H + \lambda I)\,\Delta p = J^T r - \alpha_q g`$, leaving every direction contested and weighted by how well each source knows it. `quantum_prior_sigma_ang` is the displacement over which the quantum surface is trusted, roughly the geometry error of the method, which is what makes $\alpha_q$ interpretable rather than an arbitrary knob. |
| `optimizer.objective_mode: split` with `optimizer.sv_min_abs` | The SVD partition. `sv_min_abs` is an absolute floor on the singular value. The Jacobian is σ-weighted, so $1/s$ is the parameter uncertainty along a direction — the floor means "only trust what the data resolves this well". All-or-nothing per direction. |

> **Split fits can stop short of their own answer.** With the Python API's
> `sv_min_abs`, the directions the cutoff gives to theory are often ones the data
> still weigh, and the step's acceptance test still scores the data there, so it
> refuses the quantum surface's moves and the fit stops where they are refused.
> On the benchmark molecules (HF/6-31G, from their start geometries) the report
> says so on 9 of the 18 fits (both coordinate modes): their null-space energy's
> minimum lies 9–44 mÅ (largest atom) away in Cartesian coordinates, two of them
> with a direction of negative curvature besides, and 4–28 mÅ in internal ones
> (mapped linearly; fluoroacetylene's 115 mÅ). A
> different trust radius
> moved formyl fluoride's answer by 1.5 of its widths, so the split widths
> describe neither the answer nor its bootstrap spread. The note appears when
> the gap passes 0.25 posterior widths (`split_stall`) and the step reads the
> same null space as its posterior; where they differ (fluoroethane in internal
> coordinates from its start geometry, whose bonds and angles cannot turn its
> torsion) the report's off-mode note says that instead. The check reads the
> quantum side only: a fit can also stop before it has finished fitting the data,
> and still depend on the schedule, without the note (ozone at a trust radius of
> 0.2). The `joint` objective has no such stall. With a prior target
> (`prior_target_coords`), the split posterior keeps the target's pull along
> the data's directions while its curvature does not, so its mode lies 0.02–1.0
> widths off the split fit (analytic water, σx 20–5 mÅ); the widths are
> unaffected. No script or config combines the two.

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

### Which objective to use

RMS bond error (mÅ) against each script's reference structure at RHF/6-31G,
with each script's data, start and reference and the objective and width set
explicitly (the scripts' own hybrid legs differ: `fluorobenzene_vs_published.py`
runs only `joint` at 0.005 Å, `monofluoro_benchmark.py` `split` and `joint` at
0.030); code ae82fd7:

| Case | Observables | Internal DOF | Theory | `split` | `joint`, calibrated (0.020 Å) | `joint`, 0.005 Å |
|------|-------------|--------------|--------|---------|-------------------------------|------------------|
| Fluorobenzene, parent only (`scripts/fluorobenzene_vs_published.py`) | 3 | 30 | 11.42 | 10.8-11.3 † | 11.27 | 11.18 |
| Fluorobenzene, eight isotopologues (`scripts/fluorobenzene_full_data.py`) | 24 | 30 | 11.42 | 10.6-11.6 † | 11.07 | 11.25 |
| Vinyl fluoride, parent only (`scripts/monofluoro_benchmark.py`) | 3 | 12 | 19.05 | 18.24 | 18.31 | 18.61 |
| Vinyl fluoride, all species | 24 | 12 | 19.05 | 15.87 | 17.41 | 18.38 |
| Acetyl fluoride, parent only | 3 | 15 | 15.75 | 14.78 | 15.20 | 15.61 |
| Acetyl fluoride, all species | 30 | 15 | 15.75 | 14.45 | 14.58 | 15.06 |
| Fluoroethane, parent only | 3 | 18 | 14.99 | 11.55 | 13.14 | 14.27 |
| Fluoroethane, all species | 18 | 18 | 14.99 | 12.12 | 12.49 | 13.53 |

† `split` does not settle on fluorobenzene. From the scripts' displaced start
it keeps drifting by a few hundredths of a mÅ per restart (parent only 10.78 ->
10.99 over five more restarts, eight isotopologues 10.63 -> 10.76 over four,
both still moving); from theory's minimum it ends at 11.27-11.29 and 11.60. The
fluorobenzene `joint` cells are the answers restarts settle on, the same from
either start (to 4e-6 Å). A first pass from the displaced start is not one: it
stopped about 4 mÅ (largest atomic move) from that answer, and up to 1.4 mÅ
nearer the reference on bonds than the answer keeps. The monofluoro fits start
from theory's minimum, as the script does; one restart moves them at most
2.8e-4 Å (vinyl fluoride's parent-only `split`, which settles at 18.28).

What the table shows:

- On the monofluoro molecules the calibrated `joint` is never the best setting
  scanned. `split` is best with the parent alone (by 0.07-1.59 mÅ over the
  calibrated `joint`, 0.04-1.01 over a 0.050 Å `joint` prior), and a 0.050 Å
  `joint` prior is best with all species (by 0.07-0.8 over `split`). On
  fluorobenzene only `joint` reaches an answer that does not depend on where
  the fit starts.
- The hybrid's gain over theory is small: 0.55-3.4 mÅ in the monofluoro cases
  (`split` or the calibrated `joint`), against 15-19 mÅ of theory error; on
  fluorobenzene the answers that settle are within 0.35 mÅ of theory, one
  (`split` from theory's minimum, eight isotopologues) 0.18 above it.
- 0.005 Å, the width these rows used to recommend for sparse data, is worse
  than the calibrated 0.020 in all six monofluoro cases (by 0.30-1.14 mÅ); on
  fluorobenzene it is 0.09 mÅ better with one isotopologue and 0.17 worse with
  eight.

Not reproduced: the figures this section used to give, measured at ab587e6
(2026-08-08) under `split` -- fluorobenzene with eight isotopologues 4.0 mÅ from
`split`, 8.2 from spectroscopy alone, 8.9 from `joint` at 0.005 Å -- and the rule
"joint when undersaturated, split when not". The engine has changed a great deal
since: the next day's 6f43c5a alone made `joint` the default objective, tied
`split`'s singular-value floor to the prior width and floored observation sigmas
at 0.1% of each constant. The difference is not traced to one cause. From
6f43c5a until d2078fa the scripts' legs labelled "split" passed no objective and
so ran `joint` at the calibrated width.

These scripts fit B₀ as if it were B_e against r_s or effective references,
which the next section shows is what drives this comparison. The
[Results](#results) benchmark -- `joint`, vibrational corrections, the prior
centred on the theory geometry -- is the measurement of the hybrid itself.
`python scripts/monofluoro_tune.py` scans the width on the monofluoro cases
(`--split` scans the split cutoff). Water's one-isotopologue end-to-end test
(`dev/tests/test_endtoend_water_pipeline.py`, an analytic surface) runs at
0.010 Å; its docstring records the width scan behind that.

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

**The eight-isotopologue fluorobenzene row is synthetic.** Its constants
beyond the parent's are *derived* from the reference structure and scaled by the
parent's own B₀-to-rigid ratio, so every species carries the same r_s-vs-r_0
offset and none of the species-to-species variation or measurement scatter that
real constants have. Treat it as synthetic data, not experiment; the monofluoro
rows use measured constants.

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
- **On overall bond error the hybrid is better than theory, but not by much**
  -- 0.55-3.4 mÅ in all six cases (table above), against 15-19 -- and the
  choice of objective moves the answer about as much (up to 1.6 mÅ between
  `split` and the calibrated `joint`). The larger constraint is the missing
  vibration-rotation correction: fitting r_0 constants uncorrected against an
  r_s reference leaves a one-signed ~0.5% residual that the fit removes by
  distorting the structure. With vinyl fluoride's original compilation that
  made the spectroscopy-only fit *worse* with 22 constants (31.1 mÅ) than with
  3 (21.0 mÅ); those constants have since been replaced (below). Applying `anharmonic_from_hessian` here is the obvious next
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

`run` reports the fit where it stops, on the Hessian it last computed --
usually at its start -- and that sits short of the fit's own optimum (where the
Hessian is at the answer) by 0.8-1.8% of the distance the fit moved. Measured
through `run` at HF/6-31G on formyl fluoride and chlorofluoromethane, from
their RHF minima and from 0.03 Å-displaced starts: 0.03-0.6 mÅ RMS (1.4 mÅ for
one atom), moving the RMS bond error against the reference by up to 0.5 mÅ in
either direction; one of the four varied by 0.5 mÅ between identical runs on
four threads. That was before D-146: `run` now recomputes the Hessian at
each start's answer and re-converges before ranking, as `quantize uncertainty`
does for its answer (below), which reaches the optimum and removes that
variation for one Hessian per start (+15-21% of the run here).

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
at `max_iter` is warned about: its widths describe where it stopped.
`quantize run` does the same for every start, before ranking them
(`--no-hessian-refresh` skips it, and the report then says how far the
fit's own Hessian was from the answer), and so, since D-177, does
`quantize uncertainty` for the starts it seeds from: ranked before any
refresh, the two commands picked different starts in 5 of 6 configs and a
water dimer's answers ended 5.8 mÅ apart with widths 3.5× apart (one Hessian
per start, +36–108% of the seeding time). Refreshed starts within the
spectral gate are ranked by the fit's own model energy at each answer
(D-179), which matched a fresh SCF energy there to 0.63 µHa on 36 starts,
where the energy at a start's last quantum point missed by up to 333 µHa;
with `--no-hessian-refresh` they are ranked by that last-point energy as
before (unrefreshed, the model energy was the worse of the two). Not only the widths depend on it: the
Hessian sets the quantum weight, so on a hydrogen-bonded water dimer
(HF/6-31G, eight starts) one refresh moved the answer by up to 14 mÅ (RMS) and
the weight by up to 18%, and brought the internal- and Cartesian-coordinate
answers from up to 23.5 mÅ apart to within 2.6 mÅ. It costs one Hessian per
start, 15–21% of a run on formyl fluoride and chlorofluoromethane.

The widths are not always Gauss-Newton's (JᵀJ). The fit probes its three
softest physical directions by finite differences, and along one where the
exact curvature is over 50% from Gauss-Newton's, the reported widths, the
distance to the optimum and the bootstrap's refusal limit use the exact
value, and so does the fit's step where the exact value is the larger
(`optimizer.exact_curvature_where_flagged: false` in a config, or
`--no-exact-curvature` on `quantize run` and `quantize uncertainty`, turns this
off and reproduces the Gauss-Newton fit and report). Beyond two widths from
its optimum a fit keeps Gauss-Newton's widths, and the report says why. On the
water dimer above (eight starts with no point group, both coordinate modes),
Gauss-Newton keeps at most 2.2% of the curvature along the hydrogen-bond torsion: at
the same answers it would report the widest angle at 13.5–331°, depending on
the start; the reported widths are now 6.6–9.6°, every answer stands within
0.052 widths of its optimum, and its 16-refit bootstrap keeps 14–16 refits
(8–13 before). The check runs at every step; profiled, it takes 0.3–2% of a
benchmark molecule's fit and 4% of the dimer's. In the RHF benchmark rows it
moves only isocyanic acid, by 0.26–0.45 mÅ onto another stationary point of
nearly the same objective. The interval table, the
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

> **Spectral-only Cartesian fits: the start prior's angle units (fixed
> 2026-11-26).** With `quantum.backend: none` and `coordinate_mode: cartesian`,
> the `BALANCED` (default) and `STRICT` presets add a prior that holds each fit
> near its start geometry (0.04 Å per bond, 2° per angle). Its angle rows took
> their residuals in degrees but their derivatives in radians, so the step and
> the reported uncertainty saw the angle prior 57.3 times weaker than the
> residuals did. On the nine benchmark molecules (starts displaced 0.03 Å from
> the reference, autoconfig off) the fix takes the rejected steps from 58 to 3
> over the nine fits, and the angle widths from "none" for 3 fits and up to 64°
> for others to 0.3–2.1°. Accuracy moves both ways, because the prior now holds
> the start's angles as firmly as it says: mean bond error 24.16 → 23.00 mÅ,
> mean angle error 1.578 → 1.922° (fluoroacetylene keeps its start's 4° bend,
> 1.12 → 3.86°). Dihedral residuals are also wrapped to ±180° now. The default
> `coordinate_mode: internal` was never affected.

> **Free torsions in internal mode.** `coordinate_mode: internal` (the
> default) steps in bonds and angles, and a torsion -- a methyl group's
> rotation -- changes neither, so the fit cannot turn it: acetyl fluoride and
> fluoroethane (HF/6-31G) stopped 0.19 and 0.16 posterior widths short of
> their optimum and kept 0 and 4 of 50 bootstrap refits. So by default
> (`internal_coordinates.use_dihedrals: auto`) the fit adds dihedrals exactly
> when bonds and angles miss a direction and dihedrals reach it (it says so
> when it does; a molecule whose bonds and angles reach every direction is
> unchanged); `false` keeps bonds and angles, `true` always adds dihedrals.
> With several starts, every start gets dihedrals when any one needs them,
> so all report the same coordinates. It moves those two answers to their
> optimum (bond RMS against their references −0.19 and +0.22 mÅ, autoconfig
> off; −0.08 and +0.37 on) and costs acetyl fluoride one more Hessian.
> Whenever the coordinate set misses a direction
> (bonds and angles without dihedrals, or an exactly linear chain even with
> them), the run and the uncertainty warnings say how many directions it
> reaches and how far the fit stands from its own optimum. At the benchmark
> starts that is acetone, acetyl fluoride, fluorobenzene and fluoroethane.
> A hydrogen-bonded water dimer whose start has no point group is another:
> bonds, angles and the O···H contact
> reach 10 of its 12 directions, missing the two torsions about the hydrogen
> bond. From eight such starts its fits ended 0.07 to 4.0 widths from their
> optimum, against 0.003 to 0.08 with `auto`. Some example configs
> (`configs/benzene.yaml`, `configs/fluorobenzene.yaml` and four small
> molecules) still state `use_dihedrals: false`.

> **Near-linear chains in internal mode.** Near 180° a valence angle no
> longer describes the bend (it is not differentiable there and stands for
> two bend directions), so a fit whose optimum is nearly linear can stop
> short: fluoroacetylene (HF/6-31G) at 179.995°, 0.008 posterior widths from
> its optimum (179.98°, 179.96°), and its bootstrap kept 27 of 50 refits. By
> default (`internal_coordinates.cartesian_fallback`, on since 2026-11-26,
> D-163) such a step is retaken in Cartesian coordinates on the same
> objective, and the rest of that fit steps in Cartesians: fluoroacetylene
> then reaches its optimum (3.5e-3 -> 6.9e-7 widths, bond RMS 15.396 ->
> 15.385 mÅ) and keeps 50 of 50. Of the nine benchmark molecules (HF/6-31G,
> internal mode, dihedrals auto) it acts on fluoroacetylene and formyl
> fluoride, whose answer then lands 0.018 mÅ from the Cartesian fit's after
> the CLI's Hessian refresh (0.072 without the fallback; Cartesian restarts
> differ by 0.05; bond RMS 13.76 mÅ either way, while without the refresh the
> two answers are 0.5 mÅ apart, 13.89 against 13.56); the other seven are
> bit-identical. It applies to the
> `joint` objective only; `cartesian_fallback: false` turns it off.

> **Pieces the bond rule leaves apart.** Bonds come from the covalent-radii
> rule (distance under 1.2× the sum of the radii; O–H 1.16 Å). Internal
> coordinates join pieces it leaves apart -- a stretched starting bond, or
> the molecules of a complex -- through their closest atoms; the joining
> coordinates are listed as `contact i-j` and get no rule-book prior: where
> the internal prior is on (spectral-only fits, by default) they are held to
> their starting value instead. A water dimer's hydrogen bond (1.95 Å) was
> pulled to 1.46 Å by the O–H rule; it now fits to 1.949 Å (the monomers'
> own bonds and angles keep their rule-book priors). The printed bond
> table applies the rule to the final geometry, so it lists a contact only
> if the fit ends it within bonding distance.

**Parallel multistart + ORCA:** many licenses allow only one ORCA job at a time. `run_multistart` therefore defaults to **`max_workers=1`** when `quantum_backend="orca"`. If your license allows multiple processes, set **`QUANTIZE_ALLOW_PARALLEL_ORCA=1`** before running to use the preset worker count.

**Parallel workers and in-process backends:** with more than one worker, bootstrap refits and multistart starts run in forked processes. A child forked from a process whose quantum backend has started threads (pyscf or Psi4 with `OMP_NUM_THREADS` above 1) can hang for ever: a water bootstrap hung 3 of 3 times at 50 samples with four threads. One worker (the default) runs the refits in-process and cannot hang; with several, set `OMP_NUM_THREADS=1` or use ORCA, which runs as a separate program.

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
