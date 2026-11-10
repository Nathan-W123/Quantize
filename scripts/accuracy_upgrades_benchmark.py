"""Do the five accuracy upgrades actually make the numbers better?

Four things were identified as missing from the engine and are implemented
here; this script switches each on and measures, on the same molecules, at the
same level of theory, against the same references as the existing mixed-
estimation benchmark, so the answer is a delta and not an anecdote.

    offsets  bond-class offsets on the quantum prior      backend.spectral.bond_offsets
    elec     electronic (rotational g-tensor) correction  backend.spectral.electronic_g
    lam      large-amplitude modes carried as sigma       harmonic_alpha.lam_freq_cm
    corr     VPT2 corrections at a level of their own     hessian_fn at another level

Each configuration reports all three legs -- theory alone, mixed estimation,
hybrid -- because two of these upgrades (offsets, corr) change the inputs every
method sees, so they should move the baselines too. An upgrade that improved
only the hybrid would be suspicious.

Leave-one-out
-------------
The offsets are measured on the benchmark molecules, so scoring a molecule with
an offset table that saw it is circular. Every molecule is corrected using
offsets pooled from the *other* molecules only, and a bond class with no other
member is left uncorrected. That makes the reported gain a lower bound on what
a table calibrated on an external set would give, and never an artefact.

    python scripts/accuracy_upgrades_benchmark.py [method=hf] [basis=6-31g]
                                                  [configs=base,offsets,all]
                                                  [molecule_key ...]

Results
-------
All-bond RMS in mA against the published references, each leg at its own best
configuration. Run ``--table`` to regenerate this from the cached JSON rather
than trusting the transcription; an earlier hand-assembled version of this
table had drifted from its own data, and its MEAN row did not reproduce from
the rows above it.

Scoring each leg independently matters. Picking one row per molecule -- the
configuration where the hybrid happens to do best -- lets the hybrid choose the
settings the other two are then scored under, and that alone moves the
mixed-estimation mean at RHF from 12.41 to 15.30 mA. The margins below are the
ones that survive giving every method its best case.

                              RHF/6-31G             B3LYP/6-31G(d)
                         theory    ME  hybrid    theory    ME  hybrid
  vinyl fluoride           5.72  7.80    7.21      4.17  7.18    6.84
  acetyl fluoride          6.57 12.07    8.55      6.87 11.18    8.75
  fluoroethane            10.04  8.77    7.40      6.61  7.88    8.75
  formyl fluoride          9.38  6.06    5.24      3.84  2.52    3.68
  fluoroacetylene          5.02  4.59    4.37      3.41  3.46    2.27
  chlorofluoromethane     45.42 27.26   13.15     25.14 16.70   10.18
  water                    8.14  1.38    1.36     10.89  4.20    4.13
  ozone                   21.03  2.28    2.32      7.57  1.03    1.00
  isocyanic acid          26.28 40.86   18.73      4.50  4.32    3.74
  MEAN                    15.29 12.34    7.59      8.11  6.50    5.48

The hybrid is best of the three on 6/9 at RHF and 5/9 at B3LYP, and beats
mixed estimation on 8/9 and 7/9. It improves with the level of theory --
7.59 to 5.48 mA overall, and on 6 of 9 molecules individually, by a factor of
5 on isocyanic acid -- so it is not merely rescuing a bad prior, which three
molecules into the B3LYP run looked like the conclusion.

The RHF hybrid mean is 7.59 rather than 7.76 because of ``defect``, the last
upgrade that improved anything without costing a Hessian. Against the
offset-corrected prior it moved vinyl fluoride 7.95 -> 7.21 and fluoroethane
8.30 -> 7.40, left acetyl fluoride at 8.55 -> 8.59, and did not touch the other
six molecules at all -- which is the property that matters, because the earlier
absolute-floor version of the same probe bought a similar gain by damaging the
molecules whose corrections work.

Water's row moved (1.57 -> 1.36 and 4.34 -> 4.13) when its reference structure
was corrected to the r_e it cites; see monofluoro_references.WATER and
dev/tests/test_water_reference_verification.py.

Where theory alone wins (vinyl fluoride, acetyl fluoride, fluoroethane) the
margin is 2-3 mA, and those are the molecules whose corrected targets the
inertial-defect probe independently flags as biased (see
rovib_corrections._apply_defect_bias_floor). The hybrid's wins elsewhere run
to 8-15 mA.

Water is the one genuine regression with level, 1.36 to 4.13. sigma_x comes
from a per-level table, 7 mA here, while B3LYP/6-31G(d) is atypically poor for
O-H at 10.89 mA -- so the fit was instructed to trust a bad prior. A
per-molecule sigma_x would catch it; prior_sigma_for_molecule already computes
one from the offset residuals where the classes are calibrated.

Ozone reaches 1.00 mA, but its reference is an r_s structure with a Costain
floor of 4.1 mA, so that figure is at the resolution of the yardstick rather
than of the engine. Sub-mA cannot be demonstrated against r_s references at
all; that needs r_e or r_e^SE structures, and it is the binding constraint
here rather than compute.

Centrifugal distortion as fit data: a bit-exact no-op, for want of data
-----------------------------------------------------------------------
The ``cd`` and ``offsets+cd`` configurations switch distortion constants on as
fit rows. Measured against ``base`` and ``offsets`` at RHF/6-31G, all nine
molecules, at one code vintage so the comparison is clean:

                         base     cd        offsets  offsets+cd
  vinyl fluoride        18.33  18.33           6.73        6.73
  acetyl fluoride       14.11  14.11           7.46        7.46
  fluoroethane          10.11  10.11           7.22        7.22
  formyl fluoride        7.16   7.16           7.33        7.33
  fluoroacetylene       13.59  13.59           4.42        4.42
  chlorofluoromethane   13.39  13.39          20.15       20.15
  water                  0.56   0.56           0.65        0.65
  ozone                  1.70   1.70           1.71        1.71
  isocyanic acid        23.98  23.98          21.62       21.62
  MEAN                  11.44  11.44           8.59        8.59

Largest difference anywhere: 8e-9 mA. That is solver noise, not a small
effect. The reason is that ``cd_residuals_mhz`` skips any isotopologue without
a ``cd_observed`` block, and no reference molecule in this repository carries
observed distortion constants -- so there are zero CD rows and the switch
cannot do anything. It is correctly built and correctly inert.

An earlier reading of this said CD *helped*, mean 12.75 -> 11.44. That was
entirely a code-vintage artefact: the cached ``base`` rows predated the
harmonic-term correction, and their mixed-estimation column -- which cannot
depend on ``fit_cd`` at all -- disagreed with the same-vintage rows by up to
3.4 mA. ME agreeing exactly between ``base`` and ``cd`` is the check that says
the comparison above is at one vintage; it is worth re-running whenever a
cached row is compared against a fresh one.

So the gate on this feature is data, not code: published DJ/DJK/DK for the
reference molecules, with the reduction and representation they were fitted
in. Until then ``fit_cd_constants=True`` prints a warning saying it will do
nothing (see MolecularOptimizer and
dev/tests/test_centrifugal_distortion.py). Expect a small gain when the data
arrives, in any case: distortion constants are determined mostly by the force
field at the geometry rather than by the geometry, which is why they are
normally used to validate a force field rather than to locate atoms.
"""

from __future__ import annotations

import atexit
import contextlib
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / ".github", _ROOT):
    sys.path.insert(0, str(_p))

import dev.pyscf_backend  # noqa: F401,E402  (registers "pyscf_hf")
from backend.quantize import MolecularOptimizer  # noqa: E402
from backend.registry import get_backend  # noqa: E402
from backend.spectral.bond_offsets import (  # noqa: E402
    leave_one_out_offsets,
    measure_offsets,
    offset_corrected_geometry,
    prior_sigma_for_molecule,
)
from backend.spectral.correction_models import (  # noqa: E402
    electronic_delta_b,
    get_builtin_bob_params,
)
from backend.spectral.electronic_g import (  # noqa: E402
    g_tensors_for_isotopologues,
    rotational_g_tensor,
)
from backend.spectral.hessian_cache import HessianCache  # noqa: E402
from backend.spectral.harmonic_alpha import (  # noqa: E402
    build_correction_table_from_hessian,
    self_consistent_correction_table,
)
from dev.monofluoro_references import (  # noqa: E402
    ISOCYANIC_ACID,
    MOLECULES,
    MOLECULES_SET2,
    OZONE,
    WATER_SET,
)
from scripts.mixed_estimation_baseline import (  # noqa: E402
    _SIGMA_X_BY_LEVEL,
    corrected_targets,
    mixed_estimation_fit,
    rms_angle_error,
    rms_bond_error,
)
from scripts.monofluoro_benchmark import build_isotopologues, start_geometry  # noqa: E402

METHOD, BASIS = "hf", "6-31g"
HARMONIC_SCHEME = "watson"
CORR_METHOD, CORR_BASIS = None, None
#: Distance gate for expansion-point tracking, in Angstrom. Raised on the
#: command line to measure where each scheme actually stops working, which is
#: the only way the default can be set from evidence rather than guessed.
TRACK_MAX = None
#: Weight on the distortion rows relative to the rotational constants. 1.0
#: trusts them as stated; the sweep below is what says whether that is right.
CD_WEIGHT = 1.0
#: Stamped into every result row so vintages can never be compared by accident.
#: Filled in main() once git has been consulted.
VINTAGE = "unknown"
WANT_CONFIGS: list[str] = []
HESSIAN_CACHE = True
_args: list[str] = []
for _tok in sys.argv[1:]:
    if _tok.startswith("method="):
        METHOD = _tok.split("=", 1)[1]
    elif _tok.startswith("basis="):
        BASIS = _tok.split("=", 1)[1]
    elif _tok.startswith("corr_method="):
        CORR_METHOD = _tok.split("=", 1)[1]
    elif _tok.startswith("corr_basis="):
        CORR_BASIS = _tok.split("=", 1)[1]
    elif _tok.startswith("cd_weight="):
        CD_WEIGHT = float(_tok.split("=", 1)[1])
    elif _tok.startswith("track_max="):
        TRACK_MAX = float(_tok.split("=", 1)[1])
    elif _tok.startswith("harmonic="):
        HARMONIC_SCHEME = _tok.split("=", 1)[1]
    elif _tok == "no_hessian_cache":
        HESSIAN_CACHE = False
    elif _tok.startswith("configs="):
        WANT_CONFIGS = _tok.split("=", 1)[1].split(",")
    else:
        _args.append(_tok)

SIGMA_X_ANG = _SIGMA_X_BY_LEVEL.get((METHOD.lower(), BASIS.lower()), 0.020)

#: Frequency below which a mode is treated as large-amplitude. Methyl torsions
#: in this set land at 200-300 cm-1; the next modes up are ordinary bends well
#: above 400, so the threshold is not sitting on top of a cluster.
LAM_FREQ_CM = 350.0

#: Grid points for a relaxed torsional scan in the "torsion" configuration.
#: 12 over a full turn resolves harmonics up to order 6, so for a three-fold
#: rotor it carries V3 and V6 and nothing it cannot distinguish; see
#: torsion_scan_qm on why asking for more aliases rather than refuses.
TORSION_SCAN_POINTS = 12

#: Where Hessians are cached between runs, and whether to. On by default: a
#: Hessian is a deterministic function of (level, elements, geometry), so a hit
#: is the same matrix the run would have computed, and the only cost of the
#: cache is disk. Pass `no_hessian_cache` to turn it off for a clean timing.
#: HESSIAN_CACHE itself is declared above the argument loop and set by it --
#: assigning it here too would clobber the flag, since this block runs after.
HESSIAN_CACHE_DIR = _ROOT / "output" / "hessian_cache"

#: Harmonic frequency scale factor. 1.0 disables it. Left off by default:
#: measured against water's experimental harmonic frequencies, RHF/6-31G is
#: 11% stiff on the bend (1829.4 against 1649) but only 1.5% on the stretches
#: (3903.5 against 3832), so a single factor cannot fit both and tuning one on
#: this set would be fitting the answer. The hook is here because alpha is very
#: sensitive to it -- with the corrected harmonic term water's alpha_B moves
#: from -5366 to -1331 MHz between scales of 1.00 and 0.89, a factor of four
#: from an 11% frequency change -- so it belongs to any serious attempt at the
#: correction bias, with per-mode factors from an external frequency set rather
#: than a scalar guessed here.
FREQ_SCALE = 1.0

#: Fractional uncertainty on a computed g value. The OCS cross-check puts
#: RHF/6-31G at -0.0345 against an experimental -0.028 and RHF/cc-pVTZ at
#: -0.0272, so a small basis is good to roughly 25% on g. The electronic
#: correction is linear in g, so that carries straight through.
SIGMA_G_FRACTION = 0.30

#: Which upgrades each configuration enables.
CONFIGS = {
    "base":    {"offsets": False, "elec": False, "lam": False, "corr": False},
    "offsets": {"offsets": True,  "elec": False, "lam": False, "corr": False},
    "elec":    {"offsets": False, "elec": True,  "lam": False, "corr": False},
    "lam":     {"offsets": False, "elec": False, "lam": True,  "corr": False},
    "corr":    {"offsets": False, "elec": False, "lam": False, "corr": True},
    "all":     {"offsets": True,  "elec": True,  "lam": True,  "corr": False},
    "all+corr": {"offsets": True, "elec": True,  "lam": True,  "corr": True},
    # Measured best pairing: offsets carry the gain, while elec and lam each
    # cost more than they return on this set (ozone 2.32 -> 3.23 to the RHF
    # g-tensor, isocyanic acid 18.73 -> 29.12 to the large-amplitude widening).
    #
    # The corr half of this was measured and does nothing. Corrections at
    # B3LYP/6-31G(d) over an RHF/6-31G geometry, on the four molecules the
    # diagnosis pointed at, moved the hybrid +0.43 mA and mixed estimation
    # -0.33 mA -- both inside the noise, against a predicted ~20% gain from
    # RHF's 11%-high bend frequency and the 1/omega^2 in both tau and alpha.
    #
    # The decisive one is isocyanic acid, where mixed estimation with honest
    # sigmas sits at 63.97 mA against pure theory's 26.28: the corrected
    # targets there are badly biased, and B3LYP corrections move that to
    # 63.64. So the bias is not in the force field. What is left is the data
    # itself -- two isotopologues, measured in 1950 -- and the r_s reference
    # the fit is scored against while the engine targets r_e.
    "offsets+corr": {"offsets": True, "elec": False, "lam": False, "corr": True},
    # Born-Oppenheimer breakdown, with the module's built-in u-parameters.
    # Small in absolute terms -- about -13.6/-7.1/-4.6 MHz on water's A/B/C
    # against correction sigmas of 950-1500 -- but isotope-dependent, and
    # isotopic differences are where the structural information is, so the
    # effect on geometry need not track the sigma ratio.
    #
    # MEASURED, AND IT CHANGES NOTHING HERE. Across all nine molecules and both
    # configurations, 18 rows at B3LYP/6-31G(d): the hybrid moved by 0.000 mA
    # on every one, and mixed estimation by at most 0.29 (one fluoroethane row,
    # mean -0.02). Water is the decisive case -- two thirds hydrogen, which
    # carries the largest built-in u-parameter at 0.015 against oxygen's 0.002
    # -- and it moved 0.00. If BOB does not move water it does not move
    # anything in this set.
    #
    # So the repair in bob_delta_b was worth making, since a correction that
    # silently returns zero is a liability whether or not it currently bites,
    # but it is not a lever on accuracy at this level. It becomes one only when
    # correction sigmas fall to single-digit MHz, which is the sub-mA regime
    # the module's own warning refers to.
    "bob": {"offsets": False, "elec": False, "lam": False, "corr": False,
            "bob": True},
    "offsets+bob": {"offsets": True, "elec": False, "lam": False,
                    "corr": False, "bob": True},
    # Widen sigma by how badly each species' correction failed the inertial-
    # defect test. See rovib_corrections.apply_defect_bias_scale: the probe's
    # *ratio* is what carries the signal, and the earlier attempt to use its
    # absolute residual as a sigma floor damaged the molecules whose
    # corrections work.
    "defect": {"offsets": False, "elec": False, "lam": False, "corr": False,
               "defect_scale": True},
    "offsets+defect": {"offsets": True, "elec": False, "lam": False,
                       "corr": False, "defect_scale": True},
    # The correction's expansion point follows where the fit lands.
    #
    # alpha is expanded at the quantum minimum and then frozen, but the joint
    # objective exists to move the geometry off that minimum. Measured on
    # water at B3LYP/cc-pVTZ the optimiser travels -2.97 mA, and alpha is not
    # flat over that: d(corr)/dr is +273.5 MHz/mA on A, so the frozen
    # correction is applied ~787 MHz from where it belongs, against a
    # correction error of 2825. Two extra alpha evaluations buy it back.
    #
    # Mixed estimation deliberately does NOT get this. It has no geometry
    # variable to track -- the correction is applied to the data before any
    # fitting -- so handing it a tracked table would be scoring it on
    # something it cannot do.
    "track": {"offsets": False, "elec": False, "lam": False, "corr": False,
              "track": True},
    "offsets+track": {"offsets": True, "elec": False, "lam": False,
                      "corr": False, "track": True},
    # The three ways of carrying alpha to the fitted geometry, so they can be
    # compared on the same molecules rather than argued about. "track" is
    # "track_linear" under its original name and is kept so the measurements
    # already on file stay addressable.
    "track_direct": {"offsets": False, "elec": False, "lam": False,
                     "corr": False, "track": True, "scheme": "direct"},
    "track_quadratic": {"offsets": False, "elec": False, "lam": False,
                        "corr": False, "track": True, "scheme": "quadratic"},
    # Centrifugal distortion as fit data, the switch that has been built and
    # off since it was written. The constants come from the Hessian the
    # corrections already need, so the only cost is their own model error.
    "cd": {"offsets": False, "elec": False, "lam": False, "corr": False,
           "fit_cd": True},
    "offsets+cd": {"offsets": True, "elec": False, "lam": False, "corr": False,
                   "fit_cd": True},
    # The torsional mode treated explicitly instead of perturbatively.
    #
    # Compare against "lam", not against "base": both act on exactly the same
    # modes -- everything below LAM_FREQ_CM, since the replacement is handed
    # the same freq_cm -- and they are the two answers to the same problem. A
    # methyl torsion sits at 100-250 cm^-1 with a zero-point amplitude of tens
    # of degrees, and alpha goes as 1/omega, so it is both one of the largest
    # terms in the sum and the one VPT2 is least entitled to. "lam" keeps the
    # perturbative value and widens sigma to its full size, an honest refusal
    # that leaves the suspect number in the value. "torsion" computes the
    # contribution from a relaxed scan and puts that in instead.
    #
    # Only acetyl fluoride and fluoroethane have rotors; find_rotors reports
    # none for the other seven, and those molecules fall back to the
    # untreated table so the row is still comparable.
    "torsion": {"offsets": False, "elec": False, "lam": False, "corr": False,
                "torsion": True},
    "offsets+torsion": {"offsets": True, "elec": False, "lam": False,
                        "corr": False, "torsion": True},
}
for _cfg in CONFIGS.values():
    _cfg.setdefault("bob", False)
    _cfg.setdefault("defect_scale", False)
    _cfg.setdefault("track", False)
    _cfg.setdefault("scheme", "linear")
    _cfg.setdefault("fit_cd", False)
    _cfg.setdefault("torsion", False)


def code_vintage() -> str:
    """The commit this result was produced by, or "dirty"/"unknown".

    WHY THIS EXISTS. Results accumulate in one JSON across many runs, and a
    configuration already present is skipped rather than recomputed. That makes
    the file a mix of vintages, and comparing a cached row against a fresh one
    silently compares two different programs.

    It has already produced a wrong published conclusion. Switching
    centrifugal-distortion constants on as fit data appeared to improve the
    nine-molecule mean from 12.75 to 11.44 mA. It does nothing at all -- the
    switch is a bit-exact no-op for want of observed constants -- and the
    entire apparent gain was that the cached ``base`` rows predated a
    correction to the harmonic alpha term. The tell was that those rows'
    mixed-estimation column, which cannot depend on the switch under test,
    disagreed with the fresh rows by up to 3.4 mA.

    A dirty tree returns "<sha>-dirty": an uncommitted edit means the row
    cannot be reproduced from any commit, and saying so is more useful than
    attributing it to the last one.
    """
    import subprocess
    try:
        sha = subprocess.run(
            ["git", "-C", str(_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def warn_on_mixed_vintages(out: dict, current: str) -> None:
    """Say loudly when a results file mixes code vintages.

    Not an error: re-running nine molecules to refresh one row is often not
    what the user wants, and an older row is still a valid record of what that
    code did. What is never acceptable is comparing across vintages without
    knowing it, so the warning names the rows and what to do about them.
    """
    stale: dict[str, list[str]] = {}
    for key, rec in out.items():
        for cfg, entry in rec.items():
            seen = entry.get("code_vintage") if isinstance(entry, dict) else None
            if seen != current:
                stale.setdefault(seen or "unstamped", []).append(f"{key}:{cfg}")
    if not stale:
        return
    print(f"\n  [vintage] this run is {current}; the results file also holds:")
    for vintage, rows in sorted(stale.items()):
        shown = ", ".join(rows[:4]) + (" ..." if len(rows) > 4 else "")
        print(f"            {vintage:<18} {len(rows):>3} rows   {shown}")
    print("            DO NOT compare across these. Delete the stale rows and")
    print("            re-run them, or compare only within one vintage. A"
          " cached row")
    print("            from older code has already produced one wrong"
          " conclusion here;")
    print("            see code_vintage() for which.\n")


def electronic_shifted_isotopologues(isos, g_tensor, total_mass_amu):
    """Isotopologues with the electronic correction folded into the constants.

    Both methods are handed the same numbers this way, rather than one of them
    applying the correction internally and the other not. The correction is
    ``delta = -(m_e/m_p)*g*B0``, added to B0 exactly as ``resolve_corrections``
    would, and its own uncertainty goes into sigma in quadrature.
    """
    out = []
    for iso in isos:
        new = dict(iso)
        obs = np.asarray(iso["obs_constants"], dtype=float).copy()
        sig = np.asarray(iso["sigma_constants"], dtype=float).copy()
        # g_tensor is species-keyed: the tensor is isotope-dependent, so the
        # parent's g is the wrong number for every substituted species.
        g_here = g_tensor.get(str(iso.get("name", "iso")), {})
        for k, comp in enumerate(np.asarray(iso["component_indices"], dtype=int)):
            g_val = g_here.get("ABC"[int(comp)])
            if g_val is None or not np.isfinite(g_val):
                continue
            delta = electronic_delta_b(float(obs[k]), total_mass_amu,
                                       g_value=float(g_val))
            obs[k] = float(obs[k]) + delta
            sig[k] = float(np.hypot(sig[k], abs(delta) * SIGMA_G_FRACTION))
        new["obs_constants"] = obs
        new["sigma_constants"] = sig
        out.append(new)
    return out


def hybrid_fit(mol, isos, prior_coords, ctbl, sigma_x_ang,
               defect_scale=False, fit_cd=False, cd_weight=1.0):
    """The engine's answer, with the prior centred and widened as handed.

    ``prior_target_coords`` is what makes the prior's centre follow
    ``prior_coords``. Without it the quantum term is the level of theory's own
    gradient, so the prior's minimum stays at the level of theory's minimum
    however the starting structure is chosen, and a bias-corrected prior is
    discarded on the first step -- measured on vinyl fluoride as 18.01 mA
    against 7.95 with the centre moved.

    Passing it unconditionally is deliberate. Where the prior *is* the theory
    geometry the two agree anyway, since g is zero at that minimum and
    H (x - x_theory) is its harmonic expansion, so the base configuration is
    not quietly running a different prior from the others.
    """
    opt = MolecularOptimizer(
        elems=list(mol.elems), coords=np.asarray(prior_coords, dtype=float),
        isotopologues=isos, quantum_backend="pyscf_hf",
        orca_method=METHOD, orca_basis=BASIS, coordinate_mode="cartesian",
        use_autoconfig=False, max_iter=40, hess_recalc_every=10,
        correction_table=ctbl, quantum_prior_sigma_ang=float(sigma_x_ang),
        prior_target_coords=np.asarray(prior_coords, dtype=float),
        defect_bias_scale=bool(defect_scale),
        # Distortion constants as fit data. They are computed from the same
        # Hessian the corrections come from, so turning them on costs nothing
        # extra -- the question is only whether the extra rows help or whether
        # their own model error outweighs the information they carry.
        harmonic_cd_from_hessian=bool(fit_cd),
        fit_cd_constants=bool(fit_cd),
        cd_weight=float(cd_weight) if fit_cd else 0.0,
        chi2_rescale=True, chi2_rescale_max_passes=3)
    with contextlib.redirect_stdout(io.StringIO()):
        return opt.run()


def torsion_replacement_for(mol, coords_ang, backend, freq_cm=LAM_FREQ_CM,
                            n_points=TORSION_SCAN_POINTS):
    """A computed torsional contribution for ``mol``, or None if it has no rotor.

    The rotor is found from connectivity rather than hand-entered per
    molecule: four transposed atom indices give a well-formed scan of the
    wrong coordinate with nothing in the output to say so. ``freq_cm`` is
    LAM_FREQ_CM, so the replaced modes are exactly the ones the "lam"
    configuration widens, which is what makes the two rows comparable.

    The scan runs at the level of whichever backend is handed in, at the
    geometry the correction is being expanded about, because the potential a
    torsional average needs is that level's own.
    """
    from backend.spectral.centrifugal_distortion import normal_modes
    from backend.torsion.internal_rotation import (
        find_rotors,
        torsional_mode_overlaps,
    )
    from backend.torsion.torsion_lam_integration import torsion_replacement_from_scan
    from backend.torsion.torsion_scan_qm import relaxed_torsion_scan, spec_from_scan

    coords = np.asarray(coords_ang, dtype=float)
    rotors = find_rotors(coords, list(mol.elems))
    if not rotors:
        return None, {"reason": "no internal rotor found"}

    rotor = rotors[0]
    masses = np.asarray(mol.masses, dtype=float)

    # WHICH MODE IS REPLACED, decided by projection and not by frequency.
    # Handing the replacement a freq_cm cutoff makes it cover every mode below
    # that, which is safe only while the torsion is the only one down there.
    # It is, for both rotor molecules at HF/6-31G -- 149.0 and 260.3 cm^-1 with
    # nothing else under 350 -- but a low skeletal bend in some other molecule
    # would have had its alpha removed and never added back. Projecting the
    # normal modes onto a rigid rotation of the top names the mode instead:
    # overlap 0.997 for both, with 0.999 of the torsional motion inside the
    # vibrational space.
    with contextlib.redirect_stdout(io.StringIO()):
        omega_cm, L_mw = normal_modes(backend.run_hessian(coords).hessian_bohr,
                                      masses)
    overlaps, ov_info = torsional_mode_overlaps(
        coords, masses, L_mw, rotor["frame_atom"], rotor["top_atom"],
        rotor["top_indices"])
    if ov_info["best_overlap"] < 0.7:
        return None, {"reason": "no normal mode is clearly the torsion",
                      "best_overlap": ov_info["best_overlap"],
                      "best_mode_cm": float(omega_cm[ov_info["best"]])}
    with contextlib.redirect_stdout(io.StringIO()):
        scan, scan_info = relaxed_torsion_scan(
            backend._energy_gradient, coords, list(mol.elems),
            rotor["dihedral"], n_points=int(n_points),
            top_indices=rotor["top_indices"])
        spec, spec_info = spec_from_scan(
            coords, masses, list(mol.elems), scan,
            frame_atom=rotor["frame_atom"], top_atom=rotor["top_atom"],
            symmetry_number=rotor["symmetry_number"],
            top_indices=rotor["top_indices"])
        replacement, rep_info = torsion_replacement_from_scan(
            list(mol.elems), scan, spec, coords, masses,
            symmetry_number=rotor["symmetry_number"],
            mode_indices=(int(ov_info["best"]),), freq_cm=float(freq_cm))

    # A scan whose constraint slipped is a potential at unknown angles, which
    # would be worse than no treatment because it would look like one.
    if not scan_info["dihedral_within_tol"]:
        return None, {"reason": "scan constraint slipped",
                      "max_dihedral_error_deg": scan_info["max_dihedral_error_deg"]}
    return replacement, {
        "rotor": {k: v for k, v in rotor.items() if k != "top_indices"},
        "top_indices": list(rotor["top_indices"]),
        "barrier_cm": scan_info["barrier_cm"],
        "F_cm": float(spec.F), "rho": float(spec.rho),
        "all_converged": bool(scan_info["all_converged"]),
        "max_dihedral_error_deg": float(scan_info["max_dihedral_error_deg"]),
        "n_harmonics_used": spec_info["n_harmonics_used"],
        "mode_index": int(ov_info["best"]),
        "mode_freq_cm": float(omega_cm[ov_info["best"]]),
        "mode_overlap": float(ov_info["best_overlap"]),
        "torsional_motion_captured": float(ov_info["captured"]),
        "modes_below_freq_cm": int(sum(1 for w in omega_cm if w < float(freq_cm))),
        "alpha_mhz": {k: float(v) for k, v in replacement.alpha_mhz.items()},
        "sigma_mhz": {k: float(v) for k, v in replacement.sigma_mhz.items()},
        "freq_cm": float(freq_cm),
    }


def theory_geometries(pool):
    """Theory geometry per molecule, plus each one's bond-class offsets.

    Cached to disk: it is the same optimisation for every configuration, and
    the leave-one-out table needs all of them before any molecule can be run.
    """
    tag = f"{METHOD}_{BASIS}".replace("/", "-").replace("(", "").replace(")", "")
    cache_path = _ROOT / "output" / f"theory_geometries_{tag}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) \
        if cache_path.exists() else {}
    for mol in pool:
        if mol.key in cache:
            continue
        backend = get_backend("pyscf_hf")(elems=list(mol.elems),
                                          method=METHOD, basis=BASIS)
        with contextlib.redirect_stdout(io.StringIO()):
            coords = backend.optimise(start_geometry(mol))
        cache[mol.key] = {
            "coords": np.asarray(coords, dtype=float).tolist(),
            "offsets": {k: list(map(float, v))
                        for k, v in measure_offsets(mol, coords).items()},
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        print(f"  [theory] {mol.key}", flush=True)
    return cache


def main() -> None:
    global VINTAGE
    VINTAGE = code_vintage()
    pool = (list(MOLECULES) + list(MOLECULES_SET2)
            + [WATER_SET[0], OZONE, ISOCYANIC_ACID])
    wanted = _args or [m.key for m in pool]
    configs = [c for c in (WANT_CONFIGS or ["base", "offsets", "elec", "lam", "all"])
               if c in CONFIGS]

    tag = f"{METHOD}_{BASIS}".replace("/", "-").replace("(", "").replace(")", "")
    out_path = _ROOT / "output" / f"accuracy_upgrades_{tag}.json"

    # ONE RUN PER RESULTS FILE. Every run loads the whole file at startup, adds
    # rows and writes the whole thing back after each molecule. Two runs on the
    # same tag therefore do not interleave -- the second to write silently
    # discards whatever the first added since it loaded. The tag depends only on
    # method and basis, so two runs that differ in `configs` (which is exactly
    # how one would try to add a configuration to an existing set) collide.
    #
    # Nearly done by hand while an MP2 run was in flight, which is the only
    # reason this guard exists. The lock carries the pid so a stale one left by
    # a killed run can be told from a live one and cleared.
    lock_path = out_path.with_suffix(".lock")
    if lock_path.exists():
        holder = lock_path.read_text(encoding="utf-8").strip()
        pid = holder.split()[0] if holder else ""
        alive = False
        if pid.isdigit():
            try:
                os.kill(int(pid), 0)
                alive = True
            except (OSError, ProcessLookupError):
                alive = False
        if alive:
            raise SystemExit(
                f"another run already holds {out_path.name} ({holder}).\n"
                f"Results files are rewritten whole, so a second run would "
                f"discard the first one's rows. Wait for it, or use a "
                f"different method=/basis= tag.\n"
                f"If that process is gone, delete {lock_path}."
            )
        print(f"  [lock] clearing stale lock from {holder}")
        lock_path.unlink()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(f"{os.getpid()} {' '.join(sys.argv[1:])}",
                         encoding="utf-8")
    atexit.register(lambda: lock_path.unlink(missing_ok=True))

    out = json.loads(out_path.read_text(encoding="utf-8")) \
        if out_path.exists() else {}

    print(f"  {METHOD.upper()}/{BASIS}, sigma_x = {SIGMA_X_ANG} A")
    print(f"  configs: {', '.join(configs)}")
    print(f"  code vintage: {VINTAGE}")
    warn_on_mixed_vintages(out, VINTAGE)
    if CORR_METHOD or CORR_BASIS:
        print(f"  corrections level: {CORR_METHOD or METHOD}/{CORR_BASIS or BASIS}")
    print()

    theory = theory_geometries(pool)
    per_molecule_offsets = {k: {c: [x / 1.0 for x in v]
                                for c, v in rec["offsets"].items()}
                            for k, rec in theory.items()}

    for mol in pool:
        if mol.key not in wanted:
            continue
        rec = out.setdefault(mol.key, {})
        todo = [c for c in configs if c not in rec]
        if not todo:
            continue
        t0 = time.time()

        theory_coords = np.asarray(theory[mol.key]["coords"], dtype=float)
        backend = get_backend("pyscf_hf")(elems=list(mol.elems),
                                          method=METHOD, basis=BASIS)
        # Two cache layers. The dict is the per-molecule one this script has
        # always had; the disk cache behind it survives the process, which is
        # what makes an interrupted run resumable. See hessian_cache on why a
        # Hessian is keyed by level, elements and geometry but NOT by mass --
        # one cached matrix serves every isotopologue, which is most of the
        # saving.
        hess_cache: dict = {}
        hess_disk = HessianCache(HESSIAN_CACHE_DIR, method=METHOD, basis=BASIS,
                                 elements=list(mol.elems),
                                 enabled=HESSIAN_CACHE)
        _hess_disk_fn = hess_disk.wrap(
            lambda c, _b=backend: _b.run_hessian(c).hessian_bohr)

        def hessian_fn(coords_ang, _c=hess_cache, _f=_hess_disk_fn):
            key = np.asarray(coords_ang, dtype=float).round(9).tobytes()
            if key not in _c:
                _c[key] = _f(coords_ang)
            return _c[key]

        # The correction level, when it is allowed to differ from the geometry
        # level. Corrections converge with basis and correlation treatment far
        # faster than geometries do, so the usual practice is to compute alpha
        # somewhere cheaper than the structure -- here the structure is the
        # cheap thing and the correction is the one worth improving.
        corr_backend = None
        corr_cache: dict = {}
        corr_geom = None
        if CORR_METHOD or CORR_BASIS:
            corr_backend = get_backend("pyscf_hf")(
                elems=list(mol.elems), method=CORR_METHOD or METHOD,
                basis=CORR_BASIS or BASIS)
            # VPT2 expands about a stationary point. A force field from one
            # level evaluated at another level's geometry is not at one: there
            # is a residual gradient, the normal-mode analysis is contaminated
            # by it, and alpha is computed from frequencies that are not the
            # force field's own. An earlier version of this script did exactly
            # that -- B3LYP/6-31G(d) Hessians at the RHF/6-31G geometry -- and
            # reported that the correction level does not matter, which was an
            # artefact of the protocol rather than a result.
            #
            # So the correction level gets its own minimum. alpha is a property
            # of a force field and the masses, not of whatever geometry the fit
            # is being started from, and it is applied to the observed B_0
            # either way.
            with contextlib.redirect_stdout(io.StringIO()):
                corr_geom = corr_backend.optimise(start_geometry(mol))

        # The expensive one. A correlated level has no analytic Hessian, so
        # each costs 6N gradients and the normal_mode cubic scheme needs
        # 2*n_vib of them -- about 288 MP2/cc-pVTZ gradients, an hour, for
        # formyl fluoride. Twice in one session that hour was lost to an
        # interruption because the only cache was in memory.
        corr_disk = HessianCache(HESSIAN_CACHE_DIR,
                                 method=CORR_METHOD or METHOD,
                                 basis=CORR_BASIS or BASIS,
                                 elements=list(mol.elems),
                                 enabled=HESSIAN_CACHE)
        _corr_disk_fn = corr_disk.wrap(
            lambda c, _b=corr_backend: _b.run_hessian(c).hessian_bohr)

        def corr_hessian_fn(coords_ang, _c=corr_cache, _f=_corr_disk_fn):
            key = np.asarray(coords_ang, dtype=float).round(9).tobytes()
            if key not in _c:
                _c[key] = _f(coords_ang)
            return _c[key]

        # Offsets, leave-one-out, and the geometry they produce.
        loo = leave_one_out_offsets(per_molecule_offsets, mol.key)
        offset_coords = offset_corrected_geometry(mol, theory_coords, loo)
        # Removing the bias narrows the prior, so sigma_x has to follow it or
        # the optimiser keeps treating a corrected prior as a loose one. Bond
        # by bond, because a class leave-one-out could not calibrate is not
        # narrowed at all -- quoting the corrected spread for it claimed a
        # precision the prior did not have and cost chlorofluoromethane
        # 13.16 -> 19.71 mA. Still leave-one-out, so the width is never
        # informed by the answer.
        sigma_x_offset = prior_sigma_for_molecule(
            mol, per_molecule_offsets, mol.key)

        # A correction table per (geometry, lam) pair actually needed. The
        # Hessian cache is shared, so the second lam variant is nearly free.
        tables: dict = {}

        # Torsional replacements, one per (geometry, level). The scan is the
        # expensive part, so it is computed once and shared by every
        # configuration that asks for it.
        torsions: dict = {}

        def torsion_for(coords, corr, _t=torsions):
            at = np.asarray(corr_geom if corr else coords, dtype=float)
            key = (at.round(8).tobytes(), bool(corr))
            if key not in _t:
                be = corr_backend if corr else backend
                _t[key] = torsion_replacement_for(mol, at, be)
            return _t[key]

        def table_for(coords, lam, corr, torsion=False, _tabs=tables):
            # When the corrections come from another level they are evaluated
            # at that level's own equilibrium, not at `coords`.
            at = np.asarray(corr_geom if corr else coords, dtype=float)
            key = (at.round(8).tobytes(), bool(lam), bool(corr), bool(torsion))
            if key not in _tabs:
                hfn = corr_hessian_fn if corr else hessian_fn
                isos_local = build_isotopologues(mol, None)
                replacement = torsion_for(coords, corr)[0] if torsion else None
                with contextlib.redirect_stdout(io.StringIO()):
                    _tabs[key] = build_correction_table_from_hessian(
                        hfn(at), at, isos_local, hessian_fn=hfn,
                        cubic_scheme="normal_mode",
                        lam_freq_cm=LAM_FREQ_CM if lam else 0.0,
                        freq_scale=FREQ_SCALE,
                        harmonic_scheme=HARMONIC_SCHEME,
                        torsion_replacement=replacement)
            return _tabs[key]

        g_tensor = None
        for name in todo:
            cfg = CONFIGS[name]
            prior = offset_coords if cfg["offsets"] else theory_coords
            sigma_x = sigma_x_offset if cfg["offsets"] else SIGMA_X_ANG
            isos = build_isotopologues(mol, None)

            bob_params = None
            if cfg["bob"]:
                with contextlib.redirect_stdout(io.StringIO()):
                    bob_params = get_builtin_bob_params(list(mol.elems), None)
            if cfg["elec"]:
                if g_tensor is None:
                    with contextlib.redirect_stdout(io.StringIO()):
                        g_tensor = g_tensors_for_isotopologues(
                            list(mol.elems), theory_coords, isos,
                            method=METHOD, basis=BASIS)
                isos = electronic_shifted_isotopologues(
                    isos, g_tensor, float(np.sum(np.asarray(mol.masses))))

            if cfg["corr"] and corr_backend is None:
                raise SystemExit(
                    "configs including 'corr' need corr_method=/corr_basis= "
                    "so the correction level differs from the geometry level")
            ctbl, info = table_for(prior, cfg["lam"], cfg["corr"],
                                   torsion=cfg["torsion"])
            torsion_info = (torsion_for(prior, cfg["corr"])[1]
                            if cfg["torsion"] else None)
            targets = corrected_targets(mol, isos, ctbl,
                                        bob_params=bob_params,
                                        coords_ang=prior,
                                        defect_bias_scale=cfg["defect_scale"])

            # Both methods get the same prior centre and the same width; the
            # comparison is about mechanism, not about who was told to trust
            # the theory more. Mixed estimation is always handed the untracked
            # table: tracking is a property of having a geometry variable in
            # the objective, which it does not have.
            me_geom, _chi2 = mixed_estimation_fit(mol, targets, prior, sigma_x)
            if cfg["track"]:
                at = np.asarray(corr_geom if cfg["corr"] else prior, dtype=float)
                hfn_track = corr_hessian_fn if cfg["corr"] else hessian_fn

                def _fit(table, _p=prior, _s=sigma_x, _c=cfg):
                    return hybrid_fit(mol, isos, _p, table, _s,
                                      defect_scale=_c["defect_scale"],
                                      fit_cd=_c["fit_cd"], cd_weight=CD_WEIGHT)

                with contextlib.redirect_stdout(io.StringIO()):
                    tracked, track_info = self_consistent_correction_table(
                        hfn_track, at, isos, _fit,
                        scheme=cfg["scheme"],
                        **({"max_distance_ang": TRACK_MAX}
                           if TRACK_MAX is not None else {}),
                        cubic_scheme="normal_mode",
                        lam_freq_cm=LAM_FREQ_CM if cfg["lam"] else 0.0,
                        freq_scale=FREQ_SCALE,
                        harmonic_scheme=HARMONIC_SCHEME,
                        torsion_replacement=(torsion_for(prior, cfg["corr"])[0]
                                             if cfg["torsion"] else None))
                hyb = _fit(tracked)
                ctbl = tracked
            else:
                track_info = None
                hyb = hybrid_fit(mol, isos, prior, ctbl, sigma_x,
                                 defect_scale=cfg["defect_scale"],
                                 fit_cd=cfg["fit_cd"], cd_weight=CD_WEIGHT)

            _re_geom = getattr(mol, "equilibrium_geometry", None)
            entry = {
                #: The commit that produced this row. Compare only within one
                #: vintage; see code_vintage().
                "code_vintage": VINTAGE,
                "theory_rms_ma": rms_bond_error(mol, prior)[0],
                "me_rms_ma": rms_bond_error(mol, me_geom)[0],
                "hybrid_rms_ma": rms_bond_error(mol, hyb)[0],
                "theory_rms_deg": rms_angle_error(mol, prior)[0],
                "me_rms_deg": rms_angle_error(mol, me_geom)[0],
                "hybrid_rms_deg": rms_angle_error(mol, hyb)[0],
                # WHAT THE ERROR IS MEASURED AGAINST. Only an r_e reference
                # scores the engine alone: it is the quantity the engine
                # targets. An r_s or effective reference carries a
                # vibrational-averaging offset from r_e, so part of any error
                # against it belongs to the yardstick -- which is why ozone's
                # 1.00 mA sits at the resolution of its reference rather than
                # of the engine. Recorded per row because that caveat was
                # previously only a paragraph in this file's docstring, and a
                # number read out of the JSON carried no trace of it.
                "reference_type": str(getattr(mol, "structure_type", "effective")),
                "reference_is_equilibrium": bool(
                    getattr(mol, "structure_type", "effective") == "r_e"),
                # Scored against the published EQUILIBRIUM structure as well,
                # wherever one exists, because that is the quantity the engine
                # targets. Ozone is the case that shows why it matters: its two
                # references are 1.06 mA apart, so the same fit scores ~1.0 mA
                # against its r_s structure and far less against its r_e one.
                # Null where no r_e structure is published, which is seven of
                # the nine molecules and the honest state of the art here.
                **({} if _re_geom is None else {
                    "hybrid_rms_ma_vs_re": rms_bond_error(mol, hyb, _re_geom)[0],
                    "me_rms_ma_vs_re": rms_bond_error(mol, me_geom, _re_geom)[0],
                    "theory_rms_ma_vs_re": rms_bond_error(mol, prior, _re_geom)[0],
                    "hybrid_rms_deg_vs_re": rms_angle_error(mol, hyb, _re_geom)[0],
                    "re_source": str(getattr(mol, "geometry_re_source", "")
                                     or mol.structure_source),
                }),
                "n_species": len(isos),
                "sigma_x_ang": float(sigma_x),
                # How far the fit travelled off the expansion point, and so
                # how far the correction had to be tracked. Reported because a
                # tracked run that moved nothing and an untracked run are the
                # same run, and without this they are indistinguishable after
                # the fact.
                "track_distances_ang": (
                    [float(d) for d in track_info["distances_ang"]]
                    if track_info else None),
                "track_scheme": track_info["scheme"] if track_info else None,
                "fit_cd": bool(cfg["fit_cd"]),
                # What the torsional treatment actually did, or why it did
                # nothing. A row that silently fell back to the untreated
                # table and a row that was treated must be distinguishable
                # after the fact; cd_observed taught that lesson already.
                "torsion": bool(cfg["torsion"]),
                "torsion_applied": bool(cfg["torsion"] and torsion_info is not None
                                        and "rotor" in torsion_info),
                "torsion_info": torsion_info,
                "cd_weight": float(CD_WEIGHT) if cfg["fit_cd"] else 0.0,
                "track_skipped_beyond_ang": (
                    track_info["skipped_beyond_ang"] if track_info else None),
                "track_unstable_modes": (
                    list(track_info["unstable_modes"])
                    if track_info and track_info["unstable_modes"] else None),
                "corr_level": (f"{CORR_METHOD or METHOD}/{CORR_BASIS or BASIS}"
                               if cfg["corr"] else f"{METHOD}/{BASIS}"),
                "bob": bool(cfg["bob"]),
                "defect_scale": bool(cfg["defect_scale"]),
                "harmonic_scheme": HARMONIC_SCHEME,
                "lam_modes_cm": sorted({round(w, 1) for v in info.get("lam", {}).values()
                                        for w in v.get("modes_cm", [])}),
                "g_tensor": g_tensor if cfg["elec"] else None,
                "g_per_isotopologue": bool(cfg["elec"]),
                "loo_offsets_ma": {k: round(v * 1000.0, 2) for k, v in loo.items()}
                                  if cfg["offsets"] else None,
            }
            rec[name] = entry
            out[mol.key] = rec
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            # The reference type rides on the printed line too. "vs r_e" is
            # the engine's own error; anything else shares the error with the
            # yardstick, and a figure read off a terminal should say which.
            _ref = str(getattr(mol, "structure_type", "effective"))
            _extra = ("" if _re_geom is None or _ref == "r_e"
                      else f"   (vs r_e {entry['hybrid_rms_ma_vs_re']:5.2f})")
            print(f"  {mol.name:<22} {name:<9} theory {entry['theory_rms_ma']:6.2f}"
                  f"   ME {entry['me_rms_ma']:6.2f}"
                  f"   hybrid {entry['hybrid_rms_ma']:6.2f} mA"
                  f"   vs {_ref}{_extra}", flush=True)
        print(f"    ({(time.time() - t0) / 60:.1f} min)", flush=True)

    print(f"\n  written to {out_path}")


#: Molecule order used by the summary table.
_TABLE_ORDER = [
    "vinyl_fluoride", "acetyl_fluoride", "fluoroethane", "formyl_fluoride",
    "fluoroacetylene", "chlorofluoromethane", "water", "ozone",
    "isocyanic_acid",
]

#: Levels the summary table reports, as (label, output-file tag).
_TABLE_LEVELS = [("RHF/6-31G", "hf_6-31g"), ("B3LYP/6-31G(d)", "b3lyp_6-31gd")]


def print_summary_table() -> None:
    """Regenerate the results table in this module's docstring from the JSON.

    Each leg is reported at its own best configuration, independently. Picking
    one row per molecule -- the row where the hybrid does best -- would let the
    hybrid choose the settings the other two are then scored under, which
    flatters it by 2.9 mA on the mixed-estimation mean at RHF.
    """
    legs = ("theory_rms_ma", "me_rms_ma", "hybrid_rms_ma")
    tables = {}
    for label, tag in _TABLE_LEVELS:
        path = _ROOT / "output" / f"accuracy_upgrades_{tag}.json"
        tables[label] = json.loads(path.read_text(encoding="utf-8")) \
            if path.exists() else {}

    head = " " * 23
    for label, _ in _TABLE_LEVELS:
        head += f"{label:^24s}"
    print(head)
    print(" " * 23 + 2 * f"{'theory':>8s}{'ME':>6s}{'hybrid':>8s}  ")

    totals = {lab: [0.0, 0.0, 0.0] for lab, _ in _TABLE_LEVELS}
    counts = {lab: 0 for lab, _ in _TABLE_LEVELS}
    for key in _TABLE_ORDER:
        line = f"  {key.replace('_', ' '):<21s}"
        for label, _ in _TABLE_LEVELS:
            recs = tables[label].get(key)
            if not recs:
                line += f"{'-':>8s}{'-':>6s}{'-':>8s}  "
                continue
            vals = [min(v[leg] for v in recs.values()) for leg in legs]
            for i, v in enumerate(vals):
                totals[label][i] += v
            counts[label] += 1
            line += f"{vals[0]:8.2f}{vals[1]:6.2f}{vals[2]:8.2f}  "
        print(line)
    line = f"  {'MEAN':<21s}"
    for label, _ in _TABLE_LEVELS:
        n = max(counts[label], 1)
        line += "".join(f"{t / n:8.2f}" if i != 1 else f"{t / n:6.2f}"
                        for i, t in enumerate(totals[label])) + "  "
    print(line)


if __name__ == "__main__":
    if "--table" in sys.argv:
        print_summary_table()
    else:
        main()
