"""Every choice the UI offers, read from the code that implements it.

A dropdown whose contents are typed into the HTML drifts the moment someone
adds a scheme or renames a policy, and it drifts silently -- the UI keeps
offering the old name and the backend rejects it at run time. So nothing here
is a literal list of strings: each one is imported from the module that defines
it, and a new scheme shows up in the browser without this file being touched.

Where a choice has no enumeration in the code -- basis sets, point groups --
the list is a convenience rather than a constraint, and the field stays free
text in the form so an unlisted value can still be used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paths import ensure_repo_paths

ensure_repo_paths(Path(__file__).resolve().parent.parent)


def _sorted_strs(values) -> list[str]:
    return sorted(str(v) for v in values)


#: What each backend needs before it can produce a gradient or a Hessian, and
#: how to find out without importing or launching it. Registration is not
#: availability: every backend here imports its dependency lazily, inside the
#: call that needs it, so the name appears in the registry on a machine where
#: the thing cannot run at all.
#:
#: That is not a hypothetical. pyscf publishes wheels for Linux and macOS only
#: -- there is no Windows wheel -- and it is not in requirements.txt either, so
#: "pip install -r requirements.txt" never supplies it. A Windows user picking
#: the default backend got a valid config, a successful validate, a run that
#: printed its correction table and started optimising, and then
#: ModuleNotFoundError from inside the first Hessian. The cost of finding out
#: late is the whole run.
_BACKEND_NEEDS: dict[str, dict[str, Any]] = {
    "pyscf_hf": {"module": "pyscf",
                 "hint": "pip install pyscf (Linux/macOS only -- no Windows "
                         "wheel; use WSL on Windows)"},
    "psi4": {"module": "psi4",
             "hint": "conda install -c conda-forge psi4"},
    "orca": {"executable": "orca",
             "hint": "install ORCA and put it on PATH, or set orca_executable"},
    "none": {},
}


def backend_availability() -> dict[str, dict[str, Any]]:
    """Which registered backends could actually run here, and why not if not.

    Probes without importing: importlib.util.find_spec for a module, which()
    for an executable. Both are cheap and neither pulls a heavy package into
    the server process.
    """
    import importlib.util
    import shutil

    from backend.registry import list_backends

    out: dict[str, dict[str, Any]] = {}
    for name in sorted(set(list_backends()) | {"none"}):
        need = _BACKEND_NEEDS.get(name)
        if need is None:
            # A backend nobody told us how to probe. Saying "unknown" is
            # honest; claiming it works is not.
            out[name] = {"ok": None, "why": "availability not known"}
            continue
        if not need:
            out[name] = {"ok": True, "why": ""}
            continue
        if "module" in need:
            found = importlib.util.find_spec(need["module"]) is not None
            why = "" if found else f"Python package '{need['module']}' is not installed"
        else:
            found = shutil.which(need["executable"]) is not None
            why = "" if found else f"'{need['executable']}' is not on PATH"
        out[name] = {"ok": found, "why": why, "hint": need.get("hint", "")}
    return out


def backend_options() -> dict[str, Any]:
    """Choices that come from the backend's own definitions."""
    from backend.registry import list_backends
    from backend.spectral.harmonic_alpha import (
        _HARMONIC_SCHEMES,
        _MAX_TRACK_DISTANCE_ANG,
        _NONCONVERGENT_POLICIES,
        _SCHEMES,
    )
    from backend.spectral.cd_reduction import REDUCTION_PARAMS
    from backend.spectral.rovib_corrections import _VALID_MODES
    from runner.usability import VALID_PRESETS

    # Importing the pyscf backend is what registers "pyscf_hf"; without it the
    # list is whatever happens to have been imported, which is a confusing
    # thing to show someone choosing a backend.
    try:
        import dev.pyscf_backend  # noqa: F401
    except Exception:  # pragma: no cover - pyscf absent is a valid install
        pass

    avail = backend_availability()
    usable = [n for n in sorted(set(list_backends()) | {"none"})
              if avail.get(n, {}).get("ok")]
    return {
        "quantum_backend": _sorted_strs(set(list_backends()) | {"none"}),
        "backend_availability": avail,
        # What to select on a fresh page: a backend that can actually run,
        # preferring the one the benchmarks use. Offering an unusable default
        # is how a run gets to the first Hessian before failing.
        "backend_default": next(
            (n for n in ("pyscf_hf", "psi4", "orca") if n in usable),
            usable[0] if usable else "none"),
        "preset": _sorted_strs(VALID_PRESETS),
        "corrections_mode": _sorted_strs(_VALID_MODES),
        "harmonic_scheme": list(_HARMONIC_SCHEMES),
        "nonconvergent_policy": _sorted_strs(_NONCONVERGENT_POLICIES),
        "tracking_scheme": list(_SCHEMES),
        "cd_reduction": _sorted_strs(REDUCTION_PARAMS),
        # cartesian first: it is what build_correction_table_from_hessian
        # defaults to, and the UI matching that keeps a browser run and a CLI
        # run of the same case identical unless someone changes it on purpose.
        "cubic_scheme": ["cartesian", "normal_mode"],
        "coordinate_mode": ["internal", "cartesian"],
        # SMILES first: it is the one that needs nothing but the molecule's
        # identity. run_generic checks geometry.smiles before it looks at
        # method at all, and fetches an MMFF94-quality PubChem conformer.
        "geometry_method": ["smiles", "pubchem", "bonds", "coords"],
        "internal_priors_mode": ["soft", "hard", "off"],
        "components": ["A", "B", "C"],
        "track_max_default_ang": float(_MAX_TRACK_DISTANCE_ANG),
    }


#: Basis sets worth offering first. Not a constraint -- the field accepts any
#: string the quantum backend understands, and this list only saves typing.
SUGGESTED_BASES = [
    "sto-3g", "6-31g", "6-31g(d)", "6-311+g(d,p)",
    "cc-pvdz", "cc-pvtz", "cc-pvqz", "aug-cc-pvdz", "aug-cc-pvtz",
    "def2-svp", "def2-tzvp", "def2-tzvpp", "def2-qzvpp",
]

#: Likewise for methods. HF and any DFT functional name the backend knows are
#: valid; MP2 and CCSD(T) reach the ORCA path rather than the pyscf one.
SUGGESTED_METHODS = [
    "hf", "b3lyp", "pbe", "pbe0", "wb97x-d", "m06-2x", "MP2", "CCSD(T)",
]

#: Point groups the symmetry module recognises in configs. Free text in the
#: form: an unlisted group is passed through, and omitting it lets the
#: geometry's own symmetry detection decide.
SUGGESTED_POINT_GROUPS = [
    "C1", "Cs", "C2", "C2v", "C2h", "C3v", "D2h", "D3h", "D6h",
    "Cinf_v", "Dinf_h", "Td", "Oh",
]

#: Field documentation shown next to each control, in the manner of the Aero
#: web UI: what the setting does, and what it costs or risks. Keyed by the
#: form's field id. Kept here rather than in the HTML so the physics notes sit
#: next to the code that supplies the choices.
FIELD_HELP: dict[str, str] = {
    "name": "Names the run and its output directory under output/runs.",
    "preset": "Convergence and effort preset. FAST_DEBUG is for checking a "
              "case runs at all; STRICT tightens the optimiser's tolerances.",
    "coordinate_mode": "internal fits bond lengths and angles directly, which "
                       "is what makes the per-parameter uncertainties "
                       "meaningful. cartesian fits atom positions and cannot "
                       "report them.",
    "quantum_backend": "Marked \u2014 not installed means the name is "
                       "registered but its dependency is missing here, and a "
                       "run would fail at the first Hessian. pyscf has no "
                       "Windows wheel: use WSL, or ORCA. "
                       "Supplies the energy, gradient and Hessian. pyscf_hf "
                       "covers HF and DFT; the ORCA path reaches MP2 and "
                       "CCSD(T), which matter because the cubic force field "
                       "is what limits the correction's accuracy.",
    "method": "Electronic structure method. DFT cubic force fields leave a "
              "floor of roughly 1000 MHz on water's A correction that no "
              "basis set removes; a correlated method is what moves it.",
    "basis": "Diffuse functions matter more here than cardinal number: "
             "aug-cc-pVTZ beat cc-pVQZ on water's A correction at lower cost.",
    "corrections_mode": "Where the B0 -> Be correction comes from. "
                        "hybrid_auto computes it; user_only takes the values "
                        "in the isotopologue block; none fits B0 as if it "
                        "were Be, which biases the structure.",
    "harmonic_scheme": "watson uses Watson's mu = I^-1 expansion. "
                       "eigenvalue_fd finite-differences the sorted "
                       "eigenvalues and so picks up eigenvalue repulsion that "
                       "is not a contribution to alpha -- kept only for "
                       "reproducing older results.",
    "nonconvergent_policy": "What to do with a component whose cubic term "
                            "exceeds its harmonic one. warn keeps it, inflate "
                            "widens its sigma by the divergence ratio, drop "
                            "removes it from the fit.",
    "tracking_scheme": "How the correction follows the geometry the fit lands "
                       "on. direct rebuilds it there for one extra Hessian "
                       "set; linear and quadratic extrapolate from the "
                       "quantum minimum and are gated to short displacements.",
    "fit_cd_constants": "Use centrifugal-distortion constants as fit data "
                        "alongside the rotational constants. More "
                        "information per isotopologue, and it tightens how "
                        "small a correction bias the run could detect.",
    "cd_reduction": "Watson reduction for the distortion constants. A suits "
                    "asymmetric tops, S near-prolate and symmetric ones.",
    "cubic_scheme": "How the cubic force field is transformed per "
                    "isotopologue. normal_mode transforms phi through the "
                    "mode basis; cartesian differentiates the Hessian in "
                    "Cartesians. normal_mode is what the benchmarks use.",
    "freq_scale": "Scales the computed harmonic frequencies before alpha is "
                  "built, as a crude stand-in for the anharmonicity the "
                  "force field misses. One value, or one per mode.",
    "lam_freq_cm": "Modes below this are treated as large-amplitude: their "
                   "contribution to alpha is carried as uncertainty rather "
                   "than as a value. Set it above the torsional fundamental "
                   "for a molecule with an internal rotor. 0 disables it.",
    "hess_recalc_every": "Quantum updates between Hessian rebuilds. It also "
                         "controls the correction, because alpha is "
                         "re-expanded about the CURRENT geometry on every "
                         "rebuild -- so a smaller number tracks the fitted "
                         "geometry more closely, at the cost of more "
                         "Hessians.",
    "harmonic_from_hessian": "Compute the harmonic and Coriolis parts of alpha "
                             "from the Hessian. Off means the correction has "
                             "to come from the isotopologue block instead.",
    "anharmonic_from_hessian": "Add the cubic anharmonic term. It is the "
                               "DOMINANT contribution to alpha and carries the "
                               "opposite sign to the harmonic one, so leaving "
                               "it off biases Be systematically. It costs 6N "
                               "extra Hessians, which is why it is a switch.",
    "electronic_correction": "Subtract the electron cloud's contribution to "
                             "the moment of inertia, from the rotational "
                             "g tensor. Small -- about -13.6 MHz on water's A "
                             "-- but isotope-dependent, which is where the "
                             "structural information lives.",
    "use_builtin_bob": "Born-Oppenheimer breakdown with the module's built-in "
                       "u-parameters. Measured to move nothing on this "
                       "reference set, water included, so it is cheap "
                       "insurance rather than a fix.",
    "harmonic_sigma_fraction": "Uncertainty on the harmonic part of alpha, as "
                               "a fraction of it. Sets how much the fit is "
                               "allowed to disbelieve its own correction.",
    "anharmonic_fd_delta_ang": "Step for differencing the Hessian to get cubic "
                               "force constants. Too small and SCF noise "
                               "dominates; too large and the third derivative "
                               "is not the one you wanted.",
    "harmonic_cd_from_hessian": "Compute centrifugal-distortion constants from "
                                "the Hessian. Needed before they can be used "
                                "as fit data.",
    "cd_weight": "How heavily distortion constants count against the "
                 "rotational constants. 0 computes them and fits nothing, "
                 "which the runner warns about.",
    "cd_sigma_fraction": "Uncertainty on each computed distortion constant, as "
                         "a fraction of it.",
    "use_dihedrals": "Fit dihedral angles as well as bonds and angles. Only "
                     "meaningful for a molecule that has them, and they are "
                     "usually the least determined parameters.",
    "output_root": "Where run directories are created. Each run gets a "
                   "timestamped folder holding its input.yaml, report, "
                   "exports and plots.",
    "artifacts": "Write the report, CSV exports and plots. Off leaves only the "
                 "console output, so the run cannot be read back afterwards.",
    "prior_sigma_bond": "How far the quantum geometry is trusted, per bond, "
                        "in Angstrom. It sets the weight between the quantum "
                        "surface and the spectral data in one objective.",
    "prior_sigma_angle_deg": "The same trust, for angles, in degrees.",
    "internal_priors_mode": "soft restrains parameters towards the prior; "
                            "hard freezes the ones the data cannot "
                            "determine; off leaves the spectral data alone.",
    "rng_seed": "Seeds multistart and any resampling, so a run reproduces.",
    "geometry_method": "Where the STARTING structure comes from. It only has "
                       "to be in the right basin -- the optimiser moves it, "
                       "and the spectra decide where it lands. SMILES or a "
                       "PubChem name is usually enough; typing coordinates is "
                       "the last resort.",
    "smiles": "SMILES string. Bond orders let PubChem return an "
              "MMFF94-relaxed 3D conformer, which is a better starting point "
              "than a hand-built one.",
    "pubchem_identifier": "A compound name or CID. Fetches PubChem's 3D "
                          "conformer and reorders the atoms to match your "
                          "elements list.",
    "bonds": "Atom index pairs, one bond per line, zero-based. The geometry "
             "is built from connectivity and relaxed.",
    "coords_angstrom": "Cartesian coordinates, one atom per line. A leading "
                       "element symbol is tolerated, so an .xyz block pastes "
                       "straight in.",
    "elements": "Atomic symbols in the order the masses and coordinates use. "
                "Required even when the structure comes from PubChem, because "
                "the fetched atoms are reordered to match it.",
    "symmetry": "Point group, if you want it imposed. Leave blank to let the "
                "geometry's own symmetry be detected.",
    "obs_b0_mhz": "Measured ground-state rotational constants, in MHz, for "
                  "the components selected. These are the data being fitted.",
    "sigma_mhz": "Measurement uncertainty per constant, in MHz. Fitted "
                 "weights go as 1/sigma^2, so these set what the fit "
                 "believes.",
    "masses": "Atomic masses in amu, in the same order as the elements. "
              "Isotopic substitution is where the structural information "
              "comes from, so these carry the signal.",
}


def all_options() -> dict[str, Any]:
    """The whole payload the UI asks for on load."""
    opts = backend_options()
    opts.update({
        "suggested_methods": SUGGESTED_METHODS,
        "suggested_bases": SUGGESTED_BASES,
        "suggested_point_groups": SUGGESTED_POINT_GROUPS,
        "field_help": FIELD_HELP,
    })
    return opts
