"""
Rovibrational correction model utilities.

Provides:
  - parse_correction_table: load user corrections from a dict or YAML file
  - vpt2_delta_b: VPT2 vibrational correction formula (DeltaB = 0.5 * alpha_sum)
  - electronic_delta_b: electronic mass correction (Gordy-Cook approximation)
  - bob_delta_b: Born-Oppenheimer breakdown correction from per-element u-parameters
  - propagate_sigma: quadrature uncertainty propagation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from backend.spectral.centrifugal_distortion import CDConstants

# ── Physical constants ────────────────────────────────────────────────────────

# CODATA 2018 electron mass in unified atomic mass units
M_ELECTRON_AMU: float = 5.48579909070e-4

# ── Built-in BOB u-parameter estimates ───────────────────────────────────────
#
# Dimensionless Watson u-constants for Born-Oppenheimer Breakdown corrections.
# These are ORDER-OF-MAGNITUDE ESTIMATES based on:
#   Watson, J.K.G. J. Mol. Spectrosc. 80 (1980) 411-421
#   Gordy & Cook "Microwave Molecular Spectra" 3rd ed., §11.3
#   Puzzarini et al. Int. Rev. Phys. Chem. 29 (2010) 273-367
#
# u-values are highly molecule- and bond-type-dependent; these defaults carry
# 100% relative uncertainty (sigma_u = u) so they regularise without dominating.
# Provide molecule-specific values via bob_params in the YAML for higher accuracy.
#
# Format: element → component → {"u": float, "sigma_u": float}
#
# Components A, B, C. A is present only for H, D and T. The earlier note here
# said it was omitted because it is negligible for oblate tops, which is not the
# reason and not a safe thing to rely on in this engine: A is the constant that
# determines bond angles, and a molecule with no hydrogens would get no A-axis
# BOB correction at all. The real reason is that no source consulted gives an
# A-axis u for a heavy atom.
#
# So the omission is carried as uncertainty rather than as a zero -- see
# ``bob_delta_b``'s ``unknown_component_sigma``. Measured magnitude, for anyone
# tempted to chase it: on ozone's A that adds about 0.025 MHz of sigma against a
# correction sigma of ~68 MHz, so it is currently irrelevant. It stops being
# irrelevant if a molecule-specific u is supplied that is larger than these
# placeholders, or once correction sigmas reach single-digit MHz.
#: Elements already warned about, so the notice appears once per run.
_warned_builtin_bob: set = set()

_BOB_BUILTIN: dict = {
    # Hydrogen — large BOB due to low mass; varies 0.01-0.03 in hydrides
    "H": {
        "A": {"u": 0.015, "sigma_u": 0.015},
        "B": {"u": 0.015, "sigma_u": 0.015},
        "C": {"u": 0.015, "sigma_u": 0.015},
    },
    # Deuterium — same adiabatic physics as H, smaller due to heavier mass
    "D": {
        "A": {"u": 0.008, "sigma_u": 0.008},
        "B": {"u": 0.008, "sigma_u": 0.008},
        "C": {"u": 0.008, "sigma_u": 0.008},
    },
    # Carbon — well-studied; typical range 0.002-0.005
    "C": {
        "B": {"u": 0.003, "sigma_u": 0.003},
        "C": {"u": 0.003, "sigma_u": 0.003},
    },
    # Nitrogen — similar to carbon
    "N": {
        "B": {"u": 0.003, "sigma_u": 0.003},
        "C": {"u": 0.003, "sigma_u": 0.003},
    },
    # Oxygen — typical range 0.001-0.004
    "O": {
        "B": {"u": 0.002, "sigma_u": 0.002},
        "C": {"u": 0.002, "sigma_u": 0.002},
    },
    # Fluorine — electronegative; range 0.003-0.007
    "F": {
        "B": {"u": 0.004, "sigma_u": 0.004},
        "C": {"u": 0.004, "sigma_u": 0.004},
    },
    # Sulfur — small correction; range 0.001-0.003
    "S": {
        "B": {"u": 0.001, "sigma_u": 0.002},
        "C": {"u": 0.001, "sigma_u": 0.002},
    },
    # Chlorine — similar to sulfur
    "Cl": {
        "B": {"u": 0.001, "sigma_u": 0.002},
        "C": {"u": 0.001, "sigma_u": 0.002},
    },
    # Silicon — sp3 hybridised; limited data, conservative estimate
    "Si": {
        "B": {"u": 0.002, "sigma_u": 0.003},
        "C": {"u": 0.002, "sigma_u": 0.003},
    },
    # Phosphorus — similar to silicon
    "P": {
        "B": {"u": 0.002, "sigma_u": 0.003},
        "C": {"u": 0.002, "sigma_u": 0.003},
    },
    # Bromine — heavier halogen; BOB smaller than Cl; range ~0.001-0.002
    # Ref: Puzzarini et al. Int. Rev. Phys. Chem. 29 (2010) 273-367, Table 3
    "Br": {
        "B": {"u": 0.0012, "sigma_u": 0.002},
        "C": {"u": 0.0012, "sigma_u": 0.002},
    },
    # Iodine — heaviest common halogen; BOB corrections very small; range ~0.0003-0.001
    # Ref: Watson, J.K.G. J. Mol. Spectrosc. 80 (1980) 411-421
    "I": {
        "B": {"u": 0.0006, "sigma_u": 0.001},
        "C": {"u": 0.0006, "sigma_u": 0.001},
    },
    # Selenium — similar to sulfur but slightly smaller per-atom contribution
    # Ref: Gordy & Cook "Microwave Molecular Spectra" 3rd ed., §11.3
    "Se": {
        "B": {"u": 0.0008, "sigma_u": 0.002},
        "C": {"u": 0.0008, "sigma_u": 0.002},
    },
    # Tritium (T = 3H) — same adiabatic physics as H/D, smaller still
    "T": {
        "A": {"u": 0.005, "sigma_u": 0.005},
        "B": {"u": 0.005, "sigma_u": 0.005},
        "C": {"u": 0.005, "sigma_u": 0.005},
    },
    # Lithium — alkali metal; BOB can be significant for LiX molecules; range ~0.003-0.008
    "Li": {
        "B": {"u": 0.005, "sigma_u": 0.005},
        "C": {"u": 0.005, "sigma_u": 0.005},
    },
    # Sodium — heavier alkali; BOB smaller than Li; range ~0.001-0.003
    "Na": {
        "B": {"u": 0.002, "sigma_u": 0.003},
        "C": {"u": 0.002, "sigma_u": 0.003},
    },
}


def get_builtin_bob_params(
    elems: list,
    user_params: Optional[dict] = None,
    warn: bool = True,
) -> dict:
    """
    Return a BOB parameter dict for the given element list.

    Emits a one-time warning per element when a built-in is used, because
    the built-ins are order-of-magnitude placeholders (sigma_u = u), not
    measured constants: they regularise a fit rather than correct it.

    Built-in estimates (see _BOB_BUILTIN) are used for elements not covered by
    user_params.  user_params entries take priority element-by-element.

    Parameters
    ----------
    elems      : list of str  Element symbols (duplicates are de-duplicated).
    user_params : dict or None  Per-element BOB params supplied by the user.
                                Merged with built-ins; user values win on conflicts.
    warn       : bool  If True (default), print a warning listing which elements
                       fell back to built-in estimates so users know to supply
                       molecule-specific values for higher accuracy.

    Returns
    -------
    dict  Per-element BOB parameter dict suitable for bob_delta_b().
    """
    result: dict = {}
    builtin_used: list = []
    seen = set()
    for elem in elems:
        e = str(elem).strip()
        if e in seen:
            continue
        seen.add(e)
        if user_params and e in user_params:
            result[e] = user_params[e]
        elif e in _BOB_BUILTIN:
            result[e] = _BOB_BUILTIN[e]
            builtin_used.append(e)
    # merge any user_params entries for elements not in elems (pass-through)
    if user_params:
        for e, v in user_params.items():
            if e not in result:
                result[e] = v
    if warn and builtin_used:
        print(
            f"  [BOB] Using built-in literature u-parameter estimates for: "
            f"{', '.join(sorted(set(builtin_used)))}.\n"
            "  [BOB] These carry 100% sigma_u and are order-of-magnitude only.\n"
            "  [BOB] Supply molecule-specific bob_params in the YAML for sub-mÅ accuracy."
        )
    return result


# ── Correction table loader ───────────────────────────────────────────────────

def parse_correction_table(source: Union[dict, str, Path, None]) -> dict:
    """
    Parse a correction table into a nested dict: {iso_name: {component: spec_dict}}.

    ``source`` may be:
      - None              → returns {}
      - dict              → returned as-is (validated below)
      - str or Path       → treated as a YAML file path; requires PyYAML

    Each component spec dict accepts:
      delta_mhz      : float  — direct equilibrium correction (added to B0)
      alpha_sum_mhz  : float  — sum_r alpha_r; delta = 0.5 * alpha_sum
      sigma_mhz      : float  — uncertainty on the correction (optional)
      method         : str    — "VPT2", "GVPT2", "HR", "manual" (default "VPT2")
      source         : str    — "user", "orca", "cfour" (default "user")
      basis          : str    — basis set used for the calculation (optional)
      notes          : str    — free-text provenance note (optional)
    """
    if source is None:
        return {}

    if isinstance(source, dict):
        return _validate_table(source)

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Correction table not found: {path}")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to load YAML correction tables: pip install PyYAML"
        ) from exc
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"Correction table YAML must be a mapping at top level, got {type(data).__name__}"
        )
    return _validate_table(data)


def _validate_table(tbl: dict) -> dict:
    valid_comps = {"A", "B", "C"}
    out: dict = {}
    for iso_name, comp_specs in tbl.items():
        if not isinstance(comp_specs, dict):
            raise ValueError(
                f"correction_table['{iso_name}'] must be a dict mapping components to specs, "
                f"got {type(comp_specs).__name__}"
            )
        out[str(iso_name)] = {}
        for comp, spec in comp_specs.items():
            comp = str(comp).strip().upper()
            if comp not in valid_comps:
                raise ValueError(
                    f"correction_table['{iso_name}']: unknown component '{comp}'. "
                    f"Valid: {sorted(valid_comps)}"
                )
            if not isinstance(spec, dict):
                raise ValueError(
                    f"correction_table['{iso_name}']['{comp}'] must be a dict, "
                    f"got {type(spec).__name__}"
                )
            if "delta_mhz" not in spec and "alpha_sum_mhz" not in spec:
                raise ValueError(
                    f"correction_table['{iso_name}']['{comp}'] must have "
                    f"'delta_mhz' or 'alpha_sum_mhz'."
                )
            out[str(iso_name)][comp] = dict(spec)
    return out


# ── Vibrational correction ────────────────────────────────────────────────────

def vpt2_delta_b(alpha_sum_mhz: float) -> float:
    """
    Vibrational correction from VPT2 alpha constants.

    alpha_sum_mhz is the summed alpha_r over all vibrational modes for one
    rotational constant (A, B, or C).  Returns DeltaB_vib = 0.5 * alpha_sum.
    This is added to B0: B_e,SE = B0 + DeltaB_vib - DeltaB_elec - DeltaB_BOB.
    """
    return 0.5 * float(alpha_sum_mhz)


# ── Electronic mass correction ────────────────────────────────────────────────

# CODATA 2018 electron-to-proton mass ratio
M_E_OVER_M_P: float = 5.44617021487e-4


def electronic_delta_b(
    b_obs_mhz: float,
    total_mass_amu: float,
    g_value: Optional[float] = None,
) -> float:
    """
    Electronic correction to a rotational constant.

    Returns the signed delta_mhz to ADD to B0 when building B_e,SE:
        B_e,SE = B0 + DeltaB_vib - DeltaB_elec - DeltaB_BOB

    With ``g_value`` (the rotational g-tensor component g_alpha_alpha for this
    axis) the standard relation is used:

        delta_elec = -(m_e / m_p) * g_alpha * B_obs

    This is the physically correct form. Note that g can be negative — for OCS
    g_bb is about -0.028 — in which case the correction is *positive*.

    Without a g_value it falls back to the crude 1/M_total estimate

        delta_elec = -(m_e / M_total) * B_obs

    which is NOT the standard formula and should be treated as an
    order-of-magnitude placeholder only. It scales as 1/M_total, whereas the
    real correction scales with g and is roughly mass-independent, so it
    understates the correction by about (M_total * g): ~12x for water, ~4x for
    benzene, and it gets the sign wrong whenever g is negative. Supply
    ``g_tensor`` in the config for anything beyond a rough estimate; the
    fallback's uncertainty is widened accordingly in resolve_corrections().

    Parameters
    ----------
    b_obs_mhz : float
        Observed rotational constant B0 in MHz.
    total_mass_amu : float
        Total molecular mass (sum of all atomic masses) in amu.
    g_value : float or None
        Rotational g-tensor component for this axis, dimensionless.

    Returns
    -------
    float
        delta_mhz — the signed additive correction.
    """
    if g_value is not None:
        return -M_E_OVER_M_P * float(g_value) * float(b_obs_mhz)
    return -(M_ELECTRON_AMU / float(total_mass_amu)) * float(b_obs_mhz)


# ── Born-Oppenheimer Breakdown correction ─────────────────────────────────────

def _unknown_u_bound(elem_params: dict) -> float:
    """A bound on an element's u for an axis it has no value for.

    Taken as the largest ``|u| + sigma_u`` over the axes it does have. A BOB
    parameter is a property of how one atom's electron cloud follows that atom,
    and no axis of the same atom is expected to be systematically larger than
    the others, so the measured axes bound the unmeasured one. It is a bound,
    not an estimate, which is why it lands in sigma and never in the value.
    """
    best = 0.0
    for key, entry in elem_params.items():
        if str(key).strip().upper() not in ("A", "B", "C"):
            continue
        if isinstance(entry, dict):
            u = abs(float(entry.get("u", 0.0)))
            sig = entry.get("sigma_u", None)
            u += abs(float(sig)) if sig is not None else 0.0
        else:
            u = abs(float(entry))
        best = max(best, u)
    return best


def bob_delta_b(
    elems: list,
    masses_amu: list,
    comp_label: str,
    bob_params: dict,
    b_obs_mhz: float,
    unknown_component_sigma: bool = True,
) -> tuple:
    """
    Born-Oppenheimer Breakdown (BOB) correction to one rotational constant.

    Returns the signed (delta_mhz, sigma_mhz) to ADD to B0 when building B_e,SE.
    The correction is negative (for positive u-values) because DeltaB_BOB is
    subtracted in the r_e^SE formula:
        B_e,SE = B0 + DeltaB_vib - DeltaB_elec - DeltaB_BOB
        delta_bob = -B_obs * Σ_a (m_e / m_a) * u_a

    The u-parameters are dimensionless *relative* corrections, so they scale a
    rotational constant exactly as the g-tensor does in ``electronic_delta_b``.
    Omitting that factor is not a small error: it returns a number of order
    1e-5 where the answer is of order 1e-5 * B, which for water's B is about
    7 MHz. This function did that until the factor was added, so every BOB
    correction it ever produced was zero to within printing precision.

    The u-parameters are dimensionless and mass-independent; the mass scaling
    (m_e / m_a) is applied here so that different isotopologues automatically
    get the correct isotope-specific correction from the same u-values.

    Parameters
    ----------
    elems : list of str
        Element symbols in atom order (same order as masses_amu).
    masses_amu : list of float
        Nuclear masses in amu for each atom in this isotopologue.
    comp_label : str
        Rotational constant label: "A", "B", or "C".
    bob_params : dict
        Per-element BOB u-parameters. Format::

            {
                "H": {"A": 0.0, "B": 0.012, "C": 0.009},
                "O": {"B": 0.003, "C": 0.002},
            }

        Each per-component entry is either:
          - float  : the dimensionless u-value (sigma unknown)
          - dict   : {"u": float, "sigma_u": float | None}

        Elements not present in bob_params contribute zero.
    b_obs_mhz : float
        The observed rotational constant this correction applies to, in MHz.
        The u-values are relative, so the correction is proportional to it.
    unknown_component_sigma : bool
        What to do about an element that has u-values for some axes but not the
        one being corrected -- which is every heavy atom's A axis in the
        built-in table, because no source consulted gives one.

        The historical behaviour was to contribute exactly zero with no sigma,
        which tells the fit the BOB correction on that axis is known to be zero.
        It is not known; it is unmeasured, and those are different claims. With
        this set (the default) the value still contributes zero -- there is no
        defensible non-zero estimate -- but the element's known u-values bound
        the unknown one, and that bound is carried as sigma.

        Measured magnitude: on ozone's A this adds about 0.025 MHz against a
        correction sigma of ~68 MHz, so it changes nothing today. It is here so
        that the ignorance is recorded rather than asserted away, and because it
        starts to matter if a molecule-specific u is supplied or once sigmas
        reach single-digit MHz.

    Returns
    -------
    (delta_mhz, sigma_mhz)
        delta_mhz : float — signed additive correction (negative when u > 0).
        sigma_mhz : float | None — propagated uncertainty; None if no sigma_u supplied.
    """
    comp = str(comp_label).strip().upper()
    b_obs = float(b_obs_mhz)
    total_delta = 0.0
    sigma_sq = 0.0
    any_sigma = False

    for elem, mass in zip(elems, masses_amu):
        elem_params = bob_params.get(str(elem), None)
        if elem_params is None:
            continue
        m = float(mass)
        if m <= 0.0:
            continue
        scale = M_ELECTRON_AMU / m

        comp_entry = elem_params.get(comp, None)
        if comp_entry is None:
            # This axis is unparameterised for this element. Contribute no
            # value -- there is nothing to base one on -- but bound the unknown
            # u by the ones the element does have, and carry that as sigma.
            if not unknown_component_sigma:
                continue
            bound = _unknown_u_bound(elem_params)
            if bound > 0.0:
                sigma_sq += (scale * bound * b_obs) ** 2
                any_sigma = True
            continue

        if isinstance(comp_entry, dict):
            u = float(comp_entry.get("u", 0.0))
            sigma_u = comp_entry.get("sigma_u", None)
        else:
            u = float(comp_entry)
            sigma_u = None

        total_delta -= scale * u * b_obs   # subtracted per r_e^SE formula

        if sigma_u is not None:
            sigma_sq += (scale * float(sigma_u) * b_obs) ** 2
            any_sigma = True

    sigma = float(sigma_sq ** 0.5) if any_sigma else None
    return total_delta, sigma


# ── Uncertainty propagation ───────────────────────────────────────────────────

def propagate_sigma(sigma_exp: float, *correction_sigmas: Optional[float]) -> float:
    """
    Quadrature uncertainty propagation.

    sigma_eff = sqrt(sigma_exp^2 + sigma_vib^2 + sigma_elec^2 + sigma_BOB^2 + ...)

    None entries (unknown correction uncertainty) are skipped.
    """
    total = float(sigma_exp) ** 2
    for s in correction_sigmas:
        if s is not None:
            v = float(s)
            if v > 0.0:
                total += v * v
    return float(total ** 0.5)


# ── Structured correction dataclasses ────────────────────────────────────────

COMPONENTS = ("A", "B", "C")


@dataclass
class RovibCorrection:
    """Isotopologue-specific rovibrational correction record."""

    isotopologue: str
    alpha_A: Optional[float] = None
    alpha_B: Optional[float] = None
    alpha_C: Optional[float] = None
    delta_vib_A: Optional[float] = None
    delta_vib_B: Optional[float] = None
    delta_vib_C: Optional[float] = None
    delta_elec_A: float = 0.0
    delta_elec_B: float = 0.0
    delta_elec_C: float = 0.0
    delta_bob_A: float = 0.0
    delta_bob_B: float = 0.0
    delta_bob_C: float = 0.0
    sigma_delta_A: Optional[float] = None
    sigma_delta_B: Optional[float] = None
    sigma_delta_C: Optional[float] = None
    source: str = "unknown"
    backend: Optional[str] = None
    method: Optional[str] = None
    basis: Optional[str] = None
    geometry_hash: Optional[str] = None
    status: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    pred_cd: Optional["CDConstants"] = None

    def alpha_vector(self) -> np.ndarray:
        return np.array([
            np.nan if self.alpha_A is None else self.alpha_A,
            np.nan if self.alpha_B is None else self.alpha_B,
            np.nan if self.alpha_C is None else self.alpha_C,
        ], dtype=float)

    def delta_vib_vector(self) -> np.ndarray:
        alpha = self.alpha_vector()
        out = []
        for val, a in zip(
            [self.delta_vib_A, self.delta_vib_B, self.delta_vib_C], alpha
        ):
            if val is not None:
                out.append(val)
            elif np.isfinite(a):
                out.append(0.5 * a)
            else:
                out.append(np.nan)
        return np.array(out, dtype=float)

    def delta_elec_vector(self) -> np.ndarray:
        return np.array([self.delta_elec_A, self.delta_elec_B, self.delta_elec_C], dtype=float)

    def delta_bob_vector(self) -> np.ndarray:
        return np.array([self.delta_bob_A, self.delta_bob_B, self.delta_bob_C], dtype=float)

    def delta_total_vector(self) -> np.ndarray:
        return self.delta_vib_vector() + self.delta_elec_vector() + self.delta_bob_vector()

    def sigma_delta_vector(self) -> np.ndarray:
        return np.array([
            np.nan if self.sigma_delta_A is None else self.sigma_delta_A,
            np.nan if self.sigma_delta_B is None else self.sigma_delta_B,
            np.nan if self.sigma_delta_C is None else self.sigma_delta_C,
        ], dtype=float)


@dataclass
class ParsedRovibResult:
    """Structured result from an ORCA VPT2 parse."""

    alpha_abc: np.ndarray
    frequencies: Optional[np.ndarray] = None
    warnings: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    parse_status: str = "unknown"
    units: str = "MHz"
