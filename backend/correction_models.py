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
from typing import Optional, Union

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────

# CODATA 2018 electron mass in unified atomic mass units
M_ELECTRON_AMU: float = 5.48579909070e-4


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

def electronic_delta_b(b_obs_mhz: float, total_mass_amu: float) -> float:
    """
    Electronic mass correction to a rotational constant (Gordy-Cook approximation).

    Returns the signed delta_mhz to ADD to B0 when building B_e,SE.
    The correction is negative because DeltaB_elec is subtracted:
        B_e,SE = B0 + DeltaB_vib - DeltaB_elec - DeltaB_BOB
        delta_elec = -(m_e / M_total) * B_obs

    The approximation treats total molecular mass M_total as the effective
    rotational mass and ignores the electronic g-tensor (typically valid to
    ~10-30% of the correction itself — sufficient for most semi-experimental
    structure work). Uncertainty from this approximation should be captured
    via sigma_elec_fraction in resolve_corrections().

    Parameters
    ----------
    b_obs_mhz : float
        Observed rotational constant B0 in MHz.
    total_mass_amu : float
        Total molecular mass (sum of all atomic masses) in amu.

    Returns
    -------
    float
        delta_mhz — negative value representing -DeltaB_elec.
    """
    return -(M_ELECTRON_AMU / float(total_mass_amu)) * float(b_obs_mhz)


# ── Born-Oppenheimer Breakdown correction ─────────────────────────────────────

def bob_delta_b(
    elems: list,
    masses_amu: list,
    comp_label: str,
    bob_params: dict,
) -> tuple:
    """
    Born-Oppenheimer Breakdown (BOB) correction to one rotational constant.

    Returns the signed (delta_mhz, sigma_mhz) to ADD to B0 when building B_e,SE.
    The correction is negative (for positive u-values) because DeltaB_BOB is
    subtracted in the r_e^SE formula:
        B_e,SE = B0 + DeltaB_vib - DeltaB_elec - DeltaB_BOB
        delta_bob = -Σ_a (m_e / m_a) * u_a^X

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

    Returns
    -------
    (delta_mhz, sigma_mhz)
        delta_mhz : float — signed additive correction (negative when u > 0).
        sigma_mhz : float | None — propagated uncertainty; None if no sigma_u supplied.
    """
    comp = str(comp_label).strip().upper()
    total_delta = 0.0
    sigma_sq = 0.0
    any_sigma = False

    for elem, mass in zip(elems, masses_amu):
        elem_params = bob_params.get(str(elem), None)
        if elem_params is None:
            continue
        comp_entry = elem_params.get(comp, None)
        if comp_entry is None:
            continue

        m = float(mass)
        if m <= 0.0:
            continue
        scale = M_ELECTRON_AMU / m

        if isinstance(comp_entry, dict):
            u = float(comp_entry.get("u", 0.0))
            sigma_u = comp_entry.get("sigma_u", None)
        else:
            u = float(comp_entry)
            sigma_u = None

        total_delta -= scale * u   # subtracted per r_e^SE formula

        if sigma_u is not None:
            sigma_sq += (scale * float(sigma_u)) ** 2
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
    source_files: list[str] = field(default_factory=list)
    parse_status: str = "unknown"
    units: str = "MHz"
