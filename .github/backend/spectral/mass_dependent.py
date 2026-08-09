"""Watson mass-dependent (r_m) structures.

The problem this solves
-----------------------
Rotational spectroscopy measures ground-state constants B_0, but the quantity
a structure should be compared against -- and the one theory computes -- is the
equilibrium B_e. The two differ by the rovibrational sum

    B_0 = B_e - 1/2 sum_r alpha_r

which is a few tenths of a percent, far larger than the precision of the
constants themselves. Fitting a rigid structure to B_0 gives an r_0 structure,
and the r_0-to-r_e gap is the systematic that dominates the error in every
benchmark in this repository.

There are two ways to close it. One is to compute the alpha_r from a cubic force
field, which is what `rovib_corrections` does; it needs an anharmonic force
field, and at the levels of theory that are affordable here it was measured to
be worse than applying nothing at all. The other is Watson's: note that the
rovibrational contribution to the *moment* of inertia is smooth and largely
isotope-independent when written the right way, and fit it as a parameter
instead of computing it.

The models
----------
Writing I for a principal moment (amu Ang^2) and I_m for the moment of the
fitted structure:

    r_0    I_obs = I_m
    r_m1   I_obs = I_m + c_x sqrt(I_m)
    r_m2   I_obs = I_m + c_x sqrt(I_m) + d_x rho

The c_x (one per principal axis) absorb the bulk of the rovibrational offset.
They are shared across every isotopologue -- that is the whole point, and the
reason the method needs several species: the structure and the c_x are
separated by how differently they respond to isotopic substitution, not by any
extra measurement.

`rho = (prod_i m_i / M)^(1/(2N-2))` is Watson's second-order term, which mainly
matters for hydrogen, whose large relative mass change on deuteration is poorly
described by the sqrt term alone. Both extra terms are ordinary fitted
parameters here, so the r_m2 model is only worth choosing when the data can
support the extra three.

What is asserted and what is not
--------------------------------
The fitting machinery is tested against synthetic data generated from each
model, so recovering a known structure and known c_x is verified rather than
assumed. The *physical* claim -- that these functional forms describe real
rovibrational offsets well -- is Watson's, and holds to the extent the
literature says it does; the r_m1 form is the well-established one, and it is
the default here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from .centrifugal_distortion import _INERTIA_TO_MHZ

#: Supported mass-dependence models, in increasing order of parameter count.
MODELS = ("r0", "rm1", "rm2")


def moments_amu_ang2(coords_ang, masses_amu) -> np.ndarray:
    """Principal moments I_a <= I_b <= I_c in amu Ang^2."""
    masses = np.asarray(masses_amu, dtype=float)
    coords = np.asarray(coords_ang, dtype=float)
    com = (masses[:, None] * coords).sum(0) / masses.sum()
    r = coords - com
    r2 = np.einsum("ia,ia->i", r, r)
    inertia = (np.einsum("i,jk->jk", masses * r2, np.eye(3))
               - np.einsum("i,ij,ik->jk", masses, r, r))
    return np.sort(np.linalg.eigvalsh(inertia))


def moments_from_constants_mhz(constants_mhz) -> np.ndarray:
    """Moments in amu Ang^2 from A, B, C in MHz (ordered I_a <= I_b <= I_c)."""
    c = np.asarray(constants_mhz, dtype=float)
    out = np.full(c.shape, np.nan, dtype=float)
    good = np.isfinite(c) & (c > 0)
    out[good] = _INERTIA_TO_MHZ / c[good]
    return out


def watson_rho(masses_amu) -> float:
    """Watson's second-order mass factor (prod m_i / M)^(1/(2N-2))."""
    m = np.asarray(masses_amu, dtype=float)
    n = m.size
    if n < 2:
        return 0.0
    # Accumulate in logs: the product over all atoms overflows for even
    # moderately sized molecules, and the exponent 1/(2N-2) then turns an inf
    # into a nan rather than the finite number it should be.
    log_rho = (np.sum(np.log(m)) - np.log(m.sum())) / (2.0 * n - 2.0)
    return float(np.exp(log_rho))


def predicted_moments(coords_ang, masses_amu, c=None, d=None) -> np.ndarray:
    """Observable moments predicted by a mass-dependent model.

    `c` and `d` are per-axis coefficients (length 3) or None. With both None
    this is the rigid r_0 prediction.
    """
    i_m = moments_amu_ang2(coords_ang, masses_amu)
    out = i_m.astype(float).copy()
    if c is not None:
        out = out + np.asarray(c, dtype=float) * np.sqrt(np.maximum(i_m, 0.0))
    if d is not None:
        out = out + np.asarray(d, dtype=float) * watson_rho(masses_amu)
    return out


@dataclass
class MassDependentFit:
    """Result of a mass-dependent structure fit."""

    model: str
    coords: np.ndarray
    c: Optional[np.ndarray]
    d: Optional[np.ndarray]
    n_obs: int
    n_params: int            #: parameter combinations the data determines (Jacobian rank)
    n_params_raw: int        #: parameters varied, including redundant rigid-body ones
    n_determinable: int      #: most this model could fix if every constant were measured
    rms_moment: float          #: RMS residual in amu Ang^2
    rms_constants_mhz: float   #: RMS residual back in MHz
    reduced_chi2: Optional[float]
    converged: bool
    warnings: list = field(default_factory=list)

    @property
    def dof(self) -> int:
        return self.n_obs - self.n_params

    @property
    def underdetermined(self) -> bool:
        """True when the data leave directions to the starting geometry."""
        return self.n_params < self.n_determinable


def _pack(coords, c, d, model):
    parts = [np.asarray(coords, dtype=float).ravel()]
    if model in ("rm1", "rm2"):
        parts.append(np.asarray(c, dtype=float).ravel())
    if model == "rm2":
        parts.append(np.asarray(d, dtype=float).ravel())
    return np.concatenate(parts)


def _unpack(p, n_atoms, model):
    n = 3 * n_atoms
    coords = p[:n].reshape(-1, 3)
    c = d = None
    k = n
    if model in ("rm1", "rm2"):
        c = p[k:k + 3]
        k += 3
    if model == "rm2":
        d = p[k:k + 3]
    return coords, c, d


def _rank(matrix, default: int, rel_tol: float = 1e-8) -> int:
    """Numerical rank, relative to the largest singular value."""
    m = np.asarray(matrix, dtype=float)
    if m.size == 0 or min(m.shape) == 0:
        return default
    sv = np.linalg.svd(m, compute_uv=False)
    if sv.size == 0 or sv[0] <= 0.0:
        return 0
    return int((sv > rel_tol * sv[0]).sum())


def _n_determinable(coords, n_par, symmetry) -> int:
    """How many parameter combinations this parameterisation could ever fix.

    Not every parameter is determinable even in principle. Moments do not depend
    on where the molecule sits or how it is turned, so the six rigid-body
    directions are invisible to any amount of data; and when a symmetry
    constraint is in force, the directions it forbids are invisible too. Those
    are properties of the parameterisation, so subtracting them gives the target
    an actual fit's rank should be judged against.

    Deriving this from the species at hand would be circular -- with a single
    species the "most determinable" would come out equal to what that species
    determines, and nothing would ever look under-determined.
    """
    xyz = np.asarray(coords, dtype=float)
    n = xyz.shape[0]
    com = xyz.mean(axis=0)
    rel = xyz - com
    modes = []
    for k in range(3):
        v = np.zeros((n, 3))
        v[:, k] = 1.0
        modes.append(v.ravel())
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        modes.append(np.cross(np.tile(e, (n, 1)), rel).ravel())
    rigid = np.column_stack(modes)

    if symmetry is None:
        n_blind = _rank(rigid, default=6)
    else:
        P = np.asarray(symmetry.projection, dtype=float)
        # Directions symmetry forbids outright, plus the rigid ones that survive
        # it -- both are invisible to the data, and they do not overlap.
        n_forbidden = P.shape[0] - _rank(P, default=P.shape[0])
        n_blind = n_forbidden + _rank(P @ rigid, default=0)
    return max(0, n_par - int(n_blind))


def fit_mass_dependent_structure(
    isotopologues: Sequence[dict],
    coords0,
    model: str = "rm1",
    symmetry=None,
    max_nfev: int = 400,
) -> MassDependentFit:
    """Fit a structure and its mass-dependence coefficients to measured constants.

    Parameters
    ----------
    isotopologues
        Dicts with ``masses`` (amu, per atom), ``obs_constants`` (A, B, C in
        MHz) and optionally ``component_indices`` (which of A, B, C were
        actually measured) and ``sigma_constants`` (MHz). Entries whose
        constants are missing or non-positive are skipped for that component.
    coords0
        Starting geometry, (N, 3) in Angstrom.
    model
        One of ``"r0"``, ``"rm1"``, ``"rm2"``.
    symmetry
        Optional ``PointGroupSymmetry``. When given, the geometry is confined to
        the symmetric subspace throughout, which both removes the parameters
        symmetry forbids and stops the fit spending data on breaking it.

    Notes
    -----
    Residuals are formed on the *moments*, not the constants: the model is
    linear in the correction there, whereas in constant space the same
    correction appears as 1/(I + dI) and the near-degenerate large-A cases pick
    up badly-scaled residuals. Sigmas are converted along with them.

    The six rigid-body directions are exactly null here, since moments do not
    depend on where the molecule sits or how it is turned. `least_squares` with
    a trust-region solver handles that rank deficiency without needing them
    removed; they simply never move.
    """
    model = str(model).strip().lower()
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")

    coords0 = np.asarray(coords0, dtype=float)
    n_atoms = coords0.shape[0]
    warnings: list[str] = []

    obs, sig, masses_list = [], [], []
    for iso in isotopologues:
        const = np.asarray(iso["obs_constants"], dtype=float)
        idx = np.asarray(
            iso.get("component_indices", list(range(const.size))), dtype=int)
        m = np.asarray(iso["masses"], dtype=float)
        if m.size != n_atoms:
            warnings.append(
                f"{iso.get('name', '?')}: {m.size} masses for {n_atoms} atoms; skipped")
            continue
        # `obs_constants` is compact: entry k is the constant for component
        # `component_indices[k]`, not for axis k. Species here routinely have
        # only B measured, so indexing it as a three-slot A/B/C array would pair
        # that B against A and silently corrupt the fit rather than fail.
        i_obs_local = moments_from_constants_mhz(const)
        s_const = iso.get("sigma_constants")
        s_const = (np.asarray(s_const, dtype=float) if s_const is not None
                   else np.full(const.size, np.nan))
        if idx.size != const.size:
            warnings.append(
                f"{iso.get('name', '?')}: {idx.size} component indices for "
                f"{const.size} constants; skipped")
            continue
        for k_local, comp in enumerate(idx):
            comp = int(comp)
            if not (0 <= comp < 3) or not np.isfinite(i_obs_local[k_local]):
                continue
            # dI = -I/B * dB, so a fractional error on the constant is the same
            # fractional error on the moment.
            s_b = (s_const[k_local] if k_local < s_const.size else np.nan)
            if np.isfinite(s_b) and s_b > 0 and const[k_local] > 0:
                s_i = abs(i_obs_local[k_local]) * (s_b / const[k_local])
            else:
                s_i = abs(i_obs_local[k_local]) * 1e-4
            obs.append((len(masses_list), comp,
                        float(i_obs_local[k_local]), float(s_i)))
        masses_list.append(m)

    if not obs:
        raise ValueError("no usable constants: need at least one measured A, B or C")

    n_par = 3 * n_atoms + (3 if model in ("rm1", "rm2") else 0) + (3 if model == "rm2" else 0)
    n_obs = len(obs)

    def residual(p):
        coords, c, d = _unpack(p, n_atoms, model)
        if symmetry is not None:
            coords = symmetry.symmetrize(coords)
        cache = {}
        out = np.empty(n_obs, dtype=float)
        for r, (i_iso, comp, i_obs, s_i) in enumerate(obs):
            if i_iso not in cache:
                cache[i_iso] = predicted_moments(coords, masses_list[i_iso], c, d)
            out[r] = (cache[i_iso][comp] - i_obs) / s_i
        return out

    c0 = np.zeros(3)
    d0 = np.zeros(3)
    start = _pack(coords0 if symmetry is None else symmetry.symmetrize(coords0),
                  c0, d0, model)
    sol = least_squares(residual, start, method="trf", max_nfev=max_nfev)

    coords, c, d = _unpack(sol.x, n_atoms, model)
    if symmetry is not None:
        coords = symmetry.symmetrize(coords)

    res = sol.fun
    # Effective parameter count, not the raw one. Of the 3N Cartesian
    # coordinates, six (five if linear) are rigid-body directions that no moment
    # can see, and a symmetry constraint removes more still. Counting all 3N
    # makes every real fit look under-determined -- water reports -6 degrees of
    # freedom when it genuinely has four. The rank of the residual Jacobian is
    # the number of parameter combinations the data determines, and that is the
    # count that belongs in a chi-square.
    # A looser tolerance than for an exact matrix: least_squares builds this
    # Jacobian by finite differences, so a direction the model is genuinely flat
    # along comes back with a small non-zero derivative rather than a zero one.
    n_eff = _rank(np.asarray(sol.jac, dtype=float), default=n_par, rel_tol=1e-6)

    # Rank alone cannot answer "is this determined?", because it is bounded by
    # the number of observations -- an under-determined fit would still report
    # dof >= 0. Compare instead against how many combinations the
    # parameterisation could ever fix, once the directions that are invisible by
    # construction are discounted.
    n_determinable = _n_determinable(coords, n_par, symmetry)
    # The data cannot fix more combinations than the parameterisation admits, so
    # a rank above this is finite-difference noise rather than information.
    n_eff = min(n_eff, n_determinable)
    dof = n_obs - n_eff
    # Weighted residuals are dimensionless; unweight to report in amu Ang^2.
    sig_arr = np.array([s for (_, _, _, s) in obs], dtype=float)
    rms_moment = float(np.sqrt(np.mean((res * sig_arr) ** 2)))
    mhz = []
    for r, (i_iso, comp, i_obs, s_i) in enumerate(obs):
        i_pred = i_obs + res[r] * s_i
        if i_pred > 0 and i_obs > 0:
            mhz.append(_INERTIA_TO_MHZ / i_pred - _INERTIA_TO_MHZ / i_obs)
    rms_mhz = float(np.sqrt(np.mean(np.square(mhz)))) if mhz else float("nan")

    if dof > 0:
        red_chi2 = float(np.sum(res ** 2) / dof)
    else:
        red_chi2 = None
        warnings.append(
            f"{n_obs} observations determine {n_eff} parameter combinations, "
            f"leaving no degrees of freedom: residuals cannot measure agreement."
        )
    if n_eff < n_determinable:
        warnings.append(
            f"under-determined: the data fix {n_eff} of the {n_determinable} "
            f"parameter combinations this model supports, so {n_determinable - n_eff} "
            f"direction(s) are set by the starting geometry rather than by "
            f"measurement. Supply more isotopologues, or constrain by symmetry."
        )
    if model != "r0" and len(masses_list) < 2:
        warnings.append(
            "a mass-dependent model needs more than one isotopologue: with a "
            "single species the c terms are indistinguishable from the structure."
        )
    if not sol.success:
        warnings.append(f"least_squares did not converge: {sol.message}")

    return MassDependentFit(
        model=model, coords=coords,
        c=(None if c is None else np.asarray(c, dtype=float)),
        d=(None if d is None else np.asarray(d, dtype=float)),
        n_obs=n_obs, n_params=n_eff, n_params_raw=n_par,
        n_determinable=n_determinable,
        rms_moment=rms_moment, rms_constants_mhz=rms_mhz,
        reduced_chi2=red_chi2, converged=bool(sol.success), warnings=warnings,
    )
