"""A linear molecule's distortion prediction: D only, smooth through small bends.

compute_cd_constants ran every geometry through the asymmetric-top reduction.
At a linear geometry that happened to return DJ = D (with placeholder DJK =
-2D and DK = D, which do not exist for a linear molecule), but bent by half a
degree -- a fit's start, any finite-difference step -- A turns finite and
enormous and every constant came out at 1e11-1e13 MHz (fluoroacetylene and
HCN, HF/6-31G). A near-linear geometry (I_min / I_max < CD_LINEAR_REL_TOL) now
takes D from the two perpendicular axes, D = -(tau'_bb + tau'_cc) / 8, and the
distortion rows fit DJ alone (decisions D-90).
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))

import backend.spectral.centrifugal_distortion as C  # noqa: E402
from backend.spectral.spectral import SpectralEngine  # noqa: E402

M_H, M_C, M_N = 1.00782503207, 12.0, 14.0030740048
MASSES = np.array([M_H, M_C, M_N])


def _hcn(r_ch=1.065, r_cn=1.153):
    return np.array([[0.0, 0.0, -r_ch], [0.0, 0.0, 0.0], [0.0, 0.0, r_cn]])


def _bend(x, deg):
    y = np.array(x, float)
    r = np.linalg.norm(y[0] - y[1])
    th = np.radians(deg)
    y[0] = y[1] + r * np.array([np.sin(th), 0.0, -np.cos(th)])
    return y


# HCN's Cartesian Hessian (Hartree/Bohr^2) at _hcn(), RHF/STO-3G from pyscf
# (dev/pyscf_backend, symmetrised), kept here so the test starts no quantum
# chemistry -- and no OpenMP threads, which later forking tests warn about.
_HESS = np.array([
    [0.023043081528783094, -1.488287028825036e-16, 1.0786095865043755e-16, -0.04654720510657673, 7.410605531456641e-17, -7.755069185841913e-17, 0.023504123577794286, 7.83968443079082e-17, -3.0310266347931376e-17],
    [-1.488287028825036e-16, 0.02304308152878276, -7.723766453297876e-18, 7.684259445142348e-17, -0.04654720510657662, 3.9622295681548295e-17, 6.507285770081403e-17, 0.023504123577794134, -3.189853066075592e-17],
    [1.0786095865043755e-16, -7.723766453297876e-18, 0.5277010690239063, -8.566134518569045e-17, -4.2362117001391954e-17, -0.5499878889072581, -2.2199613908837135e-17, 5.0085884887204556e-17, 0.022286819883348186],
    [-0.04654720510657673, 7.684259445142348e-17, -8.566134518569045e-17, 0.09185433833486734, -1.1868642386165365e-15, 2.0508672243422245e-16, -0.04530713322758828, 1.1488032395937921e-15, -1.1942538014949485e-16],
    [7.410605531456641e-17, -0.04654720510657662, -4.2362117001391954e-17, -1.1868642386165365e-15, 0.09185433833495171, -5.964931763955453e-16, 1.0913254748971516e-15, -0.045307133227585616, 6.388552930688527e-16],
    [-7.755069185841913e-17, 3.9622295681548295e-17, -0.5499878889072581, 2.0508672243422245e-16, -5.964931763955453e-16, 2.301279071072946, -1.2753602564008282e-16, 5.568708694581613e-16, -1.7512911821617294],
    [0.023504123577794286, 6.507285770081403e-17, -2.2199613908837135e-17, -0.04530713322758828, 1.0913254748971516e-15, -1.2753602564008282e-16, 0.02180300964823978, -1.1571317985551709e-15, 1.4973564302310993e-16],
    [7.83968443079082e-17, 0.023504123577794134, 5.0085884887204556e-17, 1.1488032395937921e-15, -0.045307133227585616, 5.568708694581613e-16, -1.1571317985551709e-15, 0.021803009648411198, -6.069567583767892e-16],
    [-3.0310266347931376e-17, -3.189853066075592e-17, 0.022286819883348186, -1.1942538014949485e-16, 6.388552930688527e-16, -1.7512911821617294, 1.4973564302310993e-16, -6.069567583767892e-16, 1.7290043622772027],
])


@pytest.fixture(scope="module")
def hess():
    return _HESS.copy()


def _d_perp_at_linear(H, x):
    # D = -tau'_bb / 4 at the linear geometry, built here from the module's
    # pieces with five rigid modes.
    om, L = C.normal_modes(H, MASSES, n_rigid=5)
    keep = om >= 50.0
    om, L = om[keep], L[:, keep]
    dB, _ = C.bk_mode_derivatives(x, MASSES, L, om, 0.05, C.rotational_constants_mhz(x, MASSES))
    tau = C.tau_prime_from_dB1_cm(dB * C._MHZ_TO_CM, om) * C._CM_TO_MHZ
    return -tau[1, 1] / 4.0


def test_a_linear_molecule_gives_d_alone(hess):
    x = _hcn()
    cd = C.compute_cd_constants(hess, x, MASSES, with_residual=False)
    assert cd.reduction == "linear"
    assert cd.DJ == pytest.approx(_d_perp_at_linear(hess, x), rel=1e-10)
    for k in ("DJK", "DK", "delta_J", "delta_K", "d1", "d2"):
        assert np.isnan(getattr(cd, k)), k
    assert set(cd.sigma) == {"DJ"}


def test_d_stays_smooth_as_the_molecule_bends(hess, monkeypatch):
    d0 = C.compute_cd_constants(hess, _hcn(), MASSES, with_residual=False).DJ
    for deg in (0.5, 1.0, 3.0):
        cd = C.compute_cd_constants(hess, _bend(_hcn(), deg), MASSES, with_residual=False)
        assert cd.reduction == "linear", deg
        # Measured on this fixture (HF/STO-3G): +0.004%, +0.016%, +0.14% at
        # 0.5, 1, 3 deg (HF/6-31G: +0.004%, +0.014%, +0.13%).
        assert cd.DJ == pytest.approx(d0, rel=3e-3), deg
    # Premise: through the asymmetric-top path the same bend is nonsense.
    monkeypatch.setattr(C, "CD_LINEAR_REL_TOL", 1e-12)
    bad = C.compute_cd_constants(hess, _bend(_hcn(), 1.0), MASSES, with_residual=False)
    assert bad.reduction != "linear" and abs(bad.DJ) > 1e3 * abs(d0)


def test_a_bent_molecule_takes_the_asymmetric_top_path(hess):
    # Bent past the threshold (HNCO's I_a / I_b is 1.3e-2 at its reference
    # structure, 3.1e-3 at its HF/6-31G minimum): the usual path,
    # finite DJK and DK.
    x = _bend(_hcn(), 30.0)
    assert not C.is_linear(x, MASSES, rel_tol=C.CD_LINEAR_REL_TOL)
    cd = C.compute_cd_constants(hess, x, MASSES, with_residual=False)
    assert cd.reduction in ("A", "S") and np.isfinite(cd.DJK) and np.isfinite(cd.DK)


def _engine(hess, cd_observed):
    b = C.rotational_constants_mhz(_hcn(), MASSES)[1]
    iso = {"name": "HCN", "masses": MASSES.tolist(), "obs_constants": [b],
           "sigma_constants": [1.0], "component_indices": [1], "cd_observed": dict(cd_observed)}
    with contextlib.redirect_stdout(io.StringIO()):
        se = SpectralEngine([iso], cd_weight=1.0, fit_cd_constants=True)
    se.set_hessian_for_cd(hess, coords=_hcn())
    return se


def test_the_rows_fit_d_alone_and_say_so(hess):
    se = _engine(hess, {"DJ": 0.08, "DJK": 0.3, "DK": 1.0})
    iso = se.isotopologues[0]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rows = se._cd_rows(iso, _hcn())
    keys, obs, pred, sigma = rows
    assert keys == ["DJ"] and np.all(np.isfinite(pred)) and np.all(np.isfinite(sigma))
    assert "linear molecule" in buf.getvalue() and "DJK, DK not fitted" in buf.getvalue()
    J = se._cd_jacobian(_bend(_hcn(), 0.5), iso)
    assert J.shape == (1, 9) and np.all(np.isfinite(J))
    # Measured: max |dD/dx| 0.26 MHz/A here, at 0 and 0.5 deg alike.
    assert np.abs(J).max() < 10.0
    Jall, r = se.stacked(_hcn())
    assert r.size == Jall.shape[0] == 1 + 1          # one rotational row, one D row


# -- The branch is pinned once per isotopologue (T-37 verifier) ----------------
#
# Deciding linear-or-not per geometry made a fit whose geometry sat just past
# the threshold predict DJ, DJK and DK on the asymmetric-top path while some
# finite-difference steps of the rows' Jacobian fell in the linear branch,
# where DJK and DK are NaN: the Jacobian filled with NaN and the SVD failed
# (a CLI HCN + DCN fit aborted at its third start). The engine now pins the
# branch with the reduction, once (CD_LINEAR_PIN_REL_TOL).

def _past_the_threshold():
    from scipy.optimize import brentq
    def ratio(x):
        ev = np.sort(np.abs(C.inertia_paf(x, MASSES)[0]))
        return ev[0] / ev[-1]
    def disp(t):
        y = _hcn()
        y[1, 0] += t
        return y
    t = brentq(lambda t: ratio(disp(t)) - C.CD_LINEAR_REL_TOL, 1e-4, 0.5)
    return disp(1.0001 * t)


def test_a_fit_just_past_the_threshold_stays_linear(hess):
    se = _engine(hess, {"DJ": 0.08, "DJK": 0.3, "DK": 1.0})
    iso = se.isotopologues[0]
    x = _past_the_threshold()
    with contextlib.redirect_stdout(io.StringIO()):
        J = se._cd_jacobian(x, iso)
        Jall, r = se.stacked(x)
    assert iso["cd_reduction"] == "linear" and iso["cd_reduction_source"].startswith("auto (linear")
    assert J.shape == (1, 9) and np.all(np.isfinite(J))
    assert np.all(np.isfinite(Jall)) and np.all(np.isfinite(r)) and r.size == 2
    np.linalg.svd(Jall)                               # raised before the pin


def test_a_molecule_that_observes_a_is_never_pinned_linear(hess):
    se = _engine(hess, {"DJ": 0.08})
    iso = se.isotopologues[0]
    iso["component_indices"] = [0, 1, 2]
    iso["obs_constants"] = [900000.0, 44500.0, 44000.0]
    iso.pop("cd_reduction", None)
    with contextlib.redirect_stdout(io.StringIO()):
        red = se._cd_reduction(iso, _bend(_hcn(), 2.0))
    assert red in ("A", "S")


def test_a_stated_reduction_is_kept(hess):
    se = _engine(hess, {"DJ": 0.08})
    assert SpectralEngine([dict(se.isotopologues[0], cd_reduction="Linear")],
                          cd_weight=1.0, fit_cd_constants=True).isotopologues[0]["cd_reduction"] == "linear"
    with pytest.raises(ValueError, match="'A' or 'S'"):
        SpectralEngine([dict(se.isotopologues[0], cd_reduction="Ir")], cd_weight=1.0,
                       fit_cd_constants=True)
    # An explicit asymmetric-top reduction is honoured even near-linear.
    cd = C.compute_cd_constants(hess, _bend(_hcn(), 0.5), MASSES, reduction="S", with_residual=False)
    assert cd.reduction == "S"
