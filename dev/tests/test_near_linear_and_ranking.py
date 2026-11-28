"""Decision page items 15-17 (D-177, D-178, D-179).

16: a geometry within 0.01 A of straight is linear for symmetry detection's
label and for the rigid-mode count everywhere (D-178; backlog N43).
17: multistart ranks gate-passing starts by the model energy at each answer
when every one carries it (D-179; backlog N44).
15: quantize uncertainty's seeding starts are refreshed before they are
ranked, as quantize run's are (D-177; backlog N39).
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

from backend.multistart import select_best_result  # noqa: E402
from backend.spectral.centrifugal_distortion import rigid_mode_count  # noqa: E402
from backend.symmetry import detect_symmetry  # noqa: E402


def _bent(zs, deg):
    """Linear chain on z, atom 0 swung about atom 1 so the 0-1-2 angle is `deg`."""
    x = np.array([[0.0, 0.0, z] for z in zs])
    th = np.radians(180.0 - deg)
    r = np.linalg.norm(x[0] - x[1])
    x[0] = x[1] + np.array([r * np.sin(th), 0.0, -r * np.cos(th)])
    return x


@pytest.mark.parametrize("deg, want", [(179.5, "Cinf_v"), (177.0, "Cinf_v"),
                                       (176.0, "Cinf_v"), (170.0, "Cs")])
def test_a_near_linear_start_is_labelled_linear(deg, want):
    """HCN bent 0.5-3 deg read C6v and 4 deg C5v; their projectors were the
    linear one already (t85/n43/detect_table.py)."""
    assert detect_symmetry(["H", "C", "N"], _bent([-1.06, 0.0, 1.156], deg))[0] == want


def test_a_near_linear_centrosymmetric_start_is_dinf_h():
    assert detect_symmetry(["O", "C", "O"], _bent([-1.16, 0.0, 1.16], 179.5))[0] == "Dinf_h"


def test_a_bent_molecule_keeps_its_group():
    th = np.radians(104.5)
    water = np.array([[0, 0, 0], [0.9578 * np.sin(th / 2), 0.9578 * np.cos(th / 2), 0],
                      [-0.9578 * np.sin(th / 2), 0.9578 * np.cos(th / 2), 0]])
    assert detect_symmetry(["O", "H", "H"], water)[0] == "C2v"


@pytest.mark.parametrize("deg, want", [(180.0, 5), (179.998, 5), (179.9, 5),
                                       (175.0, 6), (172.0, 6)])
def test_the_rigid_mode_count_treats_near_straight_as_straight(deg, want):
    x = _bent([-2.26, -1.2, 0.0, 1.28], deg)        # H-C-C-F, H swung about C
    assert rigid_mode_count(x, [1.008, 12.0, 12.0, 18.998]) == want


def test_the_fit_s_projector_counts_by_the_same_rule():
    """The projector needed an exactly straight geometry (an SVD rank), so a
    C1 fit 2.6e-5 A off axis kept six rigid modes and lost a bend."""
    from backend.quantize import MolecularOptimizer
    opt = MolecularOptimizer.__new__(MolecularOptimizer)
    m = np.array([1.008, 12.0, 12.0, 18.998])
    for deg in (180.0, 179.998, 179.9, 175.0):
        x = _bent([-2.26, -1.2, 0.0, 1.28], deg)
        P = opt._rigid_mode_projector(x, m)
        assert int(round(12 - np.trace(P))) == rigid_mode_count(x, m)


def test_a_c1_fit_a_hair_off_straight_reports_two_equal_bends():
    """HCN at HF/STO-3G without symmetry, started 1e-4 rad off straight: the
    two bends of H-C-N are the same coordinate's two directions."""
    import dev.pyscf_backend  # noqa: F401  (registers pyscf_hf)
    from backend.quantize import MolecularOptimizer
    from dev.tests.test_linear_angles import _ISOS
    x = _bent([-1.06, 0.0, 1.156], 180.0 - np.degrees(1e-4))
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MolecularOptimizer(elems=["H", "C", "N"], coords=x, isotopologues=_ISOS,
                                 quantum_backend="pyscf_hf", orca_method="hf",
                                 orca_basis="sto-3g", use_autoconfig=False, max_iter=6,
                                 symmetry="C1")
        opt.run()
        post = opt.internal_coordinate_posterior()
    bends = [s for n, s in zip(post["names"], post["std_err"]) if n.startswith("bend ")]
    assert len(bends) == 2
    assert bends[0] == pytest.approx(bends[1], rel=1e-3)


def _res(idx, freq, energy, model):
    return {"idx": idx, "freq_rms": freq, "energy": energy, "energy_at_answer_model": model}


def test_starts_are_ranked_by_the_model_energy_at_each_answer():
    """Start 2's last quantum point reads lower, its answer is higher."""
    rs = [_res(1, 1.0, -100.0002, -100.0010), _res(2, 1.0, -100.0005, -100.0001)]
    assert select_best_result(rs)["idx"] == 1


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_without_every_model_energy_the_last_point_energy_ranks(missing):
    rs = [_res(1, 1.0, -100.0002, -100.0010), _res(2, 1.0, -100.0005, missing)]
    assert select_best_result(rs)["idx"] == 2


def test_the_spectral_gate_still_comes_first():
    rs = [_res(1, 10.0, -100.0, -101.0), _res(2, 1.0, -100.0, -100.0)]
    assert select_best_result(rs)["idx"] == 2


@pytest.mark.parametrize("flag", [False, True])
def test_uncertainty_refreshes_its_seeding_starts_unless_told_not_to(flag, monkeypatch, tmp_path):
    import cli
    import runner.run_from_config as rfc
    seen = {}

    class _Stop(Exception):
        pass

    def fake_build(cfg, **kw):
        seen.update(kw)
        raise _Stop()

    monkeypatch.setattr(rfc, "build_optimizer_from_config", fake_build)
    monkeypatch.chdir(tmp_path)
    import dev.analytic_water_backend  # noqa: F401  (the config's backend)
    args = ["uncertainty", str(_ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml"),
            "--mode", "bootstrap", "--samples", "2"] + (["--no-hessian-refresh"] if flag else [])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        with pytest.raises(_Stop):
            cli.main(args)
    assert seen["refresh_hessian"] is (not flag)


def test_the_builder_passes_the_refresh_to_the_multistart(monkeypatch):
    import backend.multistart as ms
    import dev.analytic_water_backend  # noqa: F401
    from runner.run_from_config import build_optimizer_from_config
    from runner.usability import load_config
    seen = {}

    def fake_run_multistart(**kw):
        seen.update(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(ms, "run_multistart", fake_run_multistart)
    cfg = load_config(_ROOT / "dev" / "configs" / "water_analytic_endtoend.yaml")
    for value in (True, False):
        with contextlib.redirect_stdout(io.StringIO()), pytest.raises(RuntimeError):
            build_optimizer_from_config(cfg, multistart=True, refresh_hessian=value)
        assert seen["refresh_hessian"] is value
