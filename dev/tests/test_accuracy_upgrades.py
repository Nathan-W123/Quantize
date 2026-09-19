"""The four accuracy upgrades, each pinned at the point where it could silently fail.

Every one of these has a failure mode that still runs and still produces
plausible numbers, which is the only reason they are worth testing:

  bond offsets  -- an offset applied to the wrong class, or with the wrong
                   sign, moves the geometry away from the reference while the
                   fit still converges
  leave-one-out -- a table that includes the molecule being scored makes any
                   benchmark look good and means nothing
  g-tensor      -- the tensor comes back in the input frame; label it A/B/C
                   without rotating into the principal axis frame and two
                   components are swapped, with no error anywhere
  LAM sigma     -- widening sigma must not change the correction itself
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / ".github"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "dev" / "tests"))

from backend.spectral.bond_offsets import (  # noqa: E402
    bond_class,
    bond_classes_of,
    leave_one_out_offsets,
    measure_offsets,
    offset_corrected_geometry,
    residual_sigma_after_offsets,
)
from backend.spectral.harmonic_alpha import compute_harmonic_alpha  # noqa: E402
from dev.monofluoro_references import MOLECULES, WATER_SET  # noqa: E402
from reference_molecules import H2O_MASSES, h2o_coords, h2o_hessian  # noqa: E402

_VF = MOLECULES[0]          # vinyl fluoride: C-F, C-C, three C-H


# ── bond classes and offsets ───────────────────────────────────────────────

def test_bond_class_is_order_independent():
    assert bond_class("F", "C") == bond_class("C", "F") == "C-F"


def test_classes_cover_every_named_bond_of_a_real_molecule():
    classes = bond_classes_of(_VF)
    assert set(classes) == set(_VF.bonds)
    assert classes["C1-F"] == "C-F"
    assert classes["C1=C2"] == "C-C"
    assert classes["C1-H4"] == "C-H"


def test_offsets_are_signed_theory_minus_reference():
    """A geometry with one bond deliberately lengthened must report it positive.

    Sign is the whole ballgame: a flipped sign doubles the error instead of
    removing it, and the fit converges either way.
    """
    coords = np.asarray(_VF.geometry, dtype=float).copy()
    # Push F away from C1 by 0.05 A along the C1->F direction.
    i, j = _VF.bonds["C1-F"][0]
    u = coords[j] - coords[i]
    coords[j] = coords[i] + u * (1.0 + 0.05 / np.linalg.norm(u))

    offs = measure_offsets(_VF, coords)
    assert offs["C-F"][0] == pytest.approx(0.05, abs=1e-9)
    assert all(abs(v) < 1e-9 for v in offs["C-C"])


def test_leave_one_out_never_uses_the_molecule_being_scored():
    """The guarantee the benchmark rests on. Without it any gain is circular."""
    per_mol = {"a": {"C-F": [0.030]}, "b": {"C-F": [0.020, 0.024]}}
    loo_a = leave_one_out_offsets(per_mol, "a")
    assert loo_a["C-F"] == pytest.approx(0.022)
    loo_b = leave_one_out_offsets(per_mol, "b")
    assert loo_b["C-F"] == pytest.approx(0.030)


def test_a_class_seen_only_in_the_scored_molecule_gets_no_offset():
    """Silence, not a guess. Water's H-O is a singleton in the reference set,
    so leave-one-out must decline to correct it rather than borrow from
    another class."""
    per_mol = {"a": {"H-O": [0.010]}, "b": {"C-F": [0.028]}}
    assert "H-O" not in leave_one_out_offsets(per_mol, "a")


def test_correction_moves_the_named_class_and_leaves_the_others_alone():
    """Applying an offset must shorten exactly the bonds of that class."""
    theory = np.asarray(_VF.geometry, dtype=float)
    before = _VF.internal_coordinates(theory)
    fixed = offset_corrected_geometry(_VF, theory, {"C-F": 0.028})
    after = _VF.internal_coordinates(fixed)

    assert after["C1-F"] - before["C1-F"] == pytest.approx(-0.028, abs=5e-4)
    for name in ("C1=C2", "C1-H4", "C2-H5", "C2-H6"):
        assert after[name] == pytest.approx(before[name], abs=5e-4)
    for name in _VF.angles:
        assert after[name] == pytest.approx(before[name], abs=0.05)


def test_an_empty_offset_table_is_the_identity():
    theory = np.asarray(_VF.geometry, dtype=float)
    fixed = offset_corrected_geometry(_VF, theory, {})
    assert np.max(np.abs(fixed - theory)) < 1e-6


def test_measured_offsets_round_trip_through_the_correction():
    """Measure on a distorted geometry, correct by what was measured, land on
    the reference. This is the end-to-end claim the upgrade makes."""
    mol = WATER_SET[0]
    ref = np.asarray(mol.geometry, dtype=float)
    stretched = ref.copy()
    for i, j in mol.bonds["O-H"]:
        u = stretched[j] - stretched[i]
        stretched[j] = stretched[i] + u * (1.0 + 0.02 / np.linalg.norm(u))

    offs = measure_offsets(mol, stretched)
    table = {k: float(np.mean(v)) for k, v in offs.items()}
    fixed = offset_corrected_geometry(mol, stretched, table)
    assert mol.internal_coordinates(fixed)["O-H"] == pytest.approx(
        mol.internal_coordinates(ref)["O-H"], abs=1e-4)


# ── large-amplitude sigma ──────────────────────────────────────────────────

def _alpha(lam_freq_cm):
    coords = h2o_coords()
    return compute_harmonic_alpha(
        h2o_hessian(coords), coords, H2O_MASSES,
        lam_freq_cm=lam_freq_cm)


def test_lam_threshold_of_zero_changes_nothing():
    """Default off. Every existing result must be reproducible."""
    a0, _, s0, i0 = _alpha(0.0)
    assert i0["lam_modes_cm"] == []
    assert all(v == 0.0 for v in i0["lam_sigma_mhz"].values())


def test_lam_widens_sigma_without_touching_the_correction():
    """Uncertainty, not a different answer.

    A threshold above water's bend catches one real mode, so this exercises the
    mechanism on a molecule whose alpha is known rather than on a synthetic
    Hessian. If the correction itself moved, the upgrade would be silently
    changing the physics instead of describing what it does not know.
    """
    a0, _, s0, _ = _alpha(0.0)
    a1, _, s1, i1 = _alpha(2000.0)

    assert len(i1["lam_modes_cm"]) == 1
    assert i1["lam_modes_cm"][0] < 2000.0
    for k in ("A", "B", "C"):
        assert a1[k] == pytest.approx(a0[k], rel=1e-12)
        assert s1[k] > s0[k]


def test_lam_widening_is_the_flagged_modes_own_contribution():
    """The amount is not a tuning factor: it is what those modes contribute."""
    _, _, s0, _ = _alpha(0.0)
    _, _, s1, i1 = _alpha(2000.0)
    per_mode = np.asarray(i1["alpha_per_mode_mhz"], dtype=float)
    freqs = np.asarray(i1["frequencies_cm"], dtype=float)
    idx = [r for r in range(freqs.size) if freqs[r] < 2000.0]
    for i, k in enumerate(("A", "B", "C")):
        band = float(np.abs(per_mode[i, idx]).sum())
        assert s1[k] == pytest.approx(float(np.hypot(s0[k], band)), rel=1e-9)


def test_per_mode_alpha_sums_to_the_reported_total():
    """The new diagnostic has to be the same quantity the correction uses."""
    a0, _, _, i0 = _alpha(0.0)
    per_mode = np.asarray(i0["alpha_per_mode_mhz"], dtype=float)
    for i, k in enumerate(("A", "B", "C")):
        assert float(per_mode[i].sum()) == pytest.approx(a0[k], rel=1e-10)


# ── rotational g-tensor ────────────────────────────────────────────────────

_HAS_GTENSOR = __import__("importlib").util.find_spec("pyscf.prop") is not None


@pytest.mark.skipif(not _HAS_GTENSOR, reason="pyscf-properties not installed")
def test_ocs_g_bb_matches_the_experimental_value():
    """The one number in this repository with an independent experimental
    reference: correction_models cites g_bb ~ -0.028 for OCS.

    It is a good test precisely because it is small and negative -- a sign
    error, a missing electronic term or a frame mix-up all fail it, where a
    large positive g like water's would hide them.
    """
    from backend.spectral.electronic_g import rotational_g_tensor

    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5614], [0.0, 0.0, 2.7174]])
    masses = np.array([31.97207117, 12.0, 15.99491462])
    g = rotational_g_tensor(["S", "C", "O"], coords, masses,
                            method="hf", basis="cc-pvtz")
    assert g["B"] == pytest.approx(-0.028, abs=0.004)
    # Perpendicular components of a linear molecule are degenerate by symmetry.
    assert g["C"] == pytest.approx(g["B"], abs=1e-6)


@pytest.mark.skipif(not _HAS_GTENSOR, reason="pyscf-properties not installed")
def test_g_tensor_labels_follow_the_moments_not_the_input_frame():
    """Feed the same molecule in a rotated frame; the A/B/C values must not move.

    This is the failure the principal-axis rotation exists to prevent, and it
    is invisible without an explicit test: the numbers stay plausible and only
    the labels are wrong.
    """
    from backend.spectral.electronic_g import rotational_g_tensor

    coords = np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692],
                       [0.0, -0.7572, -0.4692]])
    masses = np.array([15.99491462, 1.00782503, 1.00782503])
    straight = rotational_g_tensor(["O", "H", "H"], coords, masses,
                                   method="hf", basis="6-31g")

    # An arbitrary rotation, applied to the coordinates only.
    t = 0.7
    rot = np.array([[np.cos(t), -np.sin(t), 0.0],
                    [np.sin(t), np.cos(t), 0.0],
                    [0.0, 0.0, 1.0]])
    turned = rotational_g_tensor(["O", "H", "H"], coords @ rot.T, masses,
                                 method="hf", basis="6-31g")
    for k in ("A", "B", "C"):
        assert turned[k] == pytest.approx(straight[k], abs=2e-4)


# ── prior width after the offsets ──────────────────────────────────────────

def test_residual_sigma_is_the_within_class_spread_not_the_raw_error():
    """The number sigma_x must become.

    Two classes, each with a large mean and a small spread. sigma_x should end
    up near the spread, not near the mean -- getting this wrong is what made
    the offsets briefly *hurt* the hybrid: a 5.7 mA prior described as a 20 mA
    one let the spectral residual pull the structure back out to 17.2 mA.
    """
    per_mol = {
        "a": {"C-F": [0.028, 0.030], "C-H": [-0.012]},
        "b": {"C-F": [0.026], "C-H": [-0.010, -0.014]},
    }
    sig = residual_sigma_after_offsets(per_mol, exclude_key="zzz")
    assert 0.0015 < sig < 0.004, sig
    # The raw errors are an order of magnitude larger; a mean-not-subtracted
    # implementation lands up there instead.
    assert sig < 0.010


def test_residual_sigma_excludes_the_scored_molecule():
    per_mol = {"a": {"C-F": [0.10, -0.10]}, "b": {"C-F": [0.028, 0.030]}}
    tight = residual_sigma_after_offsets(per_mol, exclude_key="a")
    loose = residual_sigma_after_offsets(per_mol, exclude_key="b")
    assert tight < loose


def test_singleton_classes_contribute_no_width():
    """One measurement cannot estimate a spread, so it must be skipped rather
    than counted as zero spread (which would claim a perfect prior)."""
    per_mol = {"a": {"C-F": [0.028]}, "b": {"C-Cl": [0.078]}}
    assert residual_sigma_after_offsets(per_mol, "zzz") == pytest.approx(0.002)


def test_a_floor_prevents_an_implausibly_tight_prior():
    per_mol = {"a": {"C-F": [0.02800, 0.02801]}, "b": {"C-F": [0.028005]}}
    assert residual_sigma_after_offsets(per_mol, "zzz", floor_ang=0.003) >= 0.003
