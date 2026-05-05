"""
Tests for rovibrational correction pipeline (M1-M4 + electronic/BOB).

Acceptance criteria from the plan:
  - Formula sign test: known alpha table gives Be = B0 + 0.5*alpha
  - Uncertainty propagation: sigma_eff matches quadrature sum
  - Missing corrections: engine warns and records uncorrected components
  - Isotope specificity: different isotopologues carry different corrections
  - Electronic correction: -(m_e/M_total)*B_obs subtracted from B_e,SE
  - BOB correction: -Σ_a (m_e/m_a)*u_a subtracted, isotope-mass-scaled
"""

import numpy as np
import pytest

from backend.correction_models import (
    parse_correction_table, vpt2_delta_b, propagate_sigma,
    M_ELECTRON_AMU, electronic_delta_b, bob_delta_b,
)
from backend.rovib_corrections import (
    CorrectionRecord,
    CorrectedSpectralTarget,
    resolve_corrections,
    apply_corrections_to_isotopologues,
    validate_correction_quality,
    correction_summary,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _water_isotopologues():
    return [
        {
            "name": "H2-16O",
            "masses": [15.9949, 1.00783, 1.00783],
            "obs_constants": [835840.3, 435351.7, 278138.7],
            "component_indices": [0, 1, 2],
            "sigma_constants": [0.2, 0.2, 0.2],
            "alpha_constants": [-43390.0, 10560.0, 6240.0],
        },
        {
            "name": "H2-18O",
            "masses": [17.9992, 1.00783, 1.00783],
            "obs_constants": [825366.1, 435331.6, 276950.5],
            "component_indices": [0, 1, 2],
            "sigma_constants": [0.2, 0.2, 0.2],
            "alpha_constants": [-42350.0, 10540.0, 6210.0],
        },
    ]


def _water_correction_table():
    return {
        "H2-16O": {
            "A": {"alpha_sum_mhz": -43390.0, "sigma_mhz": 100.0, "method": "VPT2"},
            "B": {"alpha_sum_mhz": 10560.0, "sigma_mhz": 50.0, "method": "VPT2"},
            "C": {"alpha_sum_mhz": 6240.0, "sigma_mhz": 50.0, "method": "VPT2"},
        },
        "H2-18O": {
            "A": {"delta_mhz": -21175.0, "sigma_mhz": 100.0, "method": "VPT2"},
            "B": {"delta_mhz": 5270.0, "sigma_mhz": 50.0, "method": "VPT2"},
            "C": {"delta_mhz": 3105.0, "sigma_mhz": 50.0, "method": "VPT2"},
        },
    }


# ── correction_models tests ───────────────────────────────────────────────────

class TestCorrectionModels:
    def test_vpt2_delta_b_formula(self):
        # DeltaB_vib = 0.5 * alpha_sum
        assert vpt2_delta_b(-43390.0) == pytest.approx(-21695.0)
        assert vpt2_delta_b(10560.0) == pytest.approx(5280.0)
        assert vpt2_delta_b(0.0) == pytest.approx(0.0)

    def test_propagate_sigma_quadrature(self):
        # sigma_eff = sqrt(0.2^2 + 100^2) ≈ 100.0002
        result = propagate_sigma(0.2, 100.0)
        assert result == pytest.approx(np.sqrt(0.2**2 + 100.0**2))

    def test_propagate_sigma_no_corr(self):
        assert propagate_sigma(0.5) == pytest.approx(0.5)

    def test_propagate_sigma_none_skipped(self):
        result = propagate_sigma(0.2, None, 50.0)
        assert result == pytest.approx(np.sqrt(0.2**2 + 50.0**2))

    def test_parse_correction_table_dict(self):
        tbl = parse_correction_table(_water_correction_table())
        assert "H2-16O" in tbl
        assert "A" in tbl["H2-16O"]

    def test_parse_correction_table_none(self):
        assert parse_correction_table(None) == {}

    def test_parse_correction_table_bad_component(self):
        with pytest.raises(ValueError, match="unknown component"):
            parse_correction_table({"H2-16O": {"X": {"delta_mhz": 0.0}}})

    def test_parse_correction_table_missing_delta(self):
        with pytest.raises(ValueError, match="delta_mhz.*alpha_sum_mhz"):
            parse_correction_table({"H2-16O": {"A": {"sigma_mhz": 1.0}}})


# ── Formula sign test (acceptance criterion) ──────────────────────────────────

class TestFormulaSigns:
    def test_alpha_sum_gives_correct_be(self):
        """Known alpha table → Be = B0 + 0.5*alpha for each component."""
        isos = _water_isotopologues()
        targets = resolve_corrections(isos, correction_table=None)

        for t in targets:
            iso = next(i for i in isos if i["name"] == t.isotopologue_label)
            comp_idx = {"A": 0, "B": 1, "C": 2}[t.component]
            k = list(iso["component_indices"]).index(comp_idx)
            b0 = iso["obs_constants"][k]
            alpha = iso["alpha_constants"][k]
            expected_be = b0 + 0.5 * alpha
            assert t.value_mhz == pytest.approx(expected_be, rel=1e-10), (
                f"{t.isotopologue_label}/{t.component}: expected {expected_be}, got {t.value_mhz}"
            )

    def test_user_delta_applied_directly(self):
        """delta_mhz entry → Be = B0 + delta_mhz."""
        isos = [
            {
                "name": "test_iso",
                "masses": [1.0, 1.0],
                "obs_constants": [1000.0],
                "component_indices": [1],
                "sigma_constants": [0.1],
                "alpha_constants": [0.0],
            }
        ]
        ctbl = {"test_iso": {"B": {"delta_mhz": 250.0, "sigma_mhz": 10.0}}}
        targets = resolve_corrections(isos, correction_table=ctbl)
        assert len(targets) == 1
        assert targets[0].value_mhz == pytest.approx(1000.0 + 250.0)

    def test_alpha_sum_mhz_entry(self):
        """alpha_sum_mhz entry → delta = 0.5 * alpha_sum."""
        isos = [
            {
                "name": "test_iso",
                "masses": [1.0, 1.0],
                "obs_constants": [1000.0],
                "component_indices": [1],
                "sigma_constants": [0.1],
                "alpha_constants": [0.0],
            }
        ]
        ctbl = {"test_iso": {"B": {"alpha_sum_mhz": 500.0, "sigma_mhz": 10.0}}}
        targets = resolve_corrections(isos, correction_table=ctbl)
        assert targets[0].value_mhz == pytest.approx(1000.0 + 250.0)


# ── Uncertainty propagation (acceptance criterion) ────────────────────────────

class TestUncertaintyPropagation:
    def test_sigma_eff_quadrature(self):
        """sigma_eff matches quadrature sum of exp and correction uncertainties."""
        isos = _water_isotopologues()
        ctbl = _water_correction_table()
        targets = resolve_corrections(isos, correction_table=ctbl)

        for t in targets:
            expected = np.sqrt(t.sigma_exp_mhz**2 + sum(
                r.sigma_mhz**2 for r in t.correction_records if r.sigma_mhz is not None
            ))
            assert t.sigma_mhz == pytest.approx(expected, rel=1e-10)

    def test_sigma_eff_exceeds_exp_sigma(self):
        """Effective sigma is always >= experimental sigma."""
        isos = _water_isotopologues()
        ctbl = _water_correction_table()
        targets = resolve_corrections(isos, correction_table=ctbl)
        for t in targets:
            assert t.sigma_mhz >= t.sigma_exp_mhz

    def test_unknown_sigma_uses_fraction(self):
        """When sigma_mhz is omitted, sigma_vib_fraction * |delta| is used."""
        isos = [
            {
                "name": "test_iso",
                "masses": [1.0, 1.0],
                "obs_constants": [1000.0],
                "component_indices": [1],
                "sigma_constants": [0.2],
                "alpha_constants": [0.0],
            }
        ]
        # No sigma_mhz in spec → fraction applies
        ctbl = {"test_iso": {"B": {"delta_mhz": 200.0}}}
        targets = resolve_corrections(isos, correction_table=ctbl, sigma_vib_fraction=0.1)
        sigma_vib = 200.0 * 0.1  # = 20.0
        expected = np.sqrt(0.2**2 + 20.0**2)
        assert targets[0].sigma_mhz == pytest.approx(expected)


# ── Isotope specificity (acceptance criterion) ────────────────────────────────

class TestIsotopeSpecificity:
    def test_different_deltas_per_isotopologue(self):
        """Different isotopologues carry independent corrections."""
        isos = _water_isotopologues()
        ctbl = _water_correction_table()
        targets = resolve_corrections(isos, correction_table=ctbl)

        def get(iso_name, comp):
            return next(t for t in targets if t.isotopologue_label == iso_name and t.component == comp)

        t16_A = get("H2-16O", "A")
        t18_A = get("H2-18O", "A")

        # H2-16O A: delta = 0.5 * (-43390) = -21695
        # H2-18O A: delta_mhz = -21175 (direct)
        assert t16_A.total_delta_mhz == pytest.approx(-21695.0)
        assert t18_A.total_delta_mhz == pytest.approx(-21175.0)
        assert t16_A.value_mhz != pytest.approx(t18_A.value_mhz)

    def test_component_indices_preserved(self):
        """Component indices in corrected dicts match the originals."""
        isos = _water_isotopologues()
        ctbl = _water_correction_table()
        targets = resolve_corrections(isos, correction_table=ctbl)
        corrected = apply_corrections_to_isotopologues(isos, targets)

        for orig, corr in zip(isos, corrected):
            assert list(corr["component_indices"]) == list(orig["component_indices"])


# ── apply_corrections_to_isotopologues ───────────────────────────────────────

class TestApplyCorrections:
    def test_alpha_zeroed_after_apply(self):
        isos = _water_isotopologues()
        ctbl = _water_correction_table()
        targets = resolve_corrections(isos, correction_table=ctbl)
        corrected = apply_corrections_to_isotopologues(isos, targets)
        for iso in corrected:
            assert np.allclose(iso["alpha_constants"], 0.0)

    def test_obs_constants_replaced_by_be_se(self):
        isos = _water_isotopologues()
        ctbl = _water_correction_table()
        targets = resolve_corrections(isos, correction_table=ctbl)
        corrected = apply_corrections_to_isotopologues(isos, targets)

        for iso in corrected:
            name = iso["name"]
            idx = list(iso["component_indices"])
            for k, comp in enumerate(idx):
                label = {0: "A", 1: "B", 2: "C"}[comp]
                t = next(t for t in targets if t.isotopologue_label == name and t.component == label)
                assert iso["obs_constants"][k] == pytest.approx(t.value_mhz)

    def test_original_dicts_not_mutated(self):
        isos = _water_isotopologues()
        original_b0 = isos[0]["obs_constants"][0]
        ctbl = _water_correction_table()
        targets = resolve_corrections(isos, correction_table=ctbl)
        apply_corrections_to_isotopologues(isos, targets)
        assert isos[0]["obs_constants"][0] == pytest.approx(original_b0)


# ── Missing corrections (acceptance criterion) ────────────────────────────────

class TestMissingCorrections:
    def test_no_correction_flag(self):
        """Uncorrected components get quality_flag='no_correction'."""
        isos = [
            {
                "name": "bare_iso",
                "masses": [1.0, 1.0],
                "obs_constants": [1000.0],
                "component_indices": [1],
                "sigma_constants": [0.1],
                "alpha_constants": [0.0],
            }
        ]
        targets = resolve_corrections(isos, correction_table=None)
        assert len(targets) == 1
        assert "no_correction" in targets[0].correction_records[0].quality_flags

    def test_mixed_coverage_warning(self):
        """Warning raised when one isotopologue is corrected but another is not."""
        isos = _water_isotopologues()
        # Only correct H2-16O, leave H2-18O uncorrected
        ctbl = {"H2-16O": {"A": {"delta_mhz": -21695.0, "sigma_mhz": 100.0}}}
        # Remove alpha from H2-18O so it gets no correction
        for iso in isos:
            if iso["name"] == "H2-18O":
                iso["alpha_constants"] = [0.0, 0.0, 0.0]
        targets = resolve_corrections(isos, correction_table=ctbl)
        warnings = validate_correction_quality(targets)
        assert any("Mixed correction coverage" in w for w in warnings)

    def test_large_correction_warns_when_sigma_unknown(self):
        """Large correction with unknown sigma raises a quality warning and flags the record."""
        isos = [
            {
                "name": "test_iso",
                "masses": [1.0, 1.0],
                "obs_constants": [1000.0],
                "component_indices": [1],
                "sigma_constants": [0.01],
                "alpha_constants": [0.0],
            }
        ]
        ctbl = {"test_iso": {"B": {"delta_mhz": 500.0}}}
        targets = resolve_corrections(isos, correction_table=ctbl, sigma_vib_fraction=0.0)
        warnings = validate_correction_quality(targets, sigma_ratio_warn=3.0)
        # A human-readable warning is emitted
        assert len(warnings) > 0
        assert any("correction uncertainty is unknown" in w for w in warnings)
        # The quality flag is also recorded on the CorrectionRecord
        flagged = [
            r for t in targets for r in t.correction_records
            if "large_correction_unknown_sigma" in r.quality_flags
        ]
        assert len(flagged) > 0


# ── correction_summary smoke test ─────────────────────────────────────────────

def test_correction_summary_runs():
    isos = _water_isotopologues()
    ctbl = _water_correction_table()
    targets = resolve_corrections(isos, correction_table=ctbl)
    s = correction_summary(targets)
    assert "H2-16O" in s
    assert "Be,SE" in s or "delta" in s.lower()


# ── Electronic mass correction ─────────────────────────────────────────────────

class TestElectronicCorrection:
    def test_formula_sign_negative(self):
        """delta_elec must be negative (electronic mass makes B_obs slightly too small)."""
        delta = electronic_delta_b(b_obs_mhz=435000.0, total_mass_amu=18.01)
        assert delta < 0.0

    def test_formula_magnitude(self):
        """delta_elec = -(m_e / M_total) * B_obs."""
        b_obs = 435351.7
        M = 18.01056
        expected = -(M_ELECTRON_AMU / M) * b_obs
        assert electronic_delta_b(b_obs, M) == pytest.approx(expected, rel=1e-10)

    def test_scales_with_b_obs(self):
        """Larger B gives proportionally larger (more negative) correction."""
        d1 = electronic_delta_b(100_000.0, 18.0)
        d2 = electronic_delta_b(200_000.0, 18.0)
        assert d2 == pytest.approx(2 * d1, rel=1e-10)

    def test_scales_inversely_with_mass(self):
        """Heavier molecule → smaller magnitude correction."""
        d_light = electronic_delta_b(435000.0, 18.0)
        d_heavy = electronic_delta_b(435000.0, 36.0)
        assert abs(d_heavy) == pytest.approx(abs(d_light) / 2, rel=1e-10)

    def test_added_to_resolve_corrections(self):
        """correction_elec=True adds an 'elec' CorrectionRecord to each target."""
        isos = _water_isotopologues()
        elems = ["O", "H", "H"]
        targets = resolve_corrections(
            isos, correction_table=None,
            elems=elems, correction_elec=True, sigma_elec_fraction=0.1,
        )
        for t in targets:
            elec_recs = [r for r in t.correction_records if r.method == "elec"]
            assert len(elec_recs) == 1, f"{t.isotopologue_label}/{t.component} missing elec record"
            assert elec_recs[0].delta_mhz < 0.0

    def test_elec_decreases_be_se(self):
        """Electronic correction makes B_e,SE smaller than B0 + DeltaB_vib."""
        isos = [
            {
                "name": "test",
                "masses": [16.0, 1.0, 1.0],
                "obs_constants": [500_000.0],
                "component_indices": [1],
                "sigma_constants": [0.2],
                "alpha_constants": [0.0],
            }
        ]
        targets_no_elec = resolve_corrections(isos, correction_elec=False)
        targets_elec = resolve_corrections(
            isos, elems=["O", "H", "H"], correction_elec=True
        )
        assert targets_elec[0].value_mhz < targets_no_elec[0].value_mhz

    def test_sigma_elec_propagated(self):
        """sigma_elec_fraction contributes to sigma_eff in quadrature."""
        isos = _water_isotopologues()
        elems = ["O", "H", "H"]
        frac = 0.15
        targets = resolve_corrections(
            isos, correction_elec=True, elems=elems,
            sigma_elec_fraction=frac, sigma_vib_fraction=0.0,
        )
        for t in targets:
            elec_recs = [r for r in t.correction_records if r.method == "elec"]
            assert elec_recs[0].sigma_mhz == pytest.approx(
                abs(elec_recs[0].delta_mhz) * frac, rel=1e-10
            )

    def test_isotopologue_mass_specificity(self):
        """H2-16O and H2-18O get different electronic corrections (different masses)."""
        isos = _water_isotopologues()
        elems = ["O", "H", "H"]
        targets = resolve_corrections(isos, correction_elec=True, elems=elems)
        by_iso = {t.isotopologue_label: t for t in targets if t.component == "B"}
        d16 = next(r.delta_mhz for r in by_iso["H2-16O"].correction_records if r.method == "elec")
        d18 = next(r.delta_mhz for r in by_iso["H2-18O"].correction_records if r.method == "elec")
        # H2-18O is heavier → smaller magnitude correction
        assert abs(d18) < abs(d16)


# ── Born-Oppenheimer Breakdown correction ─────────────────────────────────────

class TestBOBCorrection:
    def test_formula_sign_positive_u(self):
        """For positive u-parameters, delta_bob should be negative (subtracted)."""
        elems = ["O", "H", "H"]
        masses = [15.9949, 1.00783, 1.00783]
        bob_params = {"O": {"B": 0.01}, "H": {"B": 0.005}}
        delta, _ = bob_delta_b(elems, masses, "B", bob_params)
        assert delta < 0.0

    def test_formula_exact(self):
        """delta_bob = -Σ_a (m_e / m_a) * u_a for each atom."""
        elems = ["O", "H"]
        masses = [16.0, 1.0]
        bob_params = {"O": {"B": 1.0}, "H": {"B": 2.0}}
        delta, _ = bob_delta_b(elems, masses, "B", bob_params)
        expected = -(M_ELECTRON_AMU / 16.0) * 1.0 - (M_ELECTRON_AMU / 1.0) * 2.0
        assert delta == pytest.approx(expected, rel=1e-10)

    def test_missing_element_contributes_zero(self):
        """Elements without BOB params contribute nothing to the correction."""
        elems = ["O", "H", "H"]
        masses = [16.0, 1.0, 1.0]
        # Only O in bob_params — H contributes zero
        bob_params = {"O": {"B": 0.01}}
        delta_partial, _ = bob_delta_b(elems, masses, "B", bob_params)
        expected = -(M_ELECTRON_AMU / 16.0) * 0.01
        assert delta_partial == pytest.approx(expected, rel=1e-10)

    def test_missing_component_contributes_zero(self):
        """A component not listed for an element contributes nothing."""
        elems = ["H"]
        masses = [1.0]
        bob_params = {"H": {"B": 0.01}}  # no "A" entry
        delta_A, _ = bob_delta_b(elems, masses, "A", bob_params)
        assert delta_A == pytest.approx(0.0)

    def test_isotope_mass_scaling(self):
        """Replacing H (m=1) with D (m=2) halves the BOB correction for that atom."""
        bob_params = {"H": {"B": 0.01}}
        delta_H, _ = bob_delta_b(["H"], [1.00783], "B", bob_params)
        delta_D, _ = bob_delta_b(["H"], [2.01410], "B", bob_params)
        # Ratio should match mass ratio
        ratio = delta_H / delta_D
        assert ratio == pytest.approx(2.01410 / 1.00783, rel=1e-4)

    def test_sigma_propagation(self):
        """sigma_bob = sqrt(Σ_a ((m_e/m_a)*sigma_u_a)^2)."""
        elems = ["O", "H"]
        masses = [16.0, 1.0]
        bob_params = {
            "O": {"B": {"u": 0.01, "sigma_u": 0.001}},
            "H": {"B": {"u": 0.005, "sigma_u": 0.0005}},
        }
        _, sigma = bob_delta_b(elems, masses, "B", bob_params)
        expected = np.sqrt(
            (M_ELECTRON_AMU / 16.0 * 0.001) ** 2
            + (M_ELECTRON_AMU / 1.0 * 0.0005) ** 2
        )
        assert sigma == pytest.approx(expected, rel=1e-10)

    def test_no_sigma_u_returns_none(self):
        """When no sigma_u supplied, sigma is None."""
        elems = ["H"]
        masses = [1.0]
        bob_params = {"H": {"B": 0.01}}
        _, sigma = bob_delta_b(elems, masses, "B", bob_params)
        assert sigma is None

    def test_added_to_resolve_corrections(self):
        """correction_bob_params adds a 'BOB' CorrectionRecord to each target."""
        isos = _water_isotopologues()
        elems = ["O", "H", "H"]
        bob_params = {
            "O": {"A": 0.001, "B": 0.005, "C": 0.003},
            "H": {"A": 0.0, "B": 0.002, "C": 0.002},
        }
        targets = resolve_corrections(
            isos, correction_table=None,
            elems=elems, correction_bob_params=bob_params,
        )
        for t in targets:
            bob_recs = [r for r in t.correction_records if r.method == "BOB"]
            assert len(bob_recs) == 1, f"{t.isotopologue_label}/{t.component} missing BOB record"

    def test_bob_isotopologue_mass_specificity(self):
        """H2-16O and H2-18O get different BOB corrections (different O masses)."""
        isos = _water_isotopologues()
        elems = ["O", "H", "H"]
        bob_params = {"O": {"B": 0.01}, "H": {"B": 0.002}}
        targets = resolve_corrections(isos, elems=elems, correction_bob_params=bob_params)
        by_iso = {t.isotopologue_label: t for t in targets if t.component == "B"}
        b16 = next(r.delta_mhz for r in by_iso["H2-16O"].correction_records if r.method == "BOB")
        b18 = next(r.delta_mhz for r in by_iso["H2-18O"].correction_records if r.method == "BOB")
        # O-16 vs O-18: heavier O-18 → smaller magnitude O contribution → smaller |delta_bob|
        assert abs(b18) < abs(b16)


# ── Combined corrections end-to-end ───────────────────────────────────────────

class TestCombinedCorrections:
    def test_all_three_record_types_present(self):
        """When all corrections enabled, each target has vib + elec + BOB records."""
        isos = _water_isotopologues()
        elems = ["O", "H", "H"]
        bob_params = {"O": {"A": 0.001, "B": 0.005, "C": 0.003}, "H": {"B": 0.002, "C": 0.002}}
        targets = resolve_corrections(
            isos,
            correction_table=_water_correction_table(),
            elems=elems,
            correction_elec=True,
            correction_bob_params=bob_params,
        )
        for t in targets:
            methods = {r.method for r in t.correction_records}
            assert "VPT2" in methods or "manual" in methods or "none" in methods
            assert "elec" in methods
            assert "BOB" in methods

    def test_total_delta_is_sum_of_records(self):
        """CorrectedSpectralTarget.total_delta_mhz == sum of all record deltas."""
        isos = _water_isotopologues()
        elems = ["O", "H", "H"]
        bob_params = {"O": {"B": 0.005}, "H": {"B": 0.002}}
        targets = resolve_corrections(
            isos,
            correction_table=_water_correction_table(),
            elems=elems,
            correction_elec=True,
            correction_bob_params=bob_params,
        )
        for t in targets:
            expected = sum(r.delta_mhz for r in t.correction_records)
            assert t.total_delta_mhz == pytest.approx(expected, rel=1e-10)
            assert t.value_mhz == pytest.approx(t.b0_mhz + expected, rel=1e-10)

    def test_sigma_eff_quadrature_all_corrections(self):
        """sigma_eff = sqrt(sigma_exp^2 + sum of all correction sigmas^2)."""
        isos = _water_isotopologues()
        elems = ["O", "H", "H"]
        bob_params = {"O": {"B": {"u": 0.005, "sigma_u": 0.001}}, "H": {"B": {"u": 0.002, "sigma_u": 0.0005}}}
        targets = resolve_corrections(
            isos,
            correction_table=_water_correction_table(),
            elems=elems,
            correction_elec=True,
            sigma_elec_fraction=0.1,
            correction_bob_params=bob_params,
        )
        for t in targets:
            corr_sigmas = [r.sigma_mhz for r in t.correction_records if r.sigma_mhz is not None]
            expected = np.sqrt(t.sigma_exp_mhz**2 + sum(s**2 for s in corr_sigmas))
            assert t.sigma_mhz == pytest.approx(expected, rel=1e-10)
