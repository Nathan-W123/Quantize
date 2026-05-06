"""
Tests for backend.scan_fit.

Covers:
  - energies_to_cm1: unit conversions
  - validate_scan_coverage: coverage, gap, duplicate-endpoint detection
  - fit_fourier_potential: least-squares recovery of known Fourier coefficients
  - scan_to_torsion_potential: symmetry filtering, cosine_only default
  - scan_fit_diagnostics: residuals and RMS
  - export_scan_fit_csv: output file format
  - ingest_scan_csv: roundtrip CSV read
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.scan_fit import (
    energies_to_cm1,
    export_scan_fit_csv,
    fit_fourier_potential,
    ingest_scan_csv,
    scan_fit_diagnostics,
    scan_to_torsion_potential,
    validate_scan_coverage,
)
from backend.torsion_hamiltonian import TorsionFourierPotential


# ── energies_to_cm1 ───────────────────────────────────────────────────────────

class TestEnergiesToCm1:
    def test_cm1_passthrough(self):
        e = np.array([0.0, 100.0, 200.0])
        np.testing.assert_allclose(energies_to_cm1(e, "cm-1"), e)

    def test_hartree_conversion(self):
        # 1 hartree = 219474.6313705 cm^-1
        e = energies_to_cm1(np.array([1.0]), "hartree")
        assert float(e[0]) == pytest.approx(219474.6313705, rel=1e-8)

    def test_kcal_mol_conversion(self):
        e = energies_to_cm1(np.array([1.0]), "kcal/mol")
        assert float(e[0]) == pytest.approx(349.7550874793, rel=1e-8)

    def test_kj_mol_conversion(self):
        e = energies_to_cm1(np.array([1.0]), "kj/mol")
        assert float(e[0]) == pytest.approx(83.5934722514, rel=1e-8)

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown energy unit"):
            energies_to_cm1(np.array([1.0]), "eV")

    def test_ha_alias(self):
        e1 = energies_to_cm1(np.array([1.0]), "hartree")
        e2 = energies_to_cm1(np.array([1.0]), "ha")
        np.testing.assert_allclose(e1, e2)


# ── validate_scan_coverage ────────────────────────────────────────────────────

class TestValidateScanCoverage:
    def _full_c3_scan(self, n=12):
        phi = np.linspace(0, 2 * np.pi / 3, n, endpoint=False)
        e = np.zeros(n)
        return phi, e

    def test_valid_scan_ok_true(self):
        phi, e = self._full_c3_scan(12)
        result = validate_scan_coverage(phi, e, period_rad=2 * np.pi / 3)
        assert result["ok"] is True
        assert len(result["errors"]) == 0

    def test_n_points_returned(self):
        phi, e = self._full_c3_scan(10)
        result = validate_scan_coverage(phi, e, period_rad=2 * np.pi / 3)
        assert result["n_points"] == 10

    def test_too_few_points_is_error(self):
        phi = np.linspace(0, 2 * np.pi / 3, 3, endpoint=False)
        result = validate_scan_coverage(phi, period_rad=2 * np.pi / 3, min_points=5)
        assert result["ok"] is False
        assert any("3" in err for err in result["errors"])

    def test_undercoverage_warns(self):
        # Only half the period covered
        phi = np.linspace(0, np.pi / 3, 8)
        result = validate_scan_coverage(phi, period_rad=2 * np.pi / 3)
        assert any("%" in w for w in result["warnings"])

    def test_large_gap_warns(self):
        # 10 points clustered in first quarter, then one outlier far away
        phi = np.concatenate([np.linspace(0, 0.5, 10), [2.0]])
        result = validate_scan_coverage(phi, period_rad=2 * np.pi, max_gap_frac=0.25)
        assert any("gap" in w.lower() for w in result["warnings"])

    def test_duplicate_endpoint_warns(self):
        # First and last point are one period apart → duplicate
        period = 2 * np.pi / 3
        phi = np.linspace(0, period, 10, endpoint=True)  # phi[0]=0, phi[-1]=period → duplicate
        result = validate_scan_coverage(phi, period_rad=period, endpoint_tol_rad=0.05)
        assert result["has_duplicate_endpoint"] is True
        assert any("duplicate" in w.lower() for w in result["warnings"])

    def test_no_duplicate_for_non_periodic_endpoint(self):
        period = 2 * np.pi / 3
        phi = np.linspace(0, period * 0.95, 12)
        result = validate_scan_coverage(phi, period_rad=period)
        assert result["has_duplicate_endpoint"] is False

    def test_energy_length_mismatch_is_error(self):
        phi = np.linspace(0, 2 * np.pi, 12)
        e = np.zeros(10)
        result = validate_scan_coverage(phi, e)
        assert result["ok"] is False
        assert any("length" in err.lower() for err in result["errors"])

    def test_nonfinite_energy_is_error(self):
        phi = np.linspace(0, 2 * np.pi, 10)
        e = np.zeros(10)
        e[3] = np.nan
        result = validate_scan_coverage(phi, e)
        assert result["ok"] is False


# ── fit_fourier_potential ─────────────────────────────────────────────────────

class TestFitFourierPotential:
    def _v3_scan(self, n=36, V3=373.5, noise=0.0, rng_seed=0):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        e = 0.5 * V3 * (1 - np.cos(3 * phi))
        if noise > 0.0:
            rng = np.random.default_rng(rng_seed)
            e += rng.normal(scale=noise, size=n)
        return phi, e

    def test_recovers_v3_barrier(self):
        V3 = 373.554746
        phi, e = self._v3_scan(n=36, V3=V3)
        result = fit_fourier_potential(phi, e, include_harmonics=[3], cosine_only=True)
        # vcos[3] = -V3/2
        assert result["vcos"][3] == pytest.approx(-V3 / 2, rel=1e-6)
        assert result["rms_cm1"] == pytest.approx(0.0, abs=1e-6)
        assert result["fit_ok"] is True

    def test_recovers_v3_and_v6(self):
        V3, V6 = 373.554746, -1.319650
        phi = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        # Direct Fourier form
        e = 186.117548 + (-186.777373) * np.cos(3 * phi) + 0.659825 * np.cos(6 * phi)
        result = fit_fourier_potential(phi, e, include_harmonics=[3, 6], cosine_only=True)
        assert result["vcos"][3] == pytest.approx(-186.777373, rel=1e-6)
        assert result["vcos"][6] == pytest.approx(0.659825, rel=1e-6)
        assert result["rms_cm1"] == pytest.approx(0.0, abs=1e-6)

    def test_zero_at_minimum_shifts_v0(self):
        phi, e = self._v3_scan(n=36, V3=373.5)
        result = fit_fourier_potential(phi, e, include_harmonics=[3], cosine_only=True,
                                       zero_at_minimum=True)
        # Fitted V should be >= 0 at all fine grid points
        phi_fine = np.linspace(0, 2 * np.pi, 1801)
        V_fine = result["v0"] + result["vcos"][3] * np.cos(3 * phi_fine)
        assert float(np.min(V_fine)) >= -1e-6  # at or above zero

    def test_residuals_near_zero_for_exact_data(self):
        phi, e = self._v3_scan(n=36, V3=200.0)
        result = fit_fourier_potential(phi, e, include_harmonics=[3], cosine_only=True)
        np.testing.assert_allclose(result["residuals_cm1"], 0.0, atol=1e-6)

    def test_underdetermined_warns(self):
        phi = np.array([0.0, 1.0])  # 2 points
        e = np.array([0.0, 1.0])
        result = fit_fourier_potential(phi, e, n_harmonics=6, cosine_only=False)
        assert any("underdetermined" in w.lower() for w in result["warnings"])

    def test_harmonics_list_used(self):
        phi, e = self._v3_scan(n=36, V3=200.0)
        result = fit_fourier_potential(phi, e, include_harmonics=[3, 6], cosine_only=True)
        assert set(result["vcos"].keys()) == {3, 6}
        assert result["vsin"] == {}

    def test_sin_terms_included_by_default(self):
        phi, e = self._v3_scan(n=36)
        result = fit_fourier_potential(phi, e, include_harmonics=[3], cosine_only=False)
        assert 3 in result["vsin"]

    def test_n_points_and_params_returned(self):
        phi, e = self._v3_scan(n=36)
        result = fit_fourier_potential(phi, e, include_harmonics=[3], cosine_only=True)
        assert result["n_points"] == 36
        assert result["n_params"] == 2  # v0 + vcos3


# ── scan_to_torsion_potential ─────────────────────────────────────────────────

class TestScanToTorsionPotential:
    def _c3_scan(self, n=36, V3=373.5):
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        e = 0.5 * V3 * (1 - np.cos(3 * phi))
        return phi, e

    def test_returns_torsion_fourier_potential(self):
        phi, e = self._c3_scan()
        pot, result = scan_to_torsion_potential(phi, e, symmetry_number=3)
        assert isinstance(pot, TorsionFourierPotential)

    def test_symmetry_number_3_only_uses_3fold_harmonics(self):
        phi, e = self._c3_scan()
        pot, result = scan_to_torsion_potential(phi, e, n_harmonics=3, symmetry_number=3)
        # With n_harmonics=3 and symmetry_number=3: harmonics 3, 6, 9
        expected_harmonics = [3, 6, 9]
        assert result["harmonics"] == expected_harmonics

    def test_cosine_only_default_for_c3(self):
        phi, e = self._c3_scan()
        pot, result = scan_to_torsion_potential(phi, e, symmetry_number=3)
        assert result["cosine_only"] is True
        assert all(v == 0.0 for v in pot.vsin.values())

    def test_symmetry_number_1_uses_all_harmonics(self):
        phi = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        e = np.zeros(36)
        _, result = scan_to_torsion_potential(phi, e, n_harmonics=4, symmetry_number=1)
        assert result["harmonics"] == [1, 2, 3, 4]

    def test_v3_recovery_via_wrapper(self):
        V3 = 373.554746
        phi = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        e = 0.5 * V3 * (1 - np.cos(3 * phi))
        pot, result = scan_to_torsion_potential(phi, e, n_harmonics=1, symmetry_number=3)
        assert pot.vcos[3] == pytest.approx(-V3 / 2, rel=1e-6)


# ── scan_fit_diagnostics ──────────────────────────────────────────────────────

class TestScanFitDiagnostics:
    def test_perfect_fit_zero_residuals(self):
        v0, vcos3 = 186.0, -186.8
        phi = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        e = v0 + vcos3 * np.cos(3 * phi)
        diag = scan_fit_diagnostics(phi, e, v0=v0, vcos={3: vcos3}, vsin={})
        np.testing.assert_allclose(diag["residuals_cm1"], 0.0, atol=1e-10)
        assert diag["rms_cm1"] == pytest.approx(0.0, abs=1e-10)
        assert diag["max_abs_residual_cm1"] == pytest.approx(0.0, abs=1e-10)

    def test_constant_offset_gives_constant_residual(self):
        phi = np.linspace(0, 2 * np.pi, 18, endpoint=False)
        e = np.ones(18) * 100.0  # flat; fit v0=50 → residual = 50 everywhere
        diag = scan_fit_diagnostics(phi, e, v0=50.0, vcos={}, vsin={})
        np.testing.assert_allclose(diag["residuals_cm1"], 50.0, atol=1e-10)
        assert diag["rms_cm1"] == pytest.approx(50.0, abs=1e-10)

    def test_n_points_correct(self):
        phi = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        e = np.zeros(24)
        diag = scan_fit_diagnostics(phi, e, v0=0.0, vcos={}, vsin={})
        assert diag["n_points"] == 24


# ── export_scan_fit_csv ───────────────────────────────────────────────────────

class TestExportScanFitCsv:
    def test_csv_has_correct_columns(self):
        phi = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        e = 100.0 * (1 - np.cos(3 * phi))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scan_fit.csv"
            export_scan_fit_csv(path, phi, e, v0=100.0, vcos={3: -100.0}, vsin={})
            assert path.is_file()
            with open(path, newline="") as fh:
                reader = csv.DictReader(fh)
                cols = reader.fieldnames
            assert cols is not None
            assert set(cols) == {"phi_deg", "energy_cm1", "V_fitted_cm1", "residual_cm1"}

    def test_csv_n_rows_matches_scan(self):
        n = 18
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        e = np.zeros(n)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.csv"
            export_scan_fit_csv(path, phi, e, v0=0.0, vcos={}, vsin={})
            with open(path, newline="") as fh:
                rows = list(csv.DictReader(fh))
            assert len(rows) == n

    def test_csv_phi_deg_is_degrees(self):
        phi = np.array([0.0, np.pi / 2])
        e = np.zeros(2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.csv"
            export_scan_fit_csv(path, phi, e, v0=0.0, vcos={}, vsin={})
            with open(path, newline="") as fh:
                rows = list(csv.DictReader(fh))
            assert float(rows[1]["phi_deg"]) == pytest.approx(90.0, abs=0.01)


# ── ingest_scan_csv ───────────────────────────────────────────────────────────

class TestIngestScanCsv:
    def _write_csv(self, path, phi_deg, energies_cm1):
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["phi_deg", "energy_cm1"])
            writer.writeheader()
            for p, e in zip(phi_deg, energies_cm1):
                writer.writerow({"phi_deg": p, "energy_cm1": e})

    def test_roundtrip(self):
        phi_deg = np.linspace(0, 360, 12, endpoint=False)
        e_cm1 = np.sin(np.deg2rad(phi_deg)) * 100.0
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scan.csv"
            self._write_csv(path, phi_deg, e_cm1)
            phi_out, e_out = ingest_scan_csv(path)
        np.testing.assert_allclose(phi_out, np.deg2rad(phi_deg), atol=1e-10)
        np.testing.assert_allclose(e_out, e_cm1, atol=1e-10)

    def test_missing_phi_col_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.csv"
            with open(path, "w") as fh:
                fh.write("wrong_col,energy_cm1\n0.0,0.0\n")
            with pytest.raises(ValueError, match="phi_deg"):
                ingest_scan_csv(path)

    def test_energy_unit_hartree(self):
        phi_deg = np.array([0.0, 180.0])
        e_ha = np.array([0.0, 0.001])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scan.csv"
            with open(path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["phi_deg", "energy_cm1"])
                writer.writeheader()
                for p, e in zip(phi_deg, e_ha):
                    writer.writerow({"phi_deg": p, "energy_cm1": e})
            _, e_out = ingest_scan_csv(path, energy_unit="hartree")
        expected_cm1 = e_ha * 219474.6313705
        np.testing.assert_allclose(e_out, expected_cm1, rtol=1e-8)

    def test_radians_angle_unit(self):
        phi_rad = np.linspace(0, np.pi, 6)
        e = np.zeros(6)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scan.csv"
            with open(path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["phi_deg", "energy_cm1"])
                writer.writeheader()
                for p, en in zip(phi_rad, e):
                    writer.writerow({"phi_deg": p, "energy_cm1": en})
            phi_out, _ = ingest_scan_csv(path, angle_unit="radians")
        np.testing.assert_allclose(phi_out, phi_rad, atol=1e-12)
