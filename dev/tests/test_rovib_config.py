"""Tests for backend.rovib_tables.load_rovib_correction_table."""

from __future__ import annotations

import textwrap

import pytest

from backend.rovib_tables import load_rovib_correction_table


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return p


def test_alpha_csv_round_trip(tmp_path):
    p = _write(
        tmp_path,
        "alpha.csv",
        """
        isotopologue,component,alpha_MHz,sigma_alpha_MHz,source,method,basis,status
        parent,A,100.5,0.1,VPT2,CCSD(T),cc-pVTZ,ok
        parent,B,50.2,0.05,VPT2,CCSD(T),cc-pVTZ,ok
        parent,C,25.1,0.02,VPT2,CCSD(T),cc-pVTZ,ok
        """,
    )
    out = load_rovib_correction_table(p)
    assert "parent" in out
    corr = out["parent"]
    assert corr.alpha_A == pytest.approx(100.5)
    assert corr.alpha_B == pytest.approx(50.2)
    assert corr.alpha_C == pytest.approx(25.1)
    # sigma_alpha is converted to sigma_delta = 0.5 * sigma_alpha.
    assert corr.sigma_delta_A == pytest.approx(0.05)
    assert corr.method == "CCSD(T)"


def test_delta_csv_round_trip(tmp_path):
    p = _write(
        tmp_path,
        "delta.csv",
        """
        isotopologue,component,delta_vib_MHz,sigma_delta_vib_MHz,source,method,basis,status
        d2,A,5.0,0.05,user,VPT2,cc-pVTZ,ok
        d2,B,10.0,0.1,user,VPT2,cc-pVTZ,ok
        d2,C,15.0,0.15,user,VPT2,cc-pVTZ,ok
        """,
    )
    out = load_rovib_correction_table(p)
    assert "d2" in out
    corr = out["d2"]
    assert corr.delta_vib_A == pytest.approx(5.0)
    assert corr.delta_vib_B == pytest.approx(10.0)
    assert corr.delta_vib_C == pytest.approx(15.0)
    assert corr.sigma_delta_B == pytest.approx(0.1)


def test_invalid_component_raises(tmp_path):
    p = _write(
        tmp_path,
        "bad.csv",
        """
        isotopologue,component,alpha_MHz,sigma_alpha_MHz,source,method,basis,status
        parent,Z,1.0,0.1,user,VPT2,cc-pVTZ,ok
        """,
    )
    with pytest.raises(ValueError, match="Invalid component"):
        load_rovib_correction_table(p)


def test_negative_uncertainty_raises(tmp_path):
    p = _write(
        tmp_path,
        "neg.csv",
        """
        isotopologue,component,alpha_MHz,sigma_alpha_MHz,source,method,basis,status
        parent,A,1.0,-0.1,user,VPT2,cc-pVTZ,ok
        """,
    )
    with pytest.raises(ValueError, match="Negative uncertainty"):
        load_rovib_correction_table(p)


def test_missing_source_raises(tmp_path):
    p = _write(
        tmp_path,
        "nosrc.csv",
        """
        isotopologue,component,alpha_MHz,sigma_alpha_MHz,source,method,basis,status
        parent,A,1.0,0.1,,VPT2,cc-pVTZ,ok
        """,
    )
    with pytest.raises(ValueError, match="source"):
        load_rovib_correction_table(p)


def test_unknown_isotopologue_raises(tmp_path):
    p = _write(
        tmp_path,
        "alpha.csv",
        """
        isotopologue,component,alpha_MHz,sigma_alpha_MHz,source,method,basis,status
        parent,A,1.0,0.1,user,VPT2,cc-pVTZ,ok
        """,
    )
    with pytest.raises(ValueError, match="not in known list"):
        load_rovib_correction_table(p, known_isotopologues={"d2"})


def test_two_isotopologues_get_distinct_corrections(tmp_path):
    p = _write(
        tmp_path,
        "two.csv",
        """
        isotopologue,component,alpha_MHz,sigma_alpha_MHz,source,method,basis,status
        parent,A,100.0,0.1,VPT2,CCSD(T),cc-pVTZ,ok
        parent,B,50.0,0.05,VPT2,CCSD(T),cc-pVTZ,ok
        d1,A,90.0,0.1,VPT2,CCSD(T),cc-pVTZ,ok
        d1,B,45.0,0.05,VPT2,CCSD(T),cc-pVTZ,ok
        """,
    )
    out = load_rovib_correction_table(p)
    assert set(out.keys()) == {"parent", "d1"}
    assert out["parent"].alpha_A == pytest.approx(100.0)
    assert out["d1"].alpha_A == pytest.approx(90.0)


def test_unknown_layout_raises(tmp_path):
    p = _write(
        tmp_path,
        "weird.csv",
        """
        isotopologue,component,unrelated_field
        parent,A,1.0
        """,
    )
    with pytest.raises(ValueError, match="alpha_MHz"):
        load_rovib_correction_table(p)
