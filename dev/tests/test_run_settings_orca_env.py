import os

import pytest

from runner.run_settings import get_run_settings


@pytest.fixture
def clear_orca_env(monkeypatch):
    for k in (
        "QUANTIZE_ORCA_METHOD",
        "QUANTIZE_ORCA_BASIS",
        "ORCA_METHOD",
        "ORCA_BASIS",
    ):
        monkeypatch.delenv(k, raising=False)
    yield


def test_quantize_orca_env_overrides_molecule_defaults(clear_orca_env, monkeypatch):
    monkeypatch.setenv("QUANTIZE_ORCA_METHOD", "B3LYP")
    monkeypatch.setenv("QUANTIZE_ORCA_BASIS", "def2-SVP")
    s = get_run_settings("water", "BALANCED")
    assert s["orca_method"] == "B3LYP"
    assert s["orca_basis"] == "def2-SVP"


def test_orca_exe_env_seen_each_call(clear_orca_env, monkeypatch):
    monkeypatch.setenv("ORCA_EXE", "/opt/orca/orca")
    s = get_run_settings("co2", "FAST_DEBUG")
    assert s["orca_exe"] == "/opt/orca/orca"
