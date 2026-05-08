from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

from runner.usability import ConfigError, load_config, validate_config


ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = ROOT / "configs"
RUNS_DIR = ROOT / "runs"


def _run_cli(args: list[str]) -> tuple[int, str]:
    cmd = [sys.executable, "cli.py", *args]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return int(proc.returncode), proc.stdout


def _list_configs() -> list[Path]:
    if not CONFIGS_DIR.exists():
        return []
    out = sorted(CONFIGS_DIR.glob("*.yaml")) + sorted(CONFIGS_DIR.glob("*.yml")) + sorted(CONFIGS_DIR.glob("*.json"))
    return out


def _list_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()], reverse=True)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


st.set_page_config(page_title="Quantize Dashboard", page_icon="Q", layout="wide")
st.title("Quantize Dashboard")
st.caption("Validate configs, run jobs, and inspect run artifacts.")

tab_validate, tab_run, tab_inspect = st.tabs(["Validate Config", "Run Config", "Inspect Run"])


with tab_validate:
    st.subheader("Validate a Config")
    cfg_options = _list_configs()
    default_idx = 0 if cfg_options else None
    selected = st.selectbox(
        "Choose a config file",
        options=cfg_options,
        index=default_idx,
        format_func=lambda p: str(p.relative_to(ROOT)),
        key="validate_cfg_select",
    )
    uploaded = st.file_uploader("Or upload a YAML/JSON config", type=["yaml", "yml", "json"], key="validate_upload")
    cfg_path = selected

    if uploaded is not None:
        tmp = ROOT / ".tmp_uploaded_validate_config"
        tmp.write_bytes(uploaded.read())
        cfg_path = tmp

    if st.button("Validate", type="primary", key="validate_btn"):
        if cfg_path is None:
            st.error("No config selected.")
        else:
            try:
                cfg = load_config(cfg_path)
                validate_config(cfg)
                st.success("Config is valid.")
                st.json(
                    {
                        "name": cfg.get("name", cfg.get("molecule", "run")),
                        "schema_version": cfg.get("schema_version"),
                        "coordinate_mode": cfg.get("coordinate_mode", "internal"),
                        "elements": len(cfg.get("elements", []) or []),
                        "isotopologues": len(cfg.get("isotopologues", []) or []),
                        "preset": cfg.get("preset", "BALANCED"),
                    }
                )
            except ConfigError as exc:
                st.error(f"Config error: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

    if cfg_path is not None and cfg_path.exists():
        with st.expander("Preview config"):
            st.code(_read_text(cfg_path), language="yaml")


with tab_run:
    st.subheader("Run Quantize")
    cfg_options = _list_configs()
    default_idx = 0 if cfg_options else None
    selected = st.selectbox(
        "Config to run",
        options=cfg_options,
        index=default_idx,
        format_func=lambda p: str(p.relative_to(ROOT)),
        key="run_cfg_select",
    )
    no_run_dir = st.checkbox("Disable managed run directory (--no-run-dir)", value=False)
    if st.button("Run", type="primary", key="run_btn"):
        if selected is None:
            st.error("No config selected.")
        else:
            args = ["run", str(selected.relative_to(ROOT))]
            if no_run_dir:
                args.append("--no-run-dir")
            with st.spinner("Running..."):
                code, output = _run_cli(args)
            st.code(output, language="text")
            if code == 0:
                st.success("Run completed.")
            else:
                st.error(f"Run failed with exit code {code}.")


with tab_inspect:
    st.subheader("Inspect a Run Directory")
    runs = _list_runs()
    default_idx = 0 if runs else None
    run_dir = st.selectbox(
        "Choose a run",
        options=runs,
        index=default_idx,
        format_func=lambda p: str(p.relative_to(ROOT)),
        key="inspect_run_select",
    )
    if run_dir is None:
        st.info("No runs directory found yet.")
    else:
        cols = st.columns(2)
        with cols[0]:
            if st.button("Rebuild Report + Plots", key="rebuild_report_btn"):
                code, output = _run_cli(["report", str(run_dir)])
                st.code(output, language="text")
                if code == 0:
                    st.success("Report rebuild complete.")
                else:
                    st.error(f"Report rebuild failed with exit code {code}.")
        with cols[1]:
            st.write("")

        meta_path = run_dir / "run_metadata.json"
        if meta_path.exists():
            try:
                st.markdown("**Run metadata**")
                st.json(json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception:
                pass

        report_md = run_dir / "report.md"
        report_html = run_dir / "report.html"
        if report_md.exists():
            st.markdown("**report.md**")
            st.markdown(report_md.read_text(encoding="utf-8"))
        elif report_html.exists():
            st.markdown("**report.html**")
            st.caption("HTML report exists; open directly from filesystem for full rendering.")

        plots_dir = run_dir / "plots"
        if plots_dir.exists():
            plot_files = sorted(plots_dir.glob("*.png"))
            if plot_files:
                st.markdown("**Plots**")
                ncols = 2
                for i in range(0, len(plot_files), ncols):
                    row = st.columns(ncols)
                    for j, p in enumerate(plot_files[i : i + ncols]):
                        row[j].image(str(p), caption=p.name, use_container_width=True)

        exports_dir = run_dir / "exports"
        if exports_dir.exists():
            export_files = sorted(exports_dir.glob("*"))
            if export_files:
                st.markdown("**Exports**")
                for p in export_files:
                    st.write(f"- `{p.name}`")
