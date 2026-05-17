"""Streamlit dashboard — live view of demo.py telemetry.

Run:  .venv/bin/streamlit run dashboard/app.py --server.port 8502
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
_DASHBOARD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def _load_ui():
    """Load dashboard/ui.py from disk (avoids stale Streamlit package cache)."""
    path = _DASHBOARD_DIR / "ui.py"
    spec = importlib.util.spec_from_file_location("bridge_dashboard_ui", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load UI module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ui = _load_ui()

from telemetry import DEFAULT_CSV  # noqa: E402

REFRESH_SECONDS = 1.0


@st.cache_data(ttl=0.5, show_spinner=False)
def _load(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    str_cols = [
        "run_id", "mode", "event_type", "role", "doc_id",
        "fingerprint", "cache_hit", "message", "extra_json",
    ]
    df = pd.read_csv(p, dtype={c: "string" for c in str_cols}, keep_default_na=False)
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    for col in ("ttft_s", "total_s", "ts"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "ts" in df.columns:
        df["wall_time"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    if "cache_hit" in df.columns:
        df["cache_hit_bool"] = df["cache_hit"].str.lower() == "true"
    return df


def main() -> None:
    st.set_page_config(
        page_title="Shared Context Bridge",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    ui.inject_theme()

    with st.sidebar:
        st.markdown("### Settings")
        csv_path = st.text_input("Telemetry file", value=str(DEFAULT_CSV))
        auto_refresh = st.toggle("Live refresh", value=True)

    df_all = _load(csv_path)
    if df_all.empty:
        st.markdown(
            '<div class="hero"><h1>Waiting for data</h1>'
            '<p class="hero-tagline">Run <code>python demo.py</code> in another terminal.</p></div>',
            unsafe_allow_html=True,
        )
        if auto_refresh:
            time.sleep(REFRESH_SECONDS)
            st.rerun()
        return

    df, selected = ui.filter_latest_run(df_all)
    with st.sidebar:
        st.caption(f"Showing latest run: **{selected}**")
    before, ours = ui.split_demo_stages(df)
    after_df = df[df["mode"] == "after"] if "mode" in df.columns else df

    ui.render_hero(before, ours, selected)
    ui.render_stage_comparison(before, ours)
    ui.render_pipeline_flow()
    ui.chart_ttft_grouped(before, ours)
    ui.render_control_stats(after_df)
    ui.chart_hot_timeline(after_df)
    ui.render_event_feed(after_df)

    with st.expander("Raw telemetry (last 40 rows)"):
        cols = [c for c in [
            "wall_time", "mode", "event_type", "role", "doc_id",
            "ttft_s", "cache_hit", "message",
        ] if c in df.columns]
        st.dataframe(df[cols].tail(40), use_container_width=True, hide_index=True)

    if auto_refresh:
        time.sleep(REFRESH_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()
