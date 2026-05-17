"""Shared Context Bridge — Streamlit in Snowflake.

Deploy via scripts/deploy_sis.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import streamlit as st
from snowflake.snowpark.context import get_active_session

_DASHBOARD_DIR = Path(__file__).resolve().parent


def _load_ui():
    path = _DASHBOARD_DIR / "ui.py"
    spec = importlib.util.spec_from_file_location("bridge_dashboard_ui", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load UI module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ui = _load_ui()

session = get_active_session()


@st.cache_data(ttl=5, show_spinner=False)
def load_latest_run_id() -> str | None:
    rows = session.sql(
        "SELECT run_id "
        "FROM BRIDGE_DB.TELEMETRY.EVENTS "
        "GROUP BY run_id "
        "ORDER BY MAX(ts) DESC "
        "LIMIT 1"
    ).collect()
    return rows[0][0] if rows else None


@st.cache_data(ttl=5, show_spinner=False)
def load_events(run_id: str) -> pd.DataFrame:
    df = session.sql(
        "SELECT ts, mode, event_type, role, doc_id, fingerprint, "
        "       ttft_s, total_s, cache_hit, message, extra_json "
        "FROM BRIDGE_DB.TELEMETRY.EVENTS "
        "WHERE run_id = ? ORDER BY ts",
        params=[run_id],
    ).to_pandas()
    df.columns = [c.lower() for c in df.columns]
    df["cache_hit_bool"] = df["cache_hit"].astype(str).str.lower() == "true"
    for col in ("ttft_s", "total_s", "ts"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "ts" in df.columns:
        df["wall_time"] = pd.to_datetime(df["ts"], unit="s", errors="coerce")
    return df


@st.cache_data(ttl=10, show_spinner=False)
def load_leaderboard() -> pd.DataFrame:
    df = session.sql(
        "SELECT run_id, "
        "  TO_VARCHAR(TO_TIMESTAMP_NTZ(started_at::NUMBER),'YYYY-MM-DD HH24:MI:SS') AS started, "
        "  before_ttft_total_s, after_ttft_total_s, "
        "  gpu_seconds_saved, speedup_x, after_hit_rate_pct "
        "FROM BRIDGE_DB.TELEMETRY.RUN_SUMMARY "
        "ORDER BY gpu_seconds_saved DESC NULLS LAST"
    ).to_pandas()
    df.columns = [c.lower() for c in df.columns]
    return df


def get_cortex_narrative(run_id: str) -> str:
    result = session.sql(
        """
        WITH timeline AS (
            SELECT ROW_NUMBER() OVER (ORDER BY ts) AS step,
                   event_type, role, ROUND(ttft_s,3) AS ttft_s,
                   cache_hit, message
            FROM   BRIDGE_DB.TELEMETRY.EVENTS
            WHERE  run_id = ? AND mode = 'after'
              AND  event_type IN ('decision','task_completed','keep_resident_completed')
        )
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            'llama3.1-70b',
            'Explain this agentic-AI control plane run to a hackathon judge in 4-5 sentences. '
            || 'Cover agents in order, proactive warmups, TTFT, and one memorable takeaway.\n\n'
            || LISTAGG(
                   step || '. [' || event_type || ']'
                   || COALESCE(' role=' || role, '')
                   || COALESCE(' ttft=' || ttft_s::STRING || 's', '')
                   || COALESCE(' — ' || message, ''),
                   '\n'
               ) WITHIN GROUP (ORDER BY step)
        ) FROM timeline
        """,
        params=[run_id],
    ).collect()
    return result[0][0] if result else "No AFTER events for this run."


def main() -> None:
    st.set_page_config(
        page_title="Shared Context Bridge",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ui.inject_theme()

    selected = load_latest_run_id()
    if not selected:
        st.warning("No runs yet. Run `python demo.py` and sync telemetry to Snowflake.")
        return

    with st.sidebar:
        st.markdown("### Snowflake")
        st.caption(f"Latest run: **{selected}**")
        if st.button("Refresh", type="primary"):
            st.cache_data.clear()
            st.rerun()

    df = load_events(selected)
    before, ours = ui.split_demo_stages(df)
    after_df = df[df["mode"] == "after"] if "mode" in df.columns else df

    ui.render_hero(before, ours, selected)
    ui.render_stage_comparison(before, ours)
    ui.render_pipeline_flow()
    ui.chart_ttft_grouped(before, ours)
    ui.render_control_stats(after_df)
    ui.chart_hot_timeline(after_df)
    ui.render_event_feed(after_df)

    with st.expander("All runs leaderboard"):
        lb = load_leaderboard()
        if not lb.empty:
            st.dataframe(lb, use_container_width=True)

    with st.expander("Cortex AI narrator"):
        st.caption("Llama-70B inside Snowflake reads your orchestrator log.")
        if st.button("Generate pitch narrative"):
            with st.spinner("Calling CORTEX.COMPLETE…"):
                st.success(get_cortex_narrative(selected))


main()
