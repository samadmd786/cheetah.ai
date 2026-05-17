"""Streamlit dashboard for the Shared Context Bridge demo.

Reads `logs/telemetry.csv` live and renders the panels CLAUDE.md §6 calls for:

  1. Per-agent TTFT, BEFORE vs AFTER (line chart) — the headline.
  2. GPU-seconds saved + speedup — hero metrics.
  3. Orchestrator decision log — chronological feed of every `decision` and
     `keep_resident_completed` event, color-coded HIT/MISS.

Auto-refreshes every second so a live run is visible without the page being
reloaded by hand.

Run:  .venv/bin/streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from telemetry import DEFAULT_CSV  # noqa: E402


REFRESH_SECONDS = 1.0
ROLE_ORDER = ["Screener", "Analyst", "Auditor"]
CACHE_MISS_TTFT_SECONDS = 3.0  # mirrors bridge.py; visible as a reference line

# Palette: BEFORE is the "stateless / hot" baseline (warm red), AFTER is the
# bridge result (cool green). Picked for colorblind contrast.
COLOR_BEFORE = "#e45756"
COLOR_AFTER = "#54a24b"


# ---------------------------------------------------------------------- styling

_GLOBAL_CSS = """
<style>
  /* Tighter top padding so the hero metric sits closer to the title. */
  .block-container { padding-top: 2.2rem; padding-bottom: 2rem; }

  /* Hero metric (the big "GPU-seconds saved" number). */
  div[data-testid="stMetric"] {
    background: rgba(84, 162, 75, 0.07);
    border: 1px solid rgba(84, 162, 75, 0.25);
    border-radius: 12px;
    padding: 14px 18px;
  }
  div[data-testid="stMetricValue"] { font-size: 2.2rem; }
  div[data-testid="stMetricDelta"] { font-size: 1.0rem; }

  /* Decision log code block: dark background, generous line-height. */
  .decision-log code {
    font-size: 0.86rem !important;
    line-height: 1.55 !important;
    white-space: pre-wrap !important;
  }

  /* Subtle row striping inside the decision log. */
  .decision-log .line { display:block; padding: 1px 0; }
  .decision-log .hit  { color: #54a24b; }
  .decision-log .miss { color: #e45756; }
  .decision-log .obs  { color: #c4c4c4; }
  .decision-log .act  { color: #f4c542; }
  .decision-log .meta { color: #8a8a8a; }

  /* Pill badges shown next to the run_id and pipeline. */
  .pill {
    display: inline-block;
    padding: 2px 10px;
    margin-right: 6px;
    border-radius: 999px;
    background: rgba(120,120,120,0.18);
    font-size: 0.78rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .pill-after  { background: rgba(84,162,75,0.20);  color: #54a24b; }
  .pill-before { background: rgba(228,87,86,0.20);  color: #e45756; }
</style>
"""


# ------------------------------------------------------------------------- data

@st.cache_data(ttl=0.5, show_spinner=False)
def _load(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    # Force string columns to load as strings so empty cells become "" instead
    # of NaN (a float that would slip through `row.get("col") or ""`).
    str_cols = [
        "run_id", "mode", "event_type", "role", "doc_id",
        "fingerprint", "cache_hit", "message", "extra_json",
    ]
    df = pd.read_csv(
        p, dtype={c: "string" for c in str_cols}, keep_default_na=False
    )
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


def _s(row: pd.Series, key: str) -> str:
    v = row.get(key, "")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v)


# ------------------------------------------------------------------------ panels

def _hero(df: pd.DataFrame) -> None:
    tasks = df[df["event_type"] == "task_completed"].copy()
    before_total = float(tasks.loc[tasks["mode"] == "before", "total_s"].sum())
    after_total = float(tasks.loc[tasks["mode"] == "after", "total_s"].sum())
    saved = max(0.0, before_total - after_total)
    speedup = (before_total / after_total) if after_total > 0 else float("nan")

    n_warmups = int(
        (df["event_type"] == "keep_resident_completed").sum()
    )
    warm_hits = int(
        ((df["event_type"] == "keep_resident_completed")
         & (df["cache_hit_bool"] if "cache_hit_bool" in df else False)).sum()
    )
    n_evictions = int((df["event_type"] == "eviction").sum())
    n_near_dups = int((df["event_type"] == "near_duplicate_detected").sum())

    c1, c2, c3, c4, c5, c6 = st.columns([2, 1, 1, 1, 1, 1])
    with c1:
        st.metric(
            label="GPU-seconds saved (per pipeline)",
            value=f"{saved:6.2f} s",
            delta=(f"{speedup:.2f}× faster" if speedup == speedup else None),
        )
    with c2:
        st.metric("BEFORE wall-clock", f"{before_total:6.2f} s")
    with c3:
        st.metric("AFTER wall-clock", f"{after_total:6.2f} s")
    with c4:
        st.metric(
            label="Warmups (HIT / total)",
            value=f"{warm_hits} / {n_warmups}",
            delta="keep-resident",
            delta_color="off",
        )
    with c5:
        st.metric(
            label="LRU evictions",
            value=str(n_evictions),
            delta="budget-aware",
            delta_color="off",
        )
    with c6:
        st.metric(
            label="Near-dup matches",
            value=str(n_near_dups),
            delta="SimHash",
            delta_color="off",
        )


def _ttft_line_chart(df: pd.DataFrame) -> None:
    st.subheader("Time-to-first-token, per agent")
    tasks = df[df["event_type"] == "task_completed"].copy()
    if tasks.empty:
        st.info("No `task_completed` rows yet. Run `python run.py` to populate.")
        return

    # Keep the most recent TTFT per (mode, role); render the agents in pipeline
    # order on the x-axis so the slope tells the story.
    long = (
        tasks.sort_values("ts")
        .drop_duplicates(["mode", "role"], keep="last")
        .loc[:, ["mode", "role", "ttft_s"]]
    )
    # Use the order observed in this run as the categorical order so multi-doc
    # pipelines (Phase 3) with different roles still render coherently.
    role_order = (
        tasks.sort_values("ts")["role"].dropna().drop_duplicates().tolist()
        or ROLE_ORDER
    )

    single_mode = long["mode"].nunique() < 2
    if single_mode:
        only_mode = long["mode"].iloc[0].upper()
        st.caption(
            f"Showing **{only_mode}** only — this run did not include the "
            f"other mode. Re-run with `--mode both` for a side-by-side."
        )
    long["mode_label"] = long["mode"].str.upper()

    x_enc = alt.X(
        "role:N",
        sort=role_order,
        scale=alt.Scale(domain=role_order),
        title="agent",
        axis=alt.Axis(labelAngle=0, labelFontSize=13, titleFontSize=13),
    )
    y_enc = alt.Y(
        "ttft_s:Q",
        title="TTFT (seconds, log scale)",
        scale=alt.Scale(type="log", domainMin=0.1),
        axis=alt.Axis(format=".2f", titleFontSize=13),
    )

    base = alt.Chart(long).encode(
        x=x_enc,
        y=y_enc,
        color=alt.Color(
            "mode_label:N",
            scale=alt.Scale(
                domain=["BEFORE", "AFTER"],
                range=[COLOR_BEFORE, COLOR_AFTER],
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
        tooltip=[
            alt.Tooltip("role:N", title="agent"),
            alt.Tooltip("mode_label:N", title="mode"),
            alt.Tooltip("ttft_s:Q", title="TTFT (s)", format=".3f"),
        ],
    )

    lines = base.mark_line(strokeWidth=3, interpolate="monotone")
    points = base.mark_point(filled=True, size=170, opacity=1.0)

    # Per-mode label layers — BEFORE labels sit ABOVE the point, AFTER labels
    # sit BELOW, so on agents where the two values are close (Agent 1 in cold
    # mode, ~35s both) the text can never collide. We also drop a soft white
    # halo behind each label so it stays legible over the grid.
    def _label_layer(mode_label: str, dy: int, color: str):
        sub = long[long["mode_label"] == mode_label]
        if sub.empty:
            return None
        enc = dict(
            x=x_enc,
            y=y_enc,
            text=alt.Text("ttft_s:Q", format=".2f"),
        )
        halo = (
            alt.Chart(sub)
            .mark_text(
                align="center", baseline="middle", dx=0, dy=dy,
                fontSize=12, fontWeight="bold",
                stroke="white", strokeWidth=4, strokeOpacity=0.85,
            )
            .encode(**enc)
        )
        text = (
            alt.Chart(sub)
            .mark_text(
                align="center", baseline="middle", dx=0, dy=dy,
                fontSize=12, fontWeight="bold", color=color,
            )
            .encode(**enc)
        )
        return halo + text

    label_layers = [
        layer for layer in (
            _label_layer("BEFORE", dy=-16, color=COLOR_BEFORE),
            _label_layer("AFTER",  dy=+18, color=COLOR_AFTER),
        ) if layer is not None
    ]

    # Reference line at the cache-MISS / HIT threshold (3s in bridge.py).
    threshold_df = pd.DataFrame({"y": [CACHE_MISS_TTFT_SECONDS]})
    ref = (
        alt.Chart(threshold_df)
        .mark_rule(strokeDash=[4, 4], color="#888", opacity=0.6)
        .encode(y=alt.Y("y:Q", scale=alt.Scale(type="log", domainMin=0.1)))
    )
    # Use the same field name ("role") as every other layer so Vega-Lite
    # merges the x-scale into a single ordinal axis honoring `role_order`
    # instead of falling back to alphabetical.
    ref_label = (
        alt.Chart(pd.DataFrame({
            "y": [CACHE_MISS_TTFT_SECONDS],
            "role": [role_order[-1]],
        }))
        .mark_text(
            text=f"cache-MISS threshold ({CACHE_MISS_TTFT_SECONDS:g}s)",
            align="right", dx=-4, dy=-6,
            color="#888", fontSize=11,
        )
        .encode(x=x_enc, y=alt.Y("y:Q", scale=alt.Scale(type="log", domainMin=0.1)))
    )

    layers = [ref, ref_label, lines, points, *label_layers]
    chart = (
        alt.layer(*layers)
        .properties(height=380, padding={"top": 20, "bottom": 16})
        .configure_axis(grid=True, gridOpacity=0.15)
        .configure_view(strokeWidth=0)
        .interactive(bind_y=False)
    )
    st.altair_chart(chart, use_container_width=True)

    # Small companion table beneath the chart.
    pivot = (
        long.pivot_table(index="role", columns="mode", values="ttft_s", aggfunc="last")
        .reindex(role_order)
    )
    if {"before", "after"}.issubset(pivot.columns):
        pivot["speedup"] = pivot["before"] / pivot["after"].replace(0, pd.NA)
    st.dataframe(
        pivot.round(3).rename_axis(index="agent"),
        use_container_width=True,
    )


def _hot_set_timeline(df: pd.DataFrame) -> None:
    """Phase 3: visualize each doc's activity (observations + warmups + the
    eviction events that recycle the hot-set) over wall-clock time. With one
    doc this is a single horizontal lane; with multi-doc pipelines you can
    see eviction strikes happening on whichever doc is the LRU at the time.
    """
    activity = df[df["event_type"].isin([
        "task_completed", "keep_resident_completed", "eviction",
        "near_duplicate_detected",
    ])].copy()
    if activity.empty or activity["doc_id"].str.strip().eq("").all():
        return  # Phase 1/2-style single-doc runs: nothing to show here.

    st.subheader("Hot-doc activity timeline")
    activity = activity.sort_values("ts")
    activity["doc_id"] = activity["doc_id"].replace("", "—")
    activity["wall_time"] = pd.to_datetime(activity["ts"], unit="s")
    # Use the wall-clock seconds since first event for a clean x-axis.
    t0 = activity["ts"].min()
    activity["t_offset"] = activity["ts"] - t0

    # Map event_type → marker shape and color so each category reads at a glance.
    type_color = alt.Color(
        "event_type:N",
        scale=alt.Scale(
            domain=[
                "task_completed",
                "keep_resident_completed",
                "eviction",
                "near_duplicate_detected",
            ],
            range=["#4c78a8", "#54a24b", "#e45756", "#f4c542"],
        ),
        legend=alt.Legend(title=None, orient="top", columns=4),
    )
    type_shape = alt.Shape(
        "event_type:N",
        scale=alt.Scale(
            domain=[
                "task_completed",
                "keep_resident_completed",
                "eviction",
                "near_duplicate_detected",
            ],
            range=["circle", "triangle-up", "cross", "diamond"],
        ),
        legend=None,
    )

    chart = (
        alt.Chart(activity)
        .mark_point(filled=True, size=180, opacity=0.95, stroke="#0008", strokeWidth=0.5)
        .encode(
            x=alt.X(
                "t_offset:Q",
                title="seconds from first event",
                axis=alt.Axis(format=".1f", titleFontSize=12),
            ),
            y=alt.Y(
                "doc_id:N",
                title="document",
                axis=alt.Axis(labelFontSize=12, titleFontSize=12),
            ),
            color=type_color,
            shape=type_shape,
            tooltip=[
                alt.Tooltip("wall_time:T", title="wall time", format="%H:%M:%S"),
                alt.Tooltip("event_type:N", title="event"),
                alt.Tooltip("mode:N", title="mode"),
                alt.Tooltip("role:N", title="role"),
                alt.Tooltip("doc_id:N", title="doc"),
                alt.Tooltip("ttft_s:Q", title="TTFT (s)", format=".3f"),
                alt.Tooltip("message:N", title="message"),
            ],
        )
        .properties(height=200)
        .configure_axis(grid=True, gridOpacity=0.15)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)


def _near_dup_panel(df: pd.DataFrame) -> None:
    """Phase 3 robustness panel: every SimHash near-duplicate match logged
    by the orchestrator, with the candidate / matched doc / hamming /
    similarity. Empty when no near-dups have been seen (Phase 1/2 runs)."""
    nd = df[df["event_type"] == "near_duplicate_detected"].copy()
    if nd.empty:
        return

    st.subheader("SimHash near-duplicate detections")

    def _extract(row, key, cast=None):
        try:
            data = json.loads(row.get("extra_json") or "{}")
        except Exception:
            data = {}
        v = data.get(key)
        if v is None:
            return None
        if cast is not None:
            try:
                return cast(v)
            except Exception:
                return v
        return v

    nd["candidate"] = nd.apply(lambda r: _extract(r, "candidate_doc_id"), axis=1)
    nd["matched"] = nd.apply(lambda r: _extract(r, "matched_doc_id"), axis=1)
    nd["hamming"] = nd.apply(lambda r: _extract(r, "hamming", int), axis=1)
    nd["similarity"] = nd.apply(lambda r: _extract(r, "similarity", float), axis=1)
    nd["threshold"] = nd.apply(lambda r: _extract(r, "threshold", int), axis=1)
    view = (
        nd[["wall_time", "mode", "candidate", "matched", "hamming",
            "similarity", "threshold", "message"]]
        .sort_values("wall_time")
        .reset_index(drop=True)
    )
    view = view.rename(columns={"wall_time": "ts"})
    st.dataframe(view, use_container_width=True, hide_index=True)


def _decision_log_panel(df: pd.DataFrame) -> None:
    st.subheader("Orchestrator decision log")
    relevant = df[
        df["event_type"].isin(["decision", "keep_resident_completed"])
    ].copy()
    if relevant.empty:
        st.info("No orchestrator decisions yet.")
        return
    relevant = relevant.sort_values("ts").tail(40)

    def _fmt_html(row: pd.Series) -> str:
        ts = row.get("wall_time")
        ts_str = ts.strftime("%H:%M:%S") if pd.notna(ts) else "??:??:??"
        et = _s(row, "event_type")
        mode = _s(row, "mode")
        msg = _s(row, "message").strip()
        fp = _s(row, "fingerprint")[:8]
        doc = _s(row, "doc_id")

        if et == "keep_resident_completed":
            hit = bool(row.get("cache_hit_bool"))
            ttft = row.get("ttft_s")
            ttft_s = f"{ttft:6.3f}s" if pd.notna(ttft) else "  --  "
            cls = "hit" if hit else "miss"
            tag = "HIT " if hit else "MISS"
            return (
                f'<span class="line {cls}">'
                f'[{ts_str}] {mode.upper():<6} ▶ keep-resident {tag} '
                f'ttft={ttft_s} doc={doc:<10} fp={fp}  '
                f'← warmup completed'
                f'</span>'
            )

        # Decision rows: color by phase tag in the message body.
        cls = "obs"
        if "firing keep-resident" in msg or "act" in msg.lower():
            cls = "act"
        elif "observed" in msg:
            cls = "obs"
        elif "loaded pipeline" in msg or "evicted" in msg or "skipped" in msg:
            cls = "meta"
        return f'<span class="line {cls}">[{ts_str}] {mode.upper():<6} · {msg}</span>'

    html_lines = [_fmt_html(r) for _, r in relevant.iterrows()]
    st.markdown(
        f'<div class="decision-log"><code>{"".join(html_lines)}</code></div>',
        unsafe_allow_html=True,
    )


def _raw_panel(df: pd.DataFrame) -> None:
    with st.expander("raw telemetry tail (last 50 rows)"):
        cols = [
            "wall_time", "mode", "event_type", "role", "doc_id",
            "ttft_s", "total_s", "cache_hit", "fingerprint", "message",
        ]
        cols = [c for c in cols if c in df.columns]
        st.dataframe(df[cols].tail(50), use_container_width=True)


# ---------------------------------------------------------------------- header

def _header(df: pd.DataFrame, selected_run: str) -> None:
    st.title("Shared Context Bridge")
    st.caption(
        "Eliminating the Amnesia Tax in multi-agent pipelines. "
        "BEFORE = stateless agents (UUID-busted prefix every call). "
        "AFTER = bridge + orchestrator (doc-first shared prefix + keep-resident warmup)."
    )
    pipelines = sorted(
        {
            x for x in df.loc[df["event_type"] == "decision", "extra_json"]
            .str.extract(r'"pipeline":\s*"([^"]+)"', expand=False)
            .dropna().unique()
        }
    ) if "extra_json" in df.columns else []
    pills = [f'<span class="pill">run_id: {selected_run}</span>']
    if pipelines:
        pills.append(f'<span class="pill">pipeline: {" / ".join(pipelines)}</span>')
    modes = sorted(df["mode"].dropna().unique().tolist()) if "mode" in df else []
    for m in modes:
        if not m:
            continue
        cls = "pill-after" if m == "after" else "pill-before"
        pills.append(f'<span class="pill {cls}">mode: {m}</span>')
    st.markdown(" ".join(pills), unsafe_allow_html=True)


# ------------------------------------------------------------------------- main

def main() -> None:
    st.set_page_config(
        page_title="Shared Context Bridge — Live Demo",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)

    csv_path = st.sidebar.text_input("telemetry CSV", value=str(DEFAULT_CSV))
    run_ids_slot = st.sidebar.empty()
    auto_refresh = st.sidebar.toggle("auto-refresh", value=True)

    df_all = _load(csv_path)
    if df_all.empty:
        st.title("Shared Context Bridge")
        st.warning(
            "No telemetry yet. In another terminal run "
            "`.venv/bin/python run.py` then come back here."
        )
        if auto_refresh:
            time.sleep(REFRESH_SECONDS)
            st.rerun()
        return

    known_runs = sorted(df_all["run_id"].dropna().unique().tolist())
    selected = run_ids_slot.selectbox(
        "run_id", options=known_runs, index=len(known_runs) - 1
    )
    df = df_all[df_all["run_id"] == selected]

    _header(df, selected)
    st.divider()
    _hero(df)
    st.divider()
    _ttft_line_chart(df)
    st.divider()
    _hot_set_timeline(df)
    _near_dup_panel(df)
    st.divider()
    _decision_log_panel(df)
    _raw_panel(df)

    if auto_refresh:
        time.sleep(REFRESH_SECONDS)
        st.rerun()


if __name__ == "__main__":
    main()
