"""Shared Streamlit presentation layer for the Shared Context Bridge dashboards."""
from __future__ import annotations

import json
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

# ── palette ───────────────────────────────────────────────────────────────────
C_BEFORE = "#ff6b6b"
C_OURS = "#3dd68c"
C_BG = "#0b0f14"
C_CARD = "#141b24"
C_MUTED = "#8b9cb3"
C_GOLD = "#f4c542"
C_CYAN = "#5ec8ff"
C_MAGENTA = "#d68cff"
C_BLUE = "#6b9fff"

ROLE_ACCENT = {"Screener": C_CYAN, "Analyst": C_MAGENTA, "Auditor": C_BLUE}
ROLE_ORDER = ["Screener", "Analyst", "Auditor"]
WARM_THRESHOLD_S = 3.0

THEME_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;0,9..40,800;1,9..40,400&family=JetBrains+Mono:wght@500&display=swap');

  .stApp {
    background: radial-gradient(1200px 600px at 10% -10%, rgba(61,214,140,0.08), transparent),
                radial-gradient(900px 500px at 90% 0%, rgba(255,107,107,0.07), transparent),
                #0b0f14;
    color: #e8eef7;
    font-family: 'DM Sans', system-ui, sans-serif;
  }
  .block-container { padding-top: 1.2rem; max-width: 1280px; }
  h1, h2, h3, .stMarkdown h1, .stMarkdown h2 { letter-spacing: -0.02em; }
  [data-testid="stSidebar"] {
    background: #0e141c !important;
    border-right: 1px solid rgba(255,255,255,0.06);
  }
  hr { border-color: rgba(255,255,255,0.08) !important; }

  .hero {
    border-radius: 20px; padding: 28px 32px; margin-bottom: 8px;
    background: linear-gradient(135deg, #141b24 0%, #1a2433 50%, #121820 100%);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 24px 48px rgba(0,0,0,0.35);
  }
  .hero-eyebrow {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #3dd68c; margin-bottom: 8px;
  }
  .hero h1 {
    margin: 0 0 8px 0; font-size: 2rem; font-weight: 800;
    background: linear-gradient(90deg, #fff 0%, #b8c9e0 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero-tagline { margin: 0; font-size: 1.05rem; color: #9fb0c8; line-height: 1.5; max-width: 720px; }

  .kpi-row { display: flex; flex-wrap: wrap; gap: 14px; margin: 20px 0 4px 0; }
  .kpi {
    flex: 1; min-width: 140px; border-radius: 14px; padding: 16px 18px;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
  }
  .kpi.hero-kpi {
    flex: 1.6; min-width: 200px;
    background: linear-gradient(145deg, rgba(61,214,140,0.18), rgba(61,214,140,0.04));
    border-color: rgba(61,214,140,0.35);
  }
  .kpi-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #8b9cb3; }
  .kpi-value { font-size: 2rem; font-weight: 800; line-height: 1.1; margin-top: 4px; }
  .kpi-value.green { color: #3dd68c; }
  .kpi-value.red { color: #ff6b6b; }
  .kpi-sub { font-size: 0.82rem; color: #8b9cb3; margin-top: 4px; }

  .story-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin: 16px 0; }
  @media (max-width: 900px) { .story-grid { grid-template-columns: 1fr; } }

  .stage-card {
    border-radius: 18px; padding: 20px 22px;
    border: 1px solid rgba(255,255,255,0.08);
    background: #141b24;
  }
  .stage-card.before { border-top: 3px solid #ff6b6b; }
  .stage-card.ours   { border-top: 3px solid #3dd68c; }
  .stage-num {
    font-size: 0.68rem; font-weight: 800; letter-spacing: 0.12em;
    text-transform: uppercase; opacity: 0.9;
  }
  .stage-card.before .stage-num { color: #ff6b6b; }
  .stage-card.ours .stage-num { color: #3dd68c; }
  .stage-title { font-size: 1.15rem; font-weight: 700; margin: 6px 0 8px 0; color: #f0f4fa; }
  .stage-desc { font-size: 0.88rem; color: #8b9cb3; line-height: 1.45; margin-bottom: 14px; }
  .stage-total {
    font-size: 0.8rem; color: #8b9cb3; margin-top: 12px; padding-top: 12px;
    border-top: 1px solid rgba(255,255,255,0.06);
  }
  .stage-total strong { color: #e8eef7; font-size: 1.1rem; }

  .agent-list { display: flex; flex-direction: column; gap: 10px; }
  .agent-row {
    display: grid; grid-template-columns: 100px 1fr auto auto;
    align-items: center; gap: 10px; padding: 12px 14px;
    border-radius: 12px; background: rgba(0,0,0,0.25);
    border: 1px solid rgba(255,255,255,0.05);
  }
  .agent-role { font-weight: 700; font-size: 0.95rem; }
  .agent-doc { font-size: 0.78rem; color: #6d7f96; font-family: 'JetBrains Mono', monospace; }
  .agent-ttft { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 1rem; }
  .badge {
    font-size: 0.65rem; font-weight: 800; letter-spacing: 0.06em;
    padding: 4px 10px; border-radius: 999px; white-space: nowrap;
  }
  .badge.hit  { background: rgba(61,214,140,0.2); color: #3dd68c; }
  .badge.miss { background: rgba(255,107,107,0.2); color: #ff6b6b; }

  .compare-bar-wrap { margin: 8px 0 20px 0; }
  .compare-row { display: flex; align-items: center; gap: 12px; margin: 10px 0; }
  .compare-label { width: 200px; font-size: 0.82rem; color: #9fb0c8; flex-shrink: 0; }
  .compare-track {
    flex: 1; height: 28px; background: rgba(0,0,0,0.35);
    border-radius: 8px; overflow: hidden; position: relative;
  }
  .compare-fill {
    height: 100%; border-radius: 8px; display: flex; align-items: center;
    padding-left: 10px; font-size: 0.8rem; font-weight: 700; color: #fff;
    min-width: 48px;
  }
  .compare-fill.before { background: linear-gradient(90deg, #ff6b6b, #c44); }
  .compare-fill.ours   { background: linear-gradient(90deg, #2a9d63, #3dd68c); }

  .insight {
    border-radius: 14px; padding: 16px 20px; margin: 12px 0;
    background: linear-gradient(90deg, rgba(61,214,140,0.12), rgba(61,214,140,0.02));
    border: 1px solid rgba(61,214,140,0.25);
    font-size: 0.95rem; line-height: 1.5;
  }
  .insight strong { color: #3dd68c; font-size: 1.05rem; }

  .flow {
    display: flex; align-items: center; justify-content: center; gap: 8px;
    flex-wrap: wrap; margin: 16px 0; padding: 14px;
    background: rgba(0,0,0,0.2); border-radius: 12px;
  }
  .flow-step {
    padding: 8px 14px; border-radius: 10px; font-size: 0.8rem; font-weight: 600;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
  }
  .flow-arrow { color: #4a5a70; font-size: 1.1rem; }

  .event-feed {
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
    line-height: 1.7; padding: 16px; border-radius: 14px;
    background: #0a0e12; border: 1px solid rgba(255,255,255,0.06);
    max-height: 320px; overflow-y: auto;
  }
  .event-feed .e-act { color: #f4c542; }
  .event-feed .e-hit { color: #3dd68c; }
  .event-feed .e-miss { color: #ff6b6b; }
  .event-feed .e-sim { color: #d68cff; }
  .event-feed .e-evict { color: #ff9f6b; }
  .event-feed .e-meta { color: #6d7f96; }

  .section-label {
    font-size: 0.7rem; font-weight: 800; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6d7f96; margin: 24px 0 10px 0;
  }

  div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
  }
</style>
"""

STAGE_COPY = {
    "before": (
        "Stage 1 · The problem",
        "Stateless multi-agent systems",
        "Each agent gets a <strong>unique prompt prefix</strong> (UUID, session ID, or "
        "drifted system text). The GPU cold-prefills the same ~14k-token document "
        "<strong>every time</strong> — the amnesia tax.",
        "before",
    ),
    "ours": (
        "Stage 2 · Our solution",
        "Control plane + orchestrator",
        "Three <strong>different documents</strong>. The orchestrator reads the workflow "
        "DAG, warms the next doc <strong>before</strong> each agent fires, matches "
        "near-duplicates with SimHash, and evicts under a memory budget.",
        "ours",
    ),
}


def inject_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def latest_run_id(df: pd.DataFrame) -> str:
    """Return the run_id with the most recent event timestamp."""
    if df.empty or "run_id" not in df.columns:
        return ""
    ends = df.groupby("run_id", as_index=False)["ts"].max()
    ends = ends.sort_values("ts", ascending=False)
    return str(ends.iloc[0]["run_id"])


def trim_to_latest_session(df: pd.DataFrame, gap_s: float = 120.0) -> pd.DataFrame:
    """Keep only the most recent contiguous burst of events (one demo.py invocation)."""
    if df.empty or "ts" not in df.columns:
        return df
    ordered = df.sort_values("ts").reset_index(drop=True)
    gaps = ordered["ts"].diff() > gap_s
    if not gaps.any():
        return ordered
    start = ordered.index[gaps].tolist()[-1]
    return ordered.loc[start:].copy()


def filter_latest_run(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    rid = latest_run_id(df)
    if not rid:
        return df, ""
    chunk = df[df["run_id"] == rid].copy()
    return trim_to_latest_session(chunk), rid


def split_demo_stages(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = df[df["event_type"] == "task_completed"].sort_values("ts")
    before = tasks[tasks["mode"] == "before"].tail(3)
    after_tasks = tasks[tasks["mode"] == "after"]
    decisions = df[df["event_type"] == "decision"]
    multi = decisions[
        decisions["message"].astype(str).str.contains("multi_doc_review", na=False)
    ]
    if not multi.empty:
        t0 = float(multi["ts"].max())
        ours = after_tasks[after_tasks["ts"] >= t0].head(3)
    else:
        ours = after_tasks.tail(3)
    return before.reset_index(drop=True), ours.reset_index(drop=True)


def _role_order(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ROLE_ORDER
    seen = frame.sort_values("ts")["role"].dropna().tolist()
    ordered = [r for r in ROLE_ORDER if r in seen]
    ordered += [r for r in seen if r not in ordered]
    return ordered or ROLE_ORDER


def render_hero(before: pd.DataFrame, ours: pd.DataFrame, run_id: str) -> None:
    b = float(before["ttft_s"].sum()) if not before.empty else 0.0
    o = float(ours["ttft_s"].sum()) if not ours.empty else 0.0
    saved = max(0.0, b - o)
    speedup = b / o if o > 0 else 0.0

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-eyebrow">Shared Context Bridge · Live results</div>
          <h1>Eliminating the Amnesia Tax</h1>
          <p class="hero-tagline">
            Multi-agent pipelines re-read the same documents on every step.
            We add a <strong style="color:#e8eef7">workflow-aware control plane</strong>
            that keeps context hot <em>before</em> the next agent asks — measured on real vllm-mlx TTFT.
          </p>
          <div class="kpi-row">
            <div class="kpi hero-kpi">
              <div class="kpi-label">Speedup vs stateless</div>
              <div class="kpi-value green">{speedup:.2f}×</div>
              <div class="kpi-sub">{saved:.0f}s less agent wait time</div>
            </div>
            <div class="kpi">
              <div class="kpi-label">Before (1 doc, 3 agents)</div>
              <div class="kpi-value red">{b:.1f}s</div>
              <div class="kpi-sub">all cold prefills</div>
            </div>
            <div class="kpi">
              <div class="kpi-label">Ours (3 docs, orchestrator)</div>
              <div class="kpi-value green">{o:.1f}s</div>
              <div class="kpi-sub">harder workload, still wins</div>
            </div>
            <div class="kpi">
              <div class="kpi-label">Run</div>
              <div class="kpi-value" style="font-size:1.1rem;font-family:JetBrains Mono">{run_id}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _agent_rows_html(tasks: pd.DataFrame) -> str:
    if tasks.empty:
        return '<p style="color:#6d7f96">No agent data yet.</p>'
    rows = []
    for _, r in tasks.iterrows():
        role = str(r.get("role", "—"))
        doc = str(r.get("doc_id", "") or "—")
        ttft = float(r["ttft_s"]) if pd.notna(r.get("ttft_s")) else 0.0
        hit = bool(r.get("cache_hit_bool"))
        accent = ROLE_ACCENT.get(role, "#e8eef7")
        badge = "WARM HIT" if hit else "COLD MISS"
        badge_cls = "hit" if hit else "miss"
        rows.append(
            f'<div class="agent-row">'
            f'<span class="agent-role" style="color:{accent}">{role}</span>'
            f'<span class="agent-doc">{doc}</span>'
            f'<span class="agent-ttft">{ttft:.2f}s</span>'
            f'<span class="badge {badge_cls}">{badge}</span>'
            f"</div>"
        )
    return '<div class="agent-list">' + "".join(rows) + "</div>"


def render_stage_comparison(before: pd.DataFrame, ours: pd.DataFrame) -> None:
    b_total = float(before["ttft_s"].sum()) if not before.empty else 0.0
    o_total = float(ours["ttft_s"].sum()) if not ours.empty else 0.0
    longest = max(b_total, o_total, 1.0)

    def bar_row(label: str, value: float, css: str) -> str:
        pct = max(8, int(100 * value / longest))
        return (
            f'<div class="compare-row">'
            f'<span class="compare-label">{label}</span>'
            f'<div class="compare-track">'
            f'<div class="compare-fill {css}" style="width:{pct}%">{value:.1f}s</div>'
            f"</div></div>"
        )

    bars = ""
    if b_total > 0:
        bars += bar_row("BEFORE · stateless agents", b_total, "before")
    if o_total > 0:
        bars += bar_row("OURS · orchestrator + 3 docs", o_total, "ours")

    if b_total > 0 and o_total > 0:
        speedup = b_total / o_total
        saved = b_total - o_total
        insight = (
            f'<div class="insight"><strong>{speedup:.2f}× faster</strong> on agent-visible TTFT. '
            f"We saved <strong>{saved:.0f}s</strong> of cold-prefill wait — "
            f"and OURS used <em>three different documents</em>, not one.</div>"
        )
    else:
        insight = ""

    st.markdown(f'<div class="section-label">The headline in one glance</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="compare-bar-wrap">{bars}{insight}</div>', unsafe_allow_html=True)

    num, title, desc, css = STAGE_COPY["before"]
    _, title2, desc2, css2 = STAGE_COPY["ours"]
    st.markdown(
        f"""
        <div class="story-grid">
          <div class="stage-card {css}">
            <div class="stage-num">{num}</div>
            <div class="stage-title">{title}</div>
            <div class="stage-desc">{desc}</div>
            {_agent_rows_html(before)}
            <div class="stage-total">Total wait per agent: <strong>{b_total:.2f}s</strong></div>
          </div>
          <div class="stage-card {css2}">
            <div class="stage-num">{STAGE_COPY["ours"][0]}</div>
            <div class="stage-title">{title2}</div>
            <div class="stage-desc">{desc2}</div>
            {_agent_rows_html(ours)}
            <div class="stage-total">Total wait per agent: <strong>{o_total:.2f}s</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_flow() -> None:
    st.markdown(
        """
        <div class="section-label">How the orchestrator works</div>
        <div class="flow">
          <span class="flow-step">① Agent finishes</span>
          <span class="flow-arrow">→</span>
          <span class="flow-step">② Read workflow DAG</span>
          <span class="flow-arrow">→</span>
          <span class="flow-step">③ Warm next doc (keep-resident)</span>
          <span class="flow-arrow">→</span>
          <span class="flow-step">④ Next agent lands on warm cache</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_ttft_grouped(before: pd.DataFrame, ours: pd.DataFrame) -> None:
    st.markdown('<div class="section-label">Per-agent time to first token</div>', unsafe_allow_html=True)
    frames = []
    if not before.empty:
        b = before.copy()
        b["stage"] = "Before"
        frames.append(b)
    if not ours.empty:
        o = ours.copy()
        o["stage"] = "Ours"
        frames.append(o)
    if not frames:
        st.info("No measurements yet.")
        return

    long = pd.concat(frames, ignore_index=True)
    role_order = _role_order(long)

    chart = (
        alt.Chart(long)
        .mark_bar(size=28, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("role:N", sort=role_order, title=None, axis=alt.Axis(labelFontSize=12)),
            xOffset=alt.XOffset("stage:N"),
            y=alt.Y("ttft_s:Q", title="seconds until first token"),
            color=alt.Color(
                "stage:N",
                scale=alt.Scale(domain=["Before", "Ours"], range=[C_BEFORE, C_OURS]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("role:N", title="Agent"),
                alt.Tooltip("stage:N"),
                alt.Tooltip("doc_id:N", title="Document"),
                alt.Tooltip("ttft_s:Q", format=".2f", title="TTFT (s)"),
            ],
        )
        .properties(height=300)
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#ffffff18", labelColor="#9fb0c8", titleColor="#9fb0c8")
        .configure_legend(labelColor="#e8eef7")
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"Green zone: under **{WARM_THRESHOLD_S:g}s** = cache HIT (warm). "
        "Before: every agent cold. Ours: only the first agent on a new doc pays cold cost."
    )


def render_control_stats(df: pd.DataFrame) -> None:
    warmups = int((df["event_type"] == "keep_resident_completed").sum())
    hits = int(
        ((df["event_type"] == "keep_resident_completed") & df["cache_hit_bool"]).sum()
    )
    evictions = int((df["event_type"] == "eviction").sum())
    near_dups = int((df["event_type"] == "near_duplicate_detected").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Proactive warmups", f"{warmups}", help="keep-resident calls between agents")
    c2.metric("Warmup cache hits", f"{hits}/{warmups}" if warmups else "—")
    c3.metric("LRU evictions", str(evictions), help="budget cap forced a doc out")
    c4.metric("SimHash near-dups", str(near_dups), help="amended doc flagged before dispatch")


def render_event_feed(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-label">Orchestrator activity (live log)</div>', unsafe_allow_html=True)
    types = ["decision", "keep_resident_completed", "near_duplicate_detected", "eviction"]
    rel = df[df["event_type"].isin(types)].sort_values("ts")
    if rel.empty:
        st.info("No orchestrator events in this run.")
        return

    lines: list[str] = []
    for _, row in rel.tail(35).iterrows():
        ts = row.get("wall_time")
        ts_s = ts.strftime("%H:%M:%S") if pd.notna(ts) else "??:??:??"
        et = str(row.get("event_type", ""))
        msg = str(row.get("message", "")).strip()
        doc = str(row.get("doc_id", ""))

        if et == "near_duplicate_detected":
            lines.append(f'<div class="e-sim">[{ts_s}] 🔍 {msg[:120]}</div>')
        elif et == "eviction":
            lines.append(f'<div class="e-evict">[{ts_s}] ♻ {msg[:100]}</div>')
        elif et == "keep_resident_completed":
            hit = bool(row.get("cache_hit_bool"))
            ttft = row.get("ttft_s")
            t = f"{ttft:.1f}s" if pd.notna(ttft) else "—"
            cls = "e-hit" if hit else "e-miss"
            tag = "HIT" if hit else "MISS (absorbed between agents)"
            lines.append(f'<div class="{cls}">[{ts_s}] Warmup · {doc} · {t} · {tag}</div>')
        else:
            extra: dict[str, Any] = {}
            try:
                extra = json.loads(row.get("extra_json") or "{}")
            except Exception:
                pass
            if extra.get("phase") == "act" and "warmup" in msg.lower():
                short = msg.split(";")[0] if ";" in msg else msg[:90]
                lines.append(f'<div class="e-act">[{ts_s}] 🧭 {short}</div>')
            elif "observed" in msg:
                lines.append(f'<div class="e-meta">[{ts_s}] {msg[:100]}</div>')

    st.markdown(
        f'<div class="event-feed">{"".join(lines)}</div>',
        unsafe_allow_html=True,
    )


def chart_hot_timeline(df: pd.DataFrame) -> None:
    activity = df[
        df["event_type"].isin([
            "task_completed", "keep_resident_completed", "eviction",
            "near_duplicate_detected",
        ])
    ].copy()
    if activity.empty or activity["doc_id"].astype(str).str.strip().eq("").all():
        return

    st.markdown('<div class="section-label">Document activity map</div>', unsafe_allow_html=True)
    activity = activity.sort_values("ts")
    activity["doc_id"] = activity["doc_id"].replace("", "—")
    activity["t_offset"] = activity["ts"] - activity["ts"].min()
    labels = {
        "task_completed": "Agent task",
        "keep_resident_completed": "Warmup",
        "eviction": "Eviction",
        "near_duplicate_detected": "SimHash",
    }
    activity["kind"] = activity["event_type"].map(labels)

    chart = (
        alt.Chart(activity)
        .mark_bar(size=18, cornerRadiusEnd=3)
        .encode(
            x=alt.X("t_offset:Q", title="seconds into pipeline"),
            y=alt.Y("doc_id:N", title="document"),
            color=alt.Color("kind:N", legend=alt.Legend(title=None, orient="top")),
            tooltip=["kind", "doc_id", "role", alt.Tooltip("ttft_s:Q", format=".2f")],
        )
        .properties(height=200)
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#ffffff18", labelColor="#9fb0c8")
    )
    st.altair_chart(chart, use_container_width=True)
