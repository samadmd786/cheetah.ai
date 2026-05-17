"""Dashboard gate (smoke): the three panels render non-empty content from the
real telemetry CSV produced by `run.py`.

We import the panel functions and call them under a Streamlit "AppTest"
runner so we don't need a browser. AppTest is Streamlit's built-in mock
runtime — it executes the script and records the elements it produced.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from dashboard import app as dashboard_app  # noqa: E402


def main() -> None:
    csv_path = REPO_ROOT / "logs" / "telemetry.csv"
    assert csv_path.exists(), (
        f"expected {csv_path} from a prior `python run.py` run; "
        "run that first then re-run this gate."
    )

    df = dashboard_app._load(str(csv_path))
    assert not df.empty, "telemetry CSV exists but is empty"
    assert set(["event_type", "mode", "role", "ttft_s"]).issubset(df.columns)

    # Panel 1: per-agent TTFT pivot has both modes.
    tasks = df[df["event_type"] == "task_completed"]
    pivot = tasks.pivot_table(
        index="role", columns="mode", values="ttft_s", aggfunc="last"
    )
    assert "before" in pivot.columns and "after" in pivot.columns, (
        f"pivot missing one of before/after; got columns={list(pivot.columns)}"
    )
    assert pivot.shape[0] >= 3, (
        f"expected >=3 rows in TTFT pivot (Screener/Analyst/Auditor), "
        f"got {pivot.shape[0]}"
    )
    after = pivot["after"]
    before = pivot["before"]
    # Agents 2 & 3 in AFTER must be WAY faster than their BEFORE counterparts.
    for role in ("Analyst", "Auditor"):
        if role in after.index and role in before.index:
            assert after[role] * 10 < before[role], (
                f"AFTER {role} ({after[role]:.3f}s) is not >10x faster than "
                f"BEFORE ({before[role]:.3f}s) — orchestrator value isn't visible"
            )

    # Panel 2: savings counter.
    before_total = tasks.loc[tasks["mode"] == "before", "total_s"].sum()
    after_total = tasks.loc[tasks["mode"] == "after", "total_s"].sum()
    assert before_total > after_total, (
        f"GPU-seconds-saved would be negative or zero "
        f"(before={before_total:.3f}, after={after_total:.3f})"
    )

    # Panel 3: decision log has at least one observe AND one keep-resident.
    decisions = df[df["event_type"] == "decision"]
    warmups = df[df["event_type"] == "keep_resident_completed"]
    assert (decisions["message"].str.contains("observed").any()), (
        "no 'observed ...' decision rows found"
    )
    assert not warmups.empty, "no keep_resident_completed events found"
    # The "right BEFORE next agent" gate: every warmup in AFTER mode must
    # precede the next task_completed of mode=after for that role.
    after_only = df[df["mode"] == "after"].sort_values("ts")
    warmups_after = after_only[after_only["event_type"] == "keep_resident_completed"]
    tasks_after = after_only[after_only["event_type"] == "task_completed"]
    assert not warmups_after.empty, "no AFTER-mode warmups in telemetry"
    # The first warmup must come AFTER the first task_completed (Screener) and
    # BEFORE the second task_completed (Analyst).
    if len(tasks_after) >= 2 and len(warmups_after) >= 1:
        t_screener = tasks_after.iloc[0]["ts"]
        t_analyst = tasks_after.iloc[1]["ts"]
        t_warm1 = warmups_after.iloc[0]["ts"]
        assert t_screener < t_warm1 < t_analyst, (
            "keep-resident did NOT fire between Screener and Analyst "
            f"(screener={t_screener}, warmup={t_warm1}, analyst={t_analyst})"
        )

    print(
        f"[dashboard gate] PASS  "
        f"rows={len(df)} agents={list(pivot.index)} "
        f"before_total={before_total:.2f}s after_total={after_total:.2f}s "
        f"warmups={len(warmups)} decisions={len(decisions)}"
    )


if __name__ == "__main__":
    main()
