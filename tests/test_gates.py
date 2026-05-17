"""Phase 2 §8a module gates that don't need a live server.

Run with:  .venv/bin/python -m tests.test_gates
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

# Make repo root importable when run via `python -m tests.test_gates`.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bridge import SYSTEM_PREAMBLE, build_messages, fingerprint_heavy  # noqa: E402
from telemetry import Telemetry, read_rows  # noqa: E402


def gate_bridge_fingerprint() -> None:
    """Bridge gate: same doc + different task -> identical fingerprint."""
    doc = "ARTICLE I.\nSection 1.01. This is a sample contract clause.\n" * 50

    msgs_a, heavy_a, _ = build_messages(document=doc, task="What are the parties?")
    msgs_b, heavy_b, _ = build_messages(document=doc, task="What is the term?")

    fp_a = fingerprint_heavy(SYSTEM_PREAMBLE, doc)
    fp_b = fingerprint_heavy(SYSTEM_PREAMBLE, doc)
    assert fp_a == fp_b, "deterministic fingerprint of same heavy block expected"
    assert heavy_a == heavy_b, "heavy block should be byte-identical across agents"
    # And the system message in the assembled messages should also match.
    assert msgs_a[0]["content"] == msgs_b[0]["content"], (
        "system message (heavy block) must be byte-identical across tasks"
    )
    # Sanity: cache-bust mode breaks the fingerprint (used by BEFORE mode).
    fp_busted = fingerprint_heavy("REQUEST_ID: deadbeef\n" + SYSTEM_PREAMBLE, doc)
    assert fp_busted != fp_a, "cache_bust must change the fingerprint"
    print(
        f"[bridge gate] PASS  same-doc-different-task fp={fp_a[:10]} (matches), "
        f"busted fp={fp_busted[:10]} (differs)"
    )


def gate_telemetry_csv_roundtrip() -> None:
    """Telemetry gate: 100 synthetic events written and read back."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "t.csv"
        tel = Telemetry(csv_path=csv_path, run_id="gate-test")
        try:
            for i in range(100):
                tel.log(
                    "task_completed",
                    mode="after",
                    role=f"Role{i % 3}",
                    doc_id="discovery",
                    fingerprint=f"fp{i:04d}",
                    ttft_s=float(i),
                    total_s=float(i) + 1,
                    n_output_tokens=i,
                    cache_hit=(i % 2 == 0),
                    prompt_chars=1000 + i,
                    message=f"event {i}",
                    extra={"i": i},
                )
        finally:
            tel.close(timeout=5.0)

        rows = read_rows(csv_path)
        assert len(rows) == 100, f"expected 100 rows, got {len(rows)}"
        assert rows[0]["run_id"] == "gate-test"
        assert rows[42]["message"] == "event 42"
        assert rows[42]["ttft_s"] == "42.0"
        print(f"[telemetry gate] PASS  wrote+read {len(rows)} rows from {csv_path}")


def gate_telemetry_snowflake_fallback() -> None:
    """If the Snowflake writer raises, CSV must still capture the event.

    We simulate Snowflake failure by enabling the SF path on a stub whose
    `write()` always raises. The drain thread catches the exception and the
    CSV row is still present.
    """
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "t.csv"
        tel = Telemetry(csv_path=csv_path, run_id="sf-fallback")
        # Inject a failing Snowflake stub.
        class _Failing:
            enabled = True
            def write(self, event):  # noqa: ARG002
                raise RuntimeError("simulated Snowflake outage")

        tel._sf = _Failing()  # type: ignore[attr-defined]
        try:
            tel.log("task_completed", mode="after", role="Screener",
                    doc_id="discovery", ttft_s=0.5, message="alive")
            tel.log("task_completed", mode="after", role="Analyst",
                    doc_id="discovery", ttft_s=0.6, message="also alive")
        finally:
            # Give the worker a beat to drain before close.
            time.sleep(0.05)
            tel.close(timeout=5.0)

        rows = read_rows(csv_path)
        assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
        assert rows[0]["message"] == "alive"
        assert rows[1]["message"] == "also alive"
        print(
            "[telemetry gate] PASS  CSV captured both rows despite "
            "Snowflake stub raising"
        )


def main() -> None:
    gate_bridge_fingerprint()
    gate_telemetry_csv_roundtrip()
    gate_telemetry_snowflake_fallback()
    print("\nALL OFFLINE GATES PASS")


if __name__ == "__main__":
    main()
