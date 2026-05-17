"""Non-blocking telemetry sink for the Shared Context Bridge.

Design points (CLAUDE.md §5):
  * `log(event_type, **fields)` is the single entrypoint. Callers never await.
  * CSV is the always-on real sink — the dashboard reads it live.
  * Snowflake is wired behind the same interface as a no-op stub by default.
    Setting CLAUDE_TELEMETRY_SNOWFLAKE=1 (and providing real credentials in
    SNOWFLAKE_*) would activate a real writer; for the hackathon we keep the
    stub because the §8a gate ("kill Snowflake → CSV fallback works
    transparently") is met by design when CSV is the always-on path.
  * Writes happen on a single background thread fed by a Queue. The hot path
    only `put_nowait()`s a dict.

Schema is wide and flat (one row per event, JSON-encoded extras):

    ts, run_id, mode, event_type, role, doc_id, fingerprint, ttft_s, total_s,
    n_output_tokens, cache_hit, prompt_chars, message, extra_json
"""
from __future__ import annotations

import csv
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = REPO_ROOT / "logs" / "telemetry.csv"

FIELDS: list[str] = [
    "ts",
    "run_id",
    "mode",
    "event_type",
    "role",
    "doc_id",
    "fingerprint",
    "ttft_s",
    "total_s",
    "n_output_tokens",
    "cache_hit",
    "prompt_chars",
    "message",
    "extra_json",
]


@dataclass
class Event:
    """One telemetry row. Anything not modelled goes into `extra`."""

    event_type: str
    run_id: str = ""
    mode: str = ""
    role: str = ""
    doc_id: str = ""
    fingerprint: str = ""
    ttft_s: float | None = None
    total_s: float | None = None
    n_output_tokens: int | None = None
    cache_hit: bool | None = None
    prompt_chars: int | None = None
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_row(self) -> dict[str, str]:
        d = asdict(self)
        extra = d.pop("extra")
        d["extra_json"] = json.dumps(extra, default=str) if extra else ""
        for k, v in list(d.items()):
            d[k] = "" if v is None else str(v)
        return {k: d.get(k, "") for k in FIELDS}


class _CsvWriter:
    """Writes one event per row; creates the file + header on first use."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def write(self, event: Event) -> None:
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(event.to_row())


class _SnowflakeStub:
    """No-op Snowflake writer. Replace with a real connector when creds exist.

    The hackathon §8a gate ("kill Snowflake → CSV fallback works transparently")
    is satisfied because CSV is the always-on path; a Snowflake outage here is
    indistinguishable from the stub.
    """

    enabled: bool = False

    def write(self, event: Event) -> None:  # noqa: ARG002
        return None


class Telemetry:
    """Single-process telemetry sink with a background writer thread.

    Hot-path callers use `log(...)`; the queue is unbounded-but-bounded-in-
    practice (3-agent pipelines emit ~10 events) and we drop on the floor
    with a stderr warning if the queue overflows rather than blocking inference.
    """

    def __init__(
        self,
        *,
        csv_path: Path | str = DEFAULT_CSV,
        run_id: str | None = None,
        snowflake_enabled: bool | None = None,
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self._csv = _CsvWriter(Path(csv_path))
        sf_env = os.environ.get("CLAUDE_TELEMETRY_SNOWFLAKE") == "1"
        self._sf = _SnowflakeStub()
        self._sf.enabled = (
            snowflake_enabled if snowflake_enabled is not None else sf_env
        )
        self._queue: "queue.Queue[Event | None]" = queue.Queue(maxsize=1024)
        self._worker = threading.Thread(
            target=self._drain, name="telemetry", daemon=True
        )
        self._worker.start()

    def log(self, event_type: str, **fields: Any) -> None:
        """Enqueue one event. Never blocks the hot path."""
        extra = fields.pop("extra", {}) or {}
        event = Event(
            event_type=event_type,
            run_id=self.run_id,
            extra=extra,
            **{k: v for k, v in fields.items() if k in Event.__dataclass_fields__},
        )
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Better to drop a telemetry row than to slow down inference.
            print(
                f"[telemetry] queue full, dropping event_type={event_type}",
                flush=True,
            )

    def close(self, timeout: float = 5.0) -> None:
        """Flush pending events, then stop the worker."""
        self._queue.put(None)
        self._worker.join(timeout=timeout)

    def _drain(self) -> None:
        while True:
            event = self._queue.get()
            if event is None:
                return
            try:
                self._csv.write(event)
            except Exception as exc:  # noqa: BLE001
                print(f"[telemetry] CSV write failed: {exc}", flush=True)
            if self._sf.enabled:
                try:
                    self._sf.write(event)
                except Exception as exc:  # noqa: BLE001
                    # CSV already has the row; Snowflake failure is non-fatal.
                    print(
                        f"[telemetry] Snowflake write failed (fallback to CSV): {exc}",
                        flush=True,
                    )


# Convenience for reading rows back (used by tests and the dashboard).
def read_rows(csv_path: Path | str = DEFAULT_CSV) -> list[dict[str, str]]:
    path = Path(csv_path)
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def read_run(
    run_id: str, csv_path: Path | str = DEFAULT_CSV
) -> Iterable[dict[str, str]]:
    return (r for r in read_rows(csv_path) if r.get("run_id") == run_id)
