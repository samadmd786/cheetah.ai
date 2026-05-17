"""Non-blocking telemetry sink for the Shared Context Bridge.

Design points (CLAUDE.md §5):
  * `log(event_type, **fields)` is the single entrypoint. Callers never await.
  * CSV is the always-on real sink — the dashboard reads it live.
  * Snowflake is an optional second sink. Enable by setting
    CLAUDE_TELEMETRY_SNOWFLAKE=1 + the SNOWFLAKE_* env vars
    (USER, PASSWORD, ACCOUNT, WAREHOUSE, DATABASE, SCHEMA, TABLE). On any
    connect/write failure the writer marks itself disabled and CSV keeps
    flowing — that is the live §8a "kill Snowflake → CSV fallback" gate.
  * Writes happen on a single background thread fed by a Queue. The hot path
    only `put_nowait()`s a dict.
  * Snowflake writes are batched (default 16 events / 0.5 s) so we don't pay
    a round-trip per event.

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
    """No-op Snowflake writer used when CLAUDE_TELEMETRY_SNOWFLAKE != 1.

    Kept so the hot path can always call `self._sf.write(...)` without
    branching. When disabled, CSV is the only sink — which is also the
    behavior after a real-writer failure (see `_SnowflakeWriter`).
    """

    enabled: bool = False

    def write(self, event: Event) -> None:  # noqa: ARG002
        return None

    def write_batch(self, events: list[Event]) -> None:  # noqa: ARG002
        return None

    def close(self) -> None:
        return None


# Columns in the Snowflake EVENTS table. Order matters: it must match the
# INSERT statement below.
_SF_COLUMNS: list[str] = [
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


class _SnowflakeWriter:
    """Real Snowflake sink. Batched, fail-soft, off the hot path.

    On any exception during connect or write the writer flips `enabled = False`
    and the drain thread stops trying — CSV continues uninterrupted. That is
    the live demonstration of the §8a fallback gate.
    """

    enabled: bool = False

    def __init__(
        self,
        *,
        account: str,
        user: str,
        password: str,
        warehouse: str,
        database: str,
        schema: str,
        table: str = "EVENTS",
        role: str | None = None,
        authenticator: str | None = None,
    ) -> None:
        # Lazy import so the hard dep is only needed when SF is turned on.
        import snowflake.connector

        self._database = database
        self._schema = schema
        self._table = table
        connect_kwargs: dict = dict(
            account=account,
            user=user,
            warehouse=warehouse,
            database=database,
            schema=schema,
            role=role,
            client_session_keep_alive=True,
        )
        # MFA-required accounts use Programmatic Access Tokens (PATs). The
        # connector wants the PAT in `token=`, not `password=`, when this
        # authenticator is in play. For plain password auth it stays in
        # `password=`.
        if authenticator:
            connect_kwargs["authenticator"] = authenticator
        if authenticator and authenticator.upper() in (
            "PROGRAMMATIC_ACCESS_TOKEN",
            "OAUTH",
        ):
            connect_kwargs["token"] = password
        else:
            connect_kwargs["password"] = password
        self._conn = snowflake.connector.connect(**connect_kwargs)
        self._ensure_table()
        self.enabled = True

    @property
    def fq_table(self) -> str:
        return f'"{self._database}"."{self._schema}"."{self._table}"'

    def _ensure_table(self) -> None:
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {self.fq_table} (
            ts            FLOAT,
            run_id        STRING,
            mode          STRING,
            event_type    STRING,
            role          STRING,
            doc_id        STRING,
            fingerprint   STRING,
            ttft_s        FLOAT,
            total_s       FLOAT,
            n_output_tokens INTEGER,
            cache_hit     BOOLEAN,
            prompt_chars  INTEGER,
            message       STRING,
            extra_json    VARIANT
        )
        """
        with self._conn.cursor() as cur:
            cur.execute(ddl)

    @staticmethod
    def _event_to_tuple(ev: Event) -> tuple:
        return (
            ev.ts,
            ev.run_id,
            ev.mode,
            ev.event_type,
            ev.role,
            ev.doc_id,
            ev.fingerprint,
            ev.ttft_s,
            ev.total_s,
            ev.n_output_tokens,
            ev.cache_hit,
            ev.prompt_chars,
            ev.message,
            json.dumps(ev.extra, default=str) if ev.extra else None,
        )

    def write(self, event: Event) -> None:
        self.write_batch([event])

    def write_batch(self, events: list[Event]) -> None:
        if not events:
            return
        # VARIANT can't be bound directly; we PARSE_JSON the bound STRING.
        placeholders = (
            "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, PARSE_JSON(%s))"
        )
        sql = (
            f"INSERT INTO {self.fq_table} "
            f"({', '.join(_SF_COLUMNS)}) "
            f"SELECT column1, column2, column3, column4, column5, column6, "
            f"column7, column8, column9, column10, column11, column12, "
            f"column13, PARSE_JSON(column14) "
            f"FROM VALUES "
            + ", ".join(["(" + ", ".join(["%s"] * 14) + ")"] * len(events))
        )
        params: list = []
        for ev in events:
            params.extend(self._event_to_tuple(ev))
        with self._conn.cursor() as cur:
            cur.execute(sql, params)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass


def _build_snowflake_writer() -> _SnowflakeStub | _SnowflakeWriter:
    """Construct a real writer from env vars, or a stub on failure / disabled."""
    if os.environ.get("CLAUDE_TELEMETRY_SNOWFLAKE") != "1":
        return _SnowflakeStub()
    required = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(
            f"[telemetry] Snowflake disabled: missing env vars {missing}",
            flush=True,
        )
        return _SnowflakeStub()
    try:
        return _SnowflakeWriter(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            password=os.environ["SNOWFLAKE_PASSWORD"],
            warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
            database=os.environ.get("SNOWFLAKE_DATABASE", "BRIDGE_DB"),
            schema=os.environ.get("SNOWFLAKE_SCHEMA", "TELEMETRY"),
            table=os.environ.get("SNOWFLAKE_TABLE", "EVENTS"),
            role=os.environ.get("SNOWFLAKE_ROLE") or None,
            authenticator=os.environ.get("SNOWFLAKE_AUTHENTICATOR") or None,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"[telemetry] Snowflake connect failed, falling back to CSV-only: {exc}",
            flush=True,
        )
        return _SnowflakeStub()


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
        sf_batch_max: int = 16,
        sf_flush_interval_s: float = 0.5,
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self._csv = _CsvWriter(Path(csv_path))
        # If caller passes snowflake_enabled=False explicitly, force the stub
        # (used by the offline fallback gate). Otherwise honor env config.
        if snowflake_enabled is False:
            self._sf: _SnowflakeStub | _SnowflakeWriter = _SnowflakeStub()
        else:
            self._sf = _build_snowflake_writer()
        self._sf_batch_max = sf_batch_max
        self._sf_flush_interval_s = sf_flush_interval_s
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
        self._sf.close()

    def _flush_sf_batch(self, batch: list[Event]) -> None:
        if not batch or not self._sf.enabled:
            return
        try:
            self._sf.write_batch(batch)
        except Exception as exc:  # noqa: BLE001
            # CSV already has the rows; Snowflake failure is non-fatal.
            # Disable further attempts this session — the §8a fallback gate.
            print(
                f"[telemetry] Snowflake write failed, falling back to CSV-only: {exc}",
                flush=True,
            )
            self._sf.enabled = False

    def _drain(self) -> None:
        batch: list[Event] = []
        last_flush = time.time()
        while True:
            timeout = self._sf_flush_interval_s if self._sf.enabled else None
            try:
                event = self._queue.get(timeout=timeout)
            except queue.Empty:
                self._flush_sf_batch(batch)
                batch = []
                last_flush = time.time()
                continue
            if event is None:
                self._flush_sf_batch(batch)
                return
            try:
                self._csv.write(event)
            except Exception as exc:  # noqa: BLE001
                print(f"[telemetry] CSV write failed: {exc}", flush=True)
            if self._sf.enabled:
                batch.append(event)
                if (
                    len(batch) >= self._sf_batch_max
                    or (time.time() - last_flush) >= self._sf_flush_interval_s
                ):
                    self._flush_sf_batch(batch)
                    batch = []
                    last_flush = time.time()


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
