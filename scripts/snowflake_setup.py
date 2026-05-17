"""Snowflake setup + probe for the Shared Context Bridge telemetry sink.

Run AFTER exporting credentials:

    export SNOWFLAKE_ACCOUNT=sfeducationservices7-pxb11561
    export SNOWFLAKE_USER=...
    export SNOWFLAKE_PASSWORD=...
    # optional, sensible defaults applied if missing:
    # export SNOWFLAKE_WAREHOUSE=COMPUTE_WH
    # export SNOWFLAKE_DATABASE=BRIDGE_DB
    # export SNOWFLAKE_SCHEMA=TELEMETRY
    # export SNOWFLAKE_TABLE=EVENTS
    # export SNOWFLAKE_ROLE=ACCOUNTADMIN

    .venv/bin/python scripts/snowflake_setup.py

What it does (each step idempotent):
  1. Connect with whatever credentials are exported.
  2. List existing warehouses + databases so the user sees the lay of the land.
  3. CREATE WAREHOUSE / DATABASE / SCHEMA / TABLE IF NOT EXISTS for the
     defaults above (only the table is strictly required; the rest are a no-op
     if the user already has them).
  4. INSERT one probe row and SELECT it back to prove the round-trip works.
  5. Print the env vars the user should keep exported for `telemetry.py`.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid


REQUIRED = ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD")

DEFAULTS = {
    "SNOWFLAKE_WAREHOUSE": "COMPUTE_WH",
    "SNOWFLAKE_DATABASE": "BRIDGE_DB",
    "SNOWFLAKE_SCHEMA": "TELEMETRY",
    "SNOWFLAKE_TABLE": "EVENTS",
}


def fail(msg: str) -> "None":
    print(f"[snowflake_setup] FAIL  {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    missing = [k for k in REQUIRED if not os.environ.get(k)]
    if missing:
        fail(
            f"missing env vars: {missing}. "
            "See the docstring at the top of this file for the export commands."
        )

    cfg = {k: os.environ[k] for k in REQUIRED}
    for k, default in DEFAULTS.items():
        cfg[k] = os.environ.get(k, default)
    role = os.environ.get("SNOWFLAKE_ROLE") or None

    try:
        import snowflake.connector
    except ImportError:
        fail("snowflake-connector-python is not installed in this venv")

    authenticator = os.environ.get("SNOWFLAKE_AUTHENTICATOR")  # e.g. PROGRAMMATIC_ACCESS_TOKEN, externalbrowser
    print(
        f"[snowflake_setup] connecting account={cfg['SNOWFLAKE_ACCOUNT']} "
        f"user={cfg['SNOWFLAKE_USER']} role={role or '(default)'} "
        f"authenticator={authenticator or '(default password)'}"
    )
    connect_kwargs = dict(
        account=cfg["SNOWFLAKE_ACCOUNT"],
        user=cfg["SNOWFLAKE_USER"],
        role=role,
        client_session_keep_alive=True,
    )
    if authenticator:
        connect_kwargs["authenticator"] = authenticator
    # PATs go in `token=`, plain passwords go in `password=`.
    if authenticator and authenticator.upper() in (
        "PROGRAMMATIC_ACCESS_TOKEN",
        "OAUTH",
    ):
        connect_kwargs["token"] = cfg["SNOWFLAKE_PASSWORD"]
    else:
        connect_kwargs["password"] = cfg["SNOWFLAKE_PASSWORD"]
    try:
        conn = snowflake.connector.connect(**connect_kwargs)
    except Exception as exc:  # noqa: BLE001
        fail(f"connect failed: {exc}")

    cur = conn.cursor()
    print("[snowflake_setup] PASS  connected")

    cur.execute("SHOW WAREHOUSES")
    whs = [r[0] for r in cur.fetchall()]
    print(f"[snowflake_setup] warehouses visible: {whs or '(none)'}")

    cur.execute("SHOW DATABASES")
    dbs = [r[1] for r in cur.fetchall()]
    print(f"[snowflake_setup] databases visible: {dbs}")

    wh = cfg["SNOWFLAKE_WAREHOUSE"]
    db = cfg["SNOWFLAKE_DATABASE"]
    sch = cfg["SNOWFLAKE_SCHEMA"]
    tbl = cfg["SNOWFLAKE_TABLE"]

    if wh not in whs:
        print(f"[snowflake_setup] creating warehouse {wh} (XSMALL, auto-suspend 60s)")
        cur.execute(
            f'CREATE WAREHOUSE IF NOT EXISTS "{wh}" '
            "WITH WAREHOUSE_SIZE='XSMALL' AUTO_SUSPEND=60 AUTO_RESUME=TRUE "
            "INITIALLY_SUSPENDED=TRUE"
        )
    cur.execute(f'USE WAREHOUSE "{wh}"')

    if db not in dbs:
        print(f"[snowflake_setup] creating database {db}")
        cur.execute(f'CREATE DATABASE IF NOT EXISTS "{db}"')
    cur.execute(f'USE DATABASE "{db}"')
    cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{sch}"')
    cur.execute(f'USE SCHEMA "{db}"."{sch}"')

    print(f"[snowflake_setup] creating table {db}.{sch}.{tbl} (if needed)")
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS "{db}"."{sch}"."{tbl}" (
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
    )

    probe_run = f"probe-{uuid.uuid4().hex[:8]}"
    print(f"[snowflake_setup] writing probe row run_id={probe_run}")
    cur.execute(
        f"""
        INSERT INTO "{db}"."{sch}"."{tbl}" (
            ts, run_id, mode, event_type, role, doc_id, fingerprint,
            ttft_s, total_s, n_output_tokens, cache_hit, prompt_chars,
            message, extra_json
        )
        SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, PARSE_JSON(%s)
        """,
        (
            time.time(),
            probe_run,
            "probe",
            "setup_probe",
            "Setup",
            "n/a",
            "n/a",
            0.0,
            0.0,
            0,
            True,
            0,
            "snowflake_setup.py probe row",
            json.dumps({"script": "snowflake_setup.py"}),
        ),
    )

    cur.execute(
        f'SELECT run_id, event_type, message FROM "{db}"."{sch}"."{tbl}" '
        f"WHERE run_id = %s",
        (probe_run,),
    )
    rows = cur.fetchall()
    if not rows:
        fail("probe row did not round-trip — INSERT/SELECT inconsistent")
    print(f"[snowflake_setup] PASS  round-trip row: {rows[0]}")

    cur.close()
    conn.close()

    print()
    print("[snowflake_setup] DONE. To enable Snowflake in telemetry.py, keep these exported:")
    print("    export CLAUDE_TELEMETRY_SNOWFLAKE=1")
    for k in REQUIRED:
        print(f"    export {k}=...   # already set")
    for k, v in DEFAULTS.items():
        cur_val = os.environ.get(k)
        suffix = "(already set)" if cur_val else f"(defaulted to {v})"
        print(f"    export {k}={cur_val or v}   # {suffix}")
    if role:
        print(f"    export SNOWFLAKE_ROLE={role}")


if __name__ == "__main__":
    main()
