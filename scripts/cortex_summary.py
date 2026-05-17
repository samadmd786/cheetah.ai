"""Print the Snowflake Cortex narrative + headline table for a run.

Usage (after exporting the SNOWFLAKE_* env vars used by snowflake_setup.py):

    .venv/bin/python scripts/cortex_summary.py [run_id]

If no run_id is given, summarizes the most recent run_id in EVENTS.
"""
from __future__ import annotations

import os
import sys
import textwrap


def main() -> None:
    import snowflake.connector

    run_id = sys.argv[1] if len(sys.argv) > 1 else None

    auth = os.environ.get("SNOWFLAKE_AUTHENTICATOR")
    kw: dict = dict(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        role=os.environ.get("SNOWFLAKE_ROLE") or None,
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        database=os.environ.get("SNOWFLAKE_DATABASE", "BRIDGE_DB"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA", "TELEMETRY"),
    )
    if auth:
        kw["authenticator"] = auth
    if auth and auth.upper() in ("PROGRAMMATIC_ACCESS_TOKEN", "OAUTH"):
        kw["token"] = os.environ["SNOWFLAKE_PASSWORD"]
    else:
        kw["password"] = os.environ["SNOWFLAKE_PASSWORD"]

    conn = snowflake.connector.connect(**kw)
    cur = conn.cursor()

    if run_id is None:
        cur.execute(
            "SELECT run_id FROM EVENTS "
            "WHERE event_type = 'task_completed' "
            "ORDER BY ts DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            print("no runs in EVENTS yet", file=sys.stderr)
            sys.exit(1)
        run_id = row[0]
    print(f"summarizing run_id = {run_id}\n")

    cur.execute(
        """
        SELECT role,
               ROUND(MAX(CASE WHEN mode='before' THEN ttft_s END), 3) AS before_ttft_s,
               ROUND(MAX(CASE WHEN mode='after'  THEN ttft_s END), 3) AS after_ttft_s,
               ROUND(MAX(CASE WHEN mode='before' THEN ttft_s END)
                   / NULLIF(MAX(CASE WHEN mode='after' THEN ttft_s END), 0), 2) AS speedup_x
        FROM   EVENTS
        WHERE  run_id = %s AND event_type = 'task_completed'
        GROUP  BY role
        ORDER  BY MIN(ts)
        """,
        (run_id,),
    )
    rows = cur.fetchall()
    print("agent     | before ttft | after ttft | speedup")
    print("-" * 50)
    for r in rows:
        print(f"{(r[0] or ''):<9} |  {r[1]:>9} s |  {r[2]:>8} s |  {r[3]:>5}x")
    print()

    cur.execute(
        """
        WITH timeline AS (
            SELECT ROW_NUMBER() OVER (ORDER BY ts) AS step,
                   event_type, role, ROUND(ttft_s, 3) AS ttft_s,
                   cache_hit, message
            FROM   EVENTS
            WHERE  run_id = %s AND mode = 'after'
              AND  event_type IN ('decision','task_completed','keep_resident_completed')
        )
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            'llama3.1-70b',
            'You are explaining an agentic-AI control plane to a hackathon judge.\n'
            || 'Below is the literal event log from one pipeline run.\n'
            || 'Write a 4-6 sentence narrative that:\n'
            || '  - names each agent in order,\n'
            || '  - calls out when the orchestrator decided to keep a document resident '
            || 'BEFORE the next agent fired,\n'
            || '  - reports the per-agent time-to-first-token,\n'
            || '  - and finishes with the single sentence the judge should remember.\n\n'
            || 'EVENTS:\n'
            || LISTAGG(
                  step || '. [' || event_type || ']'
                  || COALESCE(' role=' || role, '')
                  || COALESCE(' ttft=' || ttft_s::STRING || 's', '')
                  || COALESCE(' hit=' || cache_hit::STRING, '')
                  || COALESCE(' — ' || message, ''),
                  '\n'
               ) WITHIN GROUP (ORDER BY step)
        ) AS narrative
        FROM timeline
        """,
        (run_id,),
    )
    narrative = cur.fetchone()[0]
    print("Cortex (llama3.1-70b, in Snowflake) says:\n")
    print(textwrap.fill(narrative.strip(), width=78))

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
