-- Shared Context Bridge — auto-refreshing leaderboard via Snowflake Dynamic Tables.
--
-- A Dynamic Table is Snowflake's managed materialized view: you declare the
-- query and the target staleness, and Snowflake maintains it incrementally
-- for you. No cron, no batch job, no orchestration.
--
-- RUN_SUMMARY rolls up BRIDGE_DB.TELEMETRY.EVENTS into one row per pipeline
-- run with the headline numbers the dashboard / pitch needs:
--   - GPU-seconds saved (BEFORE total - AFTER total)
--   - speedup ratio
--   - AFTER cache hit rate
--   - agent count, start time
--
-- TARGET_LAG = '1 minute' → as new run.py runs land in EVENTS, the table is
-- automatically up-to-date within 60 s. Judges can run the leaderboard
-- query live during the demo and watch new rows appear.

USE ROLE TRAINING_ROLE;
USE WAREHOUSE COMPUTE_WH;
USE DATABASE BRIDGE_DB;
USE SCHEMA TELEMETRY;

CREATE OR REPLACE DYNAMIC TABLE RUN_SUMMARY
    TARGET_LAG = '1 minute'
    WAREHOUSE  = COMPUTE_WH
    REFRESH_MODE = AUTO
    INITIALIZE  = ON_CREATE
AS
WITH task AS (
    SELECT run_id, mode, role, ttft_s, total_s, cache_hit, ts
    FROM   EVENTS
    WHERE  event_type = 'task_completed'
)
SELECT
    run_id,
    MIN(ts)                                                    AS started_at,
    COUNT(DISTINCT role)                                       AS agent_count,
    ROUND(SUM(CASE WHEN mode = 'before' THEN ttft_s END), 3)   AS before_ttft_total_s,
    ROUND(SUM(CASE WHEN mode = 'after'  THEN ttft_s END), 3)   AS after_ttft_total_s,
    ROUND(
        SUM(CASE WHEN mode = 'before' THEN ttft_s END)
      - SUM(CASE WHEN mode = 'after'  THEN ttft_s END),
        3
    )                                                          AS gpu_seconds_saved,
    ROUND(
        SUM(CASE WHEN mode = 'before' THEN ttft_s END)
        / NULLIF(SUM(CASE WHEN mode = 'after' THEN ttft_s END), 0),
        2
    )                                                          AS speedup_x,
    ROUND(
        100.0 * AVG(CASE WHEN mode = 'after' AND cache_hit THEN 1.0
                         WHEN mode = 'after'               THEN 0.0
                         ELSE NULL END),
        1
    )                                                          AS after_hit_rate_pct
FROM task
GROUP BY run_id;

-- Demo query: leaderboard of all runs by GPU-seconds saved -------------------
SELECT
    run_id,
    TO_VARCHAR(TO_TIMESTAMP_NTZ(started_at::NUMBER), 'YYYY-MM-DD HH24:MI:SS')  AS started,
    agent_count,
    before_ttft_total_s  AS before_s,
    after_ttft_total_s   AS after_s,
    gpu_seconds_saved    AS saved_s,
    speedup_x,
    after_hit_rate_pct   AS hit_rate
FROM RUN_SUMMARY
ORDER BY gpu_seconds_saved DESC NULLS LAST;
