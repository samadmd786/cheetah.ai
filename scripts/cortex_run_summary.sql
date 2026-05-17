-- Shared Context Bridge — Cortex-AI run narrator.
--
-- Snowflake Cortex (`SNOWFLAKE.CORTEX.COMPLETE`) reads the orchestrator's
-- decision log and per-agent TTFTs out of BRIDGE_DB.TELEMETRY.EVENTS and
-- writes a one-paragraph plain-English narrative of what the bridge did and
-- why TTFT improved. Runs entirely inside Snowflake — no external LLM call.
--
-- Run as TRAINING_ROLE / COMPUTE_WH. Change :run_id below to summarize a
-- different run.

USE ROLE TRAINING_ROLE;
USE WAREHOUSE COMPUTE_WH;
USE DATABASE BRIDGE_DB;
USE SCHEMA TELEMETRY;

SET run_id = 'sf_live_test';

-- 1. Headline numbers per agent (BEFORE vs AFTER) ---------------------------
WITH per_agent AS (
    SELECT mode, role, ttft_s, cache_hit, ts
    FROM   EVENTS
    WHERE  run_id = $run_id
      AND  event_type = 'task_completed'
)
SELECT
    role,
    ROUND(MAX(CASE WHEN mode = 'before' THEN ttft_s END), 3) AS before_ttft_s,
    ROUND(MAX(CASE WHEN mode = 'after'  THEN ttft_s END), 3) AS after_ttft_s,
    ROUND(
        MAX(CASE WHEN mode = 'before' THEN ttft_s END)
        / NULLIF(MAX(CASE WHEN mode = 'after' THEN ttft_s END), 0),
        2
    ) AS speedup_x,
    BOOLOR_AGG(CASE WHEN mode = 'after' THEN cache_hit END) AS after_was_hit
FROM per_agent
GROUP BY role
ORDER BY MIN(ts);

-- 2. Cortex-generated plain-English narrative of the AFTER run -------------
WITH timeline AS (
    SELECT
        ROW_NUMBER() OVER (ORDER BY ts)                 AS step,
        event_type,
        role,
        ROUND(ttft_s, 3)                                AS ttft_s,
        cache_hit,
        message
    FROM EVENTS
    WHERE run_id = $run_id
      AND mode   = 'after'
      AND event_type IN ('decision', 'task_completed', 'keep_resident_completed')
),
prompt_text AS (
    SELECT
        'You are explaining an agentic-AI control plane to a hackathon judge.\n'
        || 'Below is the literal event log from one pipeline run.\n'
        || 'Write a 4-6 sentence narrative that:\n'
        || '  - names each agent in order,\n'
        || '  - calls out when the orchestrator decided to keep a document resident\n'
        || '    BEFORE the next agent fired,\n'
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
           ) WITHIN GROUP (ORDER BY step) AS p
    FROM timeline
)
SELECT
    SNOWFLAKE.CORTEX.COMPLETE('llama3.1-70b', p) AS narrative
FROM prompt_text;
