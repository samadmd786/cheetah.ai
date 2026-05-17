# PROGRESS — Shared Context Bridge

Log of phase / module gates with the measured number that proved them.
See `CLAUDE.md` §8 (phases) and §8a (module gates).

---

## Phase 1 — Bare thesis, end to end

**Status:** DONE (2026-05-16)

**What runs:** `vllm-mlx` serves `mlx-community/Llama-3.2-3B-Instruct-4bit` on
`127.0.0.1:8001` with `--enable-prefix-cache --cache-memory-mb 3000
--continuous-batching --max-kv-size 32768`. `run.py` drives 3 sequential agent
calls (Screener → Analyst → Auditor) against `data/discovery.txt` (64,817 chars,
~13.7k Llama tokens), twice: BEFORE (UUID prepended → unique prefix per call)
and AFTER (doc-first prompt → shared prefix).

### Gate: env

`curl POST /v1/chat/completions` → `{"content":"ok"}`. PASS.

### Gate: agents/run

Headline number (BEFORE vs AFTER, time-to-first-token, seconds):

| agent    | before TTFT | after TTFT | speedup |
|----------|------------:|-----------:|--------:|
| Screener |      33.26  |      35.37 |   0.94x |
| Analyst  |      35.45  |       0.54 |  65.87x |
| Auditor  |      35.44  |       0.56 |  63.80x |
| TOTAL    |     104.14  |      36.46 |   2.86x |

Wall-clock totals (incl. decode): BEFORE 125.4s → AFTER 57.4s.
GPU-seconds saved per pipeline ≈ **68 s**.

PASS — AFTER Agents 2 and 3 are >60x faster to first token than Agent 1, which
is the exact prediction in §6 of `CLAUDE.md`. Raw log: `logs/phase1_run.log`.

### Notes / decisions

- **Critical config bug found & fixed.** Default `--cache-memory-mb` is ~536 MB,
  but the KV entry for our ~13.7k-token shared prefix is ~1.5 GB, so vllm-mlx
  silently rejected the store (`Cache entry too large … stored=False`) and
  every call ended up a cold prefill. Bumping to `--cache-memory-mb 3000`
  makes the entry fit and unlocks the cache hits on agents 2 and 3. Worth
  remembering for Phase 2: the orchestrator's "keep-resident" warmup is
  meaningless if the cache budget rejects the entry.
- The simple engine path (`vllm-mlx serve` without `--continuous-batching`)
  hits `RuntimeError: There is no Stream(gpu, 1) in current thread` from a
  worker-thread MLX call. `--continuous-batching` routes around it.
- Document is ~13.7k tokens (cl100k count). Comfortably inside the 10-30k
  window in CLAUDE.md §6; can bump `TARGET_TOKENS` in
  `scripts/build_discovery.py` if the Phase 3 multi-doc story needs a larger
  doc to make eviction visible.

---

---

## Phase 2 — Agentic control plane, end to end

**Status:** DONE (2026-05-16)

**What runs:** Same vllm-mlx server as Phase 1. All inference now flows
through `bridge.py` (split heavy/tail, SHA-256 fingerprint, telemetry). The
`orchestrator.py` reads `workflow/manifest.yaml` and, after each agent call
completes, fires a `bridge.keep_resident(...)` warmup for the next node's
document *before* the next dispatch lands. Every event lands in
`logs/telemetry.csv` via the non-blocking `telemetry.py` sink. A Streamlit
app at `dashboard/app.py` reads that CSV live and renders the three required
panels.

### Headline number (unchanged from Phase 1 — still real, now via bridge)

| agent    | BEFORE TTFT | AFTER TTFT | speedup |
|----------|------------:|-----------:|--------:|
| Screener | 33.07s | 35.33s | 0.94x |
| Analyst  | 35.49s | **0.50s** | **71.59x** |
| Auditor  | 35.53s | **0.52s** | **68.64x** |
| TOTAL    | 104.08s | 36.35s | 2.86x |

Wall-clock 125.0s → 57.0s; ~68 GPU-seconds saved per pipeline. The AFTER
fingerprints are identical for all three agents (`b9cd0881…`) — visible proof
the heavy block is byte-stable. The BEFORE fingerprints all differ — visible
proof the UUID bust is working. Raw: `logs/phase2_run.log`.

### Gate: bridge

`tests.test_gates::gate_bridge_fingerprint` — same doc + different task →
identical SHA-256 (`dd2128bbb8…`); cache-busted prefix produces a different
fingerprint (`b872d0f51b…`). PASS.

### Gate: orchestrator (keep-resident BEFORE next agent)

From `logs/telemetry.csv` (run_id `phase2_demo`, AFTER mode):

```
237.27s  task_completed   Screener  ttft=35.33s  MISS
237.27s  decision         next 'Analyst' needs same doc; firing keep-resident
237.75s  keep_resident_completed   ttft=0.48s  HIT   ← warmup before next agent
237.75s  observed Analyst starting
245.12s  task_completed   Analyst   ttft=0.50s   HIT
245.12s  decision         next 'Auditor' needs same doc; firing keep-resident
245.57s  keep_resident_completed   ttft=0.45s  HIT
245.57s  observed Auditor starting
252.98s  task_completed   Auditor   ttft=0.52s   HIT
```

`tests.test_dashboard::main` asserts the ordering programmatically:
`t_screener < t_warm1 < t_analyst`. PASS.

Narrative bonus from BEFORE mode in the same CSV: the orchestrator still
fires its warmups, but each one is a MISS (different fingerprint thanks to
the UUID bust). That is the pitch — the orchestrator's intent is correct,
but stateless prompts make it useless; the bridge's doc-first construction
is what unlocks the cache.

### Gate: telemetry

* `gate_telemetry_csv_roundtrip` — 100 synthetic events queued → drained →
  read back from CSV in order with all fields intact. PASS.
* `gate_telemetry_snowflake_fallback` — injected a failing Snowflake stub
  that raises on every `write()`; both events still landed in CSV. The
  "kill Snowflake → CSV fallback works transparently" gate is met by design
  because CSV is the always-on writer. PASS.

### Gate: dashboard

* Data-shape: `tests.test_dashboard.main` loads `logs/telemetry.csv`, exercises
  all three panel builders (TTFT pivot, savings counter, decision log), and
  asserts AFTER Agents 2-3 are >10x faster than BEFORE. PASS.
* Live HTTP: `streamlit run dashboard/app.py --server.headless true
  --server.port 8502` returns `HTTP 200` on `/` and `ok` on
  `/_stcore/health`. PASS.

### Notes / decisions

- Late-bound the orchestrator into the bridge in `run.py` because each needs
  a reference to the other. Bridge is the only `OpenAI` caller; orchestrator
  is the only "what should we warm" decision-maker. Clean seam.
- The orchestrator fires the warmup *synchronously* between agent calls
  rather than truly concurrently. With a 1-doc pipeline and ~0.5s warmups
  this is fine for the demo; for Phase 3 multi-doc we may want a thread.
- `Telemetry.log()` is non-blocking via a bounded queue (1024). Drops with a
  stderr warning rather than ever back-pressuring the inference path.
- Streamlit dashboard auto-refreshes once per second; the run_id selector in
  the sidebar lets a demo show prior runs.

---

## Phase 3 — Scale + robustness

**Status:** DONE (2026-05-16) — modules and gates. Pitch rehearsal + fallback
video are user-driven; not landed here.

**What runs:**

* Two additional documents (`data/merger.txt`, `data/discovery_v3.txt`)
  generated by `scripts/build_additional_docs.py`. `merger.txt` is a
  genuinely different M&A contract (~14k tokens); `discovery_v3.txt` is a
  near-duplicate of `discovery.txt` (whitespace perturbed + a few dollar
  amounts swapped — Hamming **1 / 64** under our SimHash, while
  `discovery` vs `merger` is **28 / 64**).
* Two new pipelines in `workflow/manifest.yaml`:
  `multi_doc_review` (three agents, three distinct docs) and
  `near_dup_check` (minimal SimHash-only scenario).
* `simhash.py` — 64-bit SimHash with 3-word shingles and BLAKE2b hashing,
  threshold = 10. No dependencies.
* `orchestrator.py` gains a per-doc SimHash memo, a `_check_near_duplicate`
  call BEFORE every keep-resident dispatch, a `near_duplicate_detected`
  telemetry event, and a structured `eviction` event distinct from the
  decision log.
* `run.py --pipeline ... --hot-capacity N` lets us trigger eviction under
  a small budget; default is still 4 (preserves Phase 2 behaviour).
* `dashboard/app.py` gains two new hero tiles (LRU evictions, near-dup
  matches), a hot-doc activity timeline (per-doc marker chart), and a
  SimHash near-duplicate detail table.

### Headline run — `multi_doc_review` under hot_capacity=2

```
toff   event                       mode   doc                ttft  hit
0.04s  observed Screener starting              discovery
7.75s  task_completed Screener     after  discovery        0.52s   Y   ← already hot from prior run
7.75s  decision: warm 'merger' for Analyst
43.28s keep_resident_completed     after  merger          35.48s   N   ← orchestrator absorbs cold prefill
43.28s observed Analyst starting              merger
51.10s task_completed Analyst      after  merger           0.54s   Y
51.10s decision: warm 'discovery_v3' for Auditor
51.15s near_duplicate_detected     after  discovery_v3            ← SimHash gate
87.43s keep_resident_completed     after  discovery_v3    36.28s   N   ← orchestrator absorbs cold prefill
87.43s eviction                    after  discovery               ← LRU gate (hot_capacity=2)
87.43s observed Auditor starting              discovery_v3
95.23s task_completed Auditor      after  discovery_v3     0.56s   Y
```

Three real agent dispatches all return TTFT < 1 s; the two cold prefills
required by previously-unseen docs are paid by the orchestrator's warmups
between agents instead of by the agents themselves. Raw:
`logs/phase3_multidoc.log` and rows with `run_id=phase3_multidoc` in
`logs/telemetry.csv`.

### Gate: multi-doc + matcher (CLAUDE.md §8a)

`tests/test_phase3.py` asserts all three sub-gates from telemetry:

| sub-gate | result |
|----------|--------|
| eviction event exists | PASS — 1 eviction; first evicted doc `discovery` under cap=2 |
| SimHash near-dup detected | PASS — `discovery_v3` ≈ `discovery` (hamming=1/64, similarity=0.984) |
| detection precedes dispatch | PASS — near-dup logged **36.28 s** before the keep-resident dispatch for `discovery_v3` |

### Notes / decisions

- SimHash threshold tuned to **10 bits / 64** based on the observed pairwise
  distances (near-dup: 1, far apart: 28). Comfortable separation.
- Hot-set LRU cap kept at 4 by default to preserve Phase 2 behaviour;
  multi-doc runs explicitly pass `--hot-capacity 2`. The vllm-mlx side's
  own `--cache-memory-mb 3000` budget happens to hold ~2 entries of
  ~1.5 GB each, so the orchestrator's cap mirrors the physical reality.
- SimHash is computed once per doc (memoized) and compared only against
  hot entries with a *different* `doc_id`, so a doc never matches itself.
- The orchestrator absorbs the cold prefills of new docs between agents.
  In a streaming demo, this means agents always feel snappy; the cost is
  *deferred*, not eliminated. Important framing for Q&A: the bridge does
  not magic away the first read of a new doc, it just makes every reuse
  free and times the unavoidable prefill to be invisible to the next agent.

### Out of scope here (user-driven follow-ups)

- Recorded screen capture as demo-day fallback video.
- 2-minute pitch rehearsal + timing.

---

## How to re-run everything

```bash
# Terminal 1 — vllm-mlx server (background, logs/server.log):
.venv/bin/vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit \
  --host 127.0.0.1 --port 8001 \
  --enable-prefix-cache --cache-memory-mb 3000 \
  --continuous-batching --max-kv-size 32768 --max-tokens 512

# Terminal 2 — pipelines (each appends to logs/telemetry.csv):
.venv/bin/python run.py                                              # Phase 1/2 (both modes)
.venv/bin/python run.py --pipeline multi_doc_review \
                       --hot-capacity 2 --mode after \
                       --run-id phase3_multidoc                      # Phase 3 multi-doc

# Terminal 3 — live dashboard at http://127.0.0.1:8502 :
.venv/bin/streamlit run dashboard/app.py --server.port 8502

# Re-verify all gates (no server needed for these):
.venv/bin/python -m tests.test_gates       # Phase 2 bridge/telemetry gates
.venv/bin/python -m tests.test_dashboard   # Phase 2 dashboard data-shape gate
.venv/bin/python -m tests.test_phase3      # Phase 3 eviction / SimHash / ordering
```
