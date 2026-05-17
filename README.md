# Shared Context Bridge

A control plane on top of an existing KV cache (vllm-mlx) that reads a
multi-agent workflow graph and **pre-warms the cache for the next agent
before it asks** — eliminating the "amnesia tax" that multi-agent AI
systems pay when every agent re-prefills the same documents from scratch.

Built in 24 hours for Uncommon Hacks 2026 (University of Chicago).

---

## The headline

| stage                                  | scenario                | total TTFT |
|----------------------------------------|-------------------------|-----------:|
| BEFORE — stateless agents (today)      | 1 doc, UUID-busted      | 110.03 s   |
| **OURS** — orchestrator + SimHash + LRU | **3 different docs**   |  **40.86 s** |

**2.69× faster on a *harder* workload** — same hardware, same model, same
KV cache engine. We just schedule it.

Run the live 2-stage demo yourself: `python demo.py` (after setup).

---

## The pitch in one paragraph

vLLM and LMCache already cache KV state for prompts you've seen before.
They're **reactive** — they cache what was asked, not what's coming. In a
real multi-agent workflow where different agents read different documents,
their prefix cache cold-misses every time the doc changes.

We add the missing control plane: it reads the workflow DAG, looks ahead
at what each agent will need, and fires a `keep-resident` warmup **in the
gap between agents** while the previous agent is still streaming. A
SimHash near-duplicate matcher catches amended documents that exact-prefix
caching would miss. A budget-aware LRU evicts based on what's coming next,
not just what was used last. Same KV cache underneath, unchanged.

---

## Key Benefits

- **Eliminate the "Amnesia Tax"**: Agents no longer waste time and compute re-reading the same foundational document.
- **Dramatically Lower TTFT (Time To First Token)**: Downstream agents see up to a **70x speedup** (e.g., from 35s to 0.5s) because their required context is pre-loaded.
- **Zero Redundant Processing**: The orchestrator absorbs the cold prefill *between* agent calls, keeping the multi-agent pipeline feeling snappy and interactive.
- **Save Expensive GPU Cycles**: By sharing KV cache prefixes intelligently, you reduce redundant token processing, freeing up GPU bandwidth for actual inference.

---

## Snowflake: The Analytics & AI Layer

This project goes beyond just running inference by turning **Snowflake** into the active analytics and AI backbone of the control plane:

- **Live Telemetry Sink**: Every event, decision, and cache hit is streamed asynchronously into `BRIDGE_DB.TELEMETRY.EVENTS` using `snowflake-connector-python`.
- **Cortex AI Run Narrator**: We use Snowflake Cortex (`llama3.1-70b`) directly within the database to read the orchestrator's decision log and automatically generate a judge-ready, natural-language summary of how the pipeline performed.
- **Dynamic Tables Leaderboard**: A Snowflake Dynamic Table (`RUN_SUMMARY`) incrementally aggregates pipeline performance (TTFT, GPU-seconds saved, cache hit rate) in real-time without needing external schedulers like Airflow.

---

## Architecture

```
data/*.txt          ← documents (litigation contract, M&A agreement, near-dup)
workflow/manifest.yaml   ← the agent DAG (declared, not inferred)
        ↓
run.py / demo.py    ← entry points
        ↓
bridge.py           ← only thing that talks to vllm-mlx
                       splits heavy block (system + doc) from task tail
                       SHA-256 fingerprint of the heavy block
        ↓
orchestrator.py     ← observe → lookahead → act → adapt
                       reads manifest, fires keep-resident warmups
                       SimHash near-duplicate detection
                       budget-aware LRU eviction
        ↓
vllm-mlx (8001)     ← unchanged: native prefix cache does the actual reuse
        ↓
telemetry.py        ← non-blocking CSV writer + live Snowflake sink
        ↓
dashboard/app.py    ← Streamlit (port 8502): live charts, decision log, hot-doc timeline
                       (Snowflake Cortex AI run narrator + Dynamic Tables leaderboard)
```

Modules, each kept deliberately small:

| file                         | role |
|------------------------------|------|
| [agents.py](agents.py)       | 3 role-specific tasks (Screener, Analyst, Auditor) |
| [bridge.py](bridge.py)       | the only OpenAI seam; fingerprint, dispatch, `keep_resident` |
| [orchestrator.py](orchestrator.py) | reads manifest, lookahead warmups, SimHash check, LRU |
| [simhash.py](simhash.py)     | 64-bit SimHash on 3-word shingles + Hamming distance |
| [telemetry.py](telemetry.py) | non-blocking CSV sink + live Snowflake writer (`BRIDGE_DB.TELEMETRY.EVENTS`) |
| [run.py](run.py)             | pipeline runner (multi-mode, multi-pipeline) |
| [demo.py](demo.py)           | 2-stage live terminal demo with token streaming |
| [dashboard/app.py](dashboard/app.py) | Streamlit dashboard |
| [workflow/manifest.yaml](workflow/manifest.yaml) | the DAG (pipelines + documents) |

---

## Quick start

Prereqs: Apple Silicon Mac, Python 3.12, ~6 GB free memory.

```bash
# 1. Install (Python 3.12 only)
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Build the documents (~14k tokens each)
.venv/bin/python scripts/build_discovery.py
.venv/bin/python scripts/build_additional_docs.py

# 3. Start the vllm-mlx server (Terminal 1) — first run downloads the model (~2 GB)
.venv/bin/vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit \
  --host 127.0.0.1 --port 8001 \
  --enable-prefix-cache --cache-memory-mb 3000 \
  --continuous-batching --max-kv-size 32768 --max-tokens 512

# 4. Start the live dashboard (Terminal 2)
.venv/bin/streamlit run dashboard/app.py --server.port 8502
#    → open http://127.0.0.1:8502

# 5. Run the live demo (Terminal 3) — ~4 minutes wall, the headline output
.venv/bin/python demo.py
```

Re-verify the module gates (no server needed for the first two):
```bash
.venv/bin/python -m tests.test_gates       # bridge / telemetry
.venv/bin/python -m tests.test_dashboard   # dashboard data-shape
.venv/bin/python -m tests.test_phase3      # eviction / SimHash / ordering
```

---

## Critical config

vllm-mlx's default `--cache-memory-mb` is ~536 MB. Our shared-prefix KV
entry is ~1.5 GB per doc. **Without `--cache-memory-mb 3000` the cache
silently rejects the store and nothing ever hits** — the orchestrator's
warmups become no-ops. See `PROGRESS.md` for the full story; the value
is hard-coded into the server command above.

vllm-mlx's simple engine has an MLX threading bug
(`There is no Stream(gpu, 1) in current thread`) on chat completions.
`--continuous-batching` routes around it.

---

## What's in scope / what isn't

**In scope (built this hackathon):**
- Sequential 3-agent pipelines (`discovery_review`, `multi_doc_review`,
  `near_dup_check`).
- Single-machine vllm-mlx on Apple Silicon (M4 Pro, 16 GB).
- CSV telemetry + live Snowflake sink (async, batched).
- Snowflake Cortex AI for run narration & Dynamic Tables for a live leaderboard.
- Streamlit dashboard with live TTFT chart, hot-doc timeline, decision log,
  SimHash detail table.

**Out of scope (not shipped):**
- Multi-node, RDMA, LMCache integration.
- Fan-out / branching workflows (manifest assumes sequential).
- Pre-flight warmup for node 0 (agent 1 still pays its cold prefill).
- Trained-prediction-of-next-doc (we keep it deterministic; the manifest
  IS the prediction).

---

## Where to look

- [PROGRESS.md](PROGRESS.md) — per-phase gates with measured numbers
  (env, bridge, orchestrator, telemetry, dashboard, eviction, SimHash).
- [CLAUDE.md](CLAUDE.md) — design constitution (the constraints we
  pinned at hour 0).
- [DEMO_SCRIPT.md](DEMO_SCRIPT.md) — speaking notes for live presentation
  including Q&A defense vs. vLLM / LMCache.
- [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md) — 50-second voiceover script for
  the recorded demo clip.
- [logs/telemetry.csv](logs/telemetry.csv) — every run's events
  (dashboard reads this live).

---

## License

Built for Uncommon Hacks 2026. No license declared yet.
