# CLAUDE.md — Shared Context Bridge

> Project instructions for Claude Code. Read fully before generating or editing code.
> Hackathon: Uncommon Hacks, University of Chicago (24h). Goal: a demoable agentic-AI
> infra project that shows a clear before/after improvement in Time-To-First-Token.

---

## 1. What we are building

A **Shared Context Bridge**: a control plane that reads an agent workflow graph and
proactively keeps the right document context hot before the next agent asks for it,
eliminating wasted GPU prefill in multi-agent pipelines (the "Amnesia Tax").

The contribution is the orchestration layer. The cache engine already exists (the
backend's native prefix cache). We add the control plane on top.

---

## 2. Strategy

- No machine learning. No training. No embeddings on the critical path.
- The orchestrator does not predict what the next agent needs. It reads it from a
  hand-written workflow manifest. The dependency is declared, not inferred.
- Three deterministic mechanisms form the spine:
  1. **Explicit handoff (manifest / DAG chaining)** — PRIMARY signal. The orchestrator
     reads the agent sequence from `workflow/manifest.yaml`. When Agent1 runs on Doc_X
     it already knows Agent2 is next and also needs Doc_X.
  2. **Prefix cache (token hash-trie)** — the correctness floor. Provided FREE by
     vllm-mlx. We do not reimplement it. It guarantees a hit is a real hit.
  3. **Structural fingerprint** — SHA-256 of the heavy block (system + document),
     computed before dispatch. Confirms identity and triggers keep-resident.
- The only adaptive component is a budget-aware LRU keep/evict policy. It is a heuristic,
  not a model. With one document it never triggers; it is the multi-doc scale story.
- SimHash near-duplicate matching is a STRETCH (Phase 7) only, for the robustness Q&A.

---

## 3. The prefix-matching design constraint — CRITICAL

vllm-mlx reuses an **exact token prefix** only. Reuse works ONLY if every agent's
prompt is constructed as:

```
SYSTEM_PREAMBLE + DOCUMENT + "\n\n" + AGENT_SPECIFIC_TASK
```

- `SYSTEM_PREAMBLE + DOCUMENT` must be byte-identical across all agents in a pipeline.
- The document goes FIRST, the agent's question LAST. Never interleave them.
- Result: the shared prefix is an exact cache hit; only the short divergent task tail
  is prefilled fresh.
- We do NOT need a full-prompt match — we need a prefix match, guaranteed by how we
  build the prompt.
- Limitation (state honestly in Q&A): if the document itself differs (whitespace,
  reordered chunks) the exact prefix breaks. Fuzzy/semantic matching is future work.

---

## 4. Hardware constraints — non-negotiable

- Machine: Apple Silicon M4 Pro MacBook Pro, 16 GB unified memory.
- Backend: **vllm-mlx** only. Single path. No mock backend.
- Model: `mlx-community/Llama-3.2-3B-Instruct-4bit` (~2 GB). Drop to a 1B model if
  memory pressure appears. Keep KV cache space modest (macOS uses ~5 GB on 16 GB).
- Python 3.12 only (not 3.13).
- No CUDA. No LMCache on this machine. No NIXL. No multi-node. No P2P/RDMA.
- No Kubernetes / Helm / vLLM Production Stack on this machine.
- Demo-day safety: record a screen capture of a successful run as a fallback video.
  This is a recording, not a code mock.

---

## 5. Architecture & modules

```
agents.py                 3 plain Python functions, openai SDK -> vllm-mlx
   (Screener, Analyst, Auditor) — NO CrewAI, NO framework
        |  each call routed through:
        v
bridge.py                 library wrapper (the "gateway")
  - split heavy block (system+doc) vs task tail
  - fingerprint heavy block (SHA-256)
  - notify orchestrator, then call vllm-mlx
        |  task_started
        v
orchestrator.py           observe -> lookahead -> act -> adapt
  - reads workflow/manifest.yaml (the DAG)
  - act: keep Doc_X resident before next agent (warmup request)
  - adapt: budget-aware LRU keep/evict (trivial for 1 doc)
        |  ttft, hit/miss
        v
telemetry.py              Snowflake batched async writer + CSV fallback
        v
dashboard/                before/after bar chart + GPU-seconds saved + decision log
```

- Agents are sequential Python functions. An "agent" = a role-specific prompt sent to
  the vllm-mlx OpenAI-compatible endpoint. No agent framework.
- `bridge.py` is a library, not a server (a FastAPI proxy is optional polish only).
- All inference goes through `bridge.py`; agents never call vllm-mlx directly.
- Telemetry is fire-and-forget; the hot path never awaits a telemetry write.
- CSV fallback sits behind the same `telemetry.log()` interface in case Snowflake
  connectivity fails during the event.

---

## 6. The demo — before/after multi-agent scenario

Document: a ~10k-token litigation/discovery contract (`data/discovery.txt`).

Pipeline (declared in `workflow/manifest.yaml`):
1. **Screener** — find conflict-of-interest mentions.
2. **Analyst** — find financial-liability clauses (same document).
3. **Auditor** — verify the Analyst's findings against the document (same document).

- **BEFORE (baseline = today's stateless agents):** prepend a unique UUID to each
  prompt so the prefix differs every call -> forces a full re-prefill of all ~10k
  tokens three times. Clean, self-evident control to explain to judges.
- **AFTER (the bridge):** no UUID; document is the shared prefix; orchestrator keeps
  it resident -> Agent1 cold (full prefill), Agents 2 and 3 warm (sub-second TTFT).
- **Show:** side-by-side per-agent TTFT bar chart (before vs after) + total
  GPU-seconds saved. ~3x prefill work collapsing toward ~1x is the headline.

The entire result is one measured quantity: cold TTFT vs warm TTFT, measured for real
on vllm-mlx and logged. Everything else is presentation around that number.

Scale/potential slide (no extra code): with N competing documents and a memory cap,
the budget-aware keep/evict policy chooses what stays hot — this is the enterprise
story (what Tensormesh's control plane does at fleet scale).

---

## 7. Tech stack

| Layer        | Choice                                       |
|--------------|----------------------------------------------|
| Inference    | vllm-mlx                                      |
| Model        | mlx-community/Llama-3.2-3B-Instruct-4bit      |
| Agents       | openai Python SDK (3 functions)               |
| Orchestrator | plain Python state loop (LangGraph optional)  |
| Bridge       | Python library module                         |
| Telemetry    | snowflake-connector-python + CSV fallback     |
| Dashboard    | Streamlit (fastest) or React + Recharts       |
| Lang         | Python 3.12                                   |

---

## 8. Phased plan — three end-to-end phases

Each phase is a complete, independently demoable system. Build the simplest end-to-end
slice first, then add features in major increments. Phase 3 is the final pitched
product. Do NOT start a phase until the prior phase's "done when" is met and the
measured number is logged in `PROGRESS.md`. The module test gates in Section 8a are the
inner loop within each phase; the three phases are the outer loop.

### Phase 1 — Bare thesis, end to end (~3-4 hrs)

Goal: prove the core claim with the minimum. No bridge, orchestrator, manifest,
dashboard, or Snowflake. This alone is a demoable result if everything else collapses.

- Build: vllm-mlx serving the model; `agents.py` with 3 functions (Screener, Analyst,
  Auditor) via the openai SDK; `run.py` that runs the pipeline twice over the same
  ~10-30k-token document — BEFORE mode (unique UUID prepended, forces full re-prefill
  every call) and AFTER mode (document as shared prefix, prefix-cache hits).
- Flow: `run.py` -> 3 sequential agent calls x 2 modes -> console table of per-agent
  TTFT and totals.
- Demo: console table. "Stateless agents re-read the doc 3x: X s. Shared context: Y s.
  Same answers, ~3x less prefill."
- Done when: AFTER Agents 2 and 3 TTFT materially below Agent 1; number real and logged;
  document sized large enough that the delta is visually obvious (bump to ~25-30k
  tokens if 10k looks weak).

### Phase 2 — Agentic control plane, end to end (~6-7 hrs)

Goal: turn the prompt trick into a real bridge with the autonomous narrative visible
and live. This is the target demo and the pitch.

- Build, on top of Phase 1: `bridge.py` (split heavy block vs task tail, fingerprint,
  route every call); `workflow/manifest.yaml` (3-node DAG); `orchestrator.py`
  (observe -> lookahead -> act -> adapt; reads manifest, sees Agent 2 next and needs
  Doc_X, issues keep-resident warmup BEFORE Agent 2's request); `telemetry.py`
  (Snowflake batched async + CSV fallback); Streamlit dashboard (before/after TTFT
  bars, live GPU-seconds-saved counter, orchestrator decision log).
- Flow: `run.py` -> `bridge.py` -> `orchestrator.py` (proactive keep-resident) ->
  vllm-mlx -> `telemetry.py` -> Streamlit reads sink live.
- Demo: full 2-minute story — problem, "we read the workflow graph and warm the cache
  before the next agent asks," live run, counter jumps, decision log scrolls.
- Done when: live run shows the decision log proving keep-resident fired BEFORE
  Agent 2; dashboard renders all three panels from real telemetry; CSV fallback works
  when Snowflake is pulled.

### Phase 3 — Final product: scale + robustness (~4-5 hrs, only if Phase 2 solid)

Goal: complete the pitched product — the enterprise scale story and the robustness
answer that wins Q&A.

- Build, on top of Phase 2: multi-document scenario (2-3 docs, several agents) that
  exercises the budget-aware LRU keep/evict under a memory cap (the fleet-scale slide,
  now real); SimHash near-duplicate matcher so a slightly modified document still
  triggers keep-resident (the "what if not an exact match?" answer); dashboard polish
  (knowledge map of hot docs, eviction events visualized); recorded screen capture of
  a clean run as the demo-day fallback video; 2-minute pitch rehearsed and timed.
- Flow: same as Phase 2, plus orchestrator arbitrates competing documents under the
  cap and the matcher catches near-duplicates before dispatch.
- Demo: complete narrative with the scale slide backed by a live multi-doc eviction,
  and a robustness answer demonstrated not just claimed.
- Done when: multi-doc run shows a real eviction in the log; SimHash catches a
  modified-document near-match live; fallback video recorded; pitch under 2 minutes.

## 8a. Module test gates (inner loop within each phase)

Build and test each module in isolation before wiring the next. Record each gate
result (with the actual measured number) in `PROGRESS.md`.

- **env** (Phase 1): vllm-mlx serves the model; `curl /v1/chat/completions` returns text.
- **agents/run** (Phase 1): BEFORE vs AFTER totals captured; AFTER Agents 2-3 TTFT
  much lower than Agent 1. This is the core proof — do not proceed until real & logged.
- **bridge** (Phase 2): same doc + different task -> identical fingerprint; doc placed
  first so the shared prefix is byte-identical across the 3 agent prompts.
- **orchestrator** (Phase 2): feed the manifest; keep-resident for Doc_X issued before
  Agent 2's request, verified in telemetry.
- **telemetry** (Phase 2): 100 synthetic events written and read back from Snowflake;
  kill Snowflake -> CSV fallback works transparently.
- **dashboard** (Phase 2): before/after chart + GPU-seconds-saved + decision log render
  live from the telemetry sink.
- **multi-doc + matcher** (Phase 3): real eviction event under the cap; SimHash catches
  a near-duplicate before dispatch.

---

## 9. Conventions

- Small single-responsibility modules.
- All inference via `bridge.py`; never call vllm-mlx directly from agents/orchestrator.
- Telemetry calls are non-blocking.
- Type hints + docstrings on public functions; minimal dependencies.
- Commit after each green module gate: `phase-N <gate>: PASS (<number>)`.
- Tag the repo at each phase's "done when": `phase-1-done`, `phase-2-done`,
  `phase-3-done`. Each tag must be an independently demoable state.

---

## 10. Out of scope — do not implement

- ML training / fine-tuning / embeddings-as-prediction / OATS / centroid logic.
- Mock backend.
- CrewAI or any heavy agent framework.
- LMCache on Apple Silicon. NIXL. RDMA. Multi-node. P2P. CUDA. Kubernetes.
- Anything that cannot be shown in a 2-minute pitch.

If any request conflicts with sections 2, 3, 4, or 10, follow this file and flag it.
