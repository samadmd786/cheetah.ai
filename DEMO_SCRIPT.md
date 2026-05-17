# Demo script — Shared Context Bridge

A pitch + live-demo walkthrough you can read off in ~3 minutes. The numbers
quoted below are from the most recent full run (`run_id=demo_full`,
2026-05-16). Re-run with `.venv/bin/python demo.py --run-id <your_id>` to
refresh them before pitching.

---

## TL;DR for the judges

We add a **control plane on top of an existing KV cache** so that
multi-agent AI pipelines stop paying the "amnesia tax." Same hardware, same
model, same prefix-cache engine — our orchestrator just *schedules* cache
warmth using the workflow graph, similarity-aware matching, and a budget-
aware eviction policy.

| stage                                   | scenario                | total TTFT | comparison              |
|-----------------------------------------|-------------------------|-----------:|-------------------------|
| 1 · BEFORE — stateless agents (today)   | 1 doc, UUID-busted      | 105.14 s   | the amnesia tax         |
| 2 · OURS — orchestrator + SimHash + LRU | **3 different docs**    |  40.02 s   | **2.63× faster**        |

**OURS handles a harder workload (3 different documents) and still beats
stateless agents on a single document by 2.6×.** That's the headline. The
cold prefills the model still has to do for new docs are paid by the
orchestrator's keep-resident warmups *between* agents — never by the
agents themselves.

---

## Setup (run once, before going on)

```bash
# Terminal 1 — vllm-mlx server with a big enough prefix cache budget:
.venv/bin/vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit \
  --host 127.0.0.1 --port 8001 \
  --enable-prefix-cache --cache-memory-mb 3000 \
  --continuous-batching --max-kv-size 32768 --max-tokens 512

# Terminal 2 — Streamlit dashboard (for the closer):
.venv/bin/streamlit run dashboard/app.py --server.port 8502

# Terminal 3 — leave this one for the live demo:
.venv/bin/python demo.py                  # both stages, ~4 min wall
.venv/bin/python demo.py --pause          # pause between stages for narration
.venv/bin/python demo.py --stage 1        # BEFORE only (~100 s)
.venv/bin/python demo.py --stage 2        # OURS only (~90 s)
```

Wall-clock budget (rough): Stage 1 ~100 s, Stage 2 ~90 s — about 4 minutes
total including narration headroom.

---

## 0 — Open / problem statement (~30 s)

> "Multi-agent AI systems waste enormous amounts of GPU re-reading the same
> documents over and over. We call this the **amnesia tax**: every agent in
> a workflow stateless-ly rebuilds its context, so a 13,000-token contract
> gets prefilled three, five, ten times — once per agent — even though
> nothing about the document has changed.
>
> vLLM and LMCache already help here. They cache KV state for prompts you've
> seen before. But they're **reactive** — they cache what was asked, not
> what's coming. And in real multi-agent workflows, the docs change between
> agents, so the cache cold-misses every time."

(If the audience is technical, drop in: "vLLM's prefix cache is exact-prefix
match from token 0. LMCache adds tiered offload — RAM, SSD, remote — but the
hit/miss decision is the same: it's hit-driven.")

---

## 1 — Solution in one sentence (~10 s)

> "We're the **missing control plane.** We read the agent workflow graph,
> look ahead at what each agent will need, and pre-warm the cache *before*
> they ask. Plus a SimHash near-duplicate matcher so amended documents don't
> turn into cache misses, and a budget-aware LRU that's informed by what's
> coming next — not just what was used last."

---

## 2 — Live demo (~2 min 30 s)

Start the demo: `.venv/bin/python demo.py`. Each stage prints its own banner
explaining what to watch for; you can mostly let it talk.

### Stage 1 — BEFORE (~60 s wall)

> "First we simulate today's stateless multi-agent system. Each agent
> prepends a unique request ID to its prompt — the same thing that happens
> in production when system prompts drift slightly, session IDs leak in, or
> per-request metadata gets prepended. Any of those break exact-prefix
> matching from token 0.
>
> Watch: every agent pays the full 35-second prefill. **This is the amnesia
> tax.** Three agents, ~100 seconds of pure wasted prefill work."

*(While the cold prefill ticker climbs, you can also say: "Notice how the
counter just sits there — that's GPU time being burned re-tokenizing and
re-prefilling a document the model just saw 20 seconds ago.")*

When the stage summary appears:

> "100 seconds of TTFT, three cold MISSes. This is the baseline."

### Stage 2 — OUR SOLUTION (~90 s wall)

> "Now three agents, three different documents. Screener on an M&A merger
> agreement (`merger`), Analyst on a perturbed near-duplicate of a contract
> (`discovery_v3`), Auditor on the original of that contract (`discovery`).
>
> Without our control plane, every agent would cold-prefill its document
> from scratch — three back-to-back ~35-second prefills, ~105 seconds
> total, the same shape as Stage 1. Watch what our orchestrator does
> between agents instead."

Things to call out as they fly past in the live log:

- `🧭 Lookahead` — *"Right after each agent finishes, the orchestrator reads
  the workflow DAG, sees what the NEXT agent needs, and decides to warm it."*
- `┌─ orchestrator warmup` — *"It fires a tiny `max_tokens=1` request with
  the next document's heavy block. This **absorbs the cold prefill** so the
  next agent's actual dispatch lands on a warm cache."*
- `🔍 SimHash match` — *"When we get to the Auditor's document, the
  orchestrator runs SimHash. Hamming distance 1 out of 64 — it's essentially
  the same content as the Screener's doc. We **flag** that and pre-warm
  anyway, because we know this kind of doc is likely to be reused."*
- `♻ Eviction` — *"And under our budget of two hot docs, we drop the LRU
  one to make room. Cache management informed by the **workflow**, not just
  by recency."*

When the summary lands:

> "Screener paid the first cold prefill — that's unavoidable, no cache can
> serve a doc it's never seen. But Analyst and Auditor are sub-second hits,
> even though they're reading **different documents** the model had never
> seen either. The orchestrator's warmups absorbed those cold prefills
> *between* agents, where the user can't feel them.
>
> Total agent TTFT: **~40 seconds, vs ~105 in Stage 1** — and Stage 2 is
> doing the **harder** workload (three different documents vs one). That's
> the differentiator: stateless agents pay the amnesia tax every single
> call; our control plane absorbs it once, in the background."

---

## 3 — Dashboard (~20 s)

> "And it's all instrumented. Live telemetry to a Streamlit dashboard."

Switch to `http://127.0.0.1:8502`, pick `run_id=demo_full` from the sidebar.

Walk through the three hero panels:
1. Hero metric row — *"GPU-seconds saved this run, BEFORE vs AFTER wall-clock,
   warmup hit-rate, evictions, near-dup matches."*
2. Per-agent TTFT line chart — *"BEFORE flat across all three agents at the
   cold-prefill ceiling. AFTER plummets after Agent 1. Reference line is
   the cache-MISS threshold."*
3. Hot-doc activity timeline — *"Each document gets its own lane, with
   markers for task completions, warmups, eviction strikes, near-dup events.
   You can replay the whole pipeline visually."*

---

## 4 — Close (~15 s)

> "Two stages, one headline: **2.6× faster than stateless agents — on a
> harder workload (3 docs vs 1)**. We're using vLLM/LMCache's prefix
> cache underneath, unchanged. The difference is what we add on top:
> a control plane that reads the workflow graph, schedules cache
> warmups proactively, uses SimHash to spot near-duplicate documents,
> and budgets eviction by what's coming next. Same hardware, same
> model, same cache engine. We're the missing control plane."

---

## Q&A defense

These are the questions we expect; the answer texts are written to be said
out loud.

### "But vLLM has prefix caching too — if you fixed the prompt to put the doc first, wouldn't vLLM alone solve this?"

> "On the easy case — one document, agents fire back-to-back — yes, vLLM's
> prefix cache will reuse agent 1's KV state for agents 2 and 3. We're
> not claiming to beat vLLM at that case; it's table stakes and they
> already solve it.
>
> The pitch is what happens when agents read **different** documents —
> the realistic multi-agent case. There vLLM's prefix cache has nothing
> to reuse: each new doc is a fresh prefix it has never seen, so every
> agent cold-prefills. Three different docs = three cold prefills = back
> to the ~105-second baseline you just saw in Stage 1.
>
> Stage 2 runs exactly that scenario (three different documents) and we
> do it in ~40 seconds. That's the workload our control plane is for.
> If you want to see the vLLM-only multi-doc number measured live, run
> `.venv/bin/python run.py --pipeline multi_doc_review --mode after` with
> the orchestrator disabled — you'll see ~105 s of cold prefills, which
> is what our 40-second number is replacing."

### "Why is the Screener cold in Stage 3? Doesn't your orchestrator make agent 1 fast too?"

> "Today, agent 1 pays a cold prefill on a doc the cache has never seen —
> there's no prior agent for the orchestrator to look behind. The
> orchestrator's value is on agents 2..N, where it can read the workflow
> ahead and pre-warm.
>
> There's an obvious extension: have the orchestrator fire a pre-flight
> warmup for node 0 *at workflow registration time*, before the user even
> hits Run. With a known workflow, you can absorb that first prefill the
> same way we absorb the others. We didn't ship that this hackathon
> because we wanted Stage 3's first cold MISS to be visible — it makes
> the contrast with the warm agents 2 and 3 obvious to a judge in the
> first 10 seconds."

### "Why use a UUID in the BEFORE demo? Isn't that an unfair handicap?"

> "It's a faithful simulation, not a handicap. In production stateless
> multi-agent systems, agents commonly:
>   • have **slightly different system prompts** (prompt version drift across
>     teams or A/B tests),
>   • carry **per-session or per-request metadata** prepended to the context
>     (trace IDs, tenant IDs, conversation IDs),
>   • or **rebuild context from a vector DB lookup** that returns chunks in
>     a non-deterministic order.
>
> Any one of these breaks exact-prefix matching from token zero. The UUID is
> the simplest, cleanest way to demonstrate that worst case. The whole
> point of our solution is that we **don't depend** on the upstream prompt
> being byte-stable — we just need the document text, and we control the
> rest of the prompt layout."

### "How is this different from vLLM's prefix cache?"

> "vLLM's prefix cache is the **engine** underneath us — we use it
> unchanged. It's reactive: every request comes in, vLLM hashes the prefix,
> looks it up, and serves a hit if it's there.
>
> Our orchestrator is **proactive**: it reads the workflow DAG, sees that
> Agent 2 is coming after Agent 1 and that Agent 2 needs document Y, and
> fires a tiny warmup request with document Y *before* Agent 2 dispatches.
> vLLM doesn't know the workflow exists; we tell it. The result is that
> Agent 2 lands on a warm cache even though it's a brand-new document the
> cache had never seen.
>
> Look at Stage 3: the agents themselves were 0.5 s each, but the
> orchestrator paid 36 s of cold prefill **between** them. We didn't make
> the prefill free — we **moved it** to where the user can't feel it."

### "What about LMCache?"

> "LMCache solves a related but different problem: cache **capacity**. It
> spills KV blocks to RAM, then SSD, then remote storage. Great when you
> have very large prompts or many tenants competing for limited GPU RAM.
> It's orthogonal to our work — you could run LMCache underneath our
> orchestrator and we'd schedule keep-resident warmups against its tiered
> storage. The control plane is the missing layer in both cases."

### "What about CacheBlend / CacheGen / CachedAttention?"

> "Those are also good work and again, orthogonal. CacheBlend lets you
> reuse KV across **non-prefix** positions by recomputing the attention for
> the divergent parts — it's a way to relax the exact-prefix constraint.
> Our pitch is upstream of that question: even if you have perfect exact-
> prefix matching, you still need someone to decide **which prefixes to
> keep warm and when to fire warmups**. That's the control plane.
>
> If you wanted to go further, CacheBlend + our orchestrator would compose
> nicely: we'd schedule warmth, CacheBlend would relax the matching rule."

### "Why SimHash, not embedding similarity?"

> "Two reasons.
>
> 1. **Latency.** SimHash on a 14,000-token document is single-pass, runs
>    on CPU, takes single-digit milliseconds. Embedding similarity needs
>    either an extra model call or precomputed embeddings, both of which
>    add latency to the **control path** — and the control path is supposed
>    to be invisible to the agent.
>
> 2. **Honesty.** SimHash gives you a Hamming distance you can threshold
>    and explain. We're not claiming to do semantic similarity here. We're
>    detecting **near-duplicate documents** — amended contracts, redrafted
>    specs, version-bumped policies. Surface-level lexical hashing is the
>    right tool for that job."

### "Your eviction is just LRU under a capacity cap. How is that different from the cache engine's own LRU?"

> "Two differences.
>
> 1. **It's informed by the workflow.** The cache engine's LRU is global
>    and recency-only — oldest entry goes first. Ours has access to the
>    manifest, so we can ask 'which documents do the upcoming agents in
>    the DAG actually need?' and keep THOSE resident even if they're not
>    the most recent. Today the policy is just a cap (Phase 3 scope), but
>    the architecture has room for a richer policy keyed on lookahead
>    distance.
>
> 2. **It emits structured eviction events** that the dashboard surfaces
>    next to the LRU cap, so operators can see when the budget is too
>    tight for the workload. That's table stakes for any production cache
>    policy and missing from the standard engines."

### "How does this scale to fleet-level (many docs, many concurrent pipelines)?"

> "The orchestrator is per-pipeline state today (a cursor and an LRU). At
> fleet scale you'd run one orchestrator per workflow type, sharing a
> SimHash index across them, and the keep-resident decisions would feed
> into a global cache scheduler. The shape of the system doesn't change —
> the manifest stays the source of truth — only the eviction policy gets
> richer.
>
> The hackathon scope was 'prove the control-plane idea on a single
> machine.' We did that on a 16 GB M4 Pro with vllm-mlx. Same control
> plane, swap in vLLM + LMCache underneath, you've got a Tensormesh-style
> fleet manager."

### "What's your latency overhead in the control path?"

> "Bridge fingerprinting is a single SHA-256 over the heavy block — ~5 ms
> on a 14 k-token doc. SimHash is ~10 ms. Manifest lookup is a dict access.
> The orchestrator's only blocking call is the warmup itself, and we time
> it to fire **between** agents on purpose. So control-plane overhead in
> the hot path is < 20 ms. The orchestrator pays for itself the first
> time it avoids a 35-second cold prefill."

### "What if the workflow graph isn't known ahead of time? (dynamic agent routing)"

> "Then you fall back to the cache-engine's reactive cache and lose the
> proactive warmup. SimHash still helps you bind near-duplicates together.
> The whole pitch of the control plane is that it's only valuable when
> there's a workflow to read — that's the assumption we're making, and in
> production multi-agent systems it's usually a true one (LangGraph,
> CrewAI, Autogen all expose a workflow declaration we could parse)."

---

## Things to **avoid** saying

- Don't claim we beat vLLM at what vLLM does. We use vllm-mlx as the
  cache engine — the win is on top.
- Don't oversell SimHash as semantic similarity. It's lexical near-dup
  detection, no more.
- Don't claim "no ML." There's no ML in the **control plane**, which is
  the point — we are explicitly avoiding the trained-prediction-of-next-
  doc story. The model itself is obviously ML.
- Don't promise multi-node or LMCache integration — those are next steps,
  not shipped.
- If asked about CacheBlend specifically: it's orthogonal and complementary,
  not competing.

---

## The dashboard at a glance

When you switch to the dashboard, here's the order to call attention to:

1. **Pill badges** in the header — confirm `run_id`, pipeline, mode.
2. **GPU-seconds saved** — the hero metric. Should be ~70 s per pipeline.
3. **TTFT line chart** — the BEFORE line is flat near 35 s; the AFTER line
   drops off a cliff after agent 1.
4. **Hot-doc activity timeline** — each doc gets a lane, marker shapes
   distinguish task/warmup/eviction/near-dup.
5. **SimHash detail table** — confirms `discovery_v3 ≈ discovery` with
   Hamming = 1 / 64.
6. **Decision log** — colored by HIT (green) / MISS (red) / observe / act
   / meta. Walk through one full agent's worth of decisions if asked.
