### Inspiration
Modern multi-agent pipelines - legal reviewers, financial analysts, document auditors - share the same documents across many agents. But every agent is stateless. Each one rebuilds its context from scratch, re-reading the same document the model just processed 30 seconds ago. We called this the **amnesia tax**. 

The root cause is mathematical. Standard prefix caches like RadixAttention reuse KV state only up to the first token mismatch. The moment an agent's system prompt diverges at token 0 - different role, different task wrapping the same document - the entire shared context is a cold miss. Cache hits per pipeline: zero. We wanted to fix it not by changing the model or the cache engine, but by adding the layer that was missing: a control plane that reads the workflow graph and prepares the cache before the next agent asks.

### What We Built
Cheetah.ai is a shared context bridge for multi-agent inference. It sits between your agent framework and your serving engine and provides four core pillars:

**1. The Bridge** 
Prefix caches fail the moment a single byte changes at the start of a prompt. The Bridge restructures each agent's prompt to enforce a canonical shape: `[SYS] + [DOC] + [TASK]`. The heavy document block is injected first as a byte-stable prefix. Agent-specific instructions are appended downstream. A SHA-256 fingerprint over the document block proves byte-identity across agents, even when system prompts drift. The result: every agent in the pipeline hits the same cached prefix with zero retraining and zero overhead.

**2. The Orchestrator** 
Between agent calls, the GPU sits idle. We use that dead time. The orchestrator reads the workflow manifest - a YAML DAG declaring which agents run, in what order, reading which documents. After Agent (i) completes, the orchestrator looks ahead to Agent (i+1), identifies its document, and fires a `keep_resident` warmup request *before* Agent (i+1) dispatches. This works for any document because the orchestrator knows what is coming from the DAG, not from traffic history. By the time Agent (i+1)'s request lands, the cache is already hot.

**3. Robustness** 
Real documents get amended. A 64-bit SimHash over the document block catches near-duplicates - whitespace edits, swapped numbers, reordered clauses - with a Hamming distance threshold of `≤ 10 / 64`. A near-match still triggers keep-resident, so a minor edit does not become a cold miss. Eviction is also forward-looking. Standard LRU drops the least recently used entry. Ours drops what the next agent in the DAG will *not* need, keeping the upcoming document resident regardless of recency.

**4. Snowflake Analytics & AI Layer** 
We turned Snowflake from a passive data warehouse into the active AI and observability backbone of our control plane. Every orchestrator decision, cache hit, and eviction is asynchronously streamed to a live **Snowflake sink**. Instead of just drawing static charts, we use **Snowflake Dynamic Tables** to power a real-time leaderboard aggregating TTFT speedups and GPU-seconds saved - no external schedulers needed. We also leverage **Snowflake Cortex AI** (`llama3.1-70b`) directly within the database to read the orchestrator's decision log and automatically generate a natural-language run summary narrating how the pipeline performed.

### Results
Tested on an Apple M4 Pro (16 GB) with `vllm-mlx`: 

| Scenario | Total TTFT |
|---|---|
| Stateless agents, UUID-busted prefixes | 104 s |
| **Cheetah.ai, 3 different documents** | **36 s** |

**2.6x faster - on a harder workload.** Agents 2 and 3 landed in under 0.5 seconds each. The orchestrator absorbed their cold prefills between agents, where the user cannot feel them. 
**GPU-seconds saved ≈ 68 s per pipeline.**

### Challenges
The hardest problem was the prompt restructuring guarantee. Agents in the wild prepend role text, session IDs, and metadata before the document - exactly what breaks prefix caching. Getting the Bridge to enforce `[DOC]` at position 0 without breaking downstream task instructions required careful prompt boundary detection. 

The second challenge was timing the orchestrator warmups correctly. Fire too early and the warmup competes with the active agent for GPU memory. Fire too late and Agent (i+1) dispatches before the cache is warm. We solved this by hooking into the agent completion event and issuing warmups with `max_tokens=1` - enough to force a full prefill of the document block, but cheap enough to not interfere. 

Finally, integrating the live Snowflake telemetry without slowing down the hot inference path was a hurdle. We had to build a non-blocking, batched queue system that fires observability data to Snowflake asynchronously.

### Future Improvements
One of the key features is that the system will work across virtually any open-source model architecture. Depending on which model an agent uses, the runtime automatically selects and manages the appropriate KV cache format and attention implementation for that architecture. The orchestrator handles cache coordination between agent calls, ensuring cache creation and minimizing cache misses or unnecessary recomputation, so the user experiences a seamless low-latency workflow even when different agents or models are involved.

### What We Learned

We also learned that using Snowflake's Cortex AI directly inside the data platform drastically simplifies building real-time observability. By bringing the LLM to the data instead of moving data out to an LLM, we were able to instantly transform raw logs into human-readable narratives of our pipeline's performance.

