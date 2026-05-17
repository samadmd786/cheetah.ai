# Video voiceover — 50 seconds

Quick note — we trimmed the waiting time out of this clip so it fits in under a minute. Every TTFT number you see on screen is real.

Multi-agent AI pays the amnesia tax — every agent re-prefills the same documents from scratch.

Stage 1, BEFORE. Three agents, one document, stateless prompts. Thirty-five seconds, cold. Another thirty-five — same doc, but stateless agents can't share the cache. And another. A hundred and ten seconds burned re-reading the same text three times.

Stage 2, our solution. Three agents — but now on three different documents, the realistic case. The first agent is cold; no cache can serve a doc it's never seen.

But watch between agents. Our orchestrator reads the workflow graph, sees what the next agent needs, and fires a keep-resident warmup in the gap — pre-prefilling the next doc while the current agent is still streaming. SimHash spots near-duplicates. Budget-aware eviction handles overflow.

Next agent: half a second. And again, half a second.

A hundred and ten seconds becomes forty. Two-point-seven times faster, on a harder workload. Same cache underneath — we just schedule it.
