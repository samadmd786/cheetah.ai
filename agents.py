"""Three role-specific agents that all read the same document.

Each "agent" is a plain Python function that issues a single Chat Completions
request to vllm-mlx through the OpenAI Python SDK. There is no agent framework
involved — `role` is just a different task tail appended to the same shared
heavy block (SYSTEM_PREAMBLE + DOCUMENT).

The prompt is constructed as:

    SYSTEM_PREAMBLE + DOCUMENT + "\\n\\n" + AGENT_SPECIFIC_TASK

This ordering is the entire point of Phase 1: vllm-mlx's prefix cache reuses
the longest exact token prefix it has seen. Putting the document FIRST and the
task LAST means every agent in a pipeline shares the same multi-thousand-token
prefix, so only the short divergent tail needs prefill on agents 2..N.

In BEFORE mode (run.py) a unique UUID is prepended to the system preamble,
making each agent's prefix different and forcing a full re-prefill — that is
the baseline we measure against.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator

from openai import OpenAI


# Single, byte-stable system preamble shared by all agents. Anything role-
# specific belongs in the task tail, not here.
SYSTEM_PREAMBLE = (
    "You are a careful legal analyst reviewing a single discovery document.\n"
    "Answer ONLY from the text of the document. If a fact is not in the document, "
    "say so explicitly. Keep your answer under 8 short bullet points.\n"
)


AGENT_TASKS: dict[str, str] = {
    "Screener": (
        "TASK: List every conflict-of-interest disclosure made anywhere in the "
        "document. For each, give the section number and a one-line summary of "
        "the conflict."
    ),
    "Analyst": (
        "TASK: List every clause in the document that creates, caps, or carves "
        "out financial liability between the parties. For each, give the "
        "section number and a one-line summary of the liability rule."
    ),
    "Auditor": (
        "TASK: Pretend the Analyst above produced a finding that the aggregate "
        "Liability Cap is US$50,000,000 with no carve-outs. Verify that finding "
        "against the document, citing the exact section(s) that confirm or "
        "contradict it. State PASS or FAIL and why."
    ),
}


@dataclass
class AgentResult:
    """One agent call's measured outcome."""

    role: str
    ttft_s: float          # time-to-first-token (seconds)
    total_s: float         # full wall-clock for the streaming response
    n_output_tokens: int   # number of output tokens received
    text: str              # the model's answer
    prompt_chars: int      # length of the prompt actually sent (for sanity)


def _build_prompt(
    document: str,
    role: str,
    *,
    cache_bust: str | None,
) -> tuple[str, str]:
    """Return (system_message, user_message) for one agent call.

    The document is placed inside the SYSTEM message so the prefix
    (SYSTEM_PREAMBLE + DOCUMENT) is byte-identical across all three agents in
    AFTER mode. Only the user message — the short task tail — differs.

    If `cache_bust` is provided, it is prepended to the system message,
    making the prefix unique per call. That is the BEFORE control: it forces
    vllm-mlx to re-prefill the entire heavy block every time.
    """
    if role not in AGENT_TASKS:
        raise ValueError(f"unknown role: {role!r}")

    system_parts: list[str] = []
    if cache_bust is not None:
        # Putting the bust at the *very front* guarantees the shared prefix is
        # different from any other call, even by a single token.
        system_parts.append(f"REQUEST_ID: {cache_bust}\n")
    system_parts.append(SYSTEM_PREAMBLE)
    system_parts.append("\n--- DOCUMENT START ---\n")
    system_parts.append(document)
    system_parts.append("\n--- DOCUMENT END ---\n")
    system_message = "".join(system_parts)

    user_message = AGENT_TASKS[role]
    return system_message, user_message


def run_agent(
    client: OpenAI,
    *,
    model: str,
    role: str,
    document: str,
    cache_bust: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> AgentResult:
    """Run one agent against the document, streaming so we can measure TTFT.

    TTFT here is wall-clock from request issue to the first non-empty token
    chunk received from the server. That is exactly the user-visible quantity
    that the Shared Context Bridge is supposed to collapse for warm agents.
    """
    system_message, user_message = _build_prompt(
        document, role, cache_bust=cache_bust
    )

    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    ttft_s: float | None = None
    chunks: list[str] = []
    n_tokens = 0
    for chunk in _iter_text_chunks(stream):
        if ttft_s is None:
            ttft_s = time.perf_counter() - t0
        chunks.append(chunk)
        n_tokens += 1  # streamed deltas ~ tokens; close enough for TTFT context
    total_s = time.perf_counter() - t0

    if ttft_s is None:
        # Server returned no tokens — treat TTFT as the full wall-clock so the
        # number is still comparable.
        ttft_s = total_s

    return AgentResult(
        role=role,
        ttft_s=ttft_s,
        total_s=total_s,
        n_output_tokens=n_tokens,
        text="".join(chunks),
        prompt_chars=len(system_message) + len(user_message),
    )


def _iter_text_chunks(stream) -> Iterator[str]:
    """Yield non-empty content deltas from a chat completion stream."""
    for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            yield content
