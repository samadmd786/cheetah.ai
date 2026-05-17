"""bridge.py — the only gateway to vllm-mlx (CLAUDE.md §5).

Every agent call goes through `Bridge.dispatch(...)`. The bridge:
  1. Splits the prompt into the heavy block (SYSTEM_PREAMBLE + DOCUMENT) and
     the divergent task tail (the AGENT_SPECIFIC_TASK). The doc is *always*
     placed first so the shared prefix is byte-identical across all agents in
     a pipeline (CLAUDE.md §3).
  2. Fingerprints the heavy block with SHA-256 — this is the identity key the
     orchestrator uses for keep-resident decisions.
  3. Notifies the orchestrator BEFORE dispatch (`observe`) so the orchestrator
     can fire a `keep_resident` warmup ahead of the next agent.
  4. Sends the actual chat completion, streamed, measuring TTFT and total.
  5. Emits a `task_completed` telemetry event.

Crucial: the bridge does not decide cache policy. It just measures, fingerprints,
and reports. The orchestrator is the brain; the bridge is the nerve.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from typing import Iterator, Protocol

from openai import OpenAI

from telemetry import Telemetry


# Single, byte-stable system preamble. Anything role-specific must NOT live here.
SYSTEM_PREAMBLE = (
    "You are a careful legal analyst reviewing a single discovery document.\n"
    "Answer ONLY from the text of the document. If a fact is not in the document, "
    "say so explicitly. Keep your answer under 8 short bullet points.\n"
)


# TTFT (in seconds) above which we conclude a call was a cache MISS. The cold
# prefill of a ~13k-token doc on M4 Pro is ~30-35s and a warm cache hit is
# ~0.5s, so any threshold in between would work; 3s leaves a wide margin.
CACHE_MISS_TTFT_SECONDS = 3.0


class OrchestratorLike(Protocol):
    """Just the slice of the orchestrator the bridge knows about."""

    def observe(self, *, role: str, doc_id: str, fingerprint: str) -> None: ...

    def on_task_completed(self, *, role: str, doc_id: str) -> None: ...


@dataclass
class DispatchResult:
    role: str
    doc_id: str
    fingerprint: str
    ttft_s: float
    total_s: float
    n_output_tokens: int
    text: str
    cache_hit: bool
    prompt_chars: int


def fingerprint_heavy(system_preamble: str, document: str) -> str:
    """SHA-256 of the (preamble + document) bytes — identity key for the prefix.

    The bridge gate (CLAUDE.md §8a) is: "same doc + different task → identical
    fingerprint". Hashing only the heavy block achieves that by construction.
    """
    h = hashlib.sha256()
    h.update(system_preamble.encode("utf-8"))
    h.update(document.encode("utf-8"))
    return h.hexdigest()


def build_messages(
    *,
    document: str,
    task: str,
    cache_bust: str | None = None,
) -> tuple[list[dict[str, str]], str, str]:
    """Build the OpenAI messages array using the doc-first prompt layout.

    Returns (messages, heavy_block, task_tail). The heavy_block is what we
    fingerprint; the task_tail is what makes each agent's call unique. In
    BEFORE mode (UUID bust), the bust is prepended to the heavy block so the
    fingerprint deliberately differs every call — that proves to the dashboard
    that the orchestrator's keep-resident is correctly seen as ineffective.
    """
    heavy_parts: list[str] = []
    if cache_bust is not None:
        heavy_parts.append(f"REQUEST_ID: {cache_bust}\n")
    heavy_parts.append(SYSTEM_PREAMBLE)
    heavy_parts.append("\n--- DOCUMENT START ---\n")
    heavy_parts.append(document)
    heavy_parts.append("\n--- DOCUMENT END ---\n")
    heavy_block = "".join(heavy_parts)

    messages = [
        {"role": "system", "content": heavy_block},
        {"role": "user", "content": task},
    ]
    return messages, heavy_block, task


class Bridge:
    """Single gateway to vllm-mlx for both real agent calls and warmups."""

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        telemetry: Telemetry,
        orchestrator: OrchestratorLike | None = None,
        mode: str = "after",
    ) -> None:
        self.client = client
        self.model = model
        self.telemetry = telemetry
        self.orchestrator = orchestrator
        self.mode = mode  # "before" | "after" — propagated to telemetry rows.

    # ------------------------------------------------------------------ dispatch

    def dispatch(
        self,
        *,
        role: str,
        doc_id: str,
        document: str,
        task: str,
        max_tokens: int = 192,
        temperature: float = 0.0,
        cache_bust: str | None = None,
    ) -> DispatchResult:
        """Run one real agent call end-to-end. Notifies orchestrator first."""
        messages, heavy_block, _ = build_messages(
            document=document, task=task, cache_bust=cache_bust
        )
        fingerprint = fingerprint_heavy(
            SYSTEM_PREAMBLE if cache_bust is None else f"REQUEST_ID: {cache_bust}\n" + SYSTEM_PREAMBLE,
            document,
        )

        # Observe BEFORE dispatch so the orchestrator can fire a keep-resident
        # for the *next* agent before this one even returns. (In our sequential
        # pipeline the orchestrator actually fires its warmup after the current
        # call completes — but the observation point belongs here at the seam.)
        if self.orchestrator is not None:
            self.orchestrator.observe(
                role=role, doc_id=doc_id, fingerprint=fingerprint
            )

        ttft_s, total_s, n_tokens, text = self._stream_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        cache_hit = ttft_s < CACHE_MISS_TTFT_SECONDS

        self.telemetry.log(
            "task_completed",
            mode=self.mode,
            role=role,
            doc_id=doc_id,
            fingerprint=fingerprint,
            ttft_s=ttft_s,
            total_s=total_s,
            n_output_tokens=n_tokens,
            cache_hit=cache_hit,
            prompt_chars=len(heavy_block) + len(task),
        )

        # Hand control to the orchestrator AFTER the real call returns. This is
        # where the lookahead keep-resident fires — before the next agent's
        # dispatch arrives. Doing this here (rather than in run.py) keeps the
        # caller dumb and ensures every dispatch goes through the same seam.
        if self.orchestrator is not None:
            self.orchestrator.on_task_completed(role=role, doc_id=doc_id)

        return DispatchResult(
            role=role,
            doc_id=doc_id,
            fingerprint=fingerprint,
            ttft_s=ttft_s,
            total_s=total_s,
            n_output_tokens=n_tokens,
            text=text,
            cache_hit=cache_hit,
            prompt_chars=len(heavy_block) + len(task),
        )

    # ------------------------------------------------------------------ warmup

    def keep_resident(
        self,
        *,
        doc_id: str,
        document: str,
        cache_bust: str | None = None,
    ) -> DispatchResult:
        """Issue a tiny request whose only purpose is to keep the heavy
        prefix hot in vllm-mlx's prefix cache.

        We send `max_tokens=1` and an empty-task tail so:
          - the request *shape* matches what real agents send (same heavy
            block, just a shorter tail), guaranteeing the prefix is identical
            in token space,
          - the model produces ~1 token and exits, so the cost is dominated
            by prefill (which is exactly what we want to amortize),
          - the LRU touch keeps the entry from being evicted by other work.
        """
        warmup_task = "Reply with the single token: OK."
        messages, heavy_block, _ = build_messages(
            document=document, task=warmup_task, cache_bust=cache_bust
        )
        fingerprint = fingerprint_heavy(
            SYSTEM_PREAMBLE if cache_bust is None else f"REQUEST_ID: {cache_bust}\n" + SYSTEM_PREAMBLE,
            document,
        )

        ttft_s, total_s, n_tokens, text = self._stream_completion(
            messages=messages, max_tokens=4, temperature=0.0
        )
        cache_hit = ttft_s < CACHE_MISS_TTFT_SECONDS

        self.telemetry.log(
            "keep_resident_completed",
            mode=self.mode,
            role="orchestrator",
            doc_id=doc_id,
            fingerprint=fingerprint,
            ttft_s=ttft_s,
            total_s=total_s,
            n_output_tokens=n_tokens,
            cache_hit=cache_hit,
            prompt_chars=len(heavy_block) + len(warmup_task),
            message="warmup issued by orchestrator",
        )

        return DispatchResult(
            role="orchestrator",
            doc_id=doc_id,
            fingerprint=fingerprint,
            ttft_s=ttft_s,
            total_s=total_s,
            n_output_tokens=n_tokens,
            text=text,
            cache_hit=cache_hit,
            prompt_chars=len(heavy_block) + len(warmup_task),
        )

    # ------------------------------------------------------------------ internal

    def _stream_completion(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> tuple[float, float, int, str]:
        t0 = time.perf_counter()
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        ttft_s: float | None = None
        chunks: list[str] = []
        n = 0
        for piece in _iter_text_chunks(stream):
            if ttft_s is None:
                ttft_s = time.perf_counter() - t0
            chunks.append(piece)
            n += 1
        total_s = time.perf_counter() - t0
        if ttft_s is None:
            ttft_s = total_s
        return ttft_s, total_s, n, "".join(chunks)


def _iter_text_chunks(stream) -> Iterator[str]:
    for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            yield content


# Helper for ad-hoc cache-bust generation, kept here so callers don't have to
# import uuid themselves.
def fresh_cache_bust() -> str:
    return uuid.uuid4().hex
