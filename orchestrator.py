"""orchestrator.py — observe / lookahead / act / adapt.

This is the brain of the Shared Context Bridge (CLAUDE.md §5).

Loop:
  * **observe**  — receive `bridge.observe(role, doc_id, fingerprint)` at the
                   start of each agent call.
  * **lookahead** — read `workflow/manifest.yaml`, find the next node in the
                   pipeline, and determine the doc it will need.
  * **act**      — if the next agent uses a doc that should be hot, fire a
                   `bridge.keep_resident(...)` warmup AFTER the current call
                   completes but BEFORE the next agent's call lands. Emit a
                   `decision` telemetry event in either case (with reason),
                   so the dashboard's decision log scrolls.
  * **adapt**    — track which docs are "hot" via a tiny LRU. With one doc this
                   is trivial; the multi-doc story is Phase 3.

The orchestrator does NOT predict what the next agent needs — it reads it from
the manifest. The dependency is declared, not inferred (CLAUDE.md §2).
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from simhash import (
    NEAR_DUP_THRESHOLD,
    SIMHASH_BITS,
    find_near_match,
    hamming,
    simhash,
)
from telemetry import Telemetry


REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class Node:
    role: str
    doc_id: str


@dataclass
class Pipeline:
    name: str
    nodes: list[Node]


@dataclass
class Document:
    doc_id: str
    path: Path


@dataclass
class Manifest:
    pipelines: dict[str, Pipeline]
    documents: dict[str, Document]

    @classmethod
    def load(cls, path: Path | str) -> "Manifest":
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        pipelines: dict[str, Pipeline] = {}
        for name, body in (data.get("pipelines") or {}).items():
            nodes = [Node(role=n["role"], doc_id=n["doc_id"]) for n in body["nodes"]]
            pipelines[name] = Pipeline(name=name, nodes=nodes)
        documents: dict[str, Document] = {}
        for doc_id, body in (data.get("documents") or {}).items():
            documents[doc_id] = Document(doc_id=doc_id, path=Path(body["path"]))
        return cls(pipelines=pipelines, documents=documents)


@dataclass
class _HotEntry:
    doc_id: str
    fingerprint: str
    simhash_int: int | None = None
    last_touched: float = field(default_factory=time.time)


class Orchestrator:
    """Lookahead controller for a single sequential pipeline run.

    A fresh `Orchestrator` instance is created per `run.py` pipeline run so
    the cursor state is scoped to that run. The same instance can be reused
    across BEFORE/AFTER modes — `reset(pipeline, mode, cache_bust_provider)`
    re-positions the cursor and rebinds the keep-resident behavior.
    """

    def __init__(
        self,
        *,
        manifest: Manifest,
        telemetry: Telemetry,
        bridge: Any,                       # avoid circular import; duck-typed
        hot_capacity: int = 4,
        simhash_threshold: int = NEAR_DUP_THRESHOLD,
    ) -> None:
        self.manifest = manifest
        self.telemetry = telemetry
        self.bridge = bridge
        self._hot: "OrderedDict[str, _HotEntry]" = OrderedDict()
        self._hot_capacity = hot_capacity
        self._simhash_threshold = simhash_threshold
        self.pipeline: Pipeline | None = None
        self.mode: str = ""
        self.cursor: int = -1   # index of the LAST observed node
        self._cache_bust_provider = None  # callable returning str | None
        self._documents_text: dict[str, str] = {}
        # Memoize per-doc simhashes so we don't rehash on every lookahead.
        self._doc_simhash: dict[str, int] = {}

    # ------------------------------------------------------------------ control

    def reset(
        self,
        *,
        pipeline_name: str,
        mode: str,
        documents_text: dict[str, str],
        cache_bust_provider=None,
    ) -> None:
        """Begin a new pipeline run. `documents_text` maps doc_id -> raw text."""
        if pipeline_name not in self.manifest.pipelines:
            raise KeyError(f"unknown pipeline: {pipeline_name!r}")
        self.pipeline = self.manifest.pipelines[pipeline_name]
        self.mode = mode
        self.cursor = -1
        self._cache_bust_provider = cache_bust_provider
        self._documents_text = documents_text
        self._hot.clear()

        self.telemetry.log(
            "decision",
            mode=mode,
            role="orchestrator",
            message=(
                f"loaded pipeline {pipeline_name!r}: "
                f"{' -> '.join(n.role for n in self.pipeline.nodes)}"
            ),
            extra={
                "pipeline": pipeline_name,
                "nodes": [
                    {"role": n.role, "doc_id": n.doc_id}
                    for n in self.pipeline.nodes
                ],
            },
        )

    # ------------------------------------------------------------------ observe

    def observe(self, *, role: str, doc_id: str, fingerprint: str) -> None:
        """Bridge calls this at the START of each real agent dispatch."""
        if self.pipeline is None:
            return  # observation outside any pipeline — nothing to lookahead.

        # Advance the cursor to the matching node so lookahead is correct.
        self.cursor = self._advance_to(role, after=self.cursor)
        self._touch_hot(doc_id=doc_id, fingerprint=fingerprint)

        self.telemetry.log(
            "decision",
            mode=self.mode,
            role="orchestrator",
            doc_id=doc_id,
            fingerprint=fingerprint,
            message=f"observed {role} starting on {doc_id} (fp={fingerprint[:10]})",
            extra={"phase": "observe", "cursor": self.cursor},
        )

    # ------------------------------------------------------------------ act

    def on_task_completed(self, *, role: str, doc_id: str) -> None:
        """Bridge calls this after each real dispatch finishes.

        This is when we have a free moment to fire keep-resident for the NEXT
        agent — before the next dispatch even arrives.
        """
        if self.pipeline is None:
            return

        next_node = self._peek_next()
        if next_node is None:
            self.telemetry.log(
                "decision",
                mode=self.mode,
                role="orchestrator",
                doc_id=doc_id,
                message=f"no successor after {role}; pipeline complete",
                extra={"phase": "lookahead", "cursor": self.cursor},
            )
            return

        # Lookahead decision: what does the next agent need?
        if next_node.doc_id == doc_id:
            reason = (
                f"next agent {next_node.role!r} needs same doc {doc_id!r}; "
                f"firing keep-resident warmup"
            )
        else:
            reason = (
                f"next agent {next_node.role!r} needs doc {next_node.doc_id!r}; "
                f"firing keep-resident warmup for that doc"
            )

        self.telemetry.log(
            "decision",
            mode=self.mode,
            role="orchestrator",
            doc_id=next_node.doc_id,
            message=reason,
            extra={
                "phase": "act",
                "current": {"role": role, "doc_id": doc_id},
                "next": {"role": next_node.role, "doc_id": next_node.doc_id},
            },
        )

        document_text = self._documents_text.get(next_node.doc_id)
        if document_text is None:
            # Can't warm a doc we don't have text for. Log and move on.
            self.telemetry.log(
                "decision",
                mode=self.mode,
                role="orchestrator",
                doc_id=next_node.doc_id,
                message=(
                    f"skipped warmup: no document text loaded for "
                    f"{next_node.doc_id!r}"
                ),
                extra={"phase": "act", "skipped": True},
            )
            return

        # Robustness check (CLAUDE.md §6 stretch): before dispatching the
        # warmup, see whether this doc is a near-duplicate of any currently-
        # hot doc. SimHash on long documents is cheap (one pass, no model).
        self._check_near_duplicate(next_node.doc_id, document_text)

        cache_bust = (
            self._cache_bust_provider() if self._cache_bust_provider else None
        )
        # Fire the warmup synchronously. It's a max_tokens=1 prefill, so even
        # on a cache hit it's ~0.5s — well below human noticing for the demo.
        result = self.bridge.keep_resident(
            doc_id=next_node.doc_id,
            document=document_text,
            cache_bust=cache_bust,
        )
        self._touch_hot(
            doc_id=next_node.doc_id, fingerprint=result.fingerprint
        )

    # ------------------------------------------------------------------ adapt

    def _touch_hot(self, *, doc_id: str, fingerprint: str) -> None:
        """Update the in-memory hot-set. With 1 doc this is trivial; the
        eviction branch is exercised in Phase 3 (multi-doc)."""
        sh = self._ensure_simhash(doc_id)
        key = f"{doc_id}:{fingerprint}"
        if key in self._hot:
            entry = self._hot.pop(key)
            entry.last_touched = time.time()
            entry.simhash_int = sh if sh is not None else entry.simhash_int
            self._hot[key] = entry
            return
        self._hot[key] = _HotEntry(
            doc_id=doc_id, fingerprint=fingerprint, simhash_int=sh
        )
        while len(self._hot) > self._hot_capacity:
            evicted_key, evicted = self._hot.popitem(last=False)
            # Emit BOTH a structured `eviction` event (dashboard counts these)
            # AND a `decision` row so the scrolling decision log shows it too.
            self.telemetry.log(
                "eviction",
                mode=self.mode,
                role="orchestrator",
                doc_id=evicted.doc_id,
                fingerprint=evicted.fingerprint,
                message=(
                    f"hot-set evicted: {evicted.doc_id} "
                    f"(fp={evicted.fingerprint[:8]}) under LRU cap "
                    f"({self._hot_capacity})"
                ),
                extra={
                    "phase": "adapt",
                    "evicted_doc_id": evicted.doc_id,
                    "evicted_fingerprint": evicted.fingerprint,
                    "hot_capacity": self._hot_capacity,
                },
            )
            self.telemetry.log(
                "decision",
                mode=self.mode,
                role="orchestrator",
                doc_id=evicted.doc_id,
                fingerprint=evicted.fingerprint,
                message=(
                    f"evicted hot entry {evicted.doc_id} (fp="
                    f"{evicted.fingerprint[:8]}) under LRU cap "
                    f"({self._hot_capacity})"
                ),
                extra={"phase": "adapt", "evicted": True},
            )

    # ------------------------------------------------------------------ simhash

    def _ensure_simhash(self, doc_id: str) -> int | None:
        if doc_id in self._doc_simhash:
            return self._doc_simhash[doc_id]
        text = self._documents_text.get(doc_id)
        if text is None:
            return None
        sh = simhash(text)
        self._doc_simhash[doc_id] = sh
        return sh

    def _check_near_duplicate(self, doc_id: str, document_text: str) -> None:
        """If `doc_id` is a near-duplicate of any currently-hot OTHER doc,
        emit a structured `near_duplicate_detected` event BEFORE dispatch
        (CLAUDE.md §8a Phase-3 gate).
        """
        candidate_sh = self._ensure_simhash(doc_id)
        if candidate_sh is None:
            return
        known: dict[str, int] = {}
        for entry in self._hot.values():
            if entry.doc_id == doc_id or entry.simhash_int is None:
                continue
            # Multiple fingerprints for same doc_id collapse — keep first.
            known.setdefault(entry.doc_id, entry.simhash_int)
        match = find_near_match(
            candidate_sh, known, threshold=self._simhash_threshold
        )
        if match is None:
            return
        self.telemetry.log(
            "near_duplicate_detected",
            mode=self.mode,
            role="orchestrator",
            doc_id=doc_id,
            message=(
                f"near-duplicate: {doc_id!r} ≈ {match.other_label!r} "
                f"(hamming={match.hamming}/{SIMHASH_BITS}, "
                f"similarity={match.similarity:.3f}); prefix cache will "
                f"MISS but the orchestrator pre-warms it anyway"
            ),
            extra={
                "phase": "act",
                "candidate_doc_id": doc_id,
                "matched_doc_id": match.other_label,
                "hamming": match.hamming,
                "similarity": match.similarity,
                "threshold": match.threshold,
            },
        )

    # ------------------------------------------------------------------ helpers

    def _peek_next(self) -> Node | None:
        assert self.pipeline is not None
        nxt = self.cursor + 1
        if nxt >= len(self.pipeline.nodes):
            return None
        return self.pipeline.nodes[nxt]

    def _advance_to(self, role: str, *, after: int) -> int:
        """Find the next index >= after+1 whose role matches `role`."""
        assert self.pipeline is not None
        for i in range(after + 1, len(self.pipeline.nodes)):
            if self.pipeline.nodes[i].role == role:
                return i
        # Role wasn't in the remaining pipeline — degrade gracefully by
        # bumping the cursor anyway so we don't double-fire on the prior node.
        return after + 1
