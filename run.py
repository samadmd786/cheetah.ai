"""Phase 2 entrypoint: run the 3-agent pipeline through bridge + orchestrator.

All inference goes through `bridge.py`. The orchestrator reads
`workflow/manifest.yaml`, sees the next agent and its document, and fires a
`keep_resident` warmup *between* agent calls so the next dispatch hits a warm
prefix cache (CLAUDE.md §5).

What the dashboard sees from one run of this script:
  * `decision` rows  — orchestrator observe / lookahead / act events
  * `task_completed` rows — per-agent TTFT/total, role, doc_id, mode
  * `keep_resident_completed` rows — the warmup the orchestrator fired

The single headline number remains per-agent TTFT, BEFORE vs AFTER.
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

from openai import OpenAI

from agents import AGENT_TASKS
from bridge import Bridge, fresh_cache_bust
from orchestrator import Manifest, Orchestrator
from telemetry import Telemetry


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = REPO_ROOT / "workflow" / "manifest.yaml"
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_BASE_URL = os.environ.get("VLLM_MLX_URL", "http://127.0.0.1:8001/v1")


def _make_client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="EMPTY")


def _warmup_model(client: OpenAI, model: str) -> None:
    """Tiny request so first-call overhead doesn't pollute the first TTFT."""
    try:
        list(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=4,
                temperature=0.0,
                stream=True,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warmup] warning: {exc}", file=sys.stderr)


def _load_documents(manifest: Manifest) -> dict[str, str]:
    out: dict[str, str] = {}
    for doc_id, doc in manifest.documents.items():
        path = doc.path
        if not path.is_absolute():
            path = REPO_ROOT / path
        out[doc_id] = path.read_text()
    return out


def _run_pipeline(
    *,
    bridge: Bridge,
    orchestrator: Orchestrator,
    pipeline_name: str,
    documents_text: dict[str, str],
    mode: str,
    max_tokens: int,
) -> list:
    """Run one pipeline (BEFORE or AFTER mode) end-to-end through the bridge."""
    bridge.mode = mode
    cache_bust_provider = (lambda: fresh_cache_bust()) if mode == "before" else None
    orchestrator.reset(
        pipeline_name=pipeline_name,
        mode=mode,
        documents_text=documents_text,
        cache_bust_provider=cache_bust_provider,
    )
    pipeline = orchestrator.manifest.pipelines[pipeline_name]

    results = []
    for node in pipeline.nodes:
        cache_bust = fresh_cache_bust() if mode == "before" else None
        document_text = documents_text[node.doc_id]
        task = AGENT_TASKS[node.role]
        result = bridge.dispatch(
            role=node.role,
            doc_id=node.doc_id,
            document=document_text,
            task=task,
            max_tokens=max_tokens,
            cache_bust=cache_bust,
        )
        print(
            f"  [{mode}] {node.role:<9} doc={node.doc_id:<10} "
            f"ttft={result.ttft_s:6.3f}s  total={result.total_s:6.3f}s  "
            f"hit={'Y' if result.cache_hit else 'N'}  "
            f"fp={result.fingerprint[:8]}"
        )
        results.append(result)
    return results


def _print_table(before: list, after: list) -> None:
    print()
    print("=" * 78)
    print("PHASE 2 RESULT — Time-To-First-Token, BEFORE vs AFTER (via bridge)")
    print("=" * 78)
    header = (
        f"{'agent':<10} | {'before ttft':>12} | {'after ttft':>12} | {'speedup':>9}"
    )
    print(header)
    print("-" * len(header))
    for b, a in zip(before, after):
        speedup = b.ttft_s / a.ttft_s if a.ttft_s > 0 else float("inf")
        print(
            f"{b.role:<10} | {b.ttft_s:>10.3f}s  | {a.ttft_s:>10.3f}s  | "
            f"{speedup:>7.2f}x"
        )
    print("-" * len(header))

    before_ttft = sum(r.ttft_s for r in before)
    after_ttft = sum(r.ttft_s for r in after)
    before_total = sum(r.total_s for r in before)
    after_total = sum(r.total_s for r in after)
    speedup_total = before_ttft / after_ttft if after_ttft else float("inf")
    print(
        f"{'TOTAL':<10} | {before_ttft:>10.3f}s  | {after_ttft:>10.3f}s  | "
        f"{speedup_total:>7.2f}x"
    )
    print()
    print(
        f"wall-clock totals (incl. decode): "
        f"before={before_total:.3f}s   after={after_total:.3f}s   "
        f"gpu-seconds saved ≈ {max(0.0, before_total - after_total):.3f}s"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2: bridge + orchestrator run.")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--pipeline", default="discovery_review")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--mode", choices=["both", "before", "after"], default="both")
    p.add_argument("--skip-warmup", action="store_true")
    p.add_argument(
        "--hot-capacity",
        type=int,
        default=4,
        help=(
            "Orchestrator hot-set LRU capacity. Drop to 2 with multi-doc "
            "pipelines to force eviction events."
        ),
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Override telemetry run_id (otherwise random 8-char hex).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = Manifest.load(args.manifest)
    documents_text = _load_documents(manifest)

    print(f"manifest:  {args.manifest}")
    print(f"pipeline:  {args.pipeline}")
    pipeline = manifest.pipelines[args.pipeline]
    print(f"nodes:     {' -> '.join(n.role for n in pipeline.nodes)}")
    for doc_id, text in documents_text.items():
        print(f"doc[{doc_id}]: {len(text):,} chars")
    print(f"model:     {args.model}")
    print(f"server:    {args.base_url}")
    print()

    telemetry = Telemetry(run_id=args.run_id)
    print(f"run_id:    {telemetry.run_id}  (telemetry -> logs/telemetry.csv)")

    client = _make_client(args.base_url)
    if not args.skip_warmup:
        print("model warmup (tiny request)...")
        _warmup_model(client, args.model)

    bridge = Bridge(
        client=client, model=args.model, telemetry=telemetry, orchestrator=None
    )
    orchestrator = Orchestrator(
        manifest=manifest,
        telemetry=telemetry,
        bridge=bridge,
        hot_capacity=args.hot_capacity,
    )
    # Late binding: orchestrator needs the bridge (to fire warmups), bridge
    # needs the orchestrator (to notify on observe/complete). Build both, then
    # wire the orchestrator into the bridge.
    bridge.orchestrator = orchestrator

    before, after = [], []
    try:
        if args.mode in ("both", "before"):
            print("\n=== BEFORE (stateless: UUID-busted prefix every call) ===")
            before = _run_pipeline(
                bridge=bridge,
                orchestrator=orchestrator,
                pipeline_name=args.pipeline,
                documents_text=documents_text,
                mode="before",
                max_tokens=args.max_tokens,
            )
        if args.mode in ("both", "after"):
            print(
                "\n=== AFTER (shared prefix + orchestrator keep-resident) ==="
            )
            after = _run_pipeline(
                bridge=bridge,
                orchestrator=orchestrator,
                pipeline_name=args.pipeline,
                documents_text=documents_text,
                mode="after",
                max_tokens=args.max_tokens,
            )
        if before and after:
            _print_table(before, after)
        else:
            for r in before or after:
                print(
                    f"  {r.role:<9} ttft={r.ttft_s:.3f}s  total={r.total_s:.3f}s "
                    f" hit={'Y' if r.cache_hit else 'N'}"
                )
    finally:
        telemetry.close()
        print(f"\ntelemetry flushed; run_id={telemetry.run_id}")


if __name__ == "__main__":
    main()
