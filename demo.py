"""demo.py — three-stage live terminal demo for the Shared Context Bridge.

Run order, no setup required beyond the running vllm-mlx server:

    .venv/bin/python demo.py              # all three stages
    .venv/bin/python demo.py --stage 1    # just BEFORE
    .venv/bin/python demo.py --pause      # pause between stages (for live narration)

Each stage streams the LLM's tokens to the terminal at their real rate, with
a ticking elapsed-seconds indicator while we wait for the first token (so the
30+ second cold prefill in Stage 1 feels as painful as it is). At the end of
each stage there's a compact metrics summary; at the very end we point at
the Streamlit dashboard for the polished view.

Stage map:

  Stage 1 — BEFORE.   stateless agents, UUID-busted prefix every call.
                       3 cold prefills of the same ~13.7k-token doc.
                       Honest simulation of today's stateless multi-agent
                       systems where slight prompt drift defeats exact-
                       prefix matching from token 0.

  Stage 2 — NAIVE.    same doc, same agents, doc-first prompt construction.
                       No orchestrator. vllm-mlx's native prefix cache
                       reuses agent 1's KV state for agents 2 + 3.
                       This is what you'd get from vLLM / LMCache today,
                       for the easy case (one doc, agents fire back-to-back).

  Stage 3 — OURS.     three different docs (one a near-duplicate of doc 1).
                       Orchestrator reads the DAG, fires keep-resident
                       warmups BETWEEN agents, SimHash flags the near-
                       duplicate, budget-aware LRU evicts under the cap.
                       Every real agent dispatch stays sub-second.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from openai import OpenAI

from agents import AGENT_TASKS
from bridge import Bridge, fresh_cache_bust
from orchestrator import Manifest, Orchestrator
from telemetry import Telemetry


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = REPO_ROOT / "workflow" / "manifest.yaml"
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_BASE_URL = "http://127.0.0.1:8001/v1"


# ─────────────────────────────────────────────────────────────────────────── ansi

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
GREY = "\033[90m"

ROLE_COLOR = {"Screener": CYAN, "Analyst": MAGENTA, "Auditor": BLUE}
STAGE = {
    "before": (RED,   "STAGE 1 / 2  ·  BEFORE — today's stateless multi-agent system"),
    "ours":   (GREEN, "STAGE 2 / 2  ·  OUR SOLUTION — bridge + orchestrator + SimHash + LRU"),
}


def _print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def banner(stage_key: str, body: str) -> None:
    color, title = STAGE[stage_key]
    _print()
    _print(color + BOLD + "═" * 88 + RESET)
    _print(color + BOLD + "  " + title + RESET)
    _print(color + BOLD + "═" * 88 + RESET)
    for line in body.strip().splitlines():
        _print("  " + DIM + line + RESET)
    _print()


# ──────────────────────────────────────────────────────────────────── prefill ticker

class PrefillTicker:
    """Background thread that updates a single line with elapsed seconds.

    Sole job: make the cold-prefill wait *legible*. While we're stuck on the
    ~35s prefill in Stage 1, the audience sees the counter climb instead of
    just dead air.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.start = time.perf_counter()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "PrefillTicker":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop.is_set():
            elapsed = time.perf_counter() - self.start
            sys.stdout.write(f"\r{self.label}  {elapsed:5.1f}s")
            sys.stdout.flush()
            self._stop.wait(0.1)

    def stop(self) -> None:
        self._stop.set()
        try:
            self._thread.join(timeout=0.5)
        except RuntimeError:
            pass
        # Erase the ticker line so the next print isn't smeared.
        sys.stdout.write("\r" + " " * (len(self.label) + 14) + "\r")
        sys.stdout.flush()


# ───────────────────────────────────────────────────────────────────── streaming

def _stream_one(
    bridge: Bridge,
    *,
    role: str,
    doc_id: str,
    document: str,
    cache_bust: str | None,
    mode_label: str,
    max_tokens: int,
):
    role_color = ROLE_COLOR.get(role, WHITE)
    head = f"  {role_color}{BOLD}{role:<9}{RESET} {DIM}[{mode_label}]{RESET}"

    _print(head + f"  {DIM}▶ dispatching · doc={doc_id}{RESET}")

    first_token_time: list[float | None] = [None]
    ticker_holder: list[PrefillTicker | None] = [None]

    def on_first_token(ttft: float) -> None:
        first_token_time[0] = ttft
        if ticker_holder[0] is not None:
            ticker_holder[0].stop()
        marker = GREEN if ttft < 3.0 else RED
        verdict = "warm HIT" if ttft < 3.0 else "cold MISS"
        _print(
            f"  {DIM}└─{RESET} {marker}⏱  TTFT {ttft:6.2f}s · {verdict}{RESET}"
        )
        sys.stdout.write(f"  {DIM}└─ tokens:{RESET} ")
        sys.stdout.flush()

    def on_token(piece: str) -> None:
        sys.stdout.write(role_color + piece + RESET)
        sys.stdout.flush()

    ticker = PrefillTicker(
        f"  {DIM}└─ prefilling ~13.7k token heavy block ...{RESET}"
    )
    ticker_holder[0] = ticker
    try:
        ticker.__enter__()
        result = bridge.dispatch(
            role=role,
            doc_id=doc_id,
            document=document,
            task=AGENT_TASKS[role],
            max_tokens=max_tokens,
            cache_bust=cache_bust,
            on_token=on_token,
            on_first_token=on_first_token,
        )
    finally:
        ticker.__exit__(None, None, None)

    _print()  # newline after streamed text
    return result


# ───────────────────────────────────────────────────────────────────── summary

def stage_summary(title: str, results: list, color: str) -> None:
    _print()
    _print("  " + BOLD + color + title + RESET)
    total_ttft = sum(r.ttft_s for r in results)
    total_total = sum(r.total_s for r in results)

    # Bar widths normalized to longest TTFT so the slowest bar fills ~50 chars.
    longest = max(r.ttft_s for r in results) if results else 1.0
    for r in results:
        bar_len = max(1, int(50 * r.ttft_s / max(longest, 1e-6)))
        hit = f"{GREEN}HIT {RESET}" if r.cache_hit else f"{RED}MISS{RESET}"
        bar_color = GREEN if r.cache_hit else RED
        bar = bar_color + "█" * bar_len + RESET
        _print(
            f"    {r.role:<10} {hit}  ttft={r.ttft_s:6.2f}s  "
            f"total={r.total_s:6.2f}s  {bar}"
        )
    _print(f"    {DIM}{'─' * 70}{RESET}")
    _print(
        f"    {BOLD}TOTAL       "
        f"ttft={total_ttft:6.2f}s  total={total_total:6.2f}s{RESET}"
    )
    _print()


# ──────────────────────────────────────────────────────────────────── stages

def stage_before(bridge: Bridge, documents: dict, *, max_tokens: int) -> list:
    banner(
        "before",
        "Each agent prepends a fresh UUID to its system prompt — the cleanest\n"
        "simulation of today's stateless multi-agent systems (slightly different\n"
        "system prompts, session IDs, or per-request metadata all break exact-\n"
        "prefix matching from token 0).\n"
        "vllm-mlx sees a different prefix every call and pays the full ~13.7k-\n"
        "token cold prefill (~35s) THREE times. This is the amnesia tax.",
    )

    bridge.mode = "before"
    pipeline = [("Screener", "discovery"), ("Analyst", "discovery"), ("Auditor", "discovery")]
    results = []
    for role, doc_id in pipeline:
        r = _stream_one(
            bridge,
            role=role,
            doc_id=doc_id,
            document=documents[doc_id],
            cache_bust=fresh_cache_bust(),
            mode_label="BEFORE · UUID-busted",
            max_tokens=max_tokens,
        )
        results.append(r)

    stage_summary(
        "STAGE 1 SUMMARY  ·  the amnesia tax — three cold prefills, paid in full",
        results,
        RED,
    )
    return results


def stage_ours(
    bridge: Bridge,
    telemetry: Telemetry,
    manifest: Manifest,
    documents: dict,
    *,
    max_tokens: int,
    hot_capacity: int,
) -> list:
    banner(
        "ours",
        "Now THREE different documents — Screener on discovery, Analyst on merger,\n"
        "Auditor on discovery_v3 (a near-duplicate of discovery: same content,\n"
        "a few dollar amounts changed, whitespace shifted).\n"
        "Without our control plane, naive prefix caching would cold-prefill\n"
        "every agent (each new doc = fresh KV state).\n"
        "Our orchestrator:\n"
        "  1. Reads the workflow DAG (workflow/manifest.yaml)\n"
        "  2. After each agent, looks ahead — what does the NEXT agent need?\n"
        "  3. Fires a keep-resident warmup for that doc BEFORE the next dispatch\n"
        "  4. Runs SimHash to spot near-duplicates that exact-prefix cache misses\n"
        "  5. Under a budget cap (hot_capacity=2 here), evicts the LRU doc\n"
        "Watch the live events between agents — those are the differentiator.",
    )

    bridge.mode = "after"
    orchestrator = Orchestrator(
        manifest=manifest,
        telemetry=telemetry,
        bridge=bridge,
        hot_capacity=hot_capacity,
    )
    bridge.orchestrator = orchestrator

    pipeline_name = "multi_doc_review"
    orchestrator.reset(
        pipeline_name=pipeline_name,
        mode="after",
        documents_text=documents,
        cache_bust_provider=None,
    )
    pipeline = orchestrator.manifest.pipelines[pipeline_name]

    # ── Hook 1: print orchestrator decisions live as they fire. ────────────
    original_log = telemetry.log

    def loud_log(event_type: str, **fields):
        msg = (fields.get("message") or "").strip()
        if event_type == "near_duplicate_detected":
            _print(
                f"    {YELLOW}{BOLD}🔍 SimHash match{RESET} {DIM}·{RESET} {msg}"
            )
        elif event_type == "eviction":
            _print(
                f"    {RED}{BOLD}♻  Eviction{RESET}   {DIM}·{RESET} {msg}"
            )
        elif event_type == "decision":
            extra = fields.get("extra") or {}
            phase = extra.get("phase") if isinstance(extra, dict) else None
            if phase == "act":
                _print(
                    f"    {GREEN}{BOLD}🧭 Lookahead{RESET}  {DIM}·{RESET} {msg}"
                )
        original_log(event_type, **fields)

    telemetry.log = loud_log  # type: ignore[method-assign]

    # ── Hook 2: print the orchestrator's keep-resident warmups loudly. ─────
    original_keep_resident = bridge.keep_resident

    def keep_resident_loud(*, doc_id: str, document: str, cache_bust=None):
        head = f"    {GREEN}{DIM}┌─ orchestrator warmup{RESET}  doc={BOLD}{doc_id}{RESET}"
        _print(head)
        with PrefillTicker(
            f"    {GREEN}{DIM}│  prefilling heavy block ...{RESET}"
        ):
            r = original_keep_resident(
                doc_id=doc_id, document=document, cache_bust=cache_bust
            )
        verdict = (
            f"{GREEN}HIT {RESET}(warm)" if r.cache_hit
            else f"{RED}MISS{RESET} (paid one-time cold cost so the agent doesn't)"
        )
        _print(
            f"    {GREEN}{DIM}└─{RESET} warmup ttft={r.ttft_s:6.2f}s · {verdict}"
        )
        return r

    bridge.keep_resident = keep_resident_loud  # type: ignore[method-assign]

    try:
        results = []
        for node in pipeline.nodes:
            r = _stream_one(
                bridge,
                role=node.role,
                doc_id=node.doc_id,
                document=documents[node.doc_id],
                cache_bust=None,
                mode_label=f"OURS · doc={node.doc_id}",
                max_tokens=max_tokens,
            )
            results.append(r)
    finally:
        telemetry.log = original_log  # type: ignore[method-assign]
        bridge.keep_resident = original_keep_resident  # type: ignore[method-assign]
        bridge.orchestrator = None

    stage_summary(
        "STAGE 2 SUMMARY  ·  orchestrator earns its keep — even on multi-doc",
        results,
        GREEN,
    )
    return results


# ──────────────────────────────────────────────────────────────────── outro

def outro(all_results: dict, run_id: str) -> None:
    _print()
    _print(BOLD + "  comparison — total TTFT per stage (lower is better)" + RESET)
    _print()
    name_map = {
        "before": ("BEFORE  (stateless agents · UUID-busted prefix) ", RED),
        "ours":   ("OURS    (orchestrator + SimHash + LRU)          ", GREEN),
    }
    stage_totals = {k: sum(r.ttft_s for r in v) for k, v in all_results.items() if v}
    if not stage_totals:
        return
    ordered = [k for k in ("before", "ours") if k in stage_totals]
    longest = max(stage_totals.values())
    for key in ordered:
        total = stage_totals[key]
        label, color = name_map[key]
        bar_len = max(1, int(60 * total / max(longest, 1e-6)))
        bar = color + "█" * bar_len + RESET
        _print(f"  {label} {bar} {color}{total:6.2f}s{RESET}")
    _print()

    if "before" in stage_totals and "ours" in stage_totals:
        v = stage_totals["before"] / max(stage_totals["ours"], 1e-6)
        saved = stage_totals["before"] - stage_totals["ours"]
        _print(
            f"  {BOLD}OURS is {v:.2f}× faster than stateless agents{RESET}  "
            f"{DIM}— {saved:.1f}s of cold-prefill work eliminated.{RESET}"
        )
        _print(
            f"\n  {DIM}BEFORE = 3 agents reading 1 doc with stateless prompts "
            f"(every call cold).{RESET}"
        )
        _print(
            f"  {DIM}OURS   = 3 agents reading 3 DIFFERENT docs (the realistic "
            f"workload).{RESET}"
        )
        _print(
            f"  {DIM}OURS handles a harder scenario and still wins by {v:.1f}×. "
            f"The cold prefills new docs require still happen — the orchestrator "
            f"just absorbs them between agents, where the user can't feel them.{RESET}"
        )

    _print()
    _print(
        f"  {BOLD}Dashboard:{RESET}  open "
        f"{CYAN}http://127.0.0.1:8502{RESET} and select "
        f"run_id={BOLD}{run_id}{RESET} from the sidebar."
    )
    _print()


# ──────────────────────────────────────────────────────────────────── main

def _make_client(base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="EMPTY")


def _warmup_model(client: OpenAI, model: str) -> None:
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
        _print(f"{DIM}[warmup] warning: {exc}{RESET}")


def _load_documents(manifest: Manifest) -> dict[str, str]:
    out: dict[str, str] = {}
    for doc_id, doc in manifest.documents.items():
        path = doc.path
        if not path.is_absolute():
            path = REPO_ROOT / path
        out[doc_id] = path.read_text()
    return out


def _maybe_pause(pause: bool, message: str) -> None:
    if not pause:
        return
    try:
        input(f"\n  {DIM}{message}{RESET}")
    except EOFError:
        pass


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=96,
        help="cap decode per agent so streaming finishes within a demo beat",
    )
    p.add_argument(
        "--stage",
        choices=["all", "1", "2"],
        default="all",
        help="run just one stage. 1=BEFORE (stateless), 2=OURS (orchestrator)",
    )
    p.add_argument(
        "--pause",
        action="store_true",
        help="pause between stages — handy for narrated live demos",
    )
    p.add_argument(
        "--hot-capacity",
        type=int,
        default=2,
        help="orchestrator LRU cap for Stage 3 (default 2 → forces eviction)",
    )
    p.add_argument("--run-id", default="live_demo")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    manifest = Manifest.load(args.manifest)
    documents = _load_documents(manifest)
    client = _make_client(args.base_url)
    telemetry = Telemetry(run_id=args.run_id)

    _print()
    _print(BOLD + "  Shared Context Bridge  ·  live demo" + RESET)
    _print(f"  model={DIM}{args.model}{RESET}  server={DIM}{args.base_url}{RESET}")
    _print(f"  run_id={DIM}{args.run_id}{RESET}  manifest={DIM}{args.manifest}{RESET}")
    _print(
        f"  docs: {', '.join(f'{k} ({len(v):,} chars)' for k, v in documents.items())}"
    )
    _print()
    _print(f"  {DIM}warming model (tiny request) ...{RESET}")
    _warmup_model(client, args.model)

    bridge = Bridge(
        client=client,
        model=args.model,
        telemetry=telemetry,
        orchestrator=None,
        mode="demo",
    )

    all_results: dict[str, list] = {"before": [], "ours": []}

    try:
        if args.stage in ("all", "1"):
            all_results["before"] = stage_before(
                bridge, documents, max_tokens=args.max_tokens
            )
            _maybe_pause(args.pause, "press Enter for Stage 2 — OUR SOLUTION ...")

        if args.stage in ("all", "2"):
            all_results["ours"] = stage_ours(
                bridge,
                telemetry,
                manifest,
                documents,
                max_tokens=args.max_tokens,
                hot_capacity=args.hot_capacity,
            )

        if args.stage == "all":
            outro(all_results, args.run_id)
    finally:
        telemetry.close()


if __name__ == "__main__":
    main()
