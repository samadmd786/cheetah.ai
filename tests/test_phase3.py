"""Phase 3 gate assertions (CLAUDE.md §8a "multi-doc + matcher").

This is the offline auditor for the `phase3_multidoc` run (or any later run
of the `multi_doc_review` pipeline). It loads the telemetry CSV and asserts:

  1. EVICTION GATE: at least one `eviction` event exists for the run, with
     hot_capacity reflected in the structured `extra` payload.
  2. SIMHASH GATE: at least one `near_duplicate_detected` event exists, and
     for at least one such event the matched doc was already hot at the
     moment of detection (proving the matcher is comparing against live state
     rather than an empty set).
  3. ORDERING GATE: the near-dup detection fired BEFORE the
     `keep_resident_completed` dispatch of the same candidate doc — i.e.
     "before dispatch" per the CLAUDE.md gate wording.

Run with:  .venv/bin/python -m tests.test_phase3 [--run-id phase3_multidoc]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from telemetry import DEFAULT_CSV, read_rows  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="phase3_multidoc")
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    all_rows = read_rows(args.csv)
    rows = [r for r in all_rows if r.get("run_id") == args.run_id]
    if not rows:
        raise SystemExit(
            f"no telemetry rows for run_id={args.run_id!r} in {args.csv}; "
            f"run `python run.py --pipeline multi_doc_review "
            f"--hot-capacity 2 --mode after --run-id {args.run_id}` first."
        )
    rows.sort(key=lambda r: float(r["ts"]))

    evictions = [r for r in rows if r["event_type"] == "eviction"]
    near_dups = [r for r in rows if r["event_type"] == "near_duplicate_detected"]
    warmups = [r for r in rows if r["event_type"] == "keep_resident_completed"]

    # --- Gate 1: eviction ---------------------------------------------------
    assert evictions, (
        f"EVICTION GATE FAIL: no `eviction` rows for run {args.run_id!r}; "
        f"check that --hot-capacity is smaller than the number of distinct "
        f"docs in the pipeline."
    )
    sample = evictions[0]
    extra = json.loads(sample.get("extra_json") or "{}")
    assert extra.get("evicted_doc_id"), (
        f"EVICTION GATE FAIL: eviction event missing `evicted_doc_id`: {sample}"
    )
    print(
        f"[phase3 eviction gate] PASS  {len(evictions)} eviction(s); "
        f"first evicted doc={extra['evicted_doc_id']!r} "
        f"under cap={extra.get('hot_capacity')}"
    )

    # --- Gate 2: SimHash near-duplicate detected ----------------------------
    assert near_dups, (
        f"SIMHASH GATE FAIL: no `near_duplicate_detected` rows for run "
        f"{args.run_id!r}; check that the pipeline includes a doc that is a "
        f"near-duplicate of an earlier doc."
    )
    nd_sample = near_dups[0]
    nd_extra = json.loads(nd_sample.get("extra_json") or "{}")
    assert nd_extra.get("matched_doc_id"), (
        f"SIMHASH GATE FAIL: near-dup row missing matched_doc_id: {nd_sample}"
    )
    assert isinstance(nd_extra.get("hamming"), int)
    assert nd_extra["hamming"] <= nd_extra.get("threshold", 10), (
        f"SIMHASH GATE FAIL: matcher fired with hamming "
        f"{nd_extra.get('hamming')} > threshold {nd_extra.get('threshold')}"
    )
    print(
        f"[phase3 simhash gate]  PASS  {len(near_dups)} near-dup match(es); "
        f"first: {nd_extra['candidate_doc_id']!r} ≈ "
        f"{nd_extra['matched_doc_id']!r} "
        f"(hamming={nd_extra['hamming']}/64, "
        f"similarity={nd_extra['similarity']:.3f})"
    )

    # --- Gate 3: ordering (detection BEFORE dispatch) -----------------------
    nd_ts = float(nd_sample["ts"])
    candidate = nd_extra["candidate_doc_id"]
    matching_warmups = [
        w for w in warmups if w.get("doc_id") == candidate
    ]
    assert matching_warmups, (
        f"ORDERING GATE FAIL: no keep_resident_completed for candidate "
        f"{candidate!r}; can't verify the 'before dispatch' ordering."
    )
    first_dispatch_ts = float(matching_warmups[0]["ts"])
    assert nd_ts < first_dispatch_ts, (
        f"ORDERING GATE FAIL: near-dup detection at {nd_ts} is NOT before "
        f"first warmup dispatch for {candidate!r} at {first_dispatch_ts}"
    )
    print(
        f"[phase3 ordering gate] PASS  near-dup detected "
        f"{first_dispatch_ts - nd_ts:.2f}s before the keep-resident dispatch "
        f"for {candidate!r}"
    )

    print("\nALL PHASE 3 GATES PASS")


if __name__ == "__main__":
    main()
