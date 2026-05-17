"""64-bit SimHash near-duplicate matcher (CLAUDE.md §2 stretch, §6 robustness).

The purpose is NOT to replace vllm-mlx's exact-prefix cache — that cache is
the correctness floor and only matches byte-identical token prefixes. The
purpose IS to let the orchestrator answer the obvious Q&A question: "what
happens when the document changes slightly?"

When a near-duplicate of an already-hot document arrives, the orchestrator
logs a `near_duplicate_detected` decision and pre-warms the new doc — the
old fingerprint can't be reused (prefix differs) but the orchestrator can
*anticipate* that the new doc will follow the same pipeline and treat it
as worth keeping resident.

Tiny, standalone, no extra dependencies.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


_TOKEN_RE = re.compile(r"\w+")
SHINGLE_K = 3                  # 3-word shingles work well for English prose
SIMHASH_BITS = 64
NEAR_DUP_THRESHOLD = 10        # Hamming distance ≤ this ⇒ "near-duplicate"


@dataclass(frozen=True)
class SimhashMatch:
    """Result of comparing a candidate fingerprint to known hot ones."""

    other_label: str            # opaque caller-supplied identifier
    hamming: int
    similarity: float           # 1 - hamming/64; closer to 1 = more similar
    threshold: int = NEAR_DUP_THRESHOLD


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _shingles(tokens: list[str], k: int = SHINGLE_K) -> list[str]:
    if len(tokens) < k:
        return [" ".join(tokens)]
    return [" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)]


def _hash64(s: str) -> int:
    """Stable 64-bit hash using BLAKE2b (faster + better-mixed than md5)."""
    digest = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def simhash(text: str, *, k: int = SHINGLE_K, bits: int = SIMHASH_BITS) -> int:
    """Compute the SimHash of `text` as an int with `bits` significant bits.

    Algorithm:
      1. Shingle the lowercased token stream.
      2. Hash each shingle to a `bits`-wide integer.
      3. For each bit position, accumulate +1 if set else -1.
      4. Final bit = 1 if cumulative sum > 0 else 0.
    """
    counters = [0] * bits
    shingles = _shingles(_tokens(text), k=k)
    for sh in shingles:
        h = _hash64(sh)
        for i in range(bits):
            counters[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if counters[i] > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    """Bit-count of the XOR — number of differing positions."""
    return (a ^ b).bit_count()


def find_near_match(
    candidate: int,
    known: dict[str, int],
    *,
    threshold: int = NEAR_DUP_THRESHOLD,
) -> SimhashMatch | None:
    """Return the closest known SimHash within `threshold` Hamming distance,
    or None. `known` is a label -> simhash mapping."""
    best: SimhashMatch | None = None
    for label, other in known.items():
        d = hamming(candidate, other)
        if d > threshold:
            continue
        if best is None or d < best.hamming:
            best = SimhashMatch(
                other_label=label,
                hamming=d,
                similarity=1.0 - d / SIMHASH_BITS,
                threshold=threshold,
            )
    return best
