"""
Shannon Retrieval Evaluation Harness
DO NOT MODIFY DURING EXPERIMENTS — this is ground truth.

Tests Shannon search quality with known-answer queries.
A query has expected tags — if any expected tag appears in the top-5 results, it's a hit.
"""

import json
import sys

import requests

SHANNON_URL = "http://localhost:8765"

# Known-answer queries: we KNOW these should return entries with these tags
KNOWN_QUERIES = [
    {"query": "blockchain transaction signing deterministic serialization",
     "expected_tags": ["blockchain", "skill-building"]},
    {"query": "UE5 Blueprint inheritance vs composition vs interface",
     "expected_tags": ["ue5", "skill-building"]},
    {"query": "dynamic programming memoization vs tabulation",
     "expected_tags": ["skill-building", "dynamic-programming"]},
    {"query": "Lattice Network ML-DSA-87 validator consensus",
     "expected_tags": ["lattice-network", "architecture"]},
    {"query": "Shannon memory service health embedding coverage",
     "expected_tags": ["heartbeat"]},
    {"query": "Unreal Engine RPG tutorial combat damage system",
     "expected_tags": ["ue5", "rpg", "gorka-games"]},
    {"query": "Pigeon Browser sovereign mode relay circuit",
     "expected_tags": ["pigeon", "browser"]},
    {"query": "course to skill pipeline decomposition filter",
     "expected_tags": ["skill-building", "course-to-skill"]},
    {"query": "The Archive game project Karu underground cosmology",
     "expected_tags": ["the-archive", "game-dev"]},
    {"query": "claw-local worker bug initial_worker REPL tool use",
     "expected_tags": ["claw-local"]},
]


def evaluate(queries: list[dict] | None = None) -> dict:
    """Run retrieval evaluation. Returns {recall_at_5, hits, total, details}."""
    qs = queries or KNOWN_QUERIES
    details = []

    for q in qs:
        try:
            r = requests.get(
                f"{SHANNON_URL}/memory/search",
                params={"q": q["query"], "limit": 5},
                timeout=30,
            )
            results = r.json().get("results", [])
        except Exception as exc:
            details.append({"query": q["query"], "hit": False, "error": str(exc)})
            continue

        result_tags: set[str] = set()
        top_scores = []
        for entry in results:
            result_tags.update(entry.get("tags", []))
            top_scores.append(round(entry.get("score", 0), 3))

        expected = set(q["expected_tags"])
        hit = bool(expected & result_tags)

        details.append({
            "query": q["query"][:60],
            "expected": list(expected),
            "found_tags": sorted(result_tags)[:10],
            "hit": hit,
            "top_scores": top_scores[:3],
        })

    hits = sum(1 for d in details if d.get("hit", False))
    total = len(details)

    return {
        "recall_at_5": round(hits / total, 4) if total else 0.0,
        "hits": hits,
        "total": total,
        "details": details,
    }


if __name__ == "__main__":
    print("Evaluating Shannon retrieval quality...", file=sys.stderr)
    result = evaluate()
    print(json.dumps(result, indent=2))
