"""
Skill Autoresearch Evaluation Harness
DO NOT MODIFY DURING EXPERIMENTS — this is ground truth.

Evaluates a skill by running N test prompts through a local model twice:
  A) Baseline (no skill)
  B) With skill injected into context

Returns a score: fraction of prompts where the response meaningfully changed.
"""

import hashlib
import json
import os
import sys
import time

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:14b"
SKILLS_DIR = os.path.expanduser("~/.openclaw/skills")
SHANNON_URL = "http://localhost:8765"
RESULTS_FILE = os.path.expanduser("~/.openclaw/workspace/autoresearch-results.tsv")

# Default test prompts per skill — expand as skills are added
DEFAULT_PROMPTS = {
    "ue5-game-building": [
        "How should I share a health system between my player character and enemy NPCs in UE5?",
        "I need pickups that give different effects - coins for score, potions for health. How to structure this in Blueprints?",
        "Everything is in my player character blueprint and it's getting unmanageable. How do I refactor?",
    ],
    "blockchain-from-scratch": [
        "How do I prevent replay attacks in my blockchain transaction system?",
        "What's the right way to validate a blockchain from genesis to tip?",
        "How should I handle peer discovery when a new node joins my P2P network?",
    ],
    "dynamic-programming-patterns": [
        "Find minimum cost path in a grid from top-left to bottom-right. Which DP pattern?",
        "My recursive solution works but is too slow for n=40. How do I optimize it?",
        "Check if an array can be split into two subsets with equal sum. How?",
    ],
}


def generate(prompt: str, seed: int = 42) -> str:
    """Send a prompt to the local Ollama model and return the response text."""
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512, "seed": seed},
            },
            timeout=300,
        )
        return r.json().get("response", "")
    except Exception as exc:
        return f"ERROR: {exc}"


def evaluate_skill(skill_name: str, test_prompts: list[str] | None = None) -> dict:
    """Run A/B evaluation on a skill.

    Returns:
        {skill, changed, total, score, details[]}
    """
    skill_path = os.path.join(SKILLS_DIR, skill_name, "SKILL.md")
    if not os.path.exists(skill_path):
        return {"error": f"Skill not found: {skill_path}"}

    skill_content = open(skill_path).read()
    prompts = test_prompts or DEFAULT_PROMPTS.get(
        skill_name,
        [
            f"Give me a practical example using {skill_name}",
            f"What are common mistakes with {skill_name}?",
            f"When should I NOT use {skill_name} patterns?",
        ],
    )

    results = []
    for prompt in prompts:
        # A: Baseline (no skill)
        baseline = generate(prompt)
        time.sleep(1)

        # B: With skill
        full_prompt = (
            f"You have this specialized knowledge:\n\n{skill_content}\n\n"
            f"Now answer this question using the knowledge above:\n\n{prompt}"
        )
        with_skill = generate(full_prompt)
        time.sleep(1)

        b_hash = hashlib.sha256(baseline.encode()).hexdigest()[:16]
        w_hash = hashlib.sha256(with_skill.encode()).hexdigest()[:16]
        changed = b_hash != w_hash

        # Simple quality heuristic: longer + more specific = likely better
        # (rough proxy until we have human ratings)
        len_ratio = len(with_skill) / max(len(baseline), 1)

        results.append(
            {
                "prompt": prompt[:80],
                "baseline_len": len(baseline),
                "withskill_len": len(with_skill),
                "changed": changed,
                "len_ratio": round(len_ratio, 2),
                "baseline_hash": b_hash,
                "withskill_hash": w_hash,
            }
        )

    changed_count = sum(1 for r in results if r["changed"])
    score = changed_count / len(results) if results else 0.0

    return {
        "skill": skill_name,
        "changed": changed_count,
        "total": len(results),
        "score": round(score, 4),
        "details": results,
    }


def evaluate_retrieval(queries: list[dict]) -> dict:
    """Evaluate Shannon retrieval quality.

    queries: [{"query": str, "expected_tags": [str]}]
    Returns recall@5.
    """
    hits = 0
    total = 0
    for q in queries:
        r = requests.get(
            f"{SHANNON_URL}/memory/search",
            params={"q": q["query"], "limit": 5},
            timeout=30,
        )
        result_entries = r.json().get("results", [])
        result_tags: set[str] = set()
        for entry in result_entries:
            result_tags.update(entry.get("tags", []))

        expected = set(q["expected_tags"])
        if expected & result_tags:
            hits += 1
        total += 1

    return {
        "recall_at_5": round(hits / total, 4) if total else 0.0,
        "hits": hits,
        "total": total,
    }


def log_result(
    target: str, commit: str, score: float, status: str, description: str
) -> None:
    """Append one row to the results TSV."""
    ts = time.strftime("%Y-%m-%dT%H:%M")
    header = "timestamp\ttarget\tcommit\tscore\tstatus\tdescription\n"
    row = f"{ts}\t{target}\t{commit}\t{score:.4f}\t{status}\t{description}\n"

    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write(header)

    with open(RESULTS_FILE, "a") as f:
        f.write(row)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_skill.py <skill_name> [prompt1] [prompt2] ...")
        print(f"\nAvailable skills: {list(DEFAULT_PROMPTS.keys())}")
        sys.exit(1)

    skill_name = sys.argv[1]
    custom_prompts = sys.argv[2:] if len(sys.argv) > 2 else None

    print(f"Evaluating skill: {skill_name}", file=sys.stderr)
    print(f"Model: {MODEL}", file=sys.stderr)
    print(f"Prompts: {len(custom_prompts or DEFAULT_PROMPTS.get(skill_name, [1,2,3]))}", file=sys.stderr)
    print("---", file=sys.stderr)

    result = evaluate_skill(skill_name, custom_prompts)
    print(json.dumps(result, indent=2))
