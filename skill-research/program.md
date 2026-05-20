# Skill Autoresearch

Autonomous experimentation loop for improving OpenClaw agent skills and infrastructure.
Based on the autoresearch pattern: modify → measure → keep/discard → repeat forever.

## Setup

To set up a new experiment run:

1. **Agree on a target**: what are we improving? A skill, a prompt template, Shannon config, etc.
2. **Create the branch**: `cd ~/.openclaw/skills && git checkout -b autoresearch/<tag>`
3. **Read the in-scope files**:
   - The target skill: `~/.openclaw/skills/<name>/SKILL.md`
   - The test script: `~/.openclaw/workspace/scripts/test-skill-quality.sh`
   - Shannon health: `curl -s http://localhost:8765/health`
4. **Establish baseline**: Run the evaluation harness on the current skill. Record results.
5. **Confirm and go**.

## What Can Be Improved

### Tier 1: Skills (primary target)
- **File:** `~/.openclaw/skills/<name>/SKILL.md`
- **Metric:** A/B score — how many test prompts produce measurably different (better) output with skill vs without
- **Evaluation:** `bash ~/.openclaw/workspace/scripts/test-skill-quality.sh "<skill>" "<prompt>"`
- **Keep condition:** Skill changes ≥2/3 test responses AND the changes are qualitatively better (more specific, fewer hallucinations, correct patterns used)
- **Discard condition:** Skill changes 0/3 responses OR makes responses worse

### Tier 2: Agent Instructions
- **Files:** `~/.qwen/QWEN.md`, project-level `CLAW.md` files
- **Metric:** Task completion rate — does the agent follow the instructions correctly?
- **Evaluation:** Give agent a task, check output against acceptance criteria
- **Keep condition:** Agent follows new instruction correctly on ≥2/3 test tasks
- **Discard condition:** Agent ignores instruction or produces worse output

### Tier 3: Shannon Configuration
- **Files:** `~/development/shannon/shannon/api.py` (tier weights, scoring formula)
- **Metric:** Retrieval relevance — do searches return the right content?
- **Evaluation:** Run known-answer queries, check if expected entries appear in top 5
- **Keep condition:** Recall@5 improves or stays same with better precision
- **Discard condition:** Recall drops

### Tier 4: Pipeline Parameters
- **Files:** `youtube-to-shannon.sh` (chunk size), `batch-skill-extract.sh` (prompt template)
- **Metric:** Downstream skill quality — does changing the pipeline produce better skills?
- **Evaluation:** Process same video with old vs new params, compare resulting skill quality
- **Keep condition:** Resulting skill scores higher on A/B test
- **Discard condition:** No improvement or worse skill

## Evaluation Harness

The evaluation is the ground truth. It CANNOT be modified during experiments.

### Skill Evaluation (eval_skill.py)

```python
"""
Evaluate a skill by running N test prompts through a local model twice:
  A) Baseline (no skill)
  B) With skill injected into context

Returns a score: number of prompts where the response meaningfully changed.
Higher = skill is having more effect. Quality judgment is separate.
"""

import requests
import json
import hashlib
import sys
import os
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:14b"
SKILLS_DIR = os.path.expanduser("~/.openclaw/skills")
SHANNON_URL = "http://localhost:8765"

def evaluate_skill(skill_name: str, test_prompts: list[str]) -> dict:
    """Run A/B evaluation. Returns {changed: int, total: int, details: list}."""
    skill_path = os.path.join(SKILLS_DIR, skill_name, "SKILL.md")
    if not os.path.exists(skill_path):
        return {"error": f"Skill not found: {skill_path}"}
    
    skill_content = open(skill_path).read()
    results = []
    
    for prompt in test_prompts:
        # A: Baseline
        r_a = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512, "seed": 42}
        }, timeout=300)
        baseline = r_a.json().get("response", "")
        
        time.sleep(1)
        
        # B: With skill
        full_prompt = f"You have this specialized knowledge:\n\n{skill_content}\n\nNow answer: {prompt}"
        r_b = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512, "seed": 42}
        }, timeout=300)
        with_skill = r_b.json().get("response", "")
        
        b_hash = hashlib.sha256(baseline.encode()).hexdigest()[:12]
        w_hash = hashlib.sha256(with_skill.encode()).hexdigest()[:12]
        changed = b_hash != w_hash
        
        results.append({
            "prompt": prompt,
            "baseline_len": len(baseline),
            "withskill_len": len(with_skill),
            "changed": changed,
            "baseline_hash": b_hash,
            "withskill_hash": w_hash
        })
        
        time.sleep(1)
    
    changed_count = sum(1 for r in results if r["changed"])
    return {
        "skill": skill_name,
        "changed": changed_count,
        "total": len(results),
        "score": changed_count / len(results) if results else 0,
        "details": results
    }


def evaluate_retrieval(queries: list[dict]) -> dict:
    """Evaluate Shannon retrieval quality.
    
    queries: list of {"query": str, "expected_tags": list[str]}
    Returns recall@5 score.
    """
    hits = 0
    total = 0
    for q in queries:
        r = requests.get(f"{SHANNON_URL}/memory/search",
            params={"q": q["query"], "limit": 5}, timeout=30)
        results = r.json().get("results", [])
        result_tags = set()
        for entry in results:
            result_tags.update(entry.get("tags", []))
        
        expected = set(q["expected_tags"])
        if expected & result_tags:
            hits += 1
        total += 1
    
    return {"recall_at_5": hits / total if total else 0, "hits": hits, "total": total}


if __name__ == "__main__":
    skill_name = sys.argv[1]
    
    # Default test prompts per skill
    default_prompts = {
        "ue5-game-building": [
            "How should I share a health system between my player and enemy NPCs in UE5?",
            "I need pickups that give different effects. How to structure this in Blueprints?",
            "Everything is in my player character blueprint and its unmanageable. How to fix?"
        ],
        "blockchain-from-scratch": [
            "How do I prevent replay attacks in my blockchain?",
            "What's the right way to validate a chain from genesis to tip?",
            "How should I handle peer discovery in a P2P network?"
        ],
        "dynamic-programming-patterns": [
            "Find minimum cost path in a grid, top-left to bottom-right.",
            "My recursive solution is too slow for n=40. How to optimize?",
            "Check if array can be split into two equal-sum subsets."
        ]
    }
    
    prompts = default_prompts.get(skill_name, [
        f"Give me a practical example using {skill_name}",
        f"What are common mistakes with {skill_name}?",
        f"When should I NOT use {skill_name} patterns?"
    ])
    
    result = evaluate_skill(skill_name, prompts)
    print(json.dumps(result, indent=2))
```

## Results Logging

Results go to `~/.openclaw/workspace/autoresearch-results.tsv` (tab-separated):

```
timestamp	target	commit	score	status	description
2026-05-20T17:30	ue5-game-building	a1b2c3d	0.67	keep	added weapon collision warning to common mistakes
2026-05-20T17:35	ue5-game-building	b2c3d4e	0.67	discard	made description pushier (no score change)
2026-05-20T17:40	blockchain-from-scratch	c3d4e5f	1.00	keep	added P2P gossip flooding to common mistakes
```

## The Experiment Loop

**Target: a specific skill or config file.**

LOOP FOREVER:

1. Read the current state of the target file
2. Query Shannon for relevant failure data, lessons learned, or new knowledge:
   ```
   curl -s "http://localhost:8765/memory/search?q=<skill>+failure+mistake+lesson&limit=10"
   ```
3. Propose a MINIMAL modification. Ideas come from:
   - Shannon failure data (things that went wrong when using this skill)
   - Missing patterns (Shannon has knowledge the skill doesn't cover)
   - Pushy description gaps (skill doesn't trigger on natural language that it should)
   - Common mistake gaps (known footguns not documented)
   - Specificity (skill is too generic — add concrete examples)
4. Make the edit. Git commit with descriptive message.
5. Run evaluation:
   ```bash
   python3 ~/.openclaw/workspace/autoresearch/skill-research/eval_skill.py <skill_name>
   ```
6. Read the score from stdout (JSON: `{"changed": N, "total": M, "score": 0.XX}`)
7. Record in results.tsv
8. If score improved or stayed same with simpler skill → KEEP (advance branch)
9. If score dropped or no change with added complexity → DISCARD (git reset)
10. GOTO 1

**Simplicity criterion**: Same as autoresearch — a small improvement that adds ugly complexity
is not worth it. Removing something and getting equal or better results is a simplification win.
A skill that's half the size with the same score is strictly better.

**NEVER STOP**: Once the loop begins, do NOT pause to ask. Keep running experiments until
manually interrupted. If stuck, try:
- Different test prompts (the current ones might not exercise the skill well)
- Rephrase the description (trigger differently)
- Add a concrete example (models respond to examples more than rules)
- Remove redundant sections (simpler = less context waste)
- Query Shannon for new topics the skill doesn't cover yet

## Experiment Ideas (Starting Points)

For skills:
1. Make description pushier (add more trigger phrases)
2. Add a common mistake from Shannon failure data
3. Add a concrete code example to a pattern section
4. Simplify — remove a section and see if score holds
5. Merge two related skills into one (less context overhead)
6. Split a section into a reference file (reduce SKILL.md size)
7. Add "when to deviate" entries from real usage

For Shannon config:
1. Adjust tier weights (try 2.0/1.0/0.3 instead of 1.5/1.0/0.5)
2. Change cosine/recency ratio (try 0.7/0.3 instead of 0.6/0.4)
3. Increase chunk size for YouTube transcripts (try 5000 instead of 3000)
4. Decrease chunk size (try 1500 — more entries, more specific)

For agent instructions:
1. Add skill-loading instructions to QWEN.md
2. Add Shannon query pattern to CLAW.md
3. Simplify instructions (remove things the agent already knows)
