#!/bin/bash
# Full benchmark suite — all skills + retrieval
# Run from anywhere: bash ~/development/autoresearch/skill-research/run_full_suite.sh

EVAL_DIR="$HOME/development/autoresearch/skill-research"
RESULTS_FILE="$HOME/.openclaw/workspace/autoresearch-results.tsv"

echo "============================================"
echo "  SKILL AUTORESEARCH — FULL BASELINE SUITE"
echo "  $(date)"
echo "============================================"
echo ""

# Skills to evaluate
SKILLS=("blockchain-from-scratch" "ue5-game-building" "dynamic-programming-patterns" "course-to-skill" "shannon-memory" "session-sync")

echo "=== Shannon Retrieval ==="
python3 "$EVAL_DIR/eval_retrieval.py" 2>/dev/null
echo ""

for skill in "${SKILLS[@]}"; do
    echo "=== Skill: $skill ==="
    echo "Started: $(date '+%H:%M:%S')"
    python3 "$EVAL_DIR/eval_skill.py" "$skill" 2>/dev/null
    echo "Finished: $(date '+%H:%M:%S')"
    echo ""
done

echo "============================================"
echo "  SUITE COMPLETE — $(date)"
echo "============================================"
