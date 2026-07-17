#!/bin/bash
# Goal-loop verifier for CT Plant ID, run as a Claude Code Stop hook.
#
# Blocks the session from ending while a goal is active until it verifies.
# This project has TWO kinds of goals (see GOALS.md), so there are three
# gates, run in order:
#   1. pytest suite green
#   2. OPTIONAL metric gate: if scripts/goal_check.sh exists, it must exit 0.
#      ML goals drop an eval-and-threshold check here (top-k accuracy, ECE,
#      OOD AUROC, coverage fraction...) so "tests pass" can't ship a model
#      that misses its numbers.
#   3. Codex review returning VERDICT: SHIP against the goal text.
# Any failure exits 2, which feeds the failure back to Claude as the next
# instruction — that's the loop. Safety valve after MAX_ITERATIONS.

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0

GOAL_FILE=".claude/current-goal.md"
COUNT_FILE=".claude/goal-iterations"
CODEX="/Applications/Codex.app/Contents/Resources/codex"
MAX_ITERATIONS=8
PY=".venv/bin/python"

[ -f "$GOAL_FILE" ] || exit 0

n=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)
if [ "$n" -ge "$MAX_ITERATIONS" ]; then
    mv "$GOAL_FILE" .claude/goal-stalled.md
    rm -f "$COUNT_FILE"
    echo "Goal loop stopped after $MAX_ITERATIONS iterations without passing." \
         "Parked in .claude/goal-stalled.md — review it manually."
    exit 0
fi

block() {
    echo $((n + 1)) > "$COUNT_FILE"
    echo "$1" >&2
    exit 2
}

# ---- gate 1: tests ----
test_out=$($PY -m pytest -q 2>&1 | tail -15)
if ! echo "$test_out" | grep -qE '^[0-9]+ passed' || echo "$test_out" | grep -qE 'failed|error'; then
    block "GOAL NOT MET (iteration $((n + 1))/$MAX_ITERATIONS): tests are not green.
Fix the failures below, then finish again.

$test_out"
fi

# ---- gate 2: optional metric gate for ML goals ----
if [ -x scripts/goal_check.sh ]; then
    metric_out=$(scripts/goal_check.sh 2>&1)
    if [ $? -ne 0 ]; then
        block "GOAL NOT MET (iteration $((n + 1))/$MAX_ITERATIONS): metric gate failed.
The eval did not meet the goal's threshold. Improve the model/pipeline or,
if the target itself is wrong, revise it honestly in GOALS.md with a reason.

$metric_out"
    fi
fi

# ---- gate 3: Codex review ----
goal=$(cat "$GOAL_FILE")
codex_out=$("$CODEX" exec --skip-git-repo-check --sandbox read-only \
    "You are the final reviewer for this project (a Connecticut plant-species
classifier — comprehensive long-tail coverage, PictureThis-style app). The
developer claims this goal is complete:

---
$goal
---

Review strictly against the goal. Tests pass and any metric gate passed.
Check for real defects, honest reporting (no train/test leakage, no metric
scoped only to trained classes to hide the coverage gap, no silently
dropped species), and that the work matches the goal. End with exactly one
line: VERDICT: SHIP  or  VERDICT: NO-SHIP - <specific reasons>" < /dev/null 2>&1 | tail -40)

if echo "$codex_out" | grep -q "VERDICT: SHIP"; then
    rm -f "$GOAL_FILE" "$COUNT_FILE"
    echo "Goal verified: tests green, metric gate passed, Codex SHIP. Goal cleared."
    exit 0
fi

block "GOAL NOT MET (iteration $((n + 1))/$MAX_ITERATIONS): Codex has not approved.
Address its findings below, then finish again.

$codex_out"
