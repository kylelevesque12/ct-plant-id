---
description: Set a verified goal (or "/goal next" to pull the top of GOALS.md) — the session cannot end until tests pass, any metric gate passes, AND Codex issues VERDICT SHIP
---
A goal loop is being started. Do these steps in order:

1. Determine the goal text:
   - If the arguments below are exactly `next` (or empty), open `GOALS.md`, take the FIRST unchecked (`- [ ]`) item in the Queue as the goal, and mark it `- [x]`. If there are none, say so and stop — do not invent a goal.
   - Otherwise, the arguments themselves are the goal.
2. Write the goal verbatim to `.claude/current-goal.md` (overwrite if it exists) and delete `.claude/goal-iterations` if present.
3. Work toward the goal autonomously. Follow README.md and GOALS.md conventions; run `.venv/bin/python -m pytest -q` as you go.
4. If this is an ML goal (its GOALS.md text names a metric threshold), you MUST create or update `scripts/goal_check.sh` so it runs the eval and exits non-zero when the threshold is not met — the Stop hook runs it as a hard gate. Report metrics stratified head/mid/tail against the FULL CT checklist, never scoped to only trained classes.
5. When you believe the goal is complete, simply finish. The Stop hook runs the test suite, the metric gate (if present), and a Codex review against the goal; any failure comes back as your next instruction. You cannot end until all gates pass or the 8-iteration safety valve trips.
6. If the goal came from GOALS.md, move its line to the Done section with today's date before your final finish attempt.
7. Do not edit or work around `.claude/current-goal.md`, `.claude/goal-iterations`, or the hook — the verifier owns those.

THE GOAL:

$ARGUMENTS
