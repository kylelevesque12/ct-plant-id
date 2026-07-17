#!/bin/bash
# Per-goal metric/integrity gate, run by the Stop hook between pytest and
# Codex (see .claude/hooks/verify_goal.sh). Exit non-zero to block the goal.
#
# Data-phase goals use the data-integrity check below. An ML goal REPLACES
# or EXTENDS this with its own eval-and-threshold (top-k accuracy, ECE, OOD
# AUROC, coverage fraction) so "tests pass" can never ship a model that
# misses its numbers.
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python scripts/data_integrity.py
