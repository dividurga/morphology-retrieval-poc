#!/bin/sh
while ! grep -q ALLDONE results/run_rest.log; do sleep 15; done
mkdir -p results/dr/checkpoints
( while true; do sleep 1200; uv run python dr_report.py results/dr/run1.jsonl > results/dr/REPORT.md 2>/dev/null; cp results/dr/REPORT.md results/dr/checkpoints/ckpt_$(date +%H%M).md; done ) &
CK=$!
uv run python dr_experiment.py --bodies 200 --workers 11 --out results/dr/run1.jsonl > results/dr/run1.log 2>&1
kill $CK
uv run python dr_report.py results/dr/run1.jsonl > results/dr/REPORT.md
echo DRDONE >> results/dr/run1.log
