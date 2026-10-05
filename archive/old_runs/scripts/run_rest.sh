#!/bin/sh
while pgrep -f "generate_dataset.py" >/dev/null || pgrep -f "build_pools.py" >/dev/null; do sleep 10; done
for r in R0 R1 R2; do uv run python build_pools.py 0.6 1.0 1.5 --twin mujoco --rung $r --data data_mj/dataset.csv; done
for pre in pybullet_R0_ pybullet_R1_ pybullet_R2_ mujoco_R0_ mujoco_R1_ mujoco_R2_; do
  uv run python analyze_disparity.py --prefix $pre > results/${pre}analysis.log 2>&1
  uv run python experiment_disparity.py --target 1.0 --prefix $pre > results/${pre}loop.log 2>&1
done
echo ALLDONE
