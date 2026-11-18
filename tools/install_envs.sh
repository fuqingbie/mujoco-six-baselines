#!/bin/bash
# Install the two project environments for the MuJoCo six-baseline experiment.
# mjb  : phase-B unified stack (robustness-gymnasium 0.29.2 fork, mujoco 2.3.7, torch 2.5.1)
# mjb5 : phase-A v5 stack (gymnasium 1.2.0, robustrllib, sparc deps)
set -e
export PIP_CACHE_DIR=/root/code/continue_learning/.cache/pip
ROOT=/root/mjbexp
MJB=$ROOT/envs/mjb/bin
MJB5=$ROOT/envs/mjb5/bin
PY311=/opt/conda/bin/python

conda create -p $ROOT/envs/mjb5 python=3.11 -y >/dev/null 2>&1 || true

echo "=== [mjb] base wheels ==="
$MJB/pip install --no-cache-dir \
  torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -2
$MJB/pip install \
  "numpy==1.26.4" "pandas==2.2.3" "scipy==1.14.1" "mujoco==2.3.7" "gym==0.26.2" \
  click glfw pyopengl tqdm tensorboard matplotlib imageio pyarrow pyyaml pytest 2>&1 | tail -2

echo "=== [mjb] robustness-gymnasium (gymnasium 0.29.2 fork) ==="
$MJB/pip install -e $ROOT/upstreams/rmbpo-tmlr/embodied/robustness-gymnasium 2>&1 | tail -2

echo "=== [mjb] unstable_baselines ==="
$MJB/pip install -e $ROOT/upstreams/unstable_baselines --no-deps 2>&1 | tail -2
$MJB/pip install cv2-python 2>/dev/null || $MJB/pip install opencv-python-headless 2>&1 | tail -1

echo "=== [mjb] rmbpo-tmlr ==="
$MJB/pip install -e $ROOT/upstreams/rmbpo-tmlr --no-deps 2>&1 | tail -2

echo "=== [mjb] SB3 ==="
$MJB/pip install "stable-baselines3==2.3.2" "sb3-contrib==2.3.2" 2>&1 | tail -2

echo "=== [mjb5] v5 stack ==="
$MJB5/pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -2
$MJB5/pip install "gymnasium[mujoco]==1.2.0" "stable-baselines3==2.3.2" "sb3-contrib==2.3.2" \
  tyro wandb pytest "numpy<2" 2>&1 | tail -2

echo "=== [mjb5] robustrllib ==="
$MJB5/pip install -e "$ROOT/upstreams/robustrllib[mujoco,methods]" 2>&1 | tail -2

echo "=== DONE ==="
