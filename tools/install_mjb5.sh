#!/bin/bash
# mjb5: phase-A v5 stack (gymnasium 1.2.0, robustrllib, sparc deps)
set -x
export PIP_CACHE_DIR=/root/code/continue_learning/.cache/pip
MJB5=/root/mjbexp/envs/mjb5/bin
ROOT=/root/mjbexp
$MJB5/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
$MJB5/pip install "gymnasium[mujoco]==1.2.0" "numpy<2" tyro wandb pytest
$MJB5/pip install "stable-baselines3==2.3.2"
$MJB5/pip install -e "$ROOT/upstreams/robustrllib[mujoco,methods]" || $MJB5/pip install -e "$ROOT/upstreams/robustrllib[mujoco]"
$MJB5/pip install opencv-python-headless glfw pyopengl
echo MJB5_DONE
