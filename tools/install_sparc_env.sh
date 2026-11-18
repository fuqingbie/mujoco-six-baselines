#!/bin/bash
set -x
export PIP_CACHE_DIR=/root/code/continue_learning/.cache/pip
ROOT=/root/mjbexp
SP=$ROOT/envs/sparc/bin
conda create -p $ROOT/envs/sparc python=3.11 -y
$SP/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
$SP/pip install "gymnasium[mujoco,other]==1.2.0" "numpy==1.26.4"
$SP/pip install "stable-baselines3==2.7.0" "tyro==0.9.27" "wandb==0.21.0" pytest
echo SPARC_ENV_DONE
