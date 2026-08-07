#!/bin/bash
# RoboTwin 仿真评估端环境变量
# 用法: conda activate RoboTwin && source eval_client_env.sh

export LINGBOT_ROOT=/home/luogang/SRC/Robot/lingbot-vla-v2
export ROBOTWIN_ROOT=/home/luogang/share/zwy/Projects/RoboTwin
export CKPT_PATH=/home/luogang/CKPT/VLA/linbVLA2/rbtwn2/10000/hf_ckpt
export WS_PORT=8006

# PYTHONPATH: RoboTwin (eval_policy.py 需要)
export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH:-}"

# CUDA (仿真侧不需要 GPU, 但 SAPIEN 渲染可能用到)
export CUDA_VISIBLE_DEVICES=0

# 渲染 (headless)
export PYOPENGL_PLATFORM=egl
export MESA_GL_VERSION_OVERRIDE=4.1
export SAPIEN_DISABLE_VULKAN_VALIDATION=1

echo "[Client] 环境变量已设置. WS_PORT=${WS_PORT}"
