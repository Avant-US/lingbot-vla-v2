#!/bin/bash
export LINGBOT_ROOT=/home/luogang/SRC/Robot/lingbot-vla-v2
export ROBOTWIN_ROOT=/home/luogang/share/zwy/Projects/RoboTwin
export WS_PORT=8007
export PYTHONPATH="${ROBOTWIN_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=1
export PYOPENGL_PLATFORM=egl
export MESA_GL_VERSION_OVERRIDE=4.1
export SAPIEN_DISABLE_VULKAN_VALIDATION=1
echo "[Client-plcbrd] GPU=1, WS_PORT=${WS_PORT}"
