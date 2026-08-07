#!/bin/bash
# lingbot-vla-v2 推理服务端环境变量
# 用法: conda activate linbvla2 && source b/d/eval_server_env.sh

export LINGBOT_ROOT=/home/luogang/SRC/Robot/lingbot-vla-v2
export CKPT_PATH=/home/luogang/CKPT/VLA/linbVLA2/rbtwn2/10000/hf_ckpt
export QWEN3VL_PATH=/home/luogang/CKPT/VLM/Qwen3-VL-4B-Instruct
export WS_PORT=8006

# PYTHONPATH: lingbot-vla-v2 代码库
export PYTHONPATH="${LINGBOT_ROOT}:${PYTHONPATH:-}"

# CUDA
export CUDA_HOME="/usr/local/cuda-12.8"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

# HuggingFace
export HF_HOME=/home/luogang/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

echo "[Server] 环境变量已设置. CKPT=${CKPT_PATH}, PORT=${WS_PORT}"
