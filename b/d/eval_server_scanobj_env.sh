#!/bin/bash
export LINGBOT_ROOT=/home/luogang/SRC/Robot/lingbot-vla-v2
export CKPT_PATH=/home/luogang/CKPT/VLA/linbVLA2_scanobj/checkpoints/global_step_5016/hf_ckpt
export QWEN3VL_PATH=/home/luogang/CKPT/VLM/Qwen3-VL-4B-Instruct
export WS_PORT=8006
export PYTHONPATH="${LINGBOT_ROOT}:${PYTHONPATH:-}"
export CUDA_HOME="/usr/local/cuda-12.8"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/home/luogang/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
echo "[Server-scanobj] CKPT=${CKPT_PATH}, PORT=${WS_PORT}"
