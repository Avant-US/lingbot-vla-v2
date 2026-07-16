# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LingBot-VLA 2.0 is a Vision-Language-Action (VLA) foundation model for robot manipulation. It builds on Qwen3-VL-4B as the vision-language backbone and adds a flow-matching action expert with sparse MoE layers, plus dual-query distillation from LingBot-Depth and DINO-Video teachers. The unified 55-dimensional action space covers arms, end-effectors, grippers, dexterous hands, waist, head, and mobile base.

## Environment Setup

```bash
bash tools/create_train_env.sh              # creates conda env, installs deps + flash-attn
bash tools/create_train_env.sh --env-name lingbotvla --recreate  # rebuild from scratch
```

Requires: Miniconda/Anaconda, Python 3.12, PyTorch 2.8.0, CUDA with flash-attn 2.8.3.

## Common Commands

### Training (distributed via torchrun)

```bash
# Post-training on RoboTwin (auto-detects GPU count)
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin/robotwin.yaml \
  --data.train_path assets/training_data/robotwin.txt \
  --data.data_name multi \
  --train.output_dir output/

# Specify GPU subset
CUDA_VISIBLE_DEVICES=0,1 bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin/robotwin.yaml ...
```

`train.sh` is a thin wrapper around `torchrun` that auto-detects `NPROC_PER_NODE` from `CUDA_VISIBLE_DEVICES` or `nvidia-smi`. It sets `HF_HUB_OFFLINE=1` and related env vars.

### Compute Normalization Statistics

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh scripts/compute_norm_stats.py ./configs/vla/robotwin/robotwin.yaml \
  --data.data_name multi \
  --data.train_path assets/training_data/robotwin.txt \
  --data.norm_path assets/norm_stats/robotwin.json \
  --data.data_ratio_for_norm_compute 1
```

### Open-Loop Evaluation

```bash
export QWEN3_PATH=Qwen/Qwen3-VL-4B-Instruct
python scripts/open_loop_eval.py \
  --model_path path_to_ckpt --robo_name robotwin \
  --data_path path_to_val_data --use_length 50
```

### Deployment (WebSocket inference server)

```bash
export QWEN3VL_PATH=path_to_Qwen3-VL-4B-Instruct
python -m deploy.lingbot_vla_v2_policy \
  --model_path path_to_ckpt --use_compile --use_length 25 --port 8080
```

### Linting and Formatting

```bash
make quality   # ruff check + format --check
make style     # ruff check --fix + format
make test      # pytest tests/
```

Ruff config: line-length 119, target py38, isort with `lingbotvla` as first-party.

## Architecture

### Package Structure

- `lingbotvla/` — core library (installed as `lingbotvla` package)
  - `models/vla/lingbot_vla/` — LingBot-VLA v1 and v2 model implementations (`modeling_lingbot_vla.py`, `modeling_lingbot_vla_v2.py`), config (`configuration_lingbot_vla.py`), MoE routing (`moe_load_balance.py`), VLM patches (`qwen3vl_in_vla.py`, `qwenvl_in_vla.py`)
  - `models/vla/pi0/` — pi0 baseline model implementation
  - `models/vla/vision_models/` — depth teacher (LingBot-Depth/MoGe) and video teacher (DINO-Video) wrappers
  - `models/registry.py` — auto-discovers model classes from `lingbotvla.models.vla` via `ModelClass` module attribute
  - `models/config_registry.py` — maps config keys (`LingbotVLAConfig`, `LingbotVLAV2Config`) to config classes
  - `data/vla_data/` — VLA dataset loading from LeRobot v2.1/v3.0 format, multi-dataset support, normalization, transforms
  - `distributed/` — FSDP1/FSDP2 parallelization, MoE expert parallelism, sequence parallelism (Ulysses)
  - `ops/` — fused MoE kernels (Triton), attention ops, loss functions
  - `optim/` — AdamW, Muon optimizer, LR schedulers
  - `checkpoint/` — distributed checkpointing (DCP and ByteCheckpoint backends)
- `tasks/vla/` — training entry points (`train_lingbotvla.py`, `train_pi0.py`)
- `configs/` — YAML training configs and robot feature-mapping configs
- `scripts/` — utility scripts (norm stats, model download, open-loop eval)
- `deploy/` — inference server with WebSocket protocol and `torch.compile` support
- `experiment/` — benchmark-specific evaluation scripts (RoboTwin)

### Config / Argument System

Arguments are defined as nested dataclasses in `lingbotvla/utils/arguments.py` and extended in each training script (e.g., `MyTrainingArguments` in `tasks/vla/train_lingbotvla.py`). The first CLI argument is a YAML config file; subsequent `--group.param value` flags override individual fields. Config hierarchy: YAML < CLI overrides.

### Model Registration

Models register via a `ModelClass` module-level variable in any `.py` file under `lingbotvla/models/vla/`. The registry is discovered at import time by `lingbotvla/models/registry.py`. Config classes register similarly via `lingbotvla/models/config_registry.py`, selected by the `config_key` argument (e.g., `LingbotVLAV2Config`).

### Data Pipeline

VLA datasets expect LeRobot v2.1 or v3.0 format. Robot configs (`configs/robot_configs/*.yaml`) map raw feature names to the unified 55-dim state/action space. For multi-dataset training, set `data.data_name: multi` and point `data.train_path` to a text file with `<robot_config_name> <dataset_path>` lines. Normalization stats must be precomputed per robot config.

### Distributed Training

Default: FSDP2 (`data_parallel_mode: fsdp2`). Supports FSDP1, DDP, expert parallelism, and Ulysses sequence parallelism. Checkpoint backends: PyTorch DCP (`ckpt_manager: dcp`) or ByteCheckpoint (`bytecheckpoint`).
