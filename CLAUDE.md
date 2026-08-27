# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LingBot-VLA 2.0 is a Vision-Language-Action (VLA) foundation model for robotics. It combines a Qwen3-VL vision-language backbone with a sparse MoE action expert and dual-query distillation (depth + video teachers) to produce a 55-dimensional unified action representation across 20+ robot embodiments.

## Common Commands

### Environment Setup
```bash
bash tools/create_train_env.sh          # Creates conda env "lingbotvla" with all deps
pip install -e . --no-deps              # Editable install (if env already exists)
```

### Training
```bash
# Distributed training (auto-detects GPUs)
bash train.sh tasks/vla/train_lingbotvla.py configs/vla/robotwin/robotwin.yaml

# Override config values via CLI
bash train.sh tasks/vla/train_lingbotvla.py configs/vla/robotwin/robotwin.yaml --lr 1e-4

# Control GPU selection
CUDA_VISIBLE_DEVICES=0,1 bash train.sh tasks/vla/train_lingbotvla.py <config.yaml>

# Multi-node (set env vars before calling train.sh)
NNODES=2 NODE_RANK=0 MASTER_ADDR=<ip> MASTER_PORT=62500 bash train.sh ...
```

### Code Quality
```bash
make quality    # Lint check (ruff)
make style      # Auto-fix lint + format
make test       # pytest tests/
```

### Deployment
```bash
python -m deploy.lingbot_vla_v2_policy --model_path <path> --port <port>
```

### Utilities
```bash
python scripts/compute_norm_stats.py    # Compute dataset normalization stats
python scripts/open_loop_eval.py        # Open-loop evaluation with MSE/MAE metrics
python scripts/download_hf_model.py     # Download model weights from HuggingFace
```

## Architecture

### Model Pipeline (LingBot-VLA v2)

The model (`QwenvlWithExpertV2Model` in `lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py`) has three stages:

1. **VLM Backbone** (Qwen3-VL-4B): Encodes images and language instructions into token embeddings. Image tokens go through the Qwen3 ViT with flash attention; text tokens are processed by the language model.

2. **Action Expert** (`Qwen2ActionExpert` in `qwen2_action_expert.py`): A 36-layer transformer (768 hidden, 32 heads, 8 KV heads) that receives projected VLM embeddings plus noised action tokens and denoises them via flow matching. Uses AdaRMSNorm with timestep conditioning and sparse MoE layers (32 experts, top-4 routing, shared expert, loss-free balancing).

3. **Dual-Query Distillation**: Appends current+future perceptual queries distilled from LingBot-Depth (geometric/depth cues via MoGe) and DINO-Video (temporal priors). Teacher models are in `lingbotvla/models/vla/vision_models/`.

### Data Pipeline

```
LeRobot dataset (HF parquet/video)
  → FeatureTransform (configs/robot_configs/*.yaml maps raw features → unified 55-dim space)
  → Normalization (assets/norm_stats/*.json, schemes: meanstd, bounds_99, minmax, sincos, identity)
  → VLADataset / MultiVLADataset (lingbotvla/data/vla_data/)
  → VLADataCollatorWithPacking (variable-length sequence packing)
  → Qwen3-VL image processor
```

Adding a new robot embodiment requires: a robot config YAML mapping its features to the 55-dim space, normalization stats, and LeRobot-format datasets. See `lingbotvla/data/vla_data/README.md`.

### Key Package Layout

- `lingbotvla/models/` — Model construction (`build_foundation_model()`, `build_processor()`), config registry
- `lingbotvla/models/vla/lingbot_vla/` — Core VLA v1/v2 architecture, action expert, MoE, flex attention
- `lingbotvla/models/vla/pi0/` — Alternative Pi0 VLA architecture
- `lingbotvla/data/vla_data/` — VLA datasets, feature transforms, video decoding
- `lingbotvla/distributed/` — FSDP1/FSDP2, MoE parallelism, sequence parallelism (Ulysses)
- `lingbotvla/ops/` — Custom ops: fused MoE, group GEMM (Triton kernels), attention, loss functions
- `lingbotvla/schedulers/` — Flow matching scheduler for action denoising
- `lingbotvla/checkpoint/` — DCP and ByteCheckpoint save/load
- `lingbotvla/optim/` — AdamW, Muon optimizers; cosine/constant LR schedulers
- `tasks/vla/` — Training entry points (`train_lingbotvla.py`, `train_pi0.py`)
- `deploy/` — WebSocket policy server for real-robot deployment
- `configs/vla/` — Training YAML configs; `configs/robot_configs/` — robot feature mappings

### Training Entry Points

- `tasks/vla/train_lingbotvla.py` — Main LingBot-VLA 2.0 training (MoE, depth/video distillation, flow matching)
- `tasks/vla/train_pi0.py` — Pi0 model training (alternative architecture)

Both are launched via `train.sh` which wraps `torchrun` for distributed training. Config is YAML-based with CLI overrides. Full parameter reference in `configs/vla/Training_Config.md`.

## Code Style

- Ruff for linting and formatting (line length 119, Python 3.8+ target)
- Double quotes, space indentation
- isort with `lingbotvla` as first-party
- The Makefile `check_dirs` references `veomni` which is a legacy name; active code is in `lingbotvla/`

# 介绍

- 这是论文[From Foundation to Application: Improving VLA Models in Practice](https://arxiv.org/abs/2607.06403)的代码库, 论文的html版在 https://arxiv.org/html/2607.06403v1 
- 论文的项目主页在 https://technology.robbyant.com/lingbot-vla-v2 , 
- 论文的GitHUb在 https://github.com/Robbyant/lingbot-vla-v2 ,
- 论文的模型权重在 https://huggingface.co/robbyant/lingbot-vla-v2-6b 


# 设计/方案/分析/解释和写文档的注意点

* 图表用mermaid, 数学相关的用LaTex, 必要时可以用py脚本画一些更能帮助读者理解的图片(图片中的文字用英文). 这些脚本和图一般放在与生成的文档同目录的`asset`子文件夹中.
* 如果在公式和内容中用到了数学符号或代号, 请在该公式或内容的附近对该符号给予解释.
* 分析,解析和撰写文档时, 可以参考论文或代码库的官网, 官方文档, GItHUb, 参考github中的issues, 代码和pull requests, 也可参考网上其它可信来源的相关文章, 但参考内容要列出, 所生产的文档中若有与被参考对象相关的内容也要指出内容的出处. 
* 分析要深入仔细, 既要包括纵向分析(算法或方法的由来与演进历史, 以及在该算法或方法的基础上又演进和优化出了些什么解决类似问题的方法, 新老方法各有什么优缺点, 各适合应用到什么场景), 纵向分析(同时期同类算法的对比分析, 不同算法或方法各有什么优缺点, 各适合应用到什么场景), 和 消融分析(算法或方法中哪些点是在benchmark实验或实践中被证明有效的, 哪些点相对来说更有效, 哪些没那么有效).
* 记得深入分析模型或方法的输入,输出,在输入输出间做了些什么处理. 当然, 各组成模块的输入输出以及中间的处理也要分析. 为了训这个模型用了什么数据集和任务, 训出来后能做什么任务, 训练和推理时的输入输出数据格式大概长什么样.
* 系统或程序的设计要包括静态架构(组件图,类图,组件和类的职责与关系等等)和动态架构(数据流图,序列图,工作流图,不同场景下的各组件或类的调用与协调图.如果是算法还会涉及forward阶段的数据流,模型组件间的调用,以及backwawrd阶段的数据流,gradient流,哪些权重冻结哪些会被更新,和模型组件间的调用等等).
* 代码还是以该代码库的本地代码为准, 但可用参考网上GitHub的issues, commits, pull requests等.
* 解释要深入浅出, 图文并茂, 可以举一些易于理解的例子帮助说明, 对关键的逻辑也要进行深入的代码解读, 要用严谨的科普论文的风格.
