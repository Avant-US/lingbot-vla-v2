# LingBot-VLA 2.0 单任务微调: stack_bowls_three — 执行日志

在预训练模型 [lingbot-vla-v2-6b](https://huggingface.co/robbyant/lingbot-vla-v2-6b) 基础上, 用 RoboTwin 2.0 的 `stack_bowls_three` clean 数据进行 post-training (fine-tune).

本文按时间顺序记录所有操作、遇到的 error / 根因 / fix、增删改的文件与关键路径。

---

## 目标配置

| 项目 | 值 |
|------|-----|
| 基础模型 | `robbyant/lingbot-vla-v2-6b` (HuggingFace) |
| 微调数据 | `/mnt/r/DATA/RoboTwin-Clean/stack_bowls_three/` |
| 目标任务 | stack_bowls_three (三碗堆叠) |
| 虚拟环境 | `/mnt/r/VENV/lbvla2/` (Python 3.12.13) |
| 权重目录 | `/mnt/r/CKPT/` |
| GPU | 8 × NVIDIA H200 |
| micro_batch_size | 16 (实测未 OOM) |
| global_batch_size | 128 (= 16 × 8) |
| max_steps | 10000 |
| 优化器 | muon |
| use_compile | **false** (首次用 true 导致 rank0 崩溃) |
| 代码仓库 | `/home/physical/SRC/Robot/lingbot-vla-v2` |
| 训练输出 | `/mnt/r/CKPT/lingbot-vla-v2-ft-stack_bowls_three/` |

### 关键路径速查

| 类型 | 路径 |
|------|------|
| 预训练 VLA 权重 | `/mnt/r/CKPT/lingbot-vla-v2-6b/` (权重在根目录; `hf_ckpt/` 为兼容 symlink) |
| Qwen3 tokenizer | `/mnt/r/CKPT/Qwen3-VL-4B-Instruct/` |
| MoGe | `/mnt/r/CKPT/moge-2-vitb-normal/moge2-vitb-normal.pt` → `model.pt` |
| Depth 教师 | `/mnt/r/CKPT/lingbot-vla-v2-6b/depth/model.pt` |
| DINO-Video 教师 | `/mnt/r/CKPT/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth` |
| Norm stats (实际使用) | `assets/norm_stats/robotwin.json` (50-task 全局) |
| 训练配置 | `configs/vla/robotwin/stack_bowls_three_ft.yaml` |
| Norm 计算配置 | `configs/vla/robotwin/stack_bowls_three_norm.yaml` |
| 训练 stdout 日志 | `/mnt/r/CKPT/lingbot-vla-v2-ft-stack_bowls_three/train_stdout.log` |
| train.sh tee 日志 | `/home/physical/SRC/Robot/lingbot-vla-v2/log.txt` |
| TensorBoard | `/mnt/r/CKPT/lingbot-vla-v2-ft-stack_bowls_three/runs/` |
| Checkpoints (DCP) | `/mnt/r/CKPT/lingbot-vla-v2-ft-stack_bowls_three/checkpoints/` (每 5000 step) |
| 原始 v2.1 数据备份 | `/mnt/r/DATA/RoboTwin-Clean/stack_bowls_three_old/` |
| 当前训练数据 (v3.0) | `/mnt/r/DATA/RoboTwin-Clean/stack_bowls_three/` |

---

## 执行记录 (按时间顺序)

### [2026-07-31] Step 0: 虚拟环境搭建

#### 0.1 现状检查

**操作**: 检查系统环境、数据、权重。

**结果**:
- 系统默认 Python: 3.10.12, **无 python3.12**
- `/mnt/r/VENV/lbvla2/` **不存在**
- `/mnt/r/CKPT/` 中无 `lingbot-vla-v2-6b` / `Qwen3-VL-4B-Instruct` / `moge-2-vitb-normal`
- 数据集 `/mnt/r/DATA/RoboTwin-Clean/stack_bowls_three/` **已存在** (LeRobot v2.1, 50 ep, 23550 frames)
- `norm_stats.json` **缺失**
- `stack_bowls_three_ft.yaml` **缺失**
- 8 × H200 GPU 可用

#### 0.2 获取 Python 3.12

**操作**:
```bash
conda create -n py312 python=3.12 -y
```

**原因**: 项目要求 Python 3.12; 系统未安装。

#### 0.3 创建 venv

**操作**:
```bash
source /mnt/r/miniforge3/etc/profile.d/conda.sh && conda activate py312
python -m venv /mnt/r/VENV/lbvla2
source /mnt/r/VENV/lbvla2/bin/activate
pip install -U pip setuptools wheel
```

**结果**: `/mnt/r/VENV/lbvla2/` 创建成功, Python 3.12.13。

#### 0.4 安装 PyTorch 与依赖

**操作** (均在 `/mnt/r/VENV/lbvla2` 中):
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 torchdata==0.11.0 torchcodec==0.6.0
pip install -r requirements.txt
pip install -r requirements-depth.txt
pip install -r requirements.txt          # 重新钉住核心版本
pip install numpydantic==1.9.0 --no-deps
pip install --no-build-isolation flash-attn==2.8.3
pip install --no-deps "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz"
pip install -e . --no-deps
pip install -e lingbotvla/models/vla/vision_models/lingbot-depth --no-deps
pip install -e lingbotvla/models/vla/vision_models/MoGe
pip install huggingface_hub==0.34.0
pip install numpy==1.26.4                # MoGe 会把 numpy 升到 2.x, 需钉回
```

**结果**: 验证通过 — torch 2.8.0+cu128, flash_attn 2.8.3, transformers 4.57.3, lingbotvla/moge/mdm 均可 import。

---

### [2026-07-31] Step 1: 下载权重

**操作**:
```bash
export HF_HOME=/mnt/r/CKPT/hf_home
python3 scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-v2-6b --local_dir /mnt/r/CKPT
python3 scripts/download_hf_model.py --repo_id Qwen/Qwen3-VL-4B-Instruct --local_dir /mnt/r/CKPT
python3 scripts/download_hf_model.py --repo_id Ruicheng/moge-2-vitb-normal --local_dir /mnt/r/CKPT
```

**结果**:
- `/mnt/r/CKPT/lingbot-vla-v2-6b/` ≈ 27G (含 6 shard safetensors + depth + dino_video)
- `/mnt/r/CKPT/Qwen3-VL-4B-Instruct/` ≈ 8.3G
- `/mnt/r/CKPT/moge-2-vitb-normal/model.pt` ≈ 400M

#### Error-1: HF 权重目录结构与文档假设不一致

| 项 | 内容 |
|----|------|
| **现象** | 参考文档假设 `model_path=.../hf_ckpt`, MoGe 文件名为 `moge2-vitb-normal.pt`; 实际 HF 仓库把 safetensors 放在 repo 根目录, MoGe 文件名为 `model.pt`, **无** `hf_ckpt/` 子目录 |
| **根因** | HuggingFace 发布布局与早期文档/路径约定不同 |
| **Fix** | 创建兼容 symlink, 不改动训练代码 |

**操作**:
```bash
# VLA: 创建 hf_ckpt/ 并 symlink 权重与 tokenizer 相关文件
mkdir -p /mnt/r/CKPT/lingbot-vla-v2-6b/hf_ckpt
cd /mnt/r/CKPT/lingbot-vla-v2-6b
# symlink model-*.safetensors, model.safetensors.index.json, config.json, tokenizer* 等 → hf_ckpt/

# MoGe: 兼容文件名
ln -sfn /mnt/r/CKPT/moge-2-vitb-normal/model.pt \
        /mnt/r/CKPT/moge-2-vitb-normal/moge2-vitb-normal.pt
```

**新增文件/链接**:
- `/mnt/r/CKPT/lingbot-vla-v2-6b/hf_ckpt/*` (symlinks)
- `/mnt/r/CKPT/moge-2-vitb-normal/moge2-vitb-normal.pt` (symlink)

---

### [2026-07-31] Step 2: 数据验证

**操作**: 读取 `meta/info.json` 校验版本、维度、相机。

**结果** (转换前): LeRobot v2.1, 50 episodes, 23550 frames, state/action dim=14, 三相机匹配 robotwin config。

---

### [2026-07-31] Step 3: 归一化统计量

#### Error-2: `compute_norm_stats` 不接受 `model.*` 配置段

| 项 | 内容 |
|----|------|
| **现象** | 用 `robotwin.yaml` 跑 norm 脚本报: `Some specified arguments are not used ... --model.model_path ...` |
| **根因** | `scripts/compute_norm_stats.py` 的 `Arguments` 只有 `data` / `train`, 无 `model`; YAML 中的 `model:` 会被解析为多余参数 |
| **Fix** | 新增无 `model` 段的专用配置 `configs/vla/robotwin/stack_bowls_three_norm.yaml` |

#### Error-3: TrainingArguments 要求 `max_steps` / `num_train_epochs`

| 项 | 内容 |
|----|------|
| **现象** | `ValueError: At least one of num_train_epochs and max_steps must be specified.` |
| **根因** | 精简 YAML 漏了 `train.max_steps` |
| **Fix** | 在 norm 配置中加 `max_steps: 1` |

#### Error-4: LeRobot 0.4.2 拒绝 v2.1 数据集

| 项 | 内容 |
|----|------|
| **现象** | `BackwardCompatibilityError: dataset ... is in 2.1 format ... new format since v3.0` |
| **根因** | 环境安装的是 lerobot==0.4.2 (`CODEBASE_VERSION=v3.0`), 加载 v2.1 元数据时强制 major 版本检查失败。仓库 README 虽写支持 v2.1/v3.0, 但 0.4.2 实际会 raise |
| **Fix** | 将数据集就地转换为 v3.0 (原数据备份为 `*_old`) |

**操作**:
```bash
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
  --repo-id=stack_bowls_three \
  --root=/mnt/r/DATA/RoboTwin-Clean \
  --push-to-hub=false
```

**结果**:
- 当前数据: `/mnt/r/DATA/RoboTwin-Clean/stack_bowls_three/` → **v3.0**
- 备份: `/mnt/r/DATA/RoboTwin-Clean/stack_bowls_three_old/` → 原 v2.1

#### Error-5: `compute_norm_stats` 多进程卡死

| 项 | 内容 |
|----|------|
| **现象** | `num_workers=8` 时进度到 ~78% 后卡在 futex; 主进程与 worker 均无进展 |
| **根因** | `mp.Pool(fork)` + 含视频元数据的 dataset 在 worker 间共享易死锁; `use_future_image=true` 仍会带 video delta |
| **尝试** | 改为 `num_workers=1` + `use_future_image=false`, 可跑但很慢 (~20 batch/s, ETA ~20min) |
| **最终方案** | **改用方案 B**: 直接使用仓库自带 50-task 全局统计 `assets/norm_stats/robotwin.json` (stack_bowls_three 是其中一 task, 分布被全局统计覆盖). 单任务 stats 可后续再补算 |

**新增文件**:
- `configs/vla/robotwin/stack_bowls_three_norm.yaml` (norm 专用配置)

---

### [2026-07-31] Step 4: 创建训练配置

**操作**: 基于 `configs/vla/robotwin/robotwin.yaml` 创建单任务 8-GPU 微调配置。

**新增文件**: `configs/vla/robotwin/stack_bowls_three_ft.yaml`

相对原配置的主要差异:

| 参数 | 原 (32 GPU / 50 task) | 微调配置 | 原因 |
|------|----------------------|----------|------|
| `data.data_name` | multi | robotwin | 单数据集 |
| `data.train_path` | assets/.../robotwin.txt | 本地 LeRobot 路径 | 指向 stack_bowls_three |
| `micro_batch_size` | 32 | 16 | 8 GPU 适配 |
| `global_batch_size` | 1024 | 128 | 16×8 |
| `max_steps` | 50000 | 10000 | 单任务微调 |
| `save_steps` | 10000 | 5000 | 中间 checkpoint |
| `enable_gradient_checkpointing` | false | true | 省显存 |
| `use_compile` | true | **false** | 见 Error-6 |
| 所有 path | 占位符 | `/mnt/r/CKPT/...` | 实际路径 |

---

### [2026-07-31] Step 5: 启动训练

#### Attempt 1 — `use_compile=true` (失败)

**操作**:
```bash
source /mnt/r/VENV/lbvla2/bin/activate
cd /home/physical/SRC/Robot/lingbot-vla-v2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bash train.sh tasks/vla/train_lingbotvla.py \
  ./configs/vla/robotwin/stack_bowls_three_ft.yaml \
  --data.norm_stats_file assets/norm_stats/robotwin.json \
  > /mnt/r/CKPT/lingbot-vla-v2-ft-stack_bowls_three/train_stdout.log 2>&1 &
```

**现象**:
- 数据加载成功 (打印 batch shapes: micro_bs=16, action `[16,50,55]`)
- 各卡显存升至 ~59GB
- **rank0 进程消失**, GPU0 显存降为 ~292MiB
- rank1–7 成为孤儿进程 (PPID=1), GPU 100% 空转 (NCCL 等待)
- 日志无完整 Traceback (rank0 异常退出后未刷出)

#### Error-6: `torch.compile` 导致 rank0 崩溃

| 项 | 内容 |
|----|------|
| **现象** | 首次 forward / compile 阶段 rank0 静默退出, 分布式作业挂死 |
| **根因** | `use_compile: true` 下 torch.compile 首次编译开销/显存尖峰或内部错误导致 rank0 退出; 其余 rank 无错误处理而挂起 |
| **Fix** | 1) `pkill` 清理孤儿进程; 2) 配置改为 `use_compile: false`; 3) CLI 再传 `--train.use_compile false`; 4) 打开 `PYTHONFAULTHANDLER` / `TORCH_SHOW_CPP_STACKTRACES` 便于下次诊断 |

**修改文件**: `configs/vla/robotwin/stack_bowls_three_ft.yaml` (`use_compile: false`)

**保留日志**: `/mnt/r/CKPT/lingbot-vla-v2-ft-stack_bowls_three/train_stdout_attempt1.log`

#### Attempt 2 — `use_compile=false` (成功, 稳定训练中)

**操作**:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTHONUNBUFFERED=1
setsid bash -c 'bash train.sh tasks/vla/train_lingbotvla.py \
  ./configs/vla/robotwin/stack_bowls_three_ft.yaml \
  --data.norm_stats_file assets/norm_stats/robotwin.json \
  --train.use_compile false \
  > /mnt/r/CKPT/lingbot-vla-v2-ft-stack_bowls_three/train_stdout.log 2>&1' </dev/null &
```

**结果** (启动后约 4 分钟观测):

| Step | Loss | VLA_Loss | Depth_Loss | FutureVideo_Loss | StepTime |
|------|------|----------|------------|------------------|----------|
| 1 | 0.3850 | 0.3506 | 3.92 | 0.76 | ~19.8s (首步含初始化) |
| 25 | 0.1597 | 0.1430 | 1.81 | 0.38 | ~3.7s |
| 50 | 0.0991 | 0.0859 | 1.39 | 0.29 | ~3.7s |
| 61 | 0.0869 | 0.0745 | 1.29 | 0.26 | ~3.7s |

- 8 GPU 显存稳定 ~57258 MiB / 143771 MiB (**未 OOM, batch=16 可用**)
- 预计总时长 ~10 小时 (10000 steps × ~3.7s)
- 每 epoch ≈ 183 steps (23550 frames / global_bs 128); 10000 steps ≈ 54.6 epochs
- Checkpoint: 将在 step 5000 / 10000 写入 `checkpoints/global_step_*`

**TensorBoard**:
```bash
tensorboard --logdir /mnt/r/CKPT/lingbot-vla-v2-ft-stack_bowls_three/runs --port 6006
```
当前 events 文件: `events.out.tfevents.*.2730614.0` (attempt2)

**监控命令**:
```bash
# 最新 step
rg -N 'Step [0-9]+/183' /mnt/r/CKPT/lingbot-vla-v2-ft-stack_bowls_three/train_stdout.log | tail -3

# 进程
pgrep -af 'torchrun.*train_lingbotvla|tasks/vla/train_lingbotvla'

# 显存
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

---

## 增删改文件汇总

### 新增

| 路径 | 原因 |
|------|------|
| `/mnt/r/VENV/lbvla2/` | 训练虚拟环境 |
| `/mnt/r/CKPT/lingbot-vla-v2-6b/` | HF 预训练权重 |
| `/mnt/r/CKPT/lingbot-vla-v2-6b/hf_ckpt/` | 兼容 `model_path` 的 symlink 目录 |
| `/mnt/r/CKPT/Qwen3-VL-4B-Instruct/` | VLM tokenizer / processor |
| `/mnt/r/CKPT/moge-2-vitb-normal/` | MoGe 深度模型 |
| `/mnt/r/CKPT/moge-2-vitb-normal/moge2-vitb-normal.pt` | 文件名兼容 symlink |
| `/mnt/r/CKPT/lingbot-vla-v2-ft-stack_bowls_three/` | 训练输出根目录 |
| `configs/vla/robotwin/stack_bowls_three_ft.yaml` | 8-GPU 单任务微调配置 |
| `configs/vla/robotwin/stack_bowls_three_norm.yaml` | norm 计算专用配置 |
| `/mnt/r/DATA/RoboTwin-Clean/stack_bowls_three_old/` | v2.1→v3.0 转换时的原数据备份 |

### 修改

| 路径 | 原因 |
|------|------|
| `/mnt/r/DATA/RoboTwin-Clean/stack_bowls_three/` | v2.1 转换为 v3.0 (lerobot 0.4.2 要求) |
| `configs/vla/robotwin/stack_bowls_three_ft.yaml` | `use_compile: true` → `false` (修复 rank0 崩溃) |
| `b/d/reprd_rbtwn_stackb3.md` | 本执行日志 |

### 未改动的仓库训练核心代码

训练代码本身未 patch; 所有兼容问题通过环境、数据转换、配置与 symlink 解决。

---

## Error 总表

| # | 阶段 | Error | 根因 | Fix |
|---|------|-------|------|-----|
| 1 | 权重 | 无 `hf_ckpt/` / MoGe 文件名不同 | HF 布局与文档不一致 | symlink 兼容 |
| 2 | Norm | YAML `model.*` 参数未使用 | norm 脚本 Arguments 无 model | 新建无 model 的 YAML |
| 3 | Norm | 缺 max_steps | TrainingArguments 校验 | 加 `max_steps: 1` |
| 4 | 数据 | BackwardCompatibilityError v2.1 | lerobot 0.4.2 仅接受 v3.0 major | `convert_dataset_v21_to_v30` |
| 5 | Norm | 多进程卡死 | fork Pool + dataset 死锁 | 改用全局 `robotwin.json` |
| 6 | 训练 | rank0 崩溃, 作业挂死 | `torch.compile` 首次编译阶段 | `use_compile: false` |

---

## 当前训练状态 (写文档时)

- **状态**: 稳定运行中
- **配置**: 8 GPU × micro_bs=16, global_bs=128, max_steps=10000, muon, compile=off
- **显存**: ~57 GB / 143 GB per GPU (batch 16 充足, 无需下调)
- **日志**: `/mnt/r/CKPT/lingbot-vla-v2-ft-stack_bowls_three/train_stdout.log`
- **TB**: `/mnt/r/CKPT/lingbot-vla-v2-ft-stack_bowls_three/runs/`
- **预计完成**: ~10 小时后, checkpoint 在 `checkpoints/global_step_5000` 与 `global_step_10000`

### 训练完成后的评测命令 (待执行)

```bash
source /mnt/r/VENV/lbvla2/bin/activate
cd /home/physical/SRC/Robot/lingbot-vla-v2
export QWEN3_PATH=/mnt/r/CKPT/Qwen3-VL-4B-Instruct

python scripts/open_loop_eval.py \
  --model_path /mnt/r/CKPT/lingbot-vla-v2-ft-stack_bowls_three/checkpoints/global_step_10000 \
  --robo_name robotwin \
  --data_path /mnt/r/DATA/RoboTwin-Clean/stack_bowls_three/ \
  --use_length 50
```

---

## 参考

- 论文: [From Foundation to Application: Improving VLA Models in Practice](https://arxiv.org/abs/2607.06403) ([HTML](https://arxiv.org/html/2607.06403v1))
- 项目主页: https://technology.robbyant.com/lingbot-vla-v2
- GitHub: https://github.com/Robbyant/lingbot-vla-v2
- 模型权重: https://huggingface.co/robbyant/lingbot-vla-v2-6b
- LeRobot v2.1→v3.0 转换: https://github.com/huggingface/lerobot/blob/v0.4.2/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py
