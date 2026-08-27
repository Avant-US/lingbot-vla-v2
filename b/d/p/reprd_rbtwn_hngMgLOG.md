# hanging_mug 微调执行日志

对应实施手册：[`reprd_rbtwn_hngMg.md`](reprd_rbtwn_hngMg.md)。

按时间记录所有操作、error / 根因 / fix、增删改文件、命令与理由、关键路径。

---

## 目标配置（执行时）

| 项目 | 值 |
|------|-----|
| 基础模型 | `robbyant/lingbot-vla-v2-6b`（HF snapshot `11c703bf`） |
| 数据 | `/tmp/Dta/RoboTwin-Clean/hanging_mug/`（执行中转为 LeRobot **v3.0**） |
| v2.1 备份 | `/tmp/Dta/RoboTwin-Clean/hanging_mug_old/`（转换器自动挪走） |
| 代码 | `/tmp/SRC/lingbot-vla-v2` |
| 虚拟环境 | `/tmp/itnvla15rbt20/`（Python 3.11.9） |
| `HF_HOME` | `/tmp/itnvla15rbt20/var/hf_home/` |
| Checkpoint | `/tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/` |
| GPU | 8 × NVIDIA A800-SXM4-80GB |
| 训练配置 | `configs/vla/robotwin/hanging_mug_ft.yaml` |
| 归一化 | `assets/norm_stats/robotwin.json`（不重算单任务 stats） |

兼容 symlink（权重本体在 `$HF_HOME/models--*`，不要再拷一份）：

| 用途 | 路径 |
|------|------|
| VLA | `$HF_HOME/ckpts/lingbot-vla-v2-6b/hf_ckpt` |
| Depth 教师 | `$HF_HOME/ckpts/lingbot-vla-v2-6b/depth/model.pt` |
| Video 教师 | `$HF_HOME/ckpts/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth` |
| Qwen3 tokenizer | `$HF_HOME/ckpts/Qwen3-VL-4B-Instruct` |
| MoGe | `$HF_HOME/ckpts/moge-2-vitb-normal/moge2-vitb-normal.pt` |

---

## [2026-08-27] Step 0: 迁移前检查

**操作**: 检查 overlay / `/tmp` 容量、venv 体积、GPU、数据集版本。

```bash
df -h /b /tmp
du -sh /b/VENV/itnvla15rbt20
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

**理由**: overlay 放不下训练 DCP（单份约 20G+）；用户要求把整个 venv+`HF_HOME` 迁到 `/tmp/itnvla15rbt20`，checkpoint 写 `/tmp/Ckp/`。

**结果**:

| 项 | 值 |
|----|-----|
| overlay (`/`) | 500G，当时约 139G 可用 |
| `/tmp` | 2.5T，约 2.4T 可用 |
| `/b/VENV/itnvla15rbt20` | **93G**（含 HF 权重缓存） |
| `/tmp/itnvla15rbt20` | 不存在 |
| GPU | 8×A800 空闲 |
| hanging_mug | LeRobot **v2.1**，50 ep / 16889 frames |
| 训练进程 | 无 |

`pyvenv.cfg` 显示该 venv **原本就创建在** `/tmp/itnvla15rbt20`（`python3 -m venv --copies /tmp/itnvla15rbt20`），后来被放到 `/b/VENV/`。`bin/activate` 的 `VIRTUAL_ENV` 和 `bin/hf` shebang 仍指向 `/tmp/itnvla15rbt20`。迁回原路径后这两处会重新生效。

---

## [2026-08-27] Step 1: 移动虚拟环境并保留旧路径 symlink

**操作**:

```bash
mv /b/VENV/itnvla15rbt20 /tmp/itnvla15rbt20
ln -sfn /tmp/itnvla15rbt20 /b/VENV/itnvla15rbt20
```

**理由**:

- `mv` 跨文件系统（overlay → kubelet 上的 `/tmp`），实际是 copy + unlink，约 93G / ~57s。
- overlay 腾出空间，避免 pip / DCP / 编译再撑爆。
- `ln -sfn` 让仍写死 `/b/VENV/itnvla15rbt20` 的旧脚本/文档继续能找到环境。

**结果**: overlay 可用从约 139G 升到约 **232G**；`/tmp` 占用约 178G。`source /tmp/itnvla15rbt20/bin/activate` 可用。

**改动文件**:

- `/b/VENV/itnvla15rbt20`：实体目录 → 指向 `/tmp/itnvla15rbt20` 的 symlink。
- `$HF_HOME/ckpts/` 下若干绝对链接按新路径重建（指向 `$HF_HOME/models--*` blobs）。

---

## [2026-08-27] Step 2: 更新手册与训练 YAML 中的路径

**改动文件与缘由**:

| 文件 | 改了什么 | 为什么 |
|------|----------|--------|
| `b/d/p/reprd_rbtwn_hngMg.md` | `/b/VENV/itnvla15rbt20` → `/tmp/itnvla15rbt20`；checkpoint `/tmp/CKPT/` → `/tmp/Ckp/`；补 `LD_LIBRARY_PATH`、MoGe `--no-deps`、数据已是 v3.0 | 与实际落地路径一致，避免后人再走 overlay |
| `configs/vla/robotwin/hanging_mug_ft.yaml` | `model_path` / tokenizer / 教师 / MoGe 路径；`output_dir: /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug` | `train.sh` 设了 `HF_HUB_OFFLINE=1`，必须本地路径；用户指定 checkpoint 目录 |

未改 `assets/norm_stats/robotwin.json`：沿用 50-task 全局统计（stack_bowls 经验：单任务重算会卡在多进程 + 视频）。

---

## [2026-08-27] Step 3: 在迁后的 venv 里补依赖

环境审计：torch **2.10.0+cu128**（官方 2.8）、transformers **5.2.0**（官方 4.57.3）、Python **3.11.9**（官方 3.12）。按用户要求 **不另建 3.12**，在本 venv 补包。

**操作**（成功的部分）:

```bash
source /tmp/itnvla15rbt20/bin/activate
export HF_HOME="/tmp/itnvla15rbt20/var/hf_home/"
python -m pip install --no-cache-dir tensorboard qwen-vl-utils peft timm
python -m pip install --no-cache-dir 'numpy==1.26.4'
python -m pip install --no-cache-dir --no-deps \
  "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz"
python -m pip install -e /tmp/SRC/lingbot-vla-v2 --no-deps
python -m pip install -e /tmp/SRC/lingbot-vla-v2/lingbotvla/models/vla/vision_models/lingbot-depth --no-deps
```

**结果**: tensorboard 2.21.0、peft 0.20.0、timm 1.0.28、qwen-vl-utils 0.0.14、numpy 1.26.4、lerobot 0.4.2、lingbotvla 0.0.1 editable、mdm 1.0.0 editable。

### Error-A: pip 钉 `numpy==1.26.4` 时报 dependency conflict

| 项 | 内容 |
|----|------|
| **现象** | resolver 打出 `ERROR: pip's dependency resolver does not currently take into account all the packages...`，但随后 `Successfully installed numpy-1.26.4` |
| **根因** | 环境里有一批声明要 numpy 2.x 的包（InternVLA 遗留）；`--no-deps` 装 lerobot/mdm 后冲突仍在，但不阻止安装 |
| **Fix** | 忽略 resolver 警告，以 `python -c "import numpy; print(numpy.__version__)"` 实测为准（1.26.4） |

### Error-B: `pip install -e MoGe`（带 deps）卡死

| 项 | 内容 |
|----|------|
| **现象** | 解析到 `gradio-6.26`、`opencv-python-5.0.0.93`（73.8 MB）、`numpy-2.4.6`；opencv wheel 下载长时间停在 0%（>6 min） |
| **根因** | `MoGe/pyproject.toml` 把 **gradio / opencv-python / numpy（未钉版本）** 写成硬依赖。训练只走 `moge.model.v2.MoGeModel` + `moge.utils.vis.colorize_depth`，不需要 Gradio demo；本机已有 **opencv 4.12.0** |
| **Fix** | `kill` 卡住的 pip（pid 23366 / 23775），改为： |

```bash
python -m pip install -e /tmp/SRC/lingbot-vla-v2/lingbotvla/models/vla/vision_models/MoGe --no-deps
python -m pip install --no-deps \
  "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183"
python -m pip install 'numpy==1.26.4'
```

`utils3d` 是 `moge.model.v2` 真正 `import` 的包（`utils3d.pt.intrinsics_from_focal_center` 等）。`--no-deps` 避免再拉 `moderngl`。

**验证**:

```
numpy 1.26.4
torch 2.10.0+cu128
OK MoGeModel / MDMModel / colorize_depth
flash_attn 2.8.3
lerobot 0.4.2
lingbotvla editable
```

### Error-C: `import mdm` 看起来失败（误报）

| 项 | 内容 |
|----|------|
| **现象** | 检查脚本对 `mdm.__file__[:80]` 抛 `TypeError: 'NoneType' object is not subscriptable` |
| **根因** | `mdm` 是 namespace package，没有顶层 `__init__.py`，`__file__` 为 `None` |
| **Fix** | 改为 `from mdm.model.v2 import MDMModel`；包实际已 editable 安装成功 |

### Error-D: torchcodec 无法加载

| 项 | 内容 |
|----|------|
| **现象 1** | `OSError: libnppicc.so.12: cannot open shared object file` |
| **根因 1** | 库已在 `$VENV/lib/python3.11/site-packages/nvidia/npp/lib/libnppicc.so.12`，但默认不在 `LD_LIBRARY_PATH` |
| **现象 2** | 加上 NPP 之后变成 `libstdc++.so.6: version 'CXXABI_1.3.15' not found (required by .../libopenvino.so.2621)` |
| **根因 2** | venv 里的 ffmpeg 8 + OpenVINO 需要比系统 `/lib/x86_64-linux-gnu/libstdc++.so.6` 更新的 ABI；venv 自带 `/tmp/itnvla15rbt20/lib/libstdc++.so.6` |
| **Fix** | 训练/解码前： |

```bash
export LD_LIBRARY_PATH="/tmp/itnvla15rbt20/lib:/tmp/itnvla15rbt20/lib/python3.11/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"
```

**验证**: `torchcodec.decoders.VideoDecoder` 能打开 hanging_mug 的 AV1 mp4（333 frames @ 15 FPS）。备选路径 `torchvision`+`pyav` 也能解同一文件。

`VLADataset` 默认 `video_backend='torchcodec'`，且 `importlib.util.find_spec("torchcodec")` 在 **加载失败时仍为 True**，所以不能靠 “没装 torchcodec 就 fallback pyav”。必须把动态库路径修好。

---

## [2026-08-27] Step 4: hanging_mug v2.1 → v3.0

**操作**:

```bash
source /tmp/itnvla15rbt20/bin/activate
export HF_HOME="/tmp/itnvla15rbt20/var/hf_home/"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export LD_LIBRARY_PATH="/tmp/itnvla15rbt20/lib:/tmp/itnvla15rbt20/lib/python3.11/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
  --repo-id=hanging_mug \
  --root=/tmp/Dta/RoboTwin-Clean \
  --push-to-hub=false
```

**理由**: lerobot 0.4.2 的 `CODEBASE_VERSION=v3.0`，加载 v2.1 会 `BackwardCompatibilityError`（与 stack_bowls Error-4 相同）。转换器会把原目录 `mv` 成 `hanging_mug_old/`，再把 `hanging_mug_v30/` 换成 `hanging_mug/`，因此 **没有再 `cp -a` 一份 `hanging_mug_v21_backup`**（省 119M，语义与 stack_bowls 的 `_old` 一致）。

**结果**:

| 路径 | 版本 | 体积 |
|------|------|------|
| `/tmp/Dta/RoboTwin-Clean/hanging_mug/` | v3.0 | 246M |
| `/tmp/Dta/RoboTwin-Clean/hanging_mug_old/` | v2.1 | 119M |

`LeRobotDataset('hanging_mug', root=...)`：`len=16889`，`total_episodes=50`，三相机 `Tensor (3,480,640)`，state/action 14 维，task 为 paraphrased 英文指令。v3 视频路径变为 `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`。

---

## [2026-08-27] Step 5: 启动微调

**未做的事（有意）**:

- 不重算 `assets/norm_stats/hanging_mug.json`（stack_bowls Error-5：多进程 + 视频易 futex 卡死）。
- `use_compile=false`（stack_bowls Error-6：`torch.compile` 导致 rank0 崩溃）。
- 全参微调，不是 LoRA。

**启动前检查**: 8×A800 空闲；端口 62500 空闲；`hf_ckpt` 六个 safetensors symlink 可解析（单 shard ~5G）；`robotwin.json` 5.4K 存在。

**启动脚本**写在 `/tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/launch_train.sh`，理由：把 `HF_HOME` / `LD_LIBRARY_PATH` / `CUDA_VISIBLE_DEVICES` 固定下来，避免交互 shell 丢环境变量导致 torchcodec 再次失败。

```bash
mkdir -p /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug
cd /tmp/SRC/lingbot-vla-v2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/launch_train.sh
```

stdout：`/tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log`（`train.sh` 同时 `tee` 仓库根 `log.txt`）。

健康标志：micro batch `actions` shape `[8, 50, 55]`；8 卡显存稳定；rank0 不掉线。

---

## 增删改文件总表

| 路径 | 动作 | 缘由 |
|------|------|------|
| `/b/VENV/itnvla15rbt20` | 实体 → symlink | overlay 腾空间，旧路径兼容 |
| `/tmp/itnvla15rbt20` | 迁入 | 用户指定的 venv+HF_HOME 位置 |
| `b/d/p/reprd_rbtwn_hngMg.md` | 改路径 / 补依赖 / LD_LIBRARY_PATH / v3.0 | 手册与实操一致 |
| `configs/vla/robotwin/hanging_mug_ft.yaml` | 本地权重路径 + `output_dir=/tmp/Ckp/...` | offline 训练 + 用户指定 ckpt 根 |
| `configs/vla/robotwin/hanging_mug_norm.yaml` | 已存在，本轮未用来算 stats | 备用 |
| `$HF_HOME/ckpts/*` | 重建绝对 symlink | 指向迁后的 blobs |
| `/tmp/Dta/RoboTwin-Clean/hanging_mug/` | v2.1 → v3.0 | lerobot 0.4.2 只要 major=3 |
| `/tmp/Dta/RoboTwin-Clean/hanging_mug_old/` | 新增（转换器） | 保留原 v2.1 |
| `lingbotvla/data/vla_data/transform.py` | `_get_stat` 把 numpy stats 转 Tensor | torch 2.10 worker `stoi` |
| `/tmp/itnvla15rbt20/bin/torchrun` | 新增 shim | venv 缺 console script |
| `/tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/launch_train.sh` | 新建 | 固定 HF_HOME / LD_LIBRARY_PATH / CC |
| venv site-packages | 安装 lerobot/moge/mdm/utils3d/tensorboard/peft/timm/qwen-vl-utils；钉 numpy 1.26.4 | 训练能 import |

未改 `train_lingbotvla.py` / `video_utils.py`。额外改了 `lingbotvla/data/vla_data/transform.py`（Error-H）和 venv `bin/torchrun` shim（Error-E）。

---

### Error-E: `train.sh: torchrun: command not found`

| 项 | 内容 |
|----|------|
| **现象** | 第一次 `setsid` 启动立刻失败，`train_stdout.log` 末行 `train.sh: line 25: torchrun: command not found` |
| **根因** | torch **2.10.0+cu128** 以 site-packages 形式存在，`python -m torch.distributed.run` 可用，但 venv 是 `--copies` 迁过来的 InternVLA 环境，**没有生成 `bin/torchrun` console script** |
| **Fix** | 在 venv 里补 shim `/tmp/itnvla15rbt20/bin/torchrun` → `torch.distributed.run.main`，不改仓库 `train.sh` |

---

### Error-F: `ModuleNotFoundError: No module named 'torchdata'`

| 项 | 内容 |
|----|------|
| **现象** | torchrun 起来约 5s 后 rank3 exitcode=1，其余 rank 被 SIGTERM。交错 traceback 指向 `lingbotvla/data/data_loader.py:21` |
| **根因** | 官方 `requirements.txt` 有 `torchdata==0.11.0`；本 InternVLA venv 没装。`StatefulDataLoader` 是训练 DataLoader 封装 |
| **Fix** | `python -m pip install torchdata==0.11.0`（纯 Python wheel，不碰 torch 2.10） |

### Error-G: `cannot import name 'AutoModelForVision2Seq'`

| 项 | 内容 |
|----|------|
| **现象** | `lingbotvla/models/loader.py:21` 从 transformers 导入 `AutoModelForVision2Seq` 失败 |
| **根因** | 本 venv 原是 **transformers 5.2.0**，该类已移除（改名为 `AutoModelForImageTextToText`）。仓库按官方 **4.57.3** 写死 |
| **Fix** | 按手册路线 B 的回退：`pip install transformers==4.57.3`（顺带把 `huggingface-hub` 从 1.28 降到 **0.36.2**，因为 4.57.3 要求 `<1.0`）。同时补 `blobfile`、`sentencepiece`。**未降 torch**。 |

预导入 `tasks/vla/train_lingbotvla.py` 在 dataclass 处的 `NoneType.__dict__` 是单测用 `importlib` 没把模块挂进 `sys.modules` 的假阳性，不是训练 bug。

---

### Error-H: DataLoader worker `RuntimeError: stoi`（normalize 减法）

| 项 | 内容 |
|----|------|
| **现象** | 模型与 Depth/Video 教师加载成功，FSDP2 wrap 完成；`next(data_iterator)` 时 rank1 worker0 在 `transform.py:101` `value - low` 抛 `RuntimeError: stoi`。伴随 `unwind.cpp: Unsupported unwinding pattern` |
| **根因** | torch **2.10** 在 **父进程已初始化 CUDA 后再 fork DataLoader worker** 时，`Tensor - numpy.ndarray` 的 numpy interop 会走坏掉的 C++ 转换（`stoi` 是内部把某种字符串当整数解析失败）。官方环境是 torch 2.8，这条路径正常。stats 本身是 float64 的 q01/q99，主进程里同样的减法可跑通 |
| **Fix** | 改 `Normalizer._get_stat`：若 `value` 是 Tensor，先 `torch.as_tensor(np.asarray(stat), dtype=..., device=...)` 再做算术。启动脚本加 `TORCH_DISABLE_ADDR2LINE=1`，避免异常符号化把进程挂死 |

**改动文件**: `lingbotvla/data/vla_data/transform.py`（兼容 torch 2.10，2.8 行为不变）。

---

### Error-I: `InductorError: Failed to find C compiler`（flex attention）

| 项 | 内容 |
|----|------|
| **现象** | Error-H 修好后 8 卡都打出 `actions's shape: torch.Size([8, 50, 55])`。随后在 DINO-Video 教师 `flex_attention_block_causal` → `torch.compile(flex_attention)` 失败 |
| **根因** | 镜像里 **没有 gcc**。`train.use_compile=false` 只关训练图 compile；视频教师仍 `torch.compile(flex_attention)`，Inductor 要 C 编译器。主 VLA 的 `flex_cached` 同样会撞 |
| **Fix** | `sudo apt-get install -y gcc g++`；启动脚本设 `CC=/usr/bin/gcc` `CXX=/usr/bin/g++`，`TORCHINDUCTOR_CACHE_DIR=/tmp/Ckp/.../inductor_cache`（编译产物不写 overlay） |

---

## 训练过程（启动后追加）

### Attempt 4 — 成功开训（2026-08-27 03:17 UTC）

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /tmp/SRC/lingbot-vla-v2
nohup bash /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/launch_train.sh \
  > /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log 2>&1 &
```

进程：`torchrun` **35392**。日志：`/tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log`（`train.sh` 同时 tee 仓库根 `log.txt`）。

| 项 | 实测 |
|----|------|
| `actions` | `[8, 50, 55]`，8 卡都打印 |
| `images` / `state` | `[8, 3, 256, 1536]` / `[8, 55]` |
| 显存 | 每卡约 **51315 MiB** / 81920 |
| GPU util | 开训后约 80–99% |
| step time | ~6.7 s/it，ETA ~19 h（10000 step） |

前几步（epoch 1，`lr=1e-4`）：step 5 Loss **0.3075** / VLA **0.2776**；step 10 **0.2701** / **0.2458**；step 14 **0.2265** / **0.2044**。Depth / FutureDepth / FutureVideo 同步下降。形态与 stack_bowls 开局同量级。

监控：

```bash
rg -N 'Step [0-9]+/' /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log | tail
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

DCP：`save_steps=5000` → `/tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/checkpoints/`。`enable_resume: true`。


