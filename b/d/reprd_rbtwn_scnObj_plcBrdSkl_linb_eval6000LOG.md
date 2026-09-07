# LingBot-VLA v2 RoboTwin scan_object & place_bread_skillet 闭环评估 — 执行日志

按 [reprd_rbtwn_scnObj_plcBrdSkl_linb_eval6000.md](reprd_rbtwn_scnObj_plcBrdSkl_linb_eval6000.md) 执行评估。

## 目标

| 项目 | 值 |
|------|-----|
| Checkpoint (scan_object) | GCS DCP → 本地 HF: `/home/luogang/CKPT/VLA/linbVLA2_scanobj/checkpoints/global_step_5016/hf_ckpt/` |
| Checkpoint (place_bread_skillet) | GCS DCP → 本地 HF: `/home/luogang/CKPT/VLA/linbVLA2_plcbrd/checkpoints/global_step_4864/hf_ckpt/` |
| 任务 | `scan_object` + `place_bread_skillet` |
| 配置 | `demo_clean` + `demo_randomized` (各 100 episodes) |
| 集成 | 双环境 WebSocket (RoboTwin + linbvla2) |

---

## 执行记录 (按时间顺序)

### [2026-09-01] Phase 0: 环境验证

| 检查项 | 状态 | 详情 |
|--------|------|------|
| linbvla2 conda env | OK | `/home/luogang/miniforge3/envs/linbvla2` |
| RoboTwin scan_object.py | OK | `/home/luogang/share/zwy/Projects/RoboTwin/envs/scan_object.py` |
| RoboTwin place_bread_skillet.py | OK | `/home/luogang/share/zwy/Projects/RoboTwin/envs/place_bread_skillet.py` |
| gcloud CLI | OK | Google Cloud SDK 582.0.0 |
| GPU | OK | 2× NVIDIA RTX PRO 6000 Blackwell (97887 MiB), 均空闲 |
| Qwen3-VL-4B-Instruct | 原不存在 → 下载 | 从 HuggingFace 下载到 `/home/luogang/CKPT/VLM/Qwen3-VL-4B-Instruct/` |
| norm_stats/robotwin.json | OK | 存在 |
| ws_client | OK | `policy/lingbotvla2_ws_client.py` 存在 |

### [2026-09-01] Phase 1: 权重准备

#### 1.1 GCS 下载

3 个并行下载任务:

| 任务 | GCS 路径 | 本地路径 | DCP shards | model_assets | 状态 |
|------|---------|---------|-----------|-------------|------|
| scan_object | `gs://physical-ai-data-eu/.../scan_object_20260827_233219/` | `/home/luogang/CKPT/VLA/linbVLA2_scanobj/` | 128 .distcp + .metadata | 10 files | OK |
| place_bread_skillet | `gs://physical-ai-data-eu/.../place_bread_skillet_20260828_104222/` | `/home/luogang/CKPT/VLA/linbVLA2_plcbrd/` | 128 .distcp + .metadata | 10 files | OK |
| Qwen3-VL-4B-Instruct | HuggingFace hub | `/home/luogang/CKPT/VLM/Qwen3-VL-4B-Instruct/` | N/A | 14 files | OK |

#### 1.2 lingbotvla_cli.yaml 创建

两个任务各创建了适配本机路径的 `lingbotvla_cli.yaml`:

- `/home/luogang/CKPT/VLA/linbVLA2_scanobj/lingbotvla_cli.yaml` — model_path 指向 scan_object HF ckpt
- `/home/luogang/CKPT/VLA/linbVLA2_plcbrd/lingbotvla_cli.yaml` — model_path 指向 place_bread_skillet HF ckpt

关键路径适配 (与 GCS 原始 YAML 的差异):
- `tokenizer_path`: `/tmp/itnvla15rbt20/var/...` → `/home/luogang/CKPT/VLM/Qwen3-VL-4B-Instruct`
- `model_path`: `/tmp/itnvla15rbt20/var/...` → 本地 HF ckpt 路径
- `norm_stats_file`: `assets/norm_stats/robotwin.json` → 绝对路径
- `output_dir`: 训练集群路径 → 本地 CKPT 路径

#### 1.3 评估配置文件

4 个 eval YAML 已创建于 `/home/luogang/share/zwy/Projects/RoboTwin/policy/`:

| 文件 | task_name | task_config | ckpt_setting |
|------|-----------|-------------|-------------|
| `eval_linbvla2_scanobj.yaml` | scan_object | demo_clean | linbvla2_scanobj_5016 |
| `eval_linbvla2_scanobj_rand.yaml` | scan_object | demo_randomized | linbvla2_scanobj_5016 |
| `eval_linbvla2_plcbrd.yaml` | place_bread_skillet | demo_clean | linbvla2_plcbrd_4864 |
| `eval_linbvla2_plcbrd_rand.yaml` | place_bread_skillet | demo_randomized | linbvla2_plcbrd_4864 |

#### 1.4 环境变量脚本

4 个 env 脚本已创建于 `b/d/`:
- `eval_server_scanobj_env.sh`, `eval_client_scanobj_env.sh`
- `eval_server_plcbrd_env.sh`, `eval_client_plcbrd_env.sh`

#### 1.5 DCP → HF 转换

##### Error #1: huggingface-hub 版本不兼容

```
ImportError: huggingface-hub>=0.34.0,<1.0 is required for a normal functioning of this module,
but found huggingface-hub==1.27.0.
```

**原因**: Qwen3-VL 下载时 `huggingface_hub.snapshot_download` 将 huggingface-hub 从 <1.0 升级到 1.27.0, 与 `transformers==4.57.3` (要求 `<1.0`) 不兼容.

**修复**: `pip install "huggingface-hub==0.36.2"` — 降级回兼容版本. Qwen3-VL 文件已下载完毕不受影响.

**分析**: 也考虑了升级 transformers 的方案, 但:
- 4.57.x 全系列 (4.57.3→4.57.6) 都要求 `huggingface-hub<1.0`
- 升级到 5.x 是大版本跳跃, API 变化可能影响模型加载
- 项目 requirements.txt 锁定 `transformers==4.57.3`

降级 huggingface-hub 是最安全的方案, 恢复下载前的环境状态.

##### scan_object DCP→HF 转换

```
[1/3] Loading DCP checkpoint ...
  Loaded 1708 tensors
[2/3] Saving HF checkpoint (safetensors, float32) ...
  Saved to /home/luogang/CKPT/VLA/linbVLA2_scanobj/checkpoints/global_step_5016/hf_ckpt
[3/3] Copying model_assets ...
  10 files copied (tokenizer.json, config.json, preprocessor_config.json, etc.)
Done! 6 safetensors shards, 1708 tensors, ~24.3 GB total
```

##### place_bread_skillet DCP→HF 转换

```
[1/3] Loading DCP checkpoint ...
  Loaded 1708 tensors
[2/3] Saving HF checkpoint (safetensors, float32) ...
  Saved to /home/luogang/CKPT/VLA/linbVLA2_plcbrd/checkpoints/global_step_4864/hf_ckpt
[3/3] Copying model_assets ...
  10 files copied
Done! 6 safetensors shards, 1708 tensors, ~24.3 GB total
```

两个 checkpoint 转换验证:

| 检查项 | scan_object | place_bread_skillet |
|--------|------------|-------------------|
| safetensors shards | 6 | 6 |
| Total tensors | 1708 | 1708 |
| config.json | OK (4781 bytes) | OK (4789 bytes) |
| tokenizer.json | OK (11422654 bytes) | OK (11422654 bytes) |
| model.safetensors.index.json | OK (207389 bytes) | OK (207389 bytes) |

### [2026-09-01] Phase 2: Run 1 — scan_object / demo_clean

#### 2.1 推理服务启动

```bash
# PID 22985, GPU 0, port 8006
CUDA_VISIBLE_DEVICES=0 nohup python -m deploy.lingbot_vla_v2_policy \
    --model_path /home/luogang/CKPT/VLA/linbVLA2_scanobj/checkpoints/global_step_5016/hf_ckpt \
    --use_length 20 --chunk_ret false --use_bf16 true --use_compile false --port 8006
```

服务端日志确认:
- `apply Qwen3-VL Lingbot patch` — 模型 patch 成功
- `Action Expert V2 init 36 Layers, hidden=768, q_heads=32, kv_heads=8` — MoE action expert 初始化
- `Using Eager Attn` — 使用 eager attention (非 flash)
- safetensors 6/6 加载完成
- GPU 0 占用 ~19.4 GB (bfloat16 模型)
- Port 8006 监听就绪

> **注意**: Python stdout 缓冲导致 "Model initialized ..." 消息未及时刷新到日志文件, 但通过 `ss -tlnp | grep 8006` 确认服务已就绪.

#### 2.2 Error #2: ffmpeg 未安装

首次启动评估客户端时报错:
```
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```

**原因**: `eval_policy.py` 使用 `subprocess.Popen` 调用 ffmpeg 录制评估视频 (`eval_video_log: true`), 但 RoboTwin conda 环境中未安装 ffmpeg 二进制.

**修复**: `conda install -y -n RoboTwin ffmpeg -c conda-forge` → ffmpeg 7.1.1 安装成功.

#### 2.3 Error #3: 工作目录错误

首次启动评估客户端时报错:
```
ModuleNotFoundError: No module named 'generate_episode_instructions'
```

**原因**: `eval_policy.py` 顶部有 `from generate_episode_instructions import *`, 需要从 RoboTwin 的 `script/` 目录或其父目录运行. 直接用绝对路径指定脚本但未 cd 到正确目录.

**修复**: 在启动前 `cd /home/luogang/share/zwy/Projects/RoboTwin` 切换到 RoboTwin 根目录.

#### 2.4 首次启动 → 异常终止

```bash
# PID 56126
cd /home/luogang/share/zwy/Projects/RoboTwin && \
nohup python script/eval_policy.py \
    --config policy/eval_linbvla2_scanobj.yaml \
    --policy_ckpt_path /home/luogang/CKPT/VLA/linbVLA2_scanobj/checkpoints/global_step_5016/hf_ckpt \
    > /home/luogang/CKPT/VLA/linbVLA2_scanobj/eval_client_demo_clean.log 2>&1 &
```

评估运行了 15 个 episodes (0/15, 0.0%), 然后服务端 (PID 22985) 和评估客户端 (PID 56126) 同时异常终止:
- 服务端日志无错误信息, 最后一行是正常推理输出 (`sample_actions batch=1 cost 0.40s`)
- 客户端日志最后在 step 278/500 处中断
- `dmesg` 中无 OOM killer 记录
- 原因未确定 (可能是渲染/SAPIEN 相关的 GPU 资源冲突)

#### 2.5 Issue #4: Python stdout 缓冲

**现象**: `nohup` 重定向日志文件时, "Model initialized ..." 等 print 消息未及时出现.

**原因**: Python 在重定向到文件时默认使用 full buffering, 而非 line buffering.

**修复**: 后续启动加 `PYTHONUNBUFFERED=1` 环境变量.

### [2026-09-01] Phase 2 (重启): 双 GPU 并行评估 — demo_clean

#### 2.6 scan_object 服务重启

```bash
# PID 224993, GPU 0, port 8006
CUDA_VISIBLE_DEVICES=0 nohup python -m deploy.lingbot_vla_v2_policy \
    --model_path /home/luogang/CKPT/VLA/linbVLA2_scanobj/checkpoints/global_step_5016/hf_ckpt \
    --use_length 20 --chunk_ret false --use_bf16 true --use_compile false --port 8006
```

50 秒后 port 8006 就绪, GPU 0 占用 12823 MiB.

#### 2.7 place_bread_skillet 服务 (续)

```bash
# PID 222980, GPU 1, port 8007 — 在 scan_object crash 期间一直存活
```

#### 2.8 双任务并行评估启动

两个评估客户端同时启动, 均设置 `PYTHONUNBUFFERED=1`:

```bash
# scan_object eval — PID 225550, GPU 0, WS_PORT=8006
cd /home/luogang/share/zwy/Projects/RoboTwin && \
CUDA_VISIBLE_DEVICES=0 WS_PORT=8006 PYTHONUNBUFFERED=1 \
python script/eval_policy.py --config policy/eval_linbvla2_scanobj.yaml ...

# plcbrd eval — PID 225764, GPU 1, WS_PORT=8007
cd /home/luogang/share/zwy/Projects/RoboTwin && \
CUDA_VISIBLE_DEVICES=1 WS_PORT=8007 PYTHONUNBUFFERED=1 \
python script/eval_policy.py --config policy/eval_linbvla2_plcbrd.yaml ...
```

#### 2.9 进度快照

| 时间 | scan_object | plcbrd | 备注 |
|------|------------|--------|------|
| 09:49 | 0/3 (0.0%) | 3/5 (60.0%) | 初次检查, 双 GPU 均满载 |
| 09:59 | 2/10 (20.0%) | 4/12 (33.3%) | 稳定运行, ~1.5 min/episode |
| 10:10 | 4/17 (23.5%) | 8/22 (36.4%) | |
| 10:26 | 5/25 (20.0%) | 14/36 (38.9%) | |
| 10:41 | 6/35 (17.1%) | 19/48 (39.6%) | |
| 10:56 | 7/43 (16.3%) | 22/58 (37.9%) | |
| 11:11 | 7/49 (14.3%) | 27/70 (38.6%) | |
| 11:27 | 7/58 (12.1%) | 29/81 (35.8%) | |
| 11:37 | 7/63 (11.1%) | 30/88 (34.1%) | |
| 11:45 | 7/69 (10.1%) | 34/96 (35.4%) | |
| 11:50 | 9/73 (12.3%) | **34/100 (34.0%)** | plcbrd demo_clean 完成 |
| 12:06 | 11/84 (13.1%) | — | |
| 12:21 | 17/93 (18.3%) | — | |
| 12:30 | 17/99 (17.2%) | — | |
| 12:34 | **17/100 (17.0%)** | — | scanobj demo_clean 完成 |

#### 2.10 Run 1 (scan_object / demo_clean) 结果

```
结果文件: eval_result/scan_object/lingbotvla2_ws_client/demo_clean/linbvla2_scanobj_5016/2026-09-01 09:43:49/_result.txt
Instruction Type: unseen
成功率: 0.17 (17/100)
视频: 100 个 episode MP4 文件
```

#### 2.11 Run 2 (place_bread_skillet / demo_clean) 结果

```
结果文件: eval_result/place_bread_skillet/lingbotvla2_ws_client/demo_clean/linbvla2_plcbrd_4864/2026-09-01 09:43:51/_result.txt
Instruction Type: unseen
成功率: 0.34 (34/100)
视频: 100 个 episode MP4 文件
```

#### 2.12 scan_object 成功率低 (17%) 排查

对 scan_object 17% 的低成功率进行了系统排查:

| 检查项 | 状态 | 详情 |
|--------|------|------|
| Checkpoint | OK | step 5016 是 GCS 上 scan_object 的最新 checkpoint |
| 语言指令 | OK | eval_policy.py → `set_instruction()` → ws_client `getattr(TASK_ENV, "instruction")` 完整; 使用 `unseen` 模板 |
| norm_stats | OK | `robotwin.json`, 包含 arm.position + effector.position |
| robot_config | OK | `configs/robot_configs/` robotwin 配置 |
| 推理参数 | OK | use_length=20, bf16, chunk_ret=false |
| 错误/警告 | 无 | 无 NaN/Inf/norm 相关警告 |

**结论: 非配置错误, 是模型性能本身**

训练完成度确认 (来自 GCS 训练日志):

| 任务 | num_train_epochs | steps/epoch | 总步数 | 数据集样本数 | global_batch_size |
|------|-----------------|-------------|--------|-------------|-------------------|
| scan_object | 76 | 66 | 5,016 | 8,463 | 128 |
| place_bread_skillet | 76 | 64 | 4,864 | ~8,192 | 128 |

> **重要修正**: 训练已完整跑完 76 个 epoch, 并非"只跑了 50%". `max_steps=10000` 是配置上限, 但实际训练由 `num_train_epochs=76` 控制, 5016 步 = 76 × 66 步/epoch, 已全部完成. GCS 上有 4 个 checkpoint (每 ~19 epoch 存一次), step 5016 是最终 checkpoint.

原因分析:
1. **任务难度差异**: scan_object 是复杂双臂协调任务 (一只手拿扫描仪、另一只手拿物体、然后扫描), 比 place_bread_skillet (放面包到锅上) 复杂得多
2. **训练数据量有限**: 仅 ~8,400 个样本 (约 50 个 episode), 对于复杂双臂任务可能不够
3. **可能需要更多 epoch 或更大数据集**: 76 epoch 的训练对 scan_object 任务可能仍不充分

### [2026-09-01] Phase 3: 双 GPU 并行评估 — demo_randomized

plcbrd demo_clean 于 11:50 完成后, scanobj demo_clean 于 12:34 完成后, 各自立即启动 demo_randomized:

#### 3.1 Run 4 (plcbrd / demo_randomized) 启动

```bash
# PID 1330422, GPU 1, WS_PORT=8007 — 复用 plcbrd 推理服务 (PID 222980)
cd /home/luogang/share/zwy/Projects/RoboTwin && \
CUDA_VISIBLE_DEVICES=1 WS_PORT=8007 PYTHONUNBUFFERED=1 \
python script/eval_policy.py --config policy/eval_linbvla2_plcbrd_rand.yaml ...
```

启动时间: 11:50

#### 3.2 Run 3 (scanobj / demo_randomized) 启动

```bash
# PID 1693001, GPU 0, WS_PORT=8006 — 复用 scanobj 推理服务 (PID 224993)
cd /home/luogang/share/zwy/Projects/RoboTwin && \
CUDA_VISIBLE_DEVICES=0 WS_PORT=8006 PYTHONUNBUFFERED=1 \
python script/eval_policy.py --config policy/eval_linbvla2_scanobj_rand.yaml ...
```

启动时间: 12:34

#### 3.3 进度快照

| 时间 | scan_object rand | plcbrd rand | 备注 |
|------|-----------------|-------------|------|
| 12:06 | — | 2/11 (18.2%) | scanobj rand 未启动 |
| 12:42 | 0/4 (0.0%) | 6/35 (17.1%) | 双 randomized 并行 |
| 12:55 | 0/9 (0.0%) | 6/43 (14.0%) | |
| 12:57 | 0/10 (0.0%) | 6/45 (13.3%) | |
| 13:12 | 2/19 (10.5%) | 9/56 (16.1%) | |
| 13:27 | 2/28 (7.1%) | 9/65 (13.8%) | |
| 13:43 | 3/37 (8.1%) | 9/75 (12.0%) | |
| 13:58 | 3/46 (6.5%) | 11/85 (12.9%) | |
| 14:08 | 4/53 (7.5%) | 14/93 (15.1%) | |
| 14:16 | 5/58 (8.6%) | 15/98 (15.3%) | |
| 14:19 | 5/60 (8.3%) | **15/100 (15.0%)** | plcbrd rand 完成 |

#### 3.4 Run 4 (plcbrd / demo_randomized) 结果

```
结果文件: eval_result/place_bread_skillet/lingbotvla2_ws_client/demo_randomized/linbvla2_plcbrd_4864/2026-09-01 11:51:06/_result.txt
Instruction Type: unseen
成功率: 0.15 (15/100)
视频: 100 个 episode MP4 文件
```

plcbrd 推理服务 (PID 222980, GPU 1, port 8007) 保持运行, 待 scanobj rand 完成后统一清理.

#### 3.5 Run 3 (scanobj / demo_randomized) 进度

| 时间 | scanobj rand | 备注 |
|------|-------------|------|
| 14:35 | 6/69 (8.7%) | |
| 14:50 | 7/78 (9.0%) | |
| 15:05 | 8/88 (9.1%) | |
| 15:15 | 8/92 (8.7%) | |
| 15:23 | 9/97 (9.3%) | |
| 15:27 | **10/100 (10.0%)** | scanobj rand 完成 |

#### 3.6 Run 3 (scanobj / demo_randomized) 结果

```
结果文件: eval_result/scan_object/lingbotvla2_ws_client/demo_randomized/linbvla2_scanobj_5016/2026-09-01 12:34:39/_result.txt
Instruction Type: unseen
成功率: 0.10 (10/100)
视频: 100 个 episode MP4 文件
```

### [2026-09-01 15:28] Phase 4: 清理与总结

#### 4.1 推理服务停止

```bash
kill 224993  # scanobj server (GPU 0, port 8006)
kill 222980  # plcbrd server (GPU 1, port 8007)
```

两台推理服务已停止, GPU 内存全部释放 (0 MiB / 0 MiB).

---

## 最终结果汇总

| Run | 任务 | 配置 | 成功率 | 结果文件 |
|-----|------|------|--------|---------|
| 1 | scan_object | demo_clean | **17/100 = 17.0%** | `eval_result/scan_object/.../demo_clean/.../2026-09-01 09:43:49/_result.txt` |
| 2 | place_bread_skillet | demo_clean | **34/100 = 34.0%** | `eval_result/place_bread_skillet/.../demo_clean/.../2026-09-01 09:43:51/_result.txt` |
| 3 | scan_object | demo_randomized | **10/100 = 10.0%** | `eval_result/scan_object/.../demo_randomized/.../2026-09-01 12:34:39/_result.txt` |
| 4 | place_bread_skillet | demo_randomized | **15/100 = 15.0%** | `eval_result/place_bread_skillet/.../demo_randomized/.../2026-09-01 11:51:06/_result.txt` |

domain randomization 对两个任务均造成显著性能下降:
- scan_object: 17% → 10% (↓7pp, -41%)
- place_bread_skillet: 34% → 15% (↓19pp, -56%)

### 时间线总结

| 时间 | 事件 |
|------|------|
| ~07:00 | Phase 0-1: 环境验证, GCS 下载, DCP→HF 转换 |
| ~09:05 | Run 1 首次启动 (scan_object demo_clean), 遇 Error #2/#3 |
| ~09:30 | 服务端+客户端异常终止 (Issue #5), 原因不明 |
| 09:42 | scan_object 服务重启 (PID 224993) |
| 09:43 | 双 GPU 并行评估启动: Run 1 (GPU 0) + Run 2 (GPU 1) |
| 11:50 | Run 2 (plcbrd demo_clean) 完成 (34%), 立即启动 Run 4 (plcbrd rand) |
| 12:34 | Run 1 (scanobj demo_clean) 完成 (17%), 立即启动 Run 3 (scanobj rand) |
| 14:19 | Run 4 (plcbrd demo_randomized) 完成 (15%) |
| 15:27 | Run 3 (scanobj demo_randomized) 完成 (10%) |
| 15:28 | 推理服务全部停止, 评估结束 |

总耗时: ~8.5 小时 (含环境准备). 双 GPU 并行评估有效减少了约 40% 的纯评估时间.

### 遇到的问题汇总

| # | 类型 | 问题 | 修复 |
|---|------|------|------|
| 1 | Error | huggingface-hub 版本不兼容 (1.27.0 vs transformers<1.0) | `pip install "huggingface-hub==0.36.2"` |
| 2 | Error | ffmpeg 未安装 | `conda install -n RoboTwin ffmpeg -c conda-forge` |
| 3 | Error | 工作目录错误 (`generate_episode_instructions` 导入失败) | `cd /home/luogang/share/zwy/Projects/RoboTwin` |
| 4 | Issue | Python stdout 缓冲 (nohup 日志不刷新) | `PYTHONUNBUFFERED=1` |
| 5 | Issue | 服务端+客户端同时异常终止 (无错误日志, 无 OOM) | 重启服务 + 客户端 |

