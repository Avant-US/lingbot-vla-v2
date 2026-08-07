# LingBot-VLA v2 RoboTwin stack_bowls_three 闭环评估 — 执行日志

按 [reprd_rbtwn_stackb3_linb_eval6000.md](reprd_rbtwn_stackb3_linb_eval6000.md) 执行 RoboTwin 2.0 `stack_bowls_three` 任务闭环评估。

## 目标

| 项目 | 值 |
|------|-----|
| Checkpoint | `/home/luogang/CKPT/VLA/linbVLA2/rbtwn2/10000/hf_ckpt/` |
| 任务 | `stack_bowls_three` |
| 配置 | `demo_clean` |
| Episodes | 100 |
| 集成 | 双环境 WebSocket (RoboTwin + linbvla2) |

---

## 执行记录 (按时间顺序)

### [2026-08-06] Phase 0: 初始化日志 + 验证 RoboTwin 环境

**操作**: 创建本日志文件，开始 Phase 0 前置检查。

**RoboTwin 依赖验证**:
```
sapien: 3.0.0b1, mplib: OK, websockets: 15.0.1, msgpack: (1,1,2), numpy: 1.26.4
```

**Patch 应用**:
- SAPIEN encoding patch: **applied** (`urdf_loader.py` 加 `encoding="utf-8"`)
- mplib collide patch: **applied** (`planner.py` 移除 `or collide`)

**Headless 渲染测试**:
```bash
export PYOPENGL_PLATFORM=egl MESA_GL_VERSION_OVERRIDE=4.1 SAPIEN_DISABLE_VULKAN_VALIDATION=1
cd /home/luogang/share/zwy/Projects/RoboTwin && python script/test_render.py
# 输出: Render Well
```

---

### [2026-08-06] Phase 1: 搭建 linbvla2 推理环境

**原因**: 双环境 WebSocket 方案，推理服务需在独立 conda 环境运行，避免污染 RoboTwin 仿真环境。

**操作**:

```bash
conda create -n linbvla2 python=3.10 -y
conda activate linbvla2
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.3 einops safetensors sentencepiece websockets msgpack numpy==1.26.4
pip install -e /home/luogang/SRC/Robot/lingbot-vla-v2 --no-deps
pip install -e lingbotvla/models/vla/vision_models/lingbot-depth --no-deps
# 额外推理依赖 (逐步补齐, 见 Error 表)
pip install psutil ninja omegaconf qwen-vl-utils opencv-python-headless scipy torchdata datasets lerobot pydantic accelerate peft av blobfile
pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir  # 从源码编译
pip install --no-deps "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz"
```

**验证结果**:
```
torch: 2.8.0+cu128, CUDA: True
flash_attn: 2.8.3
transformers: 4.57.3
lingbotvla: OK
numpy: 1.26.4
```

---

### [2026-08-06] Phase 2: 下载 Qwen3-VL + 创建训练配置

**操作**:
```bash
python scripts/download_hf_model.py \
  --repo_id Qwen/Qwen3-VL-4B-Instruct \
  --local_dir /home/luogang/CKPT/VLM
```

**结果**: 下载至 `/home/luogang/CKPT/VLM/Qwen3-VL-4B-Instruct/` (14 files, ~25s)

**新建文件**: `/home/luogang/CKPT/VLA/linbVLA2/lingbotvla_cli.yaml`
- 基于 `configs/vla/robotwin/stack_bowls_three_ft.yaml`
- `joints` / `norm_type` 使用字符串格式 (见 Error #3)

**服务端单独加载验证**:
```
Model loaded successfully!
action_key: ['action']
Result keys: ['action']
  action: shape=(14,), dtype=float32
```

---

### [2026-08-06] Phase 3: 创建桥接文件与环境脚本

| 文件 | 操作 | 原因 |
|------|------|------|
| `/home/luogang/share/zwy/Projects/RoboTwin/policy/lingbotvla2_ws_client.py` | **新建** | WebSocket 客户端策略, RoboTwin eval_policy 动态加载 |
| `/home/luogang/share/zwy/Projects/RoboTwin/policy/eval_linbvla2_stackb3.yaml` | **新建** | 评估配置: demo_clean, seed=42, 100 ep |
| `b/d/eval_server_env.sh` | **新建** | linbvla2 推理服务端环境变量 |
| `b/d/eval_client_env.sh` | **新建** | RoboTwin 仿真评估端环境变量 |

---

### [2026-08-06] Phase 4: 冒烟测试 (seed 0)

**Terminal 1 — 推理服务**:
```bash
conda activate linbvla2
source b/d/eval_server_env.sh
cd /home/luogang/SRC/Robot/lingbot-vla-v2
CUDA_VISIBLE_DEVICES=0 python -m deploy.lingbot_vla_v2_policy \
  --model_path ${CKPT_PATH} --use_length 20 --chunk_ret false \
  --use_bf16 true --use_compile false --port 8006 \
  > /home/luogang/CKPT/VLA/linbVLA2/eval_server_stackb3.log 2>&1 &
# Server PID: 2232486
```

**Terminal 2 — 冒烟评估**:
```bash
conda activate RoboTwin
source b/d/eval_client_env.sh
cd /home/luogang/share/zwy/Projects/RoboTwin
python script/eval_policy.py \
  --config policy/eval_linbvla2_stackb3.yaml \
  --policy_ckpt_path /home/luogang/CKPT/VLA/linbVLA2/rbtwn2/10000/hf_ckpt \
  --overrides --seed 0 \
  > /home/luogang/CKPT/VLA/linbVLA2/eval_smoke_seed0.log 2>&1 &
```

**冒烟结果** (终止前):
- WebSocket 连接成功: `[lingbotvla2_ws] Connected and reset for robotwin`
- 前 3 episodes: **3/3 Success (100%)**
- 动作返回格式: `action` key, shape=(14,) — 无需分段重组

---

### [2026-08-06] Phase 5: 正式评估 (seed 42, 100 episodes)

**命令**:
```bash
conda activate RoboTwin
source b/d/eval_client_env.sh
cd /home/luogang/share/zwy/Projects/RoboTwin
nohup python script/eval_policy.py \
  --config policy/eval_linbvla2_stackb3.yaml \
  --policy_ckpt_path /home/luogang/CKPT/VLA/linbVLA2/rbtwn2/10000/hf_ckpt \
  > /home/luogang/CKPT/VLA/linbVLA2/eval_client_stackb3_demo_clean.log 2>&1 &
# Eval PID: 2248462
```

**评估协议**:
- task: `stack_bowls_three`, config: `demo_clean`, instruction: `unseen`
- seed=42 → 起始 seed 4300000
- 100 episodes, 步上限 1200
- use_length=20 (每 20 步重推理)

**耗时**: 约 2.4 小时 (05:12 ~ 07:38 UTC)

---

## 最终结果

| 项目 | 值 |
|------|-----|
| **成功率** | **78% (78/100)** |
| 成功 episodes | 78 |
| 失败 episodes | 22 |
| 结果文件 | `/home/luogang/share/zwy/Projects/RoboTwin/eval_result/stack_bowls_three/lingbotvla2_ws_client/demo_clean/linbvla2_rbtwn2_10000/2026-08-06 05:12:53/_result.txt` |
| 客户端日志 | `/home/luogang/CKPT/VLA/linbVLA2/eval_client_stackb3_demo_clean.log` |
| 服务端日志 | `/home/luogang/CKPT/VLA/linbVLA2/eval_server_stackb3.log` |
| 推理服务 PID | 2232486 (已停止) |
| 评估 PID | 2248462 (exit 0) |

**基线对比** (来自手册 §8.2):

| 模型 | demo_clean 成功率 |
|------|-------------------|
| GR00T (14d, 30k steps) | 55% |
| GR00T (14d, 4k steps, Qwen3.5 0.8B) | 62% |
| GR00T (14d, 5k steps, robotwin_train) | 57% |
| **LingBot-VLA v2 (10000 steps, 本评估)** | **78%** |

---

## Error 汇总表

| # | 阶段 | Error | 根因 | Fix | 验证 |
|---|------|-------|------|-----|------|
| 1 | 环境 | `ImportError: undefined symbol: _ZN3c104cuda9SetDeviceEa` (flash_attn) | 预编译 wheel 与 torch 2.8.0+cu128 ABI 不兼容 | 卸载后 `pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir` 从源码编译; 先装 psutil+ninja | flash_attn 2.8.3 import OK |
| 2 | 环境 | `ModuleNotFoundError: No module named 'psutil'` (flash-attn 编译) | flash-attn setup.py 依赖 psutil | `pip install psutil ninja` | 编译成功 |
| 3 | 加载 | `ValueError: malformed node or string: {'arm.position': 14}` | `lingbotvla_cli.yaml` 中 joints/norm_type 用 YAML dict 格式; deploy 直接 yaml.load 后 `ast.literal_eval()` 期望字符串; 训练时 parse_args 会 `str(item)` 转换 | 修改 yaml: `joints: ["{'arm.position': 14}", ...]` 和 norm_type 同理 | server.reset('robotwin') 成功 |
| 4 | 环境 | 连锁 `ModuleNotFoundError`: torchdata, datasets, lerobot, pydantic, accelerate | `pip install -e . --no-deps` 未装 transitive deps; deploy import 链经过 data/dataset.py | 逐步安装: torchdata, datasets, lerobot 0.4.2, pydantic, accelerate, omegaconf 等 | `from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server` OK |

---

## 关键路径速查

```
Checkpoint:   /home/luogang/CKPT/VLA/linbVLA2/rbtwn2/10000/hf_ckpt/
Train config: /home/luogang/CKPT/VLA/linbVLA2/lingbotvla_cli.yaml
Qwen3-VL:     /home/luogang/CKPT/VLM/Qwen3-VL-4B-Instruct/
Norm stats:   /home/luogang/SRC/Robot/lingbot-vla-v2/assets/norm_stats/robotwin.json
Policy:       /home/luogang/share/zwy/Projects/RoboTwin/policy/lingbotvla2_ws_client.py
Eval config:  /home/luogang/share/zwy/Projects/RoboTwin/policy/eval_linbvla2_stackb3.yaml
Server env:   b/d/eval_server_env.sh
Client env:   b/d/eval_client_env.sh
Results:      eval_result/stack_bowls_three/lingbotvla2_ws_client/demo_clean/linbvla2_rbtwn2_10000/
Log (本文件): b/d/reprd_rbtwn_stackb3_linb_eval6000LOG.md
```

---

## 文件增删改汇总

| 路径 | 操作 | 原因 |
|------|------|------|
| `b/d/reprd_rbtwn_stackb3_linb_eval6000LOG.md` | 新建 | 本执行日志 |
| `b/d/eval_server_env.sh` | 新建 | 推理服务端环境变量 |
| `b/d/eval_client_env.sh` | 新建 | 仿真评估端环境变量 |
| `/home/luogang/CKPT/VLA/linbVLA2/lingbotvla_cli.yaml` | 新建 | 模型加载必需训练配置 |
| `/home/luogang/share/zwy/Projects/RoboTwin/policy/lingbotvla2_ws_client.py` | 新建 | WebSocket 策略桥接 |
| `/home/luogang/share/zwy/Projects/RoboTwin/policy/eval_linbvla2_stackb3.yaml` | 新建 | 评估配置 |
| `/home/luogang/CKPT/VLM/Qwen3-VL-4B-Instruct/` | 下载 | Base VLM processor/tokenizer |
| conda env `linbvla2` | 新建 | 推理专用环境 |
| SAPIEN `urdf_loader.py` | patch | UTF-8 encoding |
| mplib `planner.py` | patch | 移除 collide check |

---

## 评估完成确认

- [x] 双环境 WebSocket 集成运行正常
- [x] 100 episodes demo_clean 评估完成
- [x] `_result.txt` 写入成功率 0.78
- [x] 推理服务已停止
- [x] 所有 error 已记录并 fix

---

### [2026-08-06] Phase 6: demo_randomized 评估 (100 episodes)

**原因**: 补测 RoboTwin 2.0 域随机化配置, 与 demo_clean 形成对比。

**新建文件**: `/home/luogang/share/zwy/Projects/RoboTwin/policy/eval_linbvla2_stackb3_rand.yaml`
```yaml
task_name: stack_bowls_three
task_config: demo_randomized
ckpt_setting: linbvla2_rbtwn2_10000
policy_name: lingbotvla2_ws_client
instruction_type: unseen
seed: 42
```

**Terminal 1 — 推理服务** (复用同一 checkpoint, 无需改模型):
```bash
conda activate linbvla2
source b/d/eval_server_env.sh
cd /home/luogang/SRC/Robot/lingbot-vla-v2
CUDA_VISIBLE_DEVICES=0 python -m deploy.lingbot_vla_v2_policy \
  --model_path ${CKPT_PATH} --use_length 20 --chunk_ret false \
  --use_bf16 true --use_compile false --port 8006 \
  > /home/luogang/CKPT/VLA/linbVLA2/eval_server_stackb3_rand.log 2>&1 &
```

**Terminal 2 — randomized 评估**:
```bash
conda activate RoboTwin
source b/d/eval_client_env.sh
cd /home/luogang/share/zwy/Projects/RoboTwin
nohup python script/eval_policy.py \
  --config policy/eval_linbvla2_stackb3_rand.yaml \
  --policy_ckpt_path /home/luogang/CKPT/VLA/linbVLA2/rbtwn2/10000/hf_ckpt \
  > /home/luogang/CKPT/VLA/linbVLA2/eval_client_stackb3_demo_randomized.log 2>&1 &
```

**耗时**: 约 3.75 小时 (13:55 ~ 17:40 UTC+8)

**Error**: 无新增 error, 直接复用 demo_clean 阶段的环境与桥接模块。

---

## demo_randomized 最终结果

| 项目 | 值 |
|------|-----|
| **成功率** | **20% (20/100)** |
| 成功 episodes | 20 |
| 失败 episodes | 80 |
| 结果文件 | `/home/luogang/share/zwy/Projects/RoboTwin/eval_result/stack_bowls_three/lingbotvla2_ws_client/demo_randomized/linbvla2_rbtwn2_10000/2026-08-06 13:55:43/_result.txt` |
| 客户端日志 | `/home/luogang/CKPT/VLA/linbVLA2/eval_client_stackb3_demo_randomized.log` |
| 服务端日志 | `/home/luogang/CKPT/VLA/linbVLA2/eval_server_stackb3_rand.log` |

---

## 两种配置对比 (LingBot-VLA v2, 10000 steps)

| 配置 | 成功率 | Episodes | 说明 |
|------|--------|----------|------|
| **demo_clean** | **78%** | 78/100 | 无域随机化, 固定背景/光照/桌高 |
| **demo_randomized** | **20%** | 20/100 | 随机背景/光照/桌高/杂物等 |

差距 58 个百分点, 符合域随机化显著增加难度的预期。模型在 clean 环境表现强, 但对 sim-to-real 风格的随机扰动泛化仍有较大提升空间。

---

## 文件增删改 (Phase 6 追加)

| 路径 | 操作 | 原因 |
|------|------|------|
| `/home/luogang/share/zwy/Projects/RoboTwin/policy/eval_linbvla2_stackb3_rand.yaml` | **新建** | demo_randomized 评估配置 |

---

## 评估完成确认 (更新)

- [x] 100 episodes demo_clean 评估完成 (78%)
- [x] 100 episodes demo_randomized 评估完成 (20%)
- [x] 推理服务已停止

