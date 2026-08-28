# RoboTwin 三任务 SFT 执行日志（2026-08-27）

对应设计文档：[`run_ech_rbt_p012.md`](run_ech_rbt_p012.md)。  
编排器：[`b/s/rbt/run_each_robotwin.py`](../../s/rbt/run_each_robotwin.py)。  
任务顺序：`scan_object` → `place_bread_skillet` → `pick_dual_bottles`。

按发生顺序记录：GPU 巡检、命令与理由、error / 根因 / fix、改过的文件、关键路径。

---

## 目标（本轮）

| 项 | 值 |
|----|-----|
| 数据根 | `/home/a26113/Dta/RoboTwin-Clean/`（`~/Dta/RoboTwin-Clean`） |
| venv | `/tmp/itnvla15rbt20` |
| `HF_HOME` | `/tmp/itnvla15rbt20/var/hf_home/` |
| Checkpoint 根 | `/home/a26113/Ckp/lbRbt/` |
| 全局 batch | 128（micro=8 × 8 GPU × acc=2） |
| epoch | 76（按帧数换算 `max_steps`） |
| 启动条件 | **8 张 GPU 均空闲**后再开训；此前每 15 分钟巡检一次 |

---

## [2026-08-27 22:14 +08 / 14:14 UTC] GPU 巡检 #1（立即执行）

**操作**:

```bash
date
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv
ps -eo pid,etime,cmd | awk '/train_lingbotvla|torchrun/'
```

**理由**: 用户要求 8 卡都空闲才开始三任务 SFT；先确认是否可立刻开训。

**结果**: **不可开训**。8×A800 全部被占用。

| GPU | 显存 used / total | GPU util | mem util |
|-----|-------------------|----------|----------|
| 0 | 52917 / 81920 MiB | 100% | 71% |
| 1 | 51797 / 81920 MiB | 100% | 72% |
| 2 | 51593 / 81920 MiB | 100% | 70% |
| 3 | 51805 / 81920 MiB | 100% | 72% |
| 4 | 51807 / 81920 MiB | 100% | 68% |
| 5 | 51737 / 81920 MiB | 100% | 74% |
| 6 | 51797 / 81920 MiB | 100% | 69% |
| 7 | 51639 / 81920 MiB | 100% | 71% |

占用进程：hanging_mug 微调（`train.sh` pid **35386**，`torchrun` pid **35392**，已跑约 **10h57m**）。

```
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/robotwin/hanging_mug_ft.yaml
  --data.norm_stats_file assets/norm_stats/robotwin.json --train.use_compile false
```

日志 `/tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log`：约 **Step 5474/131, Epoch 42, Loss 0.0405**，目标 `max_steps=10000`。按当时速率粗估还需约 **9 小时**。

**判定空闲的标准**（后续巡检沿用）：8 卡 `memory.used ≤ 200 MiB` 且 `utilization.gpu = 0`，且无 `train_lingbotvla.py` / 本仓库 `torchrun`。

**决策**: 不杀 hanging_mug；每 15 分钟再查一次，直到 8 卡空闲后再启动三任务流水线。

空闲等待循环：shell pid **106756**，间隔 **900s**，sentinel `AGENT_LOOP_TICK_gpu_idle`。第一次巡检已完成；下一拍约 15 分钟后。

---

## [2026-08-27 22:30 +08 / 14:30 UTC] GPU 巡检 #2（15min tick）

**操作**:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla|torchrun"
tail -n 5 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；按既定标准判断 8 卡是否空闲，决定是否启动三任务 SFT。

**结果**: **仍不可开训**。8 卡均被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52797 / 81920 MiB | 73% |
| 1 | 51769 / 81920 MiB | 41% |
| 2 | 51745 / 81920 MiB | 100% |
| 3 | 51787 / 81920 MiB | 99% |
| 4 | 51637 / 81920 MiB | 89% |
| 5 | 51685 / 81920 MiB | 84% |
| 6 | 51451 / 81920 MiB | 89% |
| 7 | 51669 / 81920 MiB | 98% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，8 个 `train_lingbotvla.py` worker（35466–35473）及 DataLoader worker。

日志最新：约 **Step 5609/10000，Epoch 43，Loss 0.0244**，StepTime ~7.0s，ETA ~8h30m。相对巡检 #1 推进约 **135 step / 15 min**（约 9 step/min），与 ~7s/step 一致。

**决策**: 继续等待；不杀 hanging_mug；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-27 22:45 +08 / 14:45 UTC] GPU 巡检 #3（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52797 / 81920 MiB | 82% |
| 1 | 51769 / 81920 MiB | 54% |
| 2 | 51745 / 81920 MiB | 95% |
| 3 | 51787 / 81920 MiB | 82% |
| 4 | 51637 / 81920 MiB | 81% |
| 5 | 51685 / 81920 MiB | 77% |
| 6 | 51451 / 81920 MiB | 83% |
| 7 | 51669 / 81920 MiB | 95% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 5735/10000，Epoch 44，Loss 0.0261**，StepTime ~7.2s，ETA ~8h30m。相对巡检 #2 推进约 **126 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-27 23:00 +08 / 15:00 UTC] GPU 巡检 #4（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 3 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52797 / 81920 MiB | 70% |
| 1 | 51769 / 81920 MiB | 71% |
| 2 | 51745 / 81920 MiB | 77% |
| 3 | 51787 / 81920 MiB | 77% |
| 4 | 51637 / 81920 MiB | 78% |
| 5 | 51685 / 81920 MiB | 59% |
| 6 | 51451 / 81920 MiB | 85% |
| 7 | 51669 / 81920 MiB | 71% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 5858/10000，Epoch 45，Loss 0.0154**，StepTime ~7.0s，ETA ~8h04m。相对巡检 #3 推进约 **123 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-27 23:15 +08 / 15:15 UTC] GPU 巡检 #5（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52797 / 81920 MiB | 100% |
| 1 | 51769 / 81920 MiB | 100% |
| 2 | 51745 / 81920 MiB | 100% |
| 3 | 51787 / 81920 MiB | 100% |
| 4 | 51637 / 81920 MiB | 100% |
| 5 | 51685 / 81920 MiB | 86% |
| 6 | 51451 / 81920 MiB | 100% |
| 7 | 51669 / 81920 MiB | 100% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 5983/10000，Epoch 46，Loss 0.0123**，StepTime ~6.8s，ETA ~7h59m。相对巡检 #4 推进约 **125 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-27 23:30 +08 / 15:30 UTC] GPU 巡检 #6（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52767 / 81920 MiB | 73% |
| 1 | 51617 / 81920 MiB | 88% |
| 2 | 51685 / 81920 MiB | 91% |
| 3 | 51751 / 81920 MiB | 94% |
| 4 | 51851 / 81920 MiB | 91% |
| 5 | 51677 / 81920 MiB | 91% |
| 6 | 51677 / 81920 MiB | 82% |
| 7 | 51587 / 81920 MiB | 82% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 6108/10000，Epoch 47，Loss 0.0163**，StepTime ~7.1s，ETA ~7h40m。相对巡检 #5 推进约 **125 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-27 23:45 +08 / 15:45 UTC] GPU 巡检 #7（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52767 / 81920 MiB | 47% |
| 1 | 51617 / 81920 MiB | 73% |
| 2 | 51685 / 81920 MiB | 80% |
| 3 | 51751 / 81920 MiB | 95% |
| 4 | 51851 / 81920 MiB | 87% |
| 5 | 51677 / 81920 MiB | 65% |
| 6 | 51677 / 81920 MiB | 95% |
| 7 | 51587 / 81920 MiB | 81% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 6234/10000，Epoch 48，Loss 0.0223**，StepTime ~7.1s，ETA ~7h24m。相对巡检 #6 推进约 **126 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 00:00 +08 / 16:00 UTC] GPU 巡检 #8（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52767 / 81920 MiB | 83% |
| 1 | 51617 / 81920 MiB | 48% |
| 2 | 51685 / 81920 MiB | 79% |
| 3 | 51751 / 81920 MiB | 79% |
| 4 | 51851 / 81920 MiB | 71% |
| 5 | 51677 / 81920 MiB | 69% |
| 6 | 51677 / 81920 MiB | 70% |
| 7 | 51587 / 81920 MiB | 79% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 6359/10000，Epoch 49，Loss 0.0182**，StepTime ~7.0s，ETA ~7h08m。相对巡检 #7 推进约 **125 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 00:15 +08 / 16:15 UTC] GPU 巡检 #9（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52767 / 81920 MiB | 79% |
| 1 | 51617 / 81920 MiB | 69% |
| 2 | 51685 / 81920 MiB | 80% |
| 3 | 51751 / 81920 MiB | 90% |
| 4 | 51851 / 81920 MiB | 90% |
| 5 | 51677 / 81920 MiB | 78% |
| 6 | 51677 / 81920 MiB | 79% |
| 7 | 51587 / 81920 MiB | 85% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 6484/10000，Epoch 50，Loss 0.0121**，StepTime ~6.9s，ETA ~6h56m。相对巡检 #8 推进约 **125 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 00:30 +08 / 16:30 UTC] GPU 巡检 #10（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52935 / 81920 MiB | 72% |
| 1 | 51607 / 81920 MiB | 74% |
| 2 | 51685 / 81920 MiB | 72% |
| 3 | 51661 / 81920 MiB | 75% |
| 4 | 51801 / 81920 MiB | 77% |
| 5 | 51639 / 81920 MiB | 71% |
| 6 | 51639 / 81920 MiB | 78% |
| 7 | 51685 / 81920 MiB | 85% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 6607/10000，Epoch 51，Loss 0.0136**，StepTime ~7.3s，ETA ~6h51m。相对巡检 #9 推进约 **123 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 00:45 +08 / 16:45 UTC] GPU 巡检 #11（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52935 / 81920 MiB | 70% |
| 1 | 51607 / 81920 MiB | 77% |
| 2 | 51685 / 81920 MiB | 79% |
| 3 | 51661 / 81920 MiB | 76% |
| 4 | 51801 / 81920 MiB | 72% |
| 5 | 51639 / 81920 MiB | 79% |
| 6 | 51639 / 81920 MiB | 77% |
| 7 | 51685 / 81920 MiB | 80% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 6730/10000，Epoch 52，Loss 0.0143**，StepTime ~7.2s，ETA ~6h33m。相对巡检 #10 推进约 **123 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 01:00 +08 / 17:00 UTC] GPU 巡检 #12（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52935 / 81920 MiB | 78% |
| 1 | 51607 / 81920 MiB | 78% |
| 2 | 51685 / 81920 MiB | 74% |
| 3 | 51661 / 81920 MiB | 74% |
| 4 | 51801 / 81920 MiB | 80% |
| 5 | 51639 / 81920 MiB | 76% |
| 6 | 51639 / 81920 MiB | 78% |
| 7 | 51685 / 81920 MiB | 74% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 6852/10000，Epoch 53，Loss 0.0215**，StepTime ~7.2s，ETA ~6h19m。相对巡检 #11 推进约 **122 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 01:15 +08 / 17:15 UTC] GPU 巡检 #13（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52935 / 81920 MiB | 71% |
| 1 | 51607 / 81920 MiB | 63% |
| 2 | 51685 / 81920 MiB | 80% |
| 3 | 51661 / 81920 MiB | 64% |
| 4 | 51801 / 81920 MiB | 82% |
| 5 | 51639 / 81920 MiB | 59% |
| 6 | 51639 / 81920 MiB | 82% |
| 7 | 51685 / 81920 MiB | 82% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 6975/10000，Epoch 54，Loss 0.0131**，StepTime ~7.3s，ETA ~6h06m。相对巡检 #12 推进约 **123 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 01:30 +08 / 17:30 UTC] GPU 巡检 #14（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52797 / 81920 MiB | 82% |
| 1 | 51851 / 81920 MiB | 86% |
| 2 | 51765 / 81920 MiB | 79% |
| 3 | 51651 / 81920 MiB | 79% |
| 4 | 51795 / 81920 MiB | 70% |
| 5 | 51595 / 81920 MiB | 38% |
| 6 | 51595 / 81920 MiB | 75% |
| 7 | 51685 / 81920 MiB | 70% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 7097/10000，Epoch 55，Loss 0.0128**，StepTime ~7.3s，ETA ~5h52m。相对巡检 #13 推进约 **122 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 01:45 +08 / 17:45 UTC] GPU 巡检 #15（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52797 / 81920 MiB | 69% |
| 1 | 51851 / 81920 MiB | 82% |
| 2 | 51765 / 81920 MiB | 66% |
| 3 | 51651 / 81920 MiB | 82% |
| 4 | 51795 / 81920 MiB | 83% |
| 5 | 51595 / 81920 MiB | 80% |
| 6 | 51595 / 81920 MiB | 69% |
| 7 | 51685 / 81920 MiB | 78% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 7219/10000，Epoch 56，Loss 0.0129**，StepTime ~7.4s，ETA ~5h42m。相对巡检 #14 推进约 **122 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 02:00 +08 / 18:00 UTC] GPU 巡检 #16（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52797 / 81920 MiB | 100% |
| 1 | 51851 / 81920 MiB | 100% |
| 2 | 51765 / 81920 MiB | 100% |
| 3 | 51651 / 81920 MiB | 100% |
| 4 | 51795 / 81920 MiB | 100% |
| 5 | 51595 / 81920 MiB | 100% |
| 6 | 51595 / 81920 MiB | 100% |
| 7 | 51685 / 81920 MiB | 100% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 7340/10000，Epoch 57，Loss 0.0115**，StepTime ~7.2s，ETA ~6h19m（进度条瞬时偏高）。相对巡检 #15 推进约 **121 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 02:15 +08 / 18:16 UTC] GPU 巡检 #17（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡显存仍被 hanging_mug 占用。本拍瞬时 `utilization.gpu=0`，但每卡仍约 51–53 GB 显存，且 `train_lingbotvla` 进程仍在；日志刚打出 epoch 57 结束的 VRAM dump，属于 epoch 间隙而非空闲。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52797 / 81920 MiB | 0% |
| 1 | 51851 / 81920 MiB | 0% |
| 2 | 51765 / 81920 MiB | 0% |
| 3 | 51651 / 81920 MiB | 0% |
| 4 | 51795 / 81920 MiB | 0% |
| 5 | 51595 / 81920 MiB | 0% |
| 6 | 51595 / 81920 MiB | 0% |
| 7 | 51685 / 81920 MiB | 0% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 7467/10000，Epoch 57，Loss 0.0121**，随后 `VRAM usage after epoch 57: cur 8.47GB, max 35.20GB`。相对巡检 #16 推进约 **127 step / 15 min**。

**决策**: 继续等待（空闲标准：显存 ≤200 MiB **且** util=0 **且** 无训练进程）。不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 02:30 +08 / 18:31 UTC] GPU 巡检 #18（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用（训练已从 epoch 间隙恢复）。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52895 / 81920 MiB | 82% |
| 1 | 51801 / 81920 MiB | 100% |
| 2 | 51607 / 81920 MiB | 100% |
| 3 | 51787 / 81920 MiB | 100% |
| 4 | 51863 / 81920 MiB | 100% |
| 5 | 51783 / 81920 MiB | 87% |
| 6 | 51685 / 81920 MiB | 100% |
| 7 | 51685 / 81920 MiB | 89% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 7588/10000，Epoch 58，Loss 0.0139**，StepTime ~7.2s，ETA ~4h48m。相对巡检 #17 推进约 **121 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 02:45 +08 / 18:45 UTC] GPU 巡检 #19（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52895 / 81920 MiB | 100% |
| 1 | 51801 / 81920 MiB | 100% |
| 2 | 51607 / 81920 MiB | 100% |
| 3 | 51787 / 81920 MiB | 100% |
| 4 | 51863 / 81920 MiB | 100% |
| 5 | 51783 / 81920 MiB | 100% |
| 6 | 51685 / 81920 MiB | 100% |
| 7 | 51685 / 81920 MiB | 100% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 7708/10000，Epoch 59，Loss 0.0123**，StepTime ~7.1s，ETA ~4h33m。相对巡检 #18 推进约 **120 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 03:00 +08 / 19:00 UTC] GPU 巡检 #20（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52895 / 81920 MiB | 57% |
| 1 | 51801 / 81920 MiB | 73% |
| 2 | 51607 / 81920 MiB | 57% |
| 3 | 51787 / 81920 MiB | 83% |
| 4 | 51863 / 81920 MiB | 71% |
| 5 | 51783 / 81920 MiB | 46% |
| 6 | 51685 / 81920 MiB | 77% |
| 7 | 51685 / 81920 MiB | 90% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 7830/10000，Epoch 60，Loss 0.0174**，StepTime ~6.9s，ETA ~4h15m。相对巡检 #19 推进约 **122 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 03:15 +08 / 19:15 UTC] GPU 巡检 #21（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52895 / 81920 MiB | 88% |
| 1 | 51801 / 81920 MiB | 94% |
| 2 | 51607 / 81920 MiB | 96% |
| 3 | 51787 / 81920 MiB | 89% |
| 4 | 51863 / 81920 MiB | 97% |
| 5 | 51783 / 81920 MiB | 96% |
| 6 | 51685 / 81920 MiB | 93% |
| 7 | 51685 / 81920 MiB | 90% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 7953/10000，Epoch 61，Loss 0.0120**，StepTime ~7.4s，ETA ~4h09m。相对巡检 #20 推进约 **123 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 03:30 +08 / 19:30 UTC] GPU 巡检 #22（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52935 / 81920 MiB | 100% |
| 1 | 51617 / 81920 MiB | 100% |
| 2 | 51685 / 81920 MiB | 100% |
| 3 | 51751 / 81920 MiB | 100% |
| 4 | 51779 / 81920 MiB | 100% |
| 5 | 51677 / 81920 MiB | 100% |
| 6 | 51783 / 81920 MiB | 100% |
| 7 | 51685 / 81920 MiB | 100% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 8074/10000，Epoch 62，Loss 0.0138**，StepTime ~7.2s，ETA ~3h53m。相对巡检 #21 推进约 **121 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 03:45 +08 / 19:45 UTC] GPU 巡检 #23（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52935 / 81920 MiB | 53% |
| 1 | 51617 / 81920 MiB | 56% |
| 2 | 51685 / 81920 MiB | 81% |
| 3 | 51751 / 81920 MiB | 68% |
| 4 | 51779 / 81920 MiB | 78% |
| 5 | 51677 / 81920 MiB | 87% |
| 6 | 51783 / 81920 MiB | 81% |
| 7 | 51685 / 81920 MiB | 57% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 8197/10000，Epoch 63，Loss 0.0116**，StepTime ~7.2s，ETA ~3h37m。相对巡检 #22 推进约 **123 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 04:00 +08 / 20:00 UTC] GPU 巡检 #24（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52935 / 81920 MiB | 71% |
| 1 | 51617 / 81920 MiB | 78% |
| 2 | 51685 / 81920 MiB | 77% |
| 3 | 51751 / 81920 MiB | 78% |
| 4 | 51779 / 81920 MiB | 77% |
| 5 | 51677 / 81920 MiB | 88% |
| 6 | 51783 / 81920 MiB | 77% |
| 7 | 51685 / 81920 MiB | 88% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 8318/10000，Epoch 64，Loss 0.0117**，StepTime ~7.2s，ETA ~3h23m。相对巡检 #23 推进约 **121 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 04:15 +08 / 20:15 UTC] GPU 巡检 #25（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52935 / 81920 MiB | 74% |
| 1 | 51617 / 81920 MiB | 83% |
| 2 | 51685 / 81920 MiB | 71% |
| 3 | 51751 / 81920 MiB | 89% |
| 4 | 51779 / 81920 MiB | 85% |
| 5 | 51677 / 81920 MiB | 88% |
| 6 | 51783 / 81920 MiB | 76% |
| 7 | 51685 / 81920 MiB | 75% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 8441/10000，Epoch 65，Loss 0.0110**，StepTime ~7.2s，ETA ~3h09m。相对巡检 #24 推进约 **123 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 04:30 +08 / 20:30 UTC] GPU 巡检 #26（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 53003 / 81920 MiB | 35% |
| 1 | 51837 / 81920 MiB | 47% |
| 2 | 51745 / 81920 MiB | 29% |
| 3 | 51791 / 81920 MiB | 60% |
| 4 | 51651 / 81920 MiB | 72% |
| 5 | 51639 / 81920 MiB | 35% |
| 6 | 51637 / 81920 MiB | 61% |
| 7 | 51685 / 81920 MiB | 84% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 8565/10000，Epoch 66，Loss 0.0143**，StepTime ~7.0s，ETA ~2h48m。相对巡检 #25 推进约 **124 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 04:45 +08 / 20:45 UTC] GPU 巡检 #27（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 53003 / 81920 MiB | 40% |
| 1 | 51837 / 81920 MiB | 50% |
| 2 | 51745 / 81920 MiB | 40% |
| 3 | 51791 / 81920 MiB | 79% |
| 4 | 51651 / 81920 MiB | 82% |
| 5 | 51639 / 81920 MiB | 78% |
| 6 | 51637 / 81920 MiB | 81% |
| 7 | 51685 / 81920 MiB | 66% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 8691/10000，Epoch 67，Loss 0.0111**，StepTime ~7.0s，ETA ~2h32m。相对巡检 #26 推进约 **126 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 05:00 +08 / 21:00 UTC] GPU 巡检 #28（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 53003 / 81920 MiB | 74% |
| 1 | 51837 / 81920 MiB | 66% |
| 2 | 51745 / 81920 MiB | 54% |
| 3 | 51791 / 81920 MiB | 78% |
| 4 | 51651 / 81920 MiB | 78% |
| 5 | 51639 / 81920 MiB | 63% |
| 6 | 51637 / 81920 MiB | 83% |
| 7 | 51685 / 81920 MiB | 79% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 8816/10000，Epoch 68，Loss 0.0112**，StepTime ~7.1s，ETA ~2h20m。相对巡检 #27 推进约 **125 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 05:15 +08 / 21:15 UTC] GPU 巡检 #29（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 53003 / 81920 MiB | 67% |
| 1 | 51837 / 81920 MiB | 78% |
| 2 | 51745 / 81920 MiB | 55% |
| 3 | 51791 / 81920 MiB | 91% |
| 4 | 51651 / 81920 MiB | 58% |
| 5 | 51639 / 81920 MiB | 61% |
| 6 | 51637 / 81920 MiB | 56% |
| 7 | 51685 / 81920 MiB | 90% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 8942/10000，Epoch 69，Loss 0.0112**，StepTime ~7.1s，ETA ~2h05m。相对巡检 #28 推进约 **126 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 05:30 +08 / 21:30 UTC] GPU 巡检 #30（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52789 / 81920 MiB | 77% |
| 1 | 51669 / 81920 MiB | 78% |
| 2 | 51685 / 81920 MiB | 84% |
| 3 | 51781 / 81920 MiB | 91% |
| 4 | 51625 / 81920 MiB | 91% |
| 5 | 51595 / 81920 MiB | 76% |
| 6 | 51685 / 81920 MiB | 80% |
| 7 | 51685 / 81920 MiB | 79% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 9067/10000，Epoch 70，Loss 0.0103**，StepTime ~7.0s，ETA ~1h49m。相对巡检 #29 推进约 **125 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 05:45 +08 / 21:45 UTC] GPU 巡检 #31（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52789 / 81920 MiB | 100% |
| 1 | 51669 / 81920 MiB | 100% |
| 2 | 51685 / 81920 MiB | 100% |
| 3 | 51781 / 81920 MiB | 100% |
| 4 | 51625 / 81920 MiB | 100% |
| 5 | 51595 / 81920 MiB | 100% |
| 6 | 51685 / 81920 MiB | 100% |
| 7 | 51685 / 81920 MiB | 100% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 9192/10000，Epoch 71，Loss 0.0142**，StepTime ~7.0s，ETA ~1h34m。相对巡检 #30 推进约 **125 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 06:00 +08 / 22:00 UTC] GPU 巡检 #32（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡显存仍被 hanging_mug 占用。GPU0 瞬时 util=0，其余 7 卡 100%，显存均约 51–53 GB，训练进程仍在。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52789 / 81920 MiB | 0% |
| 1 | 51669 / 81920 MiB | 100% |
| 2 | 51685 / 81920 MiB | 100% |
| 3 | 51781 / 81920 MiB | 100% |
| 4 | 51625 / 81920 MiB | 100% |
| 5 | 51595 / 81920 MiB | 100% |
| 6 | 51685 / 81920 MiB | 100% |
| 7 | 51685 / 81920 MiB | 100% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 9318/10000，Epoch 72，Loss 0.0104**，StepTime ~7.1s，ETA ~1h21m。相对巡检 #31 推进约 **126 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 06:15 +08 / 22:15 UTC] GPU 巡检 #33（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52789 / 81920 MiB | 52% |
| 1 | 51669 / 81920 MiB | 55% |
| 2 | 51685 / 81920 MiB | 36% |
| 3 | 51781 / 81920 MiB | 60% |
| 4 | 51625 / 81920 MiB | 77% |
| 5 | 51595 / 81920 MiB | 59% |
| 6 | 51685 / 81920 MiB | 88% |
| 7 | 51685 / 81920 MiB | 69% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 9443/10000，Epoch 73，Loss 0.0231**，StepTime ~7.2s，ETA ~1h07m。相对巡检 #32 推进约 **125 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 06:30 +08 / 22:30 UTC] GPU 巡检 #34（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 2 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52797 / 81920 MiB | 63% |
| 1 | 51795 / 81920 MiB | 73% |
| 2 | 51777 / 81920 MiB | 88% |
| 3 | 51775 / 81920 MiB | 73% |
| 4 | 51851 / 81920 MiB | 73% |
| 5 | 51783 / 81920 MiB | 39% |
| 6 | 51783 / 81920 MiB | 88% |
| 7 | 51685 / 81920 MiB | 82% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 9568/10000，Epoch 74，Loss 0.0097**，StepTime ~7.1s，ETA ~58m。相对巡检 #33 推进约 **125 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。

---

## [2026-08-28 06:45 +08 / 22:45 UTC] GPU 巡检 #35（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 3 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。Epoch 74 刚结束（VRAM dump），训练继续。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52797 / 81920 MiB | 73% |
| 1 | 51795 / 81920 MiB | 10% |
| 2 | 51777 / 81920 MiB | 98% |
| 3 | 51775 / 81920 MiB | 73% |
| 4 | 51851 / 81920 MiB | 73% |
| 5 | 51783 / 81920 MiB | 11% |
| 6 | 51783 / 81920 MiB | 31% |
| 7 | 51685 / 81920 MiB | 64% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 9694/10000，Epoch 74，Loss 0.0112**，随后 `VRAM usage after epoch 74`。StepTime ~6.9s，ETA ~35m。相对巡检 #34 推进约 **126 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。hanging_mug 预计再 1–2 拍内结束。

---

## [2026-08-28 07:00 +08 / 23:00 UTC] GPU 巡检 #36（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail -n 5 /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。

**结果**: **仍不可开训**。8 卡仍被 hanging_mug 占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52797 / 81920 MiB | 63% |
| 1 | 51795 / 81920 MiB | 76% |
| 2 | 51777 / 81920 MiB | 79% |
| 3 | 51775 / 81920 MiB | 89% |
| 4 | 51851 / 81920 MiB | 90% |
| 5 | 51783 / 81920 MiB | 75% |
| 6 | 51783 / 81920 MiB | 83% |
| 7 | 51685 / 81920 MiB | 79% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 9820/10000，Epoch 75，Loss 0.0098**，StepTime ~6.7s（其中一步 15.0s），ETA ~22–26m。相对巡检 #35 推进约 **126 step / 15 min**。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后。hanging_mug 预计下一拍前后结束。

---

## [2026-08-28 07:15 +08 / 23:15 UTC] GPU 巡检 #37（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py"
tail /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；判断 8 卡是否空闲。上次 ETA ~22 分钟，本拍可能已结束。

**结果**: **仍不可开训**。hanging_mug 尚未结束，8 卡仍被占用。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0 | 52797 / 81920 MiB | 100% |
| 1 | 51795 / 81920 MiB | 85% |
| 2 | 51777 / 81920 MiB | 85% |
| 3 | 51775 / 81920 MiB | 100% |
| 4 | 51851 / 81920 MiB | 100% |
| 5 | 51783 / 81920 MiB | 82% |
| 6 | 51783 / 81920 MiB | 86% |
| 7 | 51685 / 81920 MiB | 100% |

占用进程：`train.sh` **35386**，`torchrun` **35392**，worker 35466–35473。

日志最新：约 **Step 9946/10000，Epoch 76，Loss 0.0104**，StepTime ~7.0s，ETA ~6m。相对巡检 #36 推进约 **126 step / 15 min**。终局 checkpoint 保存可能再占用数分钟。

**决策**: 继续等待；不启动三任务流水线。下一拍约 15 分钟后应已空闲。

---

## [2026-08-28 07:30 +08 / 23:30 UTC] GPU 巡检 #38：8 卡空闲，开始三任务 SFT

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
pgrep -af "train_lingbotvla.py|torchrun"
tail /tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log
```

**理由**: 15 分钟循环唤醒；上一拍 hanging_mug 已到 Step 9946、ETA ~6m，本拍应已结束。

**结果**: **可开训**。8 卡均空闲（每卡 1 MiB，util 0%），无 `train_lingbotvla` / `torchrun`。

| GPU | 显存 used / total | GPU util |
|-----|-------------------|----------|
| 0–7 | 1 / 81920 MiB | 0% |

**hanging_mug 终局**（`/tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/train_stdout.log`）：

- Step 10000/10000 @ 23:22:33 UTC，Epoch 77，Loss 0.0208
- DCP 保存：`/tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/checkpoints/global_step_10000`（23:22:50）
- HF async checkpoint 完成 23:24:52
- `Reached max_steps=10000, stopping training.`

**停循环**: `kill 106756`（sentinel `AGENT_LOOP_TICK_gpu_idle`）。理由：用户要求 8 卡空闲后启动三任务流水线，空闲等待循环应停止，避免与训练抢同一会话。

**数据预检**（`~/Dta/RoboTwin-Clean/`，尚未 convert）：

| 任务 | codebase_version | frames | episodes |
|------|------------------|--------|----------|
| scan_object | v2.1 | 8463 | 50 |
| place_bread_skillet | v2.1 | 8277 | 50 |
| pick_dual_bottles | v2.1 | 6129 | 50 |

权重路径存在：`/tmp/itnvla15rbt20/var/hf_home/ckpts/lingbot-vla-v2-6b/hf_ckpt`。  
Checkpoint 根：`/home/a26113/Ckp/lbRbt/`。

**启动命令**（理由：按 `run_ech_rbt_p012.md` 用编排器串行 convert → 按帧数算 step → `bash train.sh`；env 由 `build_train_env` 注入 HF_HOME / LD_LIBRARY_PATH / CC / TORCHINDUCTOR）：

```bash
cd /tmp/SRC/lingbot-vla-v2
/tmp/itnvla15rbt20/bin/python -u b/s/rbt/run_each_robotwin.py \
  --config /tmp/SRC/lingbot-vla-v2/b/s/rbt/config.yaml \
  --tasks scan_object place_bread_skillet pick_dual_bottles
```

编排器 PID **162090**（后台）。流水线日志：`/home/a26113/Ckp/lbRbt/pipeline_20260827_233219.log`。

### scan_object 转换（成功，无 error）

- 命令：`/tmp/itnvla15rbt20/bin/python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id=scan_object --root=/home/a26113/Dta/RoboTwin-Clean --push-to-hub=false`
- 理由：数据集仍是 v2.1，训练 collator 需要 LeRobot v3.0。
- 结果：23:32:19–23:33:12 UTC，`converted scan_object -> v3.0 frames=8463`。原树应已到 `scan_object_old/`。
- 未改仓库代码。

### scan_object 步数（按手册公式）

`steps_per_epoch = floor(8463/128) = 66`，`max_steps = 66*76 = 5016`，`save_steps = 1254`，保存点 `[1254, 2508, 3762, 5016]`。micro=8，acc=2，8 GPU，gbs=128。

写出：

- `/home/a26113/Ckp/lbRbt/scan_object_20260827_233219/ft_scan_object_20260827_233219.yaml`
- `/home/a26113/Ckp/lbRbt/scan_object_20260827_233219/run_meta_20260827_233219.json`

### scan_object 训练已起来（无 error）

```
bash train.sh tasks/vla/train_lingbotvla.py .../ft_scan_object_20260827_233219.yaml \
  --data.norm_stats_file assets/norm_stats/robotwin.json --train.use_compile false
```

- `train.sh` **162724**，`torchrun` **162729**，`MASTER_PORT=62500`
- stdout：`/home/a26113/Ckp/lbRbt/scan_object_20260827_233219/train_stdout_20260827_233219.log`
- 23:40 UTC：8 卡各约 51315 MiB、util 100%；**Step 54/66，Epoch 1，Loss 0.1109**，StepTime ~6.7s
- 日志中无 Traceback / Error。沿用 hanging_mug 已修好的 venv / `transform.py` / gcc，本任务启动未再踩那些坑。

按 6.7s/step × 5016 ≈ **9.3 小时** 完 scan_object；其后两个任务继续由同一编排器串行 convert+训。

训练监控循环：sentinel `AGENT_LOOP_TICK_sft0827`，每 15 分钟查 GPU / pipeline / train_stdout；遇错则修并重启剩余任务；三任务都成功结束后停循环。

---

## [2026-08-28 07:56 +08 / 23:56 UTC] SFT 巡检 #1（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
cat /home/a26113/Ckp/lbRbt/pipeline_20260827_233219.log
tail /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/train_stdout_20260827_233219.log
```

**理由**: 训练监控循环唤醒；确认 scan_object 是否仍健康，有无 error。

**结果**: 仍在训 **scan_object**，无 error。编排器尚未进入后两个任务（`pipeline_*.log` 仍停在 train cmd）。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0–7 | 51315 MiB | 100% |

占用：`train.sh` **162724**，`torchrun` **162729**。

日志最新：约 **Step 192/5016，Epoch 3，Loss 0.0824**，StepTime ~6.7s。相对开训（Step ~54 @ 23:40）推进约 138 step / 16 min。剩余 4824 step ≈ **8.9 小时**。

**决策**: 不干预；继续等 scan_object 跑完后编排器自动转 place_bread_skillet。

---

## [2026-08-28 08:11 +08 / 00:12 UTC] SFT 巡检 #2（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
tail /home/a26113/Ckp/lbRbt/pipeline_20260827_233219.log
tail /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/train_stdout_20260827_233219.log
```

**理由**: 训练监控循环唤醒；确认有无 error。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。编排器未进入后两个任务。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0–7 | 51315 MiB | 100% |

占用：`train.sh` **162724**，`torchrun` **162729**。

日志最新：约 **Step 317/5016，Epoch 5，Loss 0.0426**，StepTime ~6.6s。相对巡检 #1 推进约 **125 step / 15 min**。剩余 4699 step ≈ **8.6 小时**。

**决策**: 不干预。

---

## [2026-08-28 08:26 +08 / 00:27 UTC] SFT 巡检 #3（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
tail /home/a26113/Ckp/lbRbt/pipeline_20260827_233219.log
tail /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/train_stdout_20260827_233219.log
```

**理由**: 训练监控循环唤醒；确认有无 error。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。编排器未进入后两个任务。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 51315 MiB | 77% |
| 1 | 51315 MiB | 61% |
| 2 | 51315 MiB | 79% |
| 3 | 51315 MiB | 89% |
| 4 | 51315 MiB | 90% |
| 5 | 51315 MiB | 89% |
| 6 | 51315 MiB | 77% |
| 7 | 51315 MiB | 62% |

占用：`train.sh` **162724**，`torchrun` **162729**。

日志最新：约 **Step 445/5016，Epoch 7，Loss 0.0457**，StepTime ~6.8s。相对巡检 #2 推进约 **128 step / 15 min**。剩余 4571 step ≈ **8.7 小时**。

**决策**: 不干预。

---

## [2026-08-28 08:41 +08 / 00:42 UTC] SFT 巡检 #4（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
tail /home/a26113/Ckp/lbRbt/pipeline_20260827_233219.log
tail /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/train_stdout_20260827_233219.log
```

**理由**: 训练监控循环唤醒；确认有无 error。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。编排器未进入后两个任务。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 51491 MiB | 100% |
| 1 | 51555 MiB | 100% |
| 2 | 51491 MiB | 100% |
| 3 | 50803 MiB | 100% |
| 4 | 51491 MiB | 100% |
| 5 | 51491 MiB | 100% |
| 6 | 51555 MiB | 100% |
| 7 | 51491 MiB | 100% |

占用：`train.sh` **162724**，`torchrun` **162729**。

日志最新：约 **Step 574/5016，Epoch 9，Loss 0.0432**，StepTime ~6.8s。相对巡检 #3 推进约 **129 step / 15 min**。剩余 4442 step ≈ **8.4 小时**。

**决策**: 不干预。

---

## [2026-08-28 08:56 +08 / 00:57 UTC] SFT 巡检 #5（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
tail /home/a26113/Ckp/lbRbt/pipeline_20260827_233219.log
tail /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/train_stdout_20260827_233219.log
```

**理由**: 训练监控循环唤醒；确认有无 error。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。编排器未进入后两个任务。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 51491 MiB | 100% |
| 1 | 51555 MiB | 87% |
| 2 | 51491 MiB | 87% |
| 3 | 50803 MiB | 98% |
| 4 | 51491 MiB | 99% |
| 5 | 51491 MiB | 100% |
| 6 | 51555 MiB | 100% |
| 7 | 51491 MiB | 100% |

占用：`train.sh` **162724**，`torchrun` **162729**。

日志最新：约 **Step 698/5016，Epoch 11，Loss 0.0359**，StepTime ~6.8s。相对巡检 #4 推进约 **124 step / 15 min**。剩余 4318 step ≈ **8.2 小时**。

**决策**: 不干预。

---

## [2026-08-28 09:11 +08 / 01:12 UTC] SFT 巡检 #6（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
tail /home/a26113/Ckp/lbRbt/pipeline_20260827_233219.log
tail /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/train_stdout_20260827_233219.log
```

**理由**: 训练监控循环唤醒；确认有无 error。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。编排器未进入后两个任务。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 51491 MiB | 85% |
| 1 | 51555 MiB | 35% |
| 2 | 51491 MiB | 47% |
| 3 | 50803 MiB | 51% |
| 4 | 51491 MiB | 79% |
| 5 | 51491 MiB | 48% |
| 6 | 51555 MiB | 59% |
| 7 | 51491 MiB | 49% |

占用：`train.sh` **162724**，`torchrun` **162729**。

日志最新：约 **Step 824/5016，Epoch 13，Loss 0.0406**，StepTime ~6.7s。相对巡检 #5 推进约 **126 step / 15 min**。剩余 4192 step ≈ **7.8 小时**。

**决策**: 不干预。

---

## [2026-08-28 09:26 +08 / 01:27 UTC] SFT 巡检 #7（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
tail /home/a26113/Ckp/lbRbt/pipeline_20260827_233219.log
tail /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/train_stdout_20260827_233219.log
```

**理由**: 训练监控循环唤醒；确认有无 error。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。编排器未进入后两个任务。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 51491 MiB | 82% |
| 1 | 51555 MiB | 90% |
| 2 | 51491 MiB | 85% |
| 3 | 50803 MiB | 82% |
| 4 | 51491 MiB | 85% |
| 5 | 51491 MiB | 82% |
| 6 | 51555 MiB | 95% |
| 7 | 51491 MiB | 83% |

占用：`train.sh` **162724**，`torchrun` **162729**。

日志最新：约 **Step 951/5016，Epoch 15，Loss 0.0262**，StepTime ~7.0s。相对巡检 #6 推进约 **127 step / 15 min**。剩余 4065 step ≈ **7.9 小时**。距第一次保存点 1254 约 303 step。

**决策**: 不干预。

---

## [2026-08-28 09:41 +08 / 01:42 UTC] SFT 巡检 #8（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
tail /home/a26113/Ckp/lbRbt/pipeline_20260827_233219.log
tail /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/train_stdout_20260827_233219.log
```

**理由**: 训练监控循环唤醒；确认有无 error。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。编排器未进入后两个任务。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 51271 MiB | 70% |
| 1 | 51505 MiB | 93% |
| 2 | 51567 MiB | 81% |
| 3 | 51533 MiB | 79% |
| 4 | 51363 MiB | 85% |
| 5 | 51271 MiB | 87% |
| 6 | 51437 MiB | 82% |
| 7 | 51271 MiB | 83% |

占用：`train.sh` **162724**，`torchrun` **162729**。

日志最新：约 **Step 1077/5016，Epoch 17，Loss 0.0385**，StepTime ~6.9s。相对巡检 #7 推进约 **126 step / 15 min**。剩余 3939 step ≈ **7.5 小时**。距第一次保存点 1254 约 177 step。

**决策**: 不干预。

---

## [2026-08-28 09:56 +08 / 01:57 UTC] SFT 巡检 #9（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
tail /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/train_stdout_20260827_233219.log
```

**理由**: 训练监控循环唤醒；确认有无 error，以及第一次 save_steps=1254 是否已落盘。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。`checkpoints/` 下尚无 `global_step_*`（尚未到 1254）。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 51271 MiB | 100% |
| 1 | 51505 MiB | 100% |
| 2 | 51567 MiB | 100% |
| 3 | 51533 MiB | 100% |
| 4 | 51363 MiB | 100% |
| 5 | 51271 MiB | 100% |
| 6 | 51437 MiB | 100% |
| 7 | 51271 MiB | 100% |

占用：`train.sh` **162724**，`torchrun` **162729**。

日志最新：约 **Step 1204/5016，Epoch 19，Loss 0.0352**，StepTime ~6.9s。相对巡检 #8 推进约 **127 step / 15 min**。剩余 3812 step ≈ **7.3 小时**。距第一次保存点 1254 约 50 step（约 6 分钟）。

**决策**: 不干预。

---

## [2026-08-28 10:11 +08 / 02:12 UTC] SFT 巡检 #10（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls -ld /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/global_step_*
tail /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/train_stdout_20260827_233219.log
```

**理由**: 训练监控循环唤醒；确认第一次保存点 1254 是否落盘、有无 error。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。第一次 DCP 已写出：

`/home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/global_step_1254`（目录 mtime 02:08 UTC）。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52787 MiB | 83% |
| 1 | 51685 MiB | 90% |
| 2 | 51631 MiB | 65% |
| 3 | 51685 MiB | 65% |
| 4 | 51685 MiB | 83% |
| 5 | 51635 MiB | 69% |
| 6 | 51757 MiB | 76% |
| 7 | 51635 MiB | 57% |

占用：`train.sh` **162724**，`torchrun` **162729**。

日志最新：约 **Step 1280/5016，Epoch 20，Loss 0.0412**，StepTime ~7.1s。相对巡检 #9 推进约 **76 step / 15 min**（中间含 1254 存盘，故步数偏少）。剩余 3736 step ≈ **7.4 小时**。下一保存点 2508。

**决策**: 不干预。

---

## [2026-08-28 10:26 +08 / 02:27 UTC] SFT 巡检 #11（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
tail /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/train_stdout_20260827_233219.log
```

**理由**: 训练监控循环唤醒；确认有无 error。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。checkpoint 仍只有 `global_step_1254`。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52787 MiB | 57% |
| 1 | 51685 MiB | 82% |
| 2 | 51631 MiB | 79% |
| 3 | 51685 MiB | 73% |
| 4 | 51685 MiB | 48% |
| 5 | 51635 MiB | 92% |
| 6 | 51757 MiB | 79% |
| 7 | 51635 MiB | 85% |

占用：`train.sh` **162724**，`torchrun` **162729**。

日志最新：约 **Step 1402/5016，Epoch 22，Loss 0.0263**，StepTime ~7.1s。相对巡检 #10 推进约 **122 step / 15 min**。剩余 3614 step ≈ **7.2 小时**。下一保存点 2508。

**决策**: 不干预。

---

## [2026-08-28 10:41 +08 / 02:42 UTC] SFT 巡检 #12（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
tail /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/train_stdout_20260827_233219.log
```

**理由**: 训练监控循环唤醒；确认有无 error。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。checkpoint 仍只有 `global_step_1254`。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52975 MiB | 79% |
| 1 | 51685 MiB | 84% |
| 2 | 51851 MiB | 84% |
| 3 | 51685 MiB | 83% |
| 4 | 51685 MiB | 77% |
| 5 | 51747 MiB | 81% |
| 6 | 51685 MiB | 84% |
| 7 | 51747 MiB | 81% |

占用：`train.sh` **162724**，`torchrun` **162729**。

日志最新：约 **Step 1526/5016，Epoch 24，Loss 0.0229**，StepTime ~6.9s。相对巡检 #11 推进约 **124 step / 15 min**。剩余 3490 step ≈ **6.7 小时**。下一保存点 2508。

**决策**: 不干预。

---

## [2026-08-28 10:57 +08 / 02:57 UTC] SFT 巡检 #13（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
tail /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/train_stdout_20260827_233219.log
```

**理由**: 训练监控循环唤醒；确认有无 error。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。checkpoint 仍只有 `global_step_1254`。编排器仍停在该任务的 `train cmd`，后两个任务未开始。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52975 MiB | 69% |
| 1 | 51685 MiB | 74% |
| 2 | 51851 MiB | 82% |
| 3 | 51685 MiB | 79% |
| 4 | 51685 MiB | 79% |
| 5 | 51747 MiB | 80% |
| 6 | 51685 MiB | 78% |
| 7 | 51747 MiB | 86% |

占用：`train.sh` **162724**，`torchrun` **162729**，`MASTER_PORT=62500`。

日志最新：约 **Step 1652/5016，Epoch 26，Loss 0.0221**，StepTime ~7.0s（epoch 边界第一步约 15.6s）。相对巡检 #12 推进约 **126 step / 15 min**。剩余 3364 step ≈ **6.5 小时**。下一保存点 2508。

**决策**: 不干预。

---

## [2026-08-28 11:12 +08 / 03:12 UTC] SFT 巡检 #14（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|RuntimeError|Step ' .../train_stdout_20260827_233219.log | tail
```

**理由**: 训练监控循环唤醒；确认有无 error。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。checkpoint 仍只有 `global_step_1254`。编排器仍停在该任务的 `train cmd`，后两个任务未开始。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52975 MiB | 89% |
| 1 | 51685 MiB | 89% |
| 2 | 51851 MiB | 88% |
| 3 | 51685 MiB | 87% |
| 4 | 51685 MiB | 84% |
| 5 | 51747 MiB | 71% |
| 6 | 51685 MiB | 69% |
| 7 | 51747 MiB | 40% |

占用：`train.sh` **162724**，`torchrun` **162729**，`MASTER_PORT=62500`。

日志最新：约 **Step 1780/5016，Epoch 27，Loss 0.0287**。相对巡检 #13 推进约 **128 step / 15 min**。剩余 3236 step ≈ **6.3 小时**。下一保存点 2508。

**决策**: 不干预。

---

## [2026-08-28 11:27 +08 / 03:27 UTC] SFT 巡检 #15（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|RuntimeError|Step ' .../train_stdout_20260827_233219.log | tail
```

**理由**: 训练监控循环唤醒；确认有无 error。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。checkpoint 仍只有 `global_step_1254`。编排器仍停在该任务的 `train cmd`，后两个任务未开始。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52975 MiB | 51% |
| 1 | 51685 MiB | 55% |
| 2 | 51851 MiB | 73% |
| 3 | 51685 MiB | 42% |
| 4 | 51685 MiB | 53% |
| 5 | 51747 MiB | 50% |
| 6 | 51685 MiB | 65% |
| 7 | 51747 MiB | 83% |

占用：`train.sh` **162724**，`torchrun` **162729**，`MASTER_PORT=62500`。

日志最新：约 **Step 1906/5016，Epoch 29，Loss 0.0311**，StepTime ~6.8–7.9s。相对巡检 #14 推进约 **126 step / 15 min**。剩余 3110 step ≈ **6.0 小时**。下一保存点 2508。

**决策**: 不干预。

---

## [2026-08-28 11:42 +08 / 03:42 UTC] SFT 巡检 #16（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|RuntimeError|Step ' .../train_stdout_20260827_233219.log | tail
```

**理由**: 训练监控循环唤醒；确认有无 error。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。checkpoint 仍只有 `global_step_1254`。编排器仍停在该任务的 `train cmd`，后两个任务未开始。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52821 MiB | 69% |
| 1 | 51685 MiB | 79% |
| 2 | 51799 MiB | 79% |
| 3 | 51685 MiB | 79% |
| 4 | 51685 MiB | 81% |
| 5 | 51745 MiB | 79% |
| 6 | 51685 MiB | 80% |
| 7 | 51781 MiB | 91% |

占用：`train.sh` **162724**，`torchrun` **162729**，`MASTER_PORT=62500`。

日志最新：约 **Step 2031/5016，Epoch 31，Loss 0.0213**，StepTime ~6.8s。相对巡检 #15 推进约 **125 step / 15 min**。剩余 2985 step ≈ **5.6 小时**。距下一保存点 2508 约 **477 step ≈ 54 分钟**。

**决策**: 不干预。

---

## [2026-08-28 11:57 +08 / 03:57 UTC] SFT 巡检 #17（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|RuntimeError|Step ' .../train_stdout_20260827_233219.log | tail
```

**理由**: 训练监控循环唤醒；确认有无 error。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。checkpoint 仍只有 `global_step_1254`。编排器仍停在该任务的 `train cmd`，后两个任务未开始。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52821 MiB | 100% |
| 1 | 51685 MiB | 100% |
| 2 | 51799 MiB | 100% |
| 3 | 51685 MiB | 100% |
| 4 | 51685 MiB | 100% |
| 5 | 51745 MiB | 100% |
| 6 | 51685 MiB | 100% |
| 7 | 51781 MiB | 100% |

占用：`train.sh` **162724**，`torchrun` **162729**，`MASTER_PORT=62500`。

日志最新：约 **Step 2156/5016，Epoch 33，Loss 0.0250**，StepTime ~7.0s。相对巡检 #16 推进约 **125 step / 15 min**。剩余 2860 step ≈ **5.6 小时**。距下一保存点 2508 约 **352 step ≈ 41 分钟**。

**决策**: 不干预。

---

## [2026-08-28 12:12 +08 / 04:12 UTC] SFT 巡检 #18（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|RuntimeError|Step ' .../train_stdout_20260827_233219.log | tail
```

**理由**: 训练监控循环唤醒；确认有无 error。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。checkpoint 仍只有 `global_step_1254`。编排器仍停在该任务的 `train cmd`，后两个任务未开始。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52821 MiB | 75% |
| 1 | 51685 MiB | 67% |
| 2 | 51799 MiB | 68% |
| 3 | 51685 MiB | 60% |
| 4 | 51685 MiB | 73% |
| 5 | 51745 MiB | 80% |
| 6 | 51685 MiB | 79% |
| 7 | 51781 MiB | 74% |

占用：`train.sh` **162724**，`torchrun` **162729**，`MASTER_PORT=62500`。

日志最新：约 **Step 2283/5016，Epoch 35，Loss 0.0447**，StepTime ~6.9s。相对巡检 #17 推进约 **127 step / 15 min**。剩余 2733 step ≈ **5.2 小时**。距下一保存点 2508 约 **225 step ≈ 26 分钟**。

**决策**: 不干预。

---

## [2026-08-28 12:27 +08 / 04:27 UTC] SFT 巡检 #19（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|RuntimeError|Step ' .../train_stdout_20260827_233219.log | tail
```

**理由**: 训练监控循环唤醒；确认有无 error。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。checkpoint 仍只有 `global_step_1254`。编排器仍停在该任务的 `train cmd`，后两个任务未开始。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52821 MiB | 63% |
| 1 | 51685 MiB | 90% |
| 2 | 51799 MiB | 87% |
| 3 | 51685 MiB | 79% |
| 4 | 51685 MiB | 79% |
| 5 | 51745 MiB | 80% |
| 6 | 51685 MiB | 79% |
| 7 | 51781 MiB | 87% |

占用：`train.sh` **162724**，`torchrun` **162729**，`MASTER_PORT=62500`。

日志最新：约 **Step 2408/5016，Epoch 37，Loss 0.0257**，StepTime ~7.0s。相对巡检 #18 推进约 **125 step / 15 min**。剩余 2608 step ≈ **5.1 小时**。距下一保存点 2508 约 **100 step ≈ 12 分钟**。

**决策**: 不干预。

---

## [2026-08-28 12:42 +08 / 04:42 UTC] SFT 巡检 #20（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
tail .../train_stdout_20260827_233219.log
```

**理由**: 训练监控循环唤醒；确认有无 error。上次距保存点 2508 约 12 分钟，本轮重点核对第二档 DCP。

**结果**: 仍在训 **scan_object**，无 Traceback / RuntimeError。第二档 DCP 已落盘。

checkpoint：

- `global_step_1254`（既有）
- `global_step_2508`（新；04:42:03 UTC 日志：`Saved checkpoint` / `Distributed checkpoint saved ... successfully`）

随后正在做 `[async_hf] saving HF checkpoint`，因此快照时 8 卡显存从 ~52 GB 降到 ~11 GB、`utilization.gpu=0`。`train.sh` **162724** / `torchrun` **162729** 仍在，worker 进程存活。这与 1254 存盘时的行为一致，属预期，不视为故障。

编排器仍停在该任务的 `train cmd`，后两个任务未开始。

日志最新：约 **Step 2508/5016，Epoch 38，Loss 0.0193**，StepTime ~6.7s。相对巡检 #19 推进约 **100 step / 15 min**（含 2508 存盘，步数偏少）。剩余 2508 step ≈ **4.9 小时**。下一保存点 3762。

**决策**: 不干预。等 HF 异步存盘结束后应恢复 8 卡满载训练。

---

## [2026-08-28 12:53 +08 / 04:53 UTC] 流水线误切任务：scan_object 在 step 2508 因 NCCL 超时崩溃

**现象**: 编排器日志出现 `train exit_code=0` 后立刻 `==== task=place_bread_skillet`。GPU 空闲。scan_object checkpoint 只有 `1254` 与 `2508`，缺少 `3762/5016`。`place_bread_skillet` 已 convert 到 v3.0 并开训（`MASTER_PORT=62501`）。

**根因**:

1. **训练崩溃（主因）**: `async_save_hf_weights=false`，HF 转换只在 **rank 0** 同步执行。DCP 存盘后的 `dist.barrier()` 之后，其它 rank 进入下一步 FSDP `all_gather`，rank 0 仍在把 6B 权重写成 HF。1254 那次 HF 用了 353s（< NCCL 600s timeout）所以活下来；2508 从 04:42:03 开始写 HF，04:52:08 rank7 watchdog 报 `_ALLGATHER_BASE` timeout 600046ms，随后 rank7 SIGABRT（exit -6），其余 rank SIGTERM。日志：`ChildFailedError` / `c10::DistBackendError`。DCP `global_step_2508` 本身已成功落盘。
2. **编排器误判成功**: `train.sh` 用 `torchrun ... | tee log.txt` 且没有 `set -o pipefail`，`tee` 成功导致 bash 返回 0。`run_each_robotwin.py`  accordingly 记下 `train exit_code=0` 并启动下一任务。

**处理**:

1. 先杀编排器 PID **162102**，再杀误开的 `place_bread_skillet` `train.sh` **201314** / `torchrun` **201319**（当时只跑了约 1.5 分钟，无 checkpoint）。8 卡回到 1 MiB。
2. **不要**对已是 v3 的 `scan_object` / `place_bread_skillet` 再 convert。`place_bread_skillet` 数据已是 v3.0（`_old` 为 v2.1）。误开训目录 `/home/a26113/Ckp/lbRbt/place_bread_skillet_20260828_045315/` 保留，scan_object 完成后用新 stamp 重训该任务。
3. 代码修复：
   - `train.sh`：加 `set -o pipefail`，让 torchrun 失败能传到编排器。
   - `tasks/vla/train_lingbotvla.py`：`save_hf_checkpoint_best_effort` 在 rank0 写完 HF 后对所有 rank `dist.barrier()`，避免下一步 all-gather 与 HF 转换重叠。
   - `b/s/rbt/run_each_robotwin.py`：即便 exit_code=0，若 stdout 含 `ChildFailedError` / `DistBackendError` 也视为失败。

**续训 scan_object**（同一 `output_dir`，`enable_resume: true` 从 `global_step_2508` 加载）:

```bash
# MASTER_PORT=62500, yaml 仍是 ft_scan_object_20260827_233219.yaml
# stdout: /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/train_stdout_resume_20260828_045753.log
```

05:01:58 UTC 日志：`Load distributed checkpoint from .../global_step_2508 successfully!`。`train.sh` PID **203226**，`torchrun` **203232**。

后续：scan_object 跑到 5016 后，再对 `place_bread_skillet`（已 v3，跳过 convert）和 `pick_dual_bottles` 开编排器。监控看 resume 日志，不要再启第二条全量流水线。

---

## [2026-08-28 13:04 +08 / 05:04 UTC] SFT 巡检 #21（15min tick）+ 旧编排器 exit 143

**操作**: 核 GPU、续训进程、resume stdout、旧 pipeline 终端 961645。

**理由**: 15min 监控唤醒；同时旧编排器终端报 `exit_code=143` 与 `train cmd` 匹配（误开 place_bread 时写入的历史行）。

**结果**:

- 终端 961645 **exit 143 = SIGTERM**，是上一轮为拦住误开的 `place_bread_skillet` 而主动杀掉编排器（PID 162102 / 包装壳 162090）的预期结果。**不要**把这条全量流水线再拉起来。
- `scan_object` 续训正常：已从 `global_step_2508` 加载，并打出 **Step 2509/5016，Epoch 39，Loss 0.0256**（首步 StepTime 180s，含编译）。`train.sh` **203226**，`torchrun` **203232**，`MASTER_PORT=62500`。
- 8 卡约 52 GB，util 70–100%。checkpoint 仍是 `1254` + `2508`。下一保存点 3762。剩余 2507 step ≈ **4.9 小时**。
- 监控日志改为：`/home/a26113/Ckp/lbRbt/scan_object_20260827_233219/train_stdout_resume_20260828_045753.log`

**决策**: 不重启旧编排器；继续盯续训直到 5016。

---

## [2026-08-28 13:12 +08 / 05:12 UTC] SFT 巡检 #22（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|Step ' .../train_stdout_resume_20260828_045753.log | tail
```

**理由**: 训练监控循环唤醒；盯 `scan_object` 从 2508 续训是否报错。

**结果**: 续训正常，无 Traceback / NCCL / ChildFailedError。checkpoint 仍是 `1254` + `2508`。旧编排器未在跑（符合预期）。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52989 MiB | 100% |
| 1–7 | 51837 MiB | 100% |

占用：`train.sh` **203226**，`torchrun` **203232**，`MASTER_PORT=62500`。

日志最新：约 **Step 2571/5016，Epoch 39，Loss 0.0287**，StepTime ~6.7s。相对巡检 #21（Step 2509）推进约 **62 step**（含续训首步编译 ~180s）。剩余 2445 step ≈ **4.6 小时**。下一保存点 3762。

**决策**: 不干预。

---

## [2026-08-28 13:27 +08 / 05:27 UTC] SFT 巡检 #23（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|Step ' .../train_stdout_resume_20260828_045753.log | tail
```

**理由**: 训练监控循环唤醒；盯 `scan_object` 从 2508 续训是否报错。

**结果**: 续训正常，无 Traceback / ChildFailedError。checkpoint 仍是 `1254` + `2508`。旧编排器未在跑。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52989 MiB | 49% |
| 1 | 51837 MiB | 47% |
| 2 | 51837 MiB | 37% |
| 3 | 51837 MiB | 48% |
| 4 | 51837 MiB | 61% |
| 5 | 51837 MiB | 45% |
| 6 | 51837 MiB | 52% |
| 7 | 51837 MiB | 53% |

占用：`train.sh` **203226**，`torchrun` **203232**，`MASTER_PORT=62500`。

日志最新：约 **Step 2700/5016，Epoch 41，Loss 0.0430**，StepTime ~6.6s。相对巡检 #22 推进约 **129 step / 15 min**。剩余 2316 step ≈ **4.3 小时**。距下一保存点 3762 约 **1062 step ≈ 2.0 小时**。

**决策**: 不干预。

---

## [2026-08-28 13:42 +08 / 05:42 UTC] SFT 巡检 #24（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|Step ' .../train_stdout_resume_20260828_045753.log | tail
```

**理由**: 训练监控循环唤醒；盯 `scan_object` 从 2508 续训是否报错。

**结果**: 续训正常，无 Traceback / ChildFailedError。checkpoint 仍是 `1254` + `2508`。旧编排器未在跑。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52989 MiB | 100% |
| 1 | 51837 MiB | 86% |
| 2 | 51837 MiB | 100% |
| 3 | 51837 MiB | 100% |
| 4 | 51837 MiB | 99% |
| 5 | 51837 MiB | 92% |
| 6 | 51837 MiB | 100% |
| 7 | 51837 MiB | 100% |

占用：`train.sh` **203226**，`torchrun` **203232**，`MASTER_PORT=62500`。

日志最新：约 **Step 2828/5016，Epoch 43，Loss 0.0237**，StepTime ~6.5s。相对巡检 #23 推进约 **128 step / 15 min**。剩余 2188 step ≈ **4.0 小时**。距下一保存点 3762 约 **934 step ≈ 1.7 小时**。

**决策**: 不干预。

---

## [2026-08-28 13:57 +08 / 05:57 UTC] SFT 巡检 #25（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|Step ' .../train_stdout_resume_20260828_045753.log | tail
```

**理由**: 训练监控循环唤醒；盯 `scan_object` 从 2508 续训是否报错。

**结果**: 续训正常，无 Traceback / ChildFailedError。checkpoint 仍是 `1254` + `2508`。旧编排器未在跑。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52989 MiB | 81% |
| 1 | 51837 MiB | 81% |
| 2 | 51837 MiB | 58% |
| 3 | 51837 MiB | 76% |
| 4 | 51837 MiB | 77% |
| 5 | 51837 MiB | 54% |
| 6 | 51837 MiB | 63% |
| 7 | 51837 MiB | 76% |

占用：`train.sh` **203226**，`torchrun` **203232**，`MASTER_PORT=62500`。

日志最新：约 **Step 2957/5016，Epoch 45，Loss 0.0205**，StepTime ~6.8s。相对巡检 #24 推进约 **129 step / 15 min**。剩余 2059 step ≈ **3.9 小时**。距下一保存点 3762 约 **805 step ≈ 1.5 小时**。

**决策**: 不干预。

---

## [2026-08-28 14:12 +08 / 06:12 UTC] SFT 巡检 #26（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|Step ' .../train_stdout_resume_20260828_045753.log | tail
```

**理由**: 训练监控循环唤醒；盯 `scan_object` 从 2508 续训是否报错。

**结果**: 续训正常，无 Traceback / ChildFailedError。checkpoint 仍是 `1254` + `2508`。旧编排器未在跑。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52969 MiB | 80% |
| 1 | 51805 MiB | 82% |
| 2 | 51821 MiB | 72% |
| 3 | 51805 MiB | 75% |
| 4 | 51833 MiB | 69% |
| 5 | 51821 MiB | 77% |
| 6 | 51821 MiB | 87% |
| 7 | 51821 MiB | 79% |

占用：`train.sh` **203226**，`torchrun` **203232**，`MASTER_PORT=62500`。

日志最新：约 **Step 3083/5016，Epoch 47，Loss 0.0163**，StepTime ~6.9s。相对巡检 #25 推进约 **126 step / 15 min**。剩余 1933 step ≈ **3.7 小时**。距下一保存点 3762 约 **679 step ≈ 1.3 小时**。

**决策**: 不干预。

---

## [2026-08-28 14:27 +08 / 06:27 UTC] SFT 巡检 #27（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|Step ' .../train_stdout_resume_20260828_045753.log | tail
```

**理由**: 训练监控循环唤醒；盯 `scan_object` 从 2508 续训是否报错。

**结果**: 续训正常，无 Traceback / ChildFailedError。checkpoint 仍是 `1254` + `2508`。旧编排器未在跑。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52969 MiB | 87% |
| 1 | 51805 MiB | 85% |
| 2 | 51821 MiB | 100% |
| 3 | 51805 MiB | 87% |
| 4 | 51833 MiB | 86% |
| 5 | 51821 MiB | 86% |
| 6 | 51821 MiB | 100% |
| 7 | 51821 MiB | 87% |

占用：`train.sh` **203226**，`torchrun` **203232**，`MASTER_PORT=62500`。

日志最新：约 **Step 3205/5016，Epoch 49，Loss 0.0164**，StepTime ~7.0s。相对巡检 #26 推进约 **122 step / 15 min**。剩余 1811 step ≈ **3.5 小时**。距下一保存点 3762 约 **557 step ≈ 1.1 小时**。

**决策**: 不干预。

---

## [2026-08-28 14:42 +08 / 06:42 UTC] SFT 巡检 #28（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|Step ' .../train_stdout_resume_20260828_045753.log | tail
```

**理由**: 训练监控循环唤醒；盯 `scan_object` 从 2508 续训是否报错。

**结果**: 续训正常，无 Traceback / ChildFailedError。checkpoint 仍是 `1254` + `2508`。旧编排器未在跑。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52969 MiB | 85% |
| 1 | 51805 MiB | 68% |
| 2 | 51821 MiB | 73% |
| 3 | 51805 MiB | 81% |
| 4 | 51833 MiB | 78% |
| 5 | 51821 MiB | 65% |
| 6 | 51821 MiB | 78% |
| 7 | 51821 MiB | 71% |

占用：`train.sh` **203226**，`torchrun` **203232**，`MASTER_PORT=62500`。

日志最新：约 **Step 3328/5016，Epoch 51，Loss 0.0193**，StepTime ~6.9s。相对巡检 #27 推进约 **123 step / 15 min**。剩余 1688 step ≈ **3.3 小时**。距下一保存点 3762 约 **434 step ≈ 50 分钟**。

**决策**: 不干预。

---

## [2026-08-28 14:57 +08 / 06:57 UTC] SFT 巡检 #29（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|Step ' .../train_stdout_resume_20260828_045753.log | tail
```

**理由**: 训练监控循环唤醒；盯 `scan_object` 从 2508 续训是否报错。

**结果**: 续训正常，无 Traceback / ChildFailedError。checkpoint 仍是 `1254` + `2508`。旧编排器未在跑。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52969 MiB | 100% |
| 1 | 51805 MiB | 100% |
| 2 | 51821 MiB | 100% |
| 3 | 51805 MiB | 100% |
| 4 | 51833 MiB | 100% |
| 5 | 51821 MiB | 100% |
| 6 | 51821 MiB | 100% |
| 7 | 51821 MiB | 100% |

占用：`train.sh` **203226**，`torchrun` **203232**，`MASTER_PORT=62500`。

日志最新：约 **Step 3450/5016，Epoch 53，Loss 0.0183**，StepTime ~7.3s。相对巡检 #28 推进约 **122 step / 15 min**。剩余 1566 step ≈ **3.2 小时**。距下一保存点 3762 约 **312 step ≈ 38 分钟**。

**决策**: 不干预。HF 存盘已加 barrier，3762 存盘时应全卡一起等，不再出现 2508 那种 NCCL 超时。

---

## [2026-08-28 15:12 +08 / 07:12 UTC] SFT 巡检 #30（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|Step ' .../train_stdout_resume_20260828_045753.log | tail
```

**理由**: 训练监控循环唤醒；盯 `scan_object` 续训，下一保存点 3762 临近。

**结果**: 续训正常，无 Traceback / ChildFailedError / NCCL。checkpoint 仍是 `1254` + `2508`。旧编排器未在跑。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 53035 MiB | 81% |
| 1 | 51655 MiB | 81% |
| 2 | 51833 MiB | 91% |
| 3 | 51655 MiB | 82% |
| 4 | 51887 MiB | 82% |
| 5 | 51833 MiB | 77% |
| 6 | 51833 MiB | 46% |
| 7 | 51833 MiB | 84% |

占用：`train.sh` **203226**，`torchrun` **203232**，`MASTER_PORT=62500`。

日志最新：约 **Step 3573/5016，Epoch 55，Loss 0.0153**，StepTime ~7.2s。相对巡检 #29 推进约 **123 step / 15 min**。剩余 1443 step ≈ **2.9 小时**。距下一保存点 3762 约 **189 step ≈ 23 分钟**。

**决策**: 不干预。

---

## [2026-08-28 15:27 +08 / 07:27 UTC] SFT 巡检 #31（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|NCCL|Saved checkpoint|Step ' .../train_stdout_resume_20260828_045753.log | tail
```

**理由**: 训练监控循环唤醒；距 3762 保存点已很近。

**结果**: 续训正常，无 Traceback / ChildFailedError / NCCL。checkpoint 仍是 `1254` + `2508`（尚未到 3762）。旧编排器未在跑。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 53035 MiB | 100% |
| 1 | 51655 MiB | 100% |
| 2 | 51833 MiB | 100% |
| 3 | 51655 MiB | 11% |
| 4 | 51887 MiB | 100% |
| 5 | 51833 MiB | 100% |
| 6 | 51833 MiB | 100% |
| 7 | 51833 MiB | 100% |

占用：`train.sh` **203226**，`torchrun` **203232**，`MASTER_PORT=62500`。

日志最新：约 **Step 3696/5016，Epoch 56，Loss 0.0198**，StepTime ~7.0s。相对巡检 #30 推进约 **123 step / 15 min**。剩余 1320 step ≈ **2.6 小时**。距下一保存点 3762 约 **66 step ≈ 8 分钟**。

**决策**: 不干预。下一轮应看到 `global_step_3762`（DCP + 全卡 barrier 等 HF）。

---

## [2026-08-28 15:42 +08 / 07:42 UTC] SFT 巡检 #32（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|NCCL|Saved checkpoint|async_hf|Step ' .../train_stdout_resume_20260828_045753.log | tail
```

**理由**: 训练监控循环唤醒；确认 3762 存盘是否成功、有无重现 2508 的 NCCL 超时。

**结果**: DCP **已落盘**，无 Traceback / ChildFailedError / NCCL timeout。barrier 修复生效：全卡在等 rank0 写 HF，而不是一边训一边写。

checkpoint：

- `global_step_1254`
- `global_step_2508`
- `global_step_3762`（新；07:38:33 UTC `Saved checkpoint` / `Distributed checkpoint saved ... successfully`；含 model / optimizer / extra_state）

随后 `[async_hf] saving HF checkpoint`（同步 + barrier）。快照时 GPU0 约 13 GB / util 0%，其余卡约 12 GB / util 100%（等 barrier）。进程仍在：`train.sh` **203226**，`torchrun` **203232**。`hf_ckpt` 尚未出现，属存盘中，不视为故障。

日志最新：约 **Step 3762/5016，Epoch 57，Loss 0.0178**。相对巡检 #31 推进约 **66 step / 15 min**（含 DCP 存盘）。剩余 1254 step ≈ **2.4 小时**（HF 结束后恢复步进）。下一保存点 5016。

**决策**: 不干预。

---

## [2026-08-28 15:57 +08 / 07:57 UTC] SFT 巡检 #33：3762 HF 再次 NCCL 超时，已关 HF 续训

**现象**: 8 卡 1 MiB、训练进程消失。最后一步仍是 3762。`global_step_3762` DCP 在，无 `hf_ckpt`。

**根因**: 07:38:33 开始 rank0 同步写 HF；其它 rank 在 `dist.barrier()`（NCCL ALLREDUCE numel=1）上等待。HF 超过 **600s**，07:48:33 watchdog timeout → 07:49:39 `ChildFailedError`（rank4 exit -6）。`barrier` 修不了「HF 本身长于 NCCL timeout」这个问题。1254 的 HF 用了 353s 所以侥幸活过；2508/3762 都 >600s。

**处理**:

1. 编排器 `b/s/rbt/run_each_robotwin.py` 的 `launch_train` 增加 `--train.save_hf_weights false --train.async_save_hf_weights false`。微调只保留 DCP；HF 可事后从 DCP 导出。
2. 从 `global_step_3762` 续训（同一 yaml，`enable_resume: true`）：

```bash
# MASTER_PORT=62500
# stdout: /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/train_stdout_resume_20260828_075821.log
# train.sh PID 224851, torchrun 224857
# dump 已确认 save_hf_weights=false
```

启动后 FSDP 已占约 42 GB，正在 load DCP 3762。后续监控该 resume 日志。剩余 1254 step ≈ **2.4 小时** 到 5016（下一保存点即终档）。

**决策**: 不重开旧编排器；等 scan_object 跑完再训后两个任务。

---

## [2026-08-28 16:12 +08 / 08:12 UTC] SFT 巡检 #34（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Load distributed|Traceback|ChildFailedError|Step ' .../train_stdout_resume_20260828_075821.log | tail
```

**理由**: 训练监控循环唤醒；确认从 3762、关闭 HF 后的续训是否正常。

**结果**: 续训正常。08:04:49 UTC `Load distributed checkpoint from .../global_step_3762 successfully!`，`save_hf_weights=false`。无 Traceback / ChildFailedError。checkpoint 仍是 `1254` + `2508` + `3762`。旧编排器未在跑。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52989 MiB | 90% |
| 1 | 51837 MiB | 57% |
| 2 | 51837 MiB | 83% |
| 3 | 51837 MiB | 87% |
| 4 | 51837 MiB | 42% |
| 5 | 51837 MiB | 77% |
| 6 | 51837 MiB | 78% |
| 7 | 51837 MiB | 72% |

占用：`train.sh` **224851**，`torchrun` **224857**，`MASTER_PORT=62500`。

日志最新：约 **Step 3823/5016，Epoch 58，Loss 0.0190**，StepTime ~6.7s。相对 3762 已推进约 **61 step**。剩余 1193 step ≈ **2.2 小时**。下一保存点 5016（终档，且不再写 HF）。

**决策**: 不干预。

---

## [2026-08-28 16:27 +08 / 08:27 UTC] SFT 巡检 #35（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|Step ' .../train_stdout_resume_20260828_075821.log | tail
```

**理由**: 训练监控循环唤醒；盯从 3762、关闭 HF 后的续训。

**结果**: 续训正常，无 Traceback / ChildFailedError。checkpoint 仍是 `1254` + `2508` + `3762`。旧编排器未在跑。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52989 MiB | 80% |
| 1–7 | 51837 MiB | 71–80% |

占用：`train.sh` **224851**，`torchrun` **224857**，`MASTER_PORT=62500`。

日志最新：约 **Step 3951/5016，Epoch 60，Loss 0.0124**，StepTime ~6.7s。相对巡检 #34 推进约 **128 step / 15 min**。剩余 1065 step ≈ **2.0 小时**。下一保存点 5016。

**决策**: 不干预。

---

## [2026-08-28 16:42 +08 / 08:42 UTC] SFT 巡检 #36（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|Step ' .../train_stdout_resume_20260828_075821.log | tail
```

**理由**: 训练监控循环唤醒；盯从 3762、关闭 HF 后的续训。

**结果**: 续训正常，无 Traceback / ChildFailedError。checkpoint 仍是 `1254` + `2508` + `3762`。旧编排器未在跑。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52985 MiB | 78% |
| 1 | 51805 MiB | 78% |
| 2 | 51829 MiB | 74% |
| 3 | 51829 MiB | 65% |
| 4 | 51829 MiB | 63% |
| 5 | 51657 MiB | 89% |
| 6 | 51641 MiB | 78% |
| 7 | 51657 MiB | 89% |

占用：`train.sh` **224851**，`torchrun` **224857**，`MASTER_PORT=62500`。

日志最新：约 **Step 4079/5016，Epoch 62，Loss 0.0138**，StepTime ~6.7s。相对巡检 #35 推进约 **128 step / 15 min**。剩余 937 step ≈ **1.8 小时**。下一保存点 5016。

**决策**: 不干预。

---

## [2026-08-28 16:57 +08 / 08:57 UTC] SFT 巡检 #37（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|Step ' .../train_stdout_resume_20260828_075821.log | tail
```

**理由**: 训练监控循环唤醒；盯从 3762、关闭 HF 后的续训。

**结果**: 续训正常，无 Traceback / ChildFailedError。checkpoint 仍是 `1254` + `2508` + `3762`。旧编排器未在跑。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52985 MiB | 84% |
| 1 | 51805 MiB | 70% |
| 2 | 51829 MiB | 74% |
| 3 | 51829 MiB | 59% |
| 4 | 51829 MiB | 63% |
| 5 | 51657 MiB | 88% |
| 6 | 51641 MiB | 71% |
| 7 | 51657 MiB | 94% |

占用：`train.sh` **224851**，`torchrun` **224857**，`MASTER_PORT=62500`。

日志最新：约 **Step 4208/5016，Epoch 64，Loss 0.0124**，StepTime ~6.7s。相对巡检 #36 推进约 **129 step / 15 min**。剩余 808 step ≈ **1.5 小时**。下一保存点 5016。

**决策**: 不干预。

---

## [2026-08-28 17:12 +08 / 09:12 UTC] SFT 巡检 #38（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|Step ' .../train_stdout_resume_20260828_075821.log | tail
```

**理由**: 训练监控循环唤醒；盯从 3762、关闭 HF 后的续训。

**结果**: 续训正常，无 Traceback / ChildFailedError。checkpoint 仍是 `1254` + `2508` + `3762`。旧编排器未在跑。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52985 MiB | 67% |
| 1 | 51805 MiB | 82% |
| 2 | 51829 MiB | 70% |
| 3 | 51829 MiB | 80% |
| 4 | 51829 MiB | 44% |
| 5 | 51657 MiB | 58% |
| 6 | 51641 MiB | 81% |
| 7 | 51657 MiB | 71% |

占用：`train.sh` **224851**，`torchrun` **224857**，`MASTER_PORT=62500`。

日志最新：约 **Step 4334/5016，Epoch 66，Loss 0.0149**，StepTime ~6.7s。相对巡检 #37 推进约 **126 step / 15 min**。剩余 682 step ≈ **1.3 小时**。下一保存点 5016（终档）。

**决策**: 不干预。

---

## [2026-08-28 17:27 +08 / 09:27 UTC] SFT 巡检 #39（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|Step ' .../train_stdout_resume_20260828_075821.log | tail
```

**理由**: 训练监控循环唤醒；盯从 3762、关闭 HF 后的续训。

**结果**: 续训正常，无 Traceback / ChildFailedError。checkpoint 仍是 `1254` + `2508` + `3762`。旧编排器未在跑。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0 | 52985 MiB | 66% |
| 1 | 51805 MiB | 80% |
| 2 | 51829 MiB | 81% |
| 3 | 51829 MiB | 82% |
| 4 | 51829 MiB | 71% |
| 5 | 51657 MiB | 73% |
| 6 | 51641 MiB | 81% |
| 7 | 51657 MiB | 87% |

占用：`train.sh` **224851**，`torchrun` **224857**，`MASTER_PORT=62500`。

日志最新：约 **Step 4462/5016，Epoch 68，Loss 0.0128**，StepTime ~6.8s。相对巡检 #38 推进约 **128 step / 15 min**。剩余 554 step ≈ **1.0 小时**。下一保存点 5016（终档）。

**决策**: 不干预。

---

## [2026-08-28 17:42 +08 / 09:42 UTC] SFT 巡检 #40（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|Step ' .../train_stdout_resume_20260828_075821.log | tail
```

**理由**: 训练监控循环唤醒；盯从 3762、关闭 HF 后的续训。

**结果**: 续训正常，无 Traceback / ChildFailedError。checkpoint 仍是 `1254` + `2508` + `3762`。旧编排器未在跑。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0–7 | ~52 GB | 100% |

占用：`train.sh` **224851**，`torchrun` **224857**，`MASTER_PORT=62500`。

日志最新：约 **Step 4587/5016，Epoch 70，Loss 0.0135**，StepTime ~7.1s。相对巡检 #39 推进约 **125 step / 15 min**。剩余 429 step ≈ **50 分钟**。下一保存点 5016（终档）。

**决策**: 不干预。

---

## [2026-08-28 17:57 +08 / 09:57 UTC] SFT 巡检 #41（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|Reached max_steps|Step ' .../train_stdout_resume_20260828_075821.log | tail
```

**理由**: 训练监控循环唤醒；`scan_object` 接近 5016 终档。

**结果**: 续训正常，无 Traceback / ChildFailedError。checkpoint 仍是 `1254` + `2508` + `3762`（终档 5016 尚未落盘）。旧编排器未在跑。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0–7 | ~52 GB | 80–100% |

占用：`train.sh` **224851**，`torchrun` **224857**，`MASTER_PORT=62500`。

日志最新：约 **Step 4711/5016，Epoch 72，Loss 0.0133**，StepTime ~7.1s。相对巡检 #40 推进约 **124 step / 15 min**。剩余 305 step ≈ **36 分钟**。下一保存点 5016（终档，仅 DCP）。

**决策**: 不干预。`scan_object` 完成后需启动编排器训 `place_bread_skillet` + `pick_dual_bottles`。

---

## [2026-08-28 18:12 +08 / 10:12 UTC] SFT 巡检 #42（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|Reached max_steps|Step ' .../train_stdout_resume_20260828_075821.log | tail
```

**理由**: 训练监控循环唤醒；`scan_object` 接近 5016 终档。

**结果**: 续训正常，无 Traceback / ChildFailedError。checkpoint 仍是 `1254` + `2508` + `3762`。旧编排器未在跑。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0–7 | ~52 GB | 71–83% |

占用：`train.sh` **224851**，`torchrun` **224857**，`MASTER_PORT=62500`。

日志最新：约 **Step 4835/5016，Epoch 74，Loss 0.0133**，StepTime ~7.1s。相对巡检 #41 推进约 **124 step / 15 min**。剩余 181 step ≈ **21 分钟**。下一保存点 5016（终档，仅 DCP）。

**决策**: 不干预。下一轮若 `Reached max_steps=5016` 且 `train exit_code=0`，启动编排器：`--tasks place_bread_skillet pick_dual_bottles`（`place_bread_skillet` 已 v3，应 skip convert）。

---

## [2026-08-28 18:27 +08 / 10:27 UTC] SFT 巡检 #43（15min tick）

**操作**:

```bash
date -u
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ls /home/a26113/Ckp/lbRbt/scan_object_20260827_233219/checkpoints/
grep -E 'Traceback|ChildFailedError|Reached max_steps|Step ' .../train_stdout_resume_20260828_075821.log | tail
```

**理由**: 训练监控循环唤醒；`scan_object` 即将到 5016。

**结果**: 续训正常，无 Traceback / ChildFailedError。已到 **Epoch 76**（最后一轮 epoch）。checkpoint 仍是 `1254` + `2508` + `3762`（终档 5016 尚未落盘）。

| GPU | 显存 used | GPU util |
|-----|-----------|----------|
| 0–7 | ~52 GB | 69–85% |

占用：`train.sh` **224851**，`torchrun` **224857**，`MASTER_PORT=62500`。

日志最新：约 **Step 4956/5016，Epoch 76，Loss 0.0119**。相对巡检 #42 推进约 **121 step / 15 min**。剩余 **60 step ≈ 7 分钟**。下一保存点 5016（终档）。

**决策**: 不干预。约 7 分钟后应完成 `scan_object`；完成后启动编排器训后两任务。

---

## [2026-08-28 18:42 +08 / 10:42 UTC] SFT 巡检 #44（15min tick）+ `scan_object` 完成，启动后两任务

**操作**:

```bash
date -u
nvidia-smi ...
ls .../scan_object_.../checkpoints/
tail .../train_stdout_resume_20260828_075821.log
# scan_object 成功后：
nohup python -u b/s/rbt/run_each_robotwin.py --config ... --tasks place_bread_skillet pick_dual_bottles &
```

**理由**: 巡检唤醒；`scan_object` 应已到 5016。

**结果**:

### `scan_object` ✅ 完成

- **Step 5016/5016**，`Reached max_steps=5016, stopping training.`
- DCP 四档齐全：`1254` / `2508` / `3762` / **`5016`**（10:36:35 UTC）
- 无 Traceback / ChildFailedError；`save_hf_weights=false`，无 HF 超时
- `train.sh` **224851** 已退出

### 已启动后两任务编排器

```bash
# PID 244925, pipeline log: /home/a26113/Ckp/lbRbt/pipeline_20260828_104222.log
```

| 任务 | 状态 |
|------|------|
| **place_bread_skillet** | 已 skip convert（v3.0）；**开训中** `MASTER_PORT=62500`，output `place_bread_skillet_20260828_104222`，max_steps=4864 |
| **pick_dual_bottles** | 待 `place_bread_skillet` 完成后 convert(v2.1→v3.0)+训，max_steps=3572，`MASTER_PORT=62501` |

**决策**: 继续 15min 巡检；监控 `pipeline_20260828_104222.log` 与 `place_bread_skillet_.../train_stdout_*.log`。

---

## [2026-08-28 18:57 +08 / 10:57 UTC] SFT 巡检 #45（15min tick）

**操作**:

```bash
date -u
nvidia-smi ...
tail /home/a26113/Ckp/lbRbt/pipeline_20260828_104222.log
grep Step .../place_bread_skillet_20260828_104222/train_stdout_*.log | tail
```

**理由**: 巡检唤醒；`scan_object` 已完成，盯 `place_bread_skillet` 开训是否正常。

**结果**:

- **`scan_object`**：已完成，checkpoint `1254/2508/3762/5016` 齐全 ✅
- **`place_bread_skillet`**：编排器 PID **244925** 正常；已 skip convert；**训中** Step **110/4864**（Epoch 2，Loss 0.0619），无 error。8 卡 ~51 GB。尚无 checkpoint。粗估剩余 ≈ **8.8 小时**。
- **`pick_dual_bottles`**：未开始（v2.1，待上任务完成后 convert+训）

占用：`train.sh` **244928**，`torchrun` **244933**，`MASTER_PORT=62500`。

**决策**: 不干预。

---

## [2026-08-28 19:12 +08 / 11:12 UTC] SFT 巡检 #46（15min tick）

**操作**:

```bash
date -u
nvidia-smi ...
tail pipeline_20260828_104222.log
grep Step .../place_bread_skillet_20260828_104222/train_stdout_*.log | tail
ps -p 244925,244928,244933
```

**理由**: 15min 巡检唤醒；确认 `place_bread_skillet` 训进是否正常。

**结果**:

- **`scan_object`**：已完成 ✅
- **`place_bread_skillet`**：编排器 PID **244925** 正常；训中 **global Step ~239 / 4864**（Epoch 4，Loss ~0.046），无 error。8 卡 ~51 GB、利用率 61–91%。尚无 checkpoint（首存 **1216**）。粗估剩余 ≈ **8.6 小时**。
- **`pick_dual_bottles`**：未开始

进程：`train.sh` **244928**，`torchrun` **244933**，`MASTER_PORT=62500`，已运行 ~30min。

**决策**: 不干预。

---

## [2026-08-28 19:27 +08 / 11:27 UTC] SFT 巡检 #47（15min tick）

**操作**:

```bash
date -u
nvidia-smi ...
grep Step .../place_bread_skillet_20260828_104222/train_stdout_*.log | tail
ps -p 244925,244928,244933
```

**理由**: 15min 巡检唤醒。

**结果**:

- **`scan_object`**：已完成 ✅
- **`place_bread_skillet`**：训中 **global Step ~367 / 4864**（Epoch 6，Loss ~0.041），无 error。8 卡 ~51 GB、利用率 38–84%。尚无 checkpoint（首存 **1216**）。粗估剩余 ≈ **8.5 小时**。
- **`pick_dual_bottles`**：未开始

进程：编排器 **244925**，已运行 ~45min。

**决策**: 不干预。

---

## [2026-08-28 19:42 +08 / 11:42 UTC] SFT 巡检 #48（15min tick）

**操作**:

```bash
date -u
nvidia-smi ...
grep Step .../place_bread_skillet_20260828_104222/train_stdout_*.log | tail
ps -p 244925,244928,244933
```

**理由**: 15min 巡检唤醒。

**结果**:

- **`scan_object`**：已完成 ✅
- **`place_bread_skillet`**：训中 **global Step ~494 / 4864**（Epoch 8，Loss ~0.033），无 error。8 卡 ~51 GB、利用率 48–86%。尚无 checkpoint（首存 **1216**）。粗估剩余 ≈ **8.1 小时**。
- **`pick_dual_bottles`**：未开始

进程：编排器 **244925**，已运行 ~60min。

**决策**: 不干预。

---

## [2026-08-28 19:57 +08 / 11:57 UTC] SFT 巡检 #49（15min tick）

**操作**:

```bash
date -u
nvidia-smi ...
grep Step .../place_bread_skillet_20260828_104222/train_stdout_*.log | tail
ps -p 244925,244928,244933
```

**理由**: 15min 巡检唤醒。

**结果**:

- **`scan_object`**：已完成 ✅
- **`place_bread_skillet`**：训中 **global Step ~620 / 4864**（Epoch 10，Loss ~0.034），无 error。8 卡 ~49–51 GB、利用率 100%。尚无 checkpoint（首存 **1216**，约再 **1 小时**）。粗估总剩余 ≈ **8.0 小时**。
- **`pick_dual_bottles`**：未开始

进程：编排器 **244925**，已运行 ~1h15m。

**决策**: 不干预。

---

## [2026-08-28 20:12 +08 / 12:12 UTC] SFT 巡检 #50（15min tick）

**操作**:

```bash
date -u
nvidia-smi ...
grep Step .../place_bread_skillet_20260828_104222/train_stdout_*.log | tail
ls checkpoints/
ps -p 244925,244928,244933
```

**理由**: 15min 巡检唤醒。

**结果**:

- **`scan_object`**：已完成 ✅
- **`place_bread_skillet`**：训中 **global Step ~747 / 4864**（Epoch 12，Loss ~0.034），无 error。8 卡 ~49–51 GB、利用率 61–90%。尚无 checkpoint（首存 **1216**，约再 **50 分钟**）。粗估总剩余 ≈ **7.9 小时**。
- **`pick_dual_bottles`**：未开始

进程：编排器 **244925**，已运行 ~1h30m。

**决策**: 不干预。

---

## [2026-08-28 20:27 +08 / 12:27 UTC] SFT 巡检 #51（15min tick）

**操作**:

```bash
date -u
nvidia-smi ...
grep Step .../place_bread_skillet_20260828_104222/train_stdout_*.log | tail
ls checkpoints/
ps -p 244925,244928,244933
```

**理由**: 15min 巡检唤醒。

**结果**:

- **`scan_object`**：已完成 ✅
- **`place_bread_skillet`**：训中 **global Step ~874 / 4864**（Epoch 14，Loss ~0.027），无 error。8 卡 ~49–51 GB、利用率 48–86%。尚无 checkpoint（首存 **1216**，约再 **40 分钟**）。粗估总剩余 ≈ **7.5 小时**。
- **`pick_dual_bottles`**：未开始

进程：编排器 **244925**，已运行 ~1h45m。

**决策**: 不干预。

---

## [2026-08-28 20:42 +08 / 12:42 UTC] SFT 巡检 #52（15min tick）

**操作**:

```bash
date -u
nvidia-smi ...
grep Step .../place_bread_skillet_20260828_104222/train_stdout_*.log | tail
ls checkpoints/
ps -p 244925,244928,244933
```

**理由**: 15min 巡检唤醒。

**结果**:

- **`scan_object`**：已完成 ✅
- **`place_bread_skillet`**：训中 **global Step ~1000 / 4864**（Epoch 16，Loss ~0.028），无 error。GPU0 ~36 GB/16%（其余卡 ~36–37 GB、85–99%），训练仍在推进。尚无 checkpoint（首存 **1216**，约再 **25 分钟**）。粗估总剩余 ≈ **7.3 小时**。
- **`pick_dual_bottles`**：未开始

进程：编排器 **244925**，已运行 ~2h。

**决策**: 不干预。

---

## [2026-08-28 20:57 +08 / 12:57 UTC] SFT 巡检 #53（15min tick）

**操作**:

```bash
date -u
nvidia-smi ...
grep Step .../place_bread_skillet_20260828_104222/train_stdout_*.log | tail
ls checkpoints/
ps -p 244925,244928,244933
```

**理由**: 15min 巡检唤醒。

**结果**:

- **`scan_object`**：已完成 ✅
- **`place_bread_skillet`**：训中 **global Step ~1127 / 4864**（Epoch 18，Loss ~0.026），无 error。8 卡 ~51 GB、利用率 84–87%。尚无 checkpoint（首存 **1216**，约再 **10 分钟**）。粗估总剩余 ≈ **7.2 小时**。
- **`pick_dual_bottles`**：未开始

进程：编排器 **244925**，已运行 ~2h15m。

**决策**: 不干预。

---

## [2026-08-28 21:12 +08 / 13:12 UTC] SFT 巡检 #54（15min tick）

**操作**:

```bash
date -u
nvidia-smi ...
grep Step .../place_bread_skillet_20260828_104222/train_stdout_*.log | tail
ls checkpoints/
ps -p 244925,244928,244933
```

**理由**: 15min 巡检唤醒；预期首 checkpoint **1216** 应已落盘。

**结果**:

- **`scan_object`**：已完成 ✅
- **`place_bread_skillet`**：训中 **global Step ~1230 / 4864**（Epoch 20，Loss ~0.033），无 error。8 卡 ~52 GB、利用率 45–88%。**首 checkpoint `global_step_1216` 已于 13:10 UTC 成功保存** ✅（DCP，无 HF 超时）。下一存盘点 **2432**。
- **`pick_dual_bottles`**：未开始

进程：编排器 **244925**，已运行 ~2h30m。粗估 `place_bread_skillet` 剩余 ≈ **6.9 小时**。

**决策**: 不干预。

---

## [2026-08-28 21:27 +08 / 13:27 UTC] SFT 巡检 #55（15min tick）

**操作**:

```bash
date -u
nvidia-smi ...
grep Step .../place_bread_skillet_20260828_104222/train_stdout_*.log | tail
ls checkpoints/
ps -p 244925,244928,244933
```

**理由**: 15min 巡检唤醒。

**结果**:

- **`scan_object`**：已完成 ✅
- **`place_bread_skillet`**：训中 **global Step ~1357 / 4864**（Epoch 22，Loss ~0.028），无 error。8 卡 ~52 GB、利用率 78–94%。checkpoint **`global_step_1216`** 已存在。下一存盘点 **2432**（约 **1.9 小时**）。粗估总剩余 ≈ **6.9 小时**。
- **`pick_dual_bottles`**：未开始

进程：编排器 **244925**，已运行 ~2h45m。

**决策**: 不干预。

---

## [2026-08-28 21:42 +08 / 13:42 UTC] SFT 巡检 #56（15min tick）

**操作**:

```bash
date -u
nvidia-smi ...
grep Step .../place_bread_skillet_20260828_104222/train_stdout_*.log | tail
ls checkpoints/
ps -p 244925,244928,244933
```

**理由**: 15min 巡检唤醒。

**结果**:

- **`scan_object`**：已完成 ✅
- **`place_bread_skillet`**：训中 **global Step ~1483 / 4864**（Epoch 24，Loss ~0.041），无 error。8 卡 ~52 GB、利用率 76–95%。checkpoint **`global_step_1216`**。下一存盘点 **2432**（约 **1.7 小时**）。粗估总剩余 ≈ **6.7 小时**。
- **`pick_dual_bottles`**：未开始

进程：编排器 **244925**，已运行 ~3h。

**决策**: 不干预。

---

## [2026-08-28 21:57 +08 / 13:57 UTC] SFT 巡检 #57（15min tick）

**操作**:

```bash
date -u
nvidia-smi ...
grep Step .../place_bread_skillet_20260828_104222/train_stdout_*.log | tail
ls checkpoints/
ps -p 244925,244928,244933
```

**理由**: 15min 巡检唤醒。

**结果**:

- **`scan_object`**：已完成 ✅
- **`place_bread_skillet`**：训中 **global Step ~1609 / 4864**（Epoch 26，Loss ~0.021），无 error。8 卡 ~52 GB、利用率 66–94%。checkpoint **`global_step_1216`**。下一存盘点 **2432**（约 **1.6 小时**）。粗估总剩余 ≈ **6.5 小时**。
- **`pick_dual_bottles`**：未开始

进程：编排器 **244925**，已运行 ~3h15m。

**决策**: 不干预。

---

## [2026-08-28 22:12 +08 / 14:12 UTC] SFT 巡检 #58（15min tick）

**操作**:

```bash
date -u
nvidia-smi ...
grep Step .../place_bread_skillet_20260828_104222/train_stdout_*.log | tail
ls checkpoints/
ps -p 244925,244928,244933
```

**理由**: 15min 巡检唤醒。

**结果**:

- **`scan_object`**：已完成 ✅
- **`place_bread_skillet`**：训中 **global Step ~1736 / 4864**（Epoch 28，Loss ~0.019），无 error。8 卡 ~52 GB、利用率 58–90%。checkpoint **`global_step_1216`**。下一存盘点 **2432**（约 **1.4 小时**）。粗估总剩余 ≈ **6.3 小时**。
- **`pick_dual_bottles`**：未开始

进程：编排器 **244925**，已运行 ~3h30m。

**决策**: 不干预。

---

## [2026-08-28 22:27 +08 / 14:27 UTC] SFT 巡检 #59（15min tick）

**操作**:

```bash
date -u
nvidia-smi ...
grep Step .../place_bread_skillet_20260828_104222/train_stdout_*.log | tail
ls checkpoints/
ps -p 244925,244928,244933
```

**理由**: 15min 巡检唤醒。

**结果**:

- **`scan_object`**：已完成 ✅
- **`place_bread_skillet`**：训中 **global Step ~1861 / 4864**（Epoch 30，Loss ~0.028），无 error。8 卡 ~52 GB、利用率 40–76%。checkpoint **`global_step_1216`**。下一存盘点 **2432**（约 **1.1 小时**）。粗估总剩余 ≈ **5.8 小时**。
- **`pick_dual_bottles`**：未开始

进程：编排器 **244925**，已运行 ~3h45m。

**决策**: 不干预。

---

## [2026-08-28 22:42 +08 / 14:42 UTC] SFT 巡检 #60（15min tick）

**操作**:

```bash
date -u
nvidia-smi ...
grep Step .../place_bread_skillet_20260828_104222/train_stdout_*.log | tail
ls checkpoints/
ps -p 244925,244928,244933
```

**理由**: 15min 巡检唤醒。

**结果**:

- **`scan_object`**：已完成 ✅
- **`place_bread_skillet`**：训中 **global Step ~1987 / 4864**（Epoch 32，Loss ~0.020），无 error。8 卡 ~52 GB、利用率 100%。checkpoint **`global_step_1216`**。下一存盘点 **2432**（约 **50 分钟**）。粗估总剩余 ≈ **5.6 小时**。
- **`pick_dual_bottles`**：未开始

进程：编排器 **244925**，已运行 ~4h。

**决策**: 不干预。

---
