# hanging_mug 微调执行日志

对应实施手册：[`reprd_rbtwn_hngMg.md`](reprd_rbtwn_hngMg.md)。

按时间记录所有操作、error / 根因 / fix、增删改文件、命令与理由、关键路径。

---

## 目标配置（执行时）

| 项目 | 值 |
|------|-----|
| 基础模型 | `robbyant/lingbot-vla-v2-6b` |
| 数据 | `/tmp/Dta/RoboTwin-Clean/hanging_mug/` |
| 代码 | `/tmp/SRC/lingbot-vla-v2` |
| 虚拟环境（迁移后） | `/tmp/itnvla15rbt20/` |
| `HF_HOME`（迁移后） | `/tmp/itnvla15rbt20/var/hf_home/` |
| Checkpoint | `/tmp/Ckp/lingbot-vla-v2-ft-hanging_mug/` |
| GPU | 8 × NVIDIA A800-SXM4-80GB |
| 训练配置 | `configs/vla/robotwin/hanging_mug_ft.yaml` |

---

## [2026-08-27] Step 0: 迁移前检查

**操作**: 检查 overlay / `/tmp` 容量、venv 体积、GPU、数据集版本。

```bash
df -h /b /tmp
du -sh /b/VENV/itnvla15rbt20
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

**结果**:

| 项 | 值 |
|----|-----|
| overlay (`/`) | 500G，当时约 139G 可用 |
| `/tmp` | 2.5T，约 2.4T 可用 |
| `/b/VENV/itnvla15rbt20` | **93G**（含 HF 权重缓存） |
| `/tmp/itnvla15rbt20` | 不存在 |
| GPU | 8×A800 空闲 |
| hanging_mug | LeRobot **v2.1** |
| 训练进程 | 无 |

**理由**: overlay 放不下训练 DCP（单份约 20G+）；手册已要求权重走 `HF_HOME`、checkpoint 走 `/tmp`。用户进一步要求把整个 venv+HF_HOME 迁到 `/tmp/itnvla15rbt20`，checkpoint 目录改为 `/tmp/Ckp/`。

`pyvenv.cfg` 显示该 venv **原本就创建在** `/tmp/itnvla15rbt20`（`python3 -m venv --copies /tmp/itnvla15rbt20`），后来被放到 `/b/VENV/`。`bin/activate` 的 `VIRTUAL_ENV` 和 `bin/hf` shebang 仍指向 `/tmp/itnvla15rbt20`。迁回原路径后这两处会重新生效。

---

## [2026-08-27] Step 1: 移动虚拟环境

**操作**: 跨文件系统 `mv`（overlay → `/tmp` 所在 kubelet 盘，实际是 copy + unlink）。

```bash
mv /b/VENV/itnvla15rbt20 /tmp/itnvla15rbt20
```

**理由**: 把 93G 的 Python 环境与 HF 权重从 overlay 挪到空间充足的 `/tmp`，避免训练写 checkpoint / 编译 / pip 再撑爆 overlay。
