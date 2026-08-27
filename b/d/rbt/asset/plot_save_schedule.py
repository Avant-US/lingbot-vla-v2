#!/usr/bin/env python3
"""Plot per-task save schedule for the first RoboTwin FT batch (English labels)."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent

TASKS = [
    ("scan_object", 8463),
    ("place_bread_skillet", 8277),
    ("pick_dual_bottles", 6129),
    ("hanging_mug (ref)", 16889),
]

GLOBAL_BSZ = 128
NUM_EPOCHS = 76


def schedule(n_frames: int) -> tuple[int, int, int, list[int]]:
    spe = math.floor(n_frames / GLOBAL_BSZ)
    max_steps = spe * NUM_EPOCHS
    save_steps = max(1, max_steps // 4)
    points = list(range(save_steps, max_steps + 1, save_steps))
    if points[-1] != max_steps:
        points.append(max_steps)
    return spe, max_steps, save_steps, points


def main() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    colors = ["#2563eb", "#059669", "#d97706", "#6b7280"]
    for color, (name, n_frames) in zip(colors, TASKS):
        spe, max_steps, save_steps, points = schedule(n_frames)
        ax.plot([0, max_steps], [name, name], color=color, linewidth=6, alpha=0.25, solid_capstyle="round")
        ax.scatter(points, [name] * len(points), color=color, s=48, zorder=3, label=None)
        ax.text(max_steps + 80, name, f"{spe} st/ep, max={max_steps}, save={save_steps}", va="center", fontsize=8, color="#111827")
    ax.set_xlabel("Optimizer step (global_batch_size=128, 76 epochs)")
    ax.set_title("Checkpoint save points: every 1/4 of the epoch budget, plus the final step")
    ax.set_xlim(left=0)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    out = OUT_DIR / "save_schedule.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
