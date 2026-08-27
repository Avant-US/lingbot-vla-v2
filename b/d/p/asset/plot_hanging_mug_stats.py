#!/usr/bin/env python3
"""Analyze hanging_mug LeRobot v2.1 stats and draw English-label figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/tmp/Dta/RoboTwin-Clean/hanging_mug")
OUT = Path(__file__).resolve().parent
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def load_episode_lengths() -> list[int]:
    lengths = []
    with open(ROOT / "meta/episodes.jsonl") as f:
        for line in f:
            lengths.append(int(json.loads(line)["length"]))
    return lengths


def load_episode_arrays(ep_idx: int) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(ROOT / f"data/chunk-000/episode_{ep_idx:06d}.parquet")
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    return state, action


def plot_episode_lengths(lengths: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bins = np.arange(min(lengths) - 2, max(lengths) + 4, 2)
    ax.hist(lengths, bins=bins, color="#4C78A8", edgecolor="white", linewidth=0.6)
    ax.axvline(np.mean(lengths), color="#E45756", ls="--", lw=1.4, label=f"mean = {np.mean(lengths):.1f}")
    ax.axvline(340, color="#F58518", ls=":", lw=1.4, label="official avg steps = 340")
    ax.set_xlabel("Episode length (frames @ 15 FPS)")
    ax.set_ylabel("Number of episodes")
    ax.set_title("hanging_mug clean-50 episode length distribution")
    ax.legend(frameon=False)
    fig.savefig(OUT / "hanging_mug_episode_length.png")
    plt.close(fig)


def plot_bimanual_phases(ep_idx: int = 0) -> None:
    state, action = load_episode_arrays(ep_idx)
    t = np.arange(state.shape[0]) / 15.0
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 6.4), sharex=True)

    axes[0].plot(t, state[:, 6], color="#4C78A8", label="left gripper (state)")
    axes[0].plot(t, state[:, 13], color="#E45756", label="right gripper (state)")
    axes[0].set_ylabel("Gripper opening")
    axes[0].set_ylim(-0.05, 1.15)
    axes[0].set_title(f"Episode {ep_idx}: pick-rotate-place then hang (Aloha-AgileX, 15 FPS)")
    axes[0].legend(frameon=False, loc="upper right", ncol=2)

    left_arm = state[:, :6]
    right_arm = state[:, 7:13]
    axes[1].plot(t, np.linalg.norm(np.diff(left_arm, axis=0, prepend=left_arm[:1]), axis=1), color="#4C78A8", label="left arm |dq|")
    axes[1].plot(t, np.linalg.norm(np.diff(right_arm, axis=0, prepend=right_arm[:1]), axis=1), color="#E45756", label="right arm |dq|")
    axes[1].set_ylabel("Joint speed proxy")
    axes[1].legend(frameon=False, loc="upper right", ncol=2)

    delta = np.abs(action - state).mean(axis=1)
    axes[2].plot(t, delta, color="#54A24B")
    axes[2].set_ylabel("Mean |action-state|")
    axes[2].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(OUT / "hanging_mug_bimanual_phases.png")
    plt.close(fig)


def plot_action_abs_vs_rel() -> None:
    abs_std = []
    rel_std = []
    for i in range(50):
        state, action = load_episode_arrays(i)
        abs_std.append(action.std(axis=0))
        rel_std.append((action - state).std(axis=0))
    abs_std = np.mean(abs_std, axis=0)
    rel_std = np.mean(rel_std, axis=0)
    names = [
        "L-waist",
        "L-shoulder",
        "L-elbow",
        "L-forearm",
        "L-wristA",
        "L-wristR",
        "L-grip",
        "R-waist",
        "R-shoulder",
        "R-elbow",
        "R-forearm",
        "R-wristA",
        "R-wristR",
        "R-grip",
    ]
    x = np.arange(14)
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    w = 0.38
    ax.bar(x - w / 2, abs_std, w, color="#4C78A8", label="absolute action std")
    ax.bar(x + w / 2, rel_std, w, color="#F58518", label="relative (action-state) std")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Std over time (rad / gripper)")
    ax.set_title("hanging_mug: absolute vs relative action scale (50 episodes)")
    ax.legend(frameon=False)
    fig.savefig(OUT / "hanging_mug_abs_vs_rel.png")
    plt.close(fig)


def main() -> None:
    lengths = load_episode_lengths()
    print(
        f"episodes={len(lengths)} frames={sum(lengths)} "
        f"mean={np.mean(lengths):.2f} min={min(lengths)} max={max(lengths)}"
    )
    plot_episode_lengths(lengths)
    plot_bimanual_phases(0)
    plot_action_abs_vs_rel()
    print(f"Wrote figures under {OUT}")


if __name__ == "__main__":
    main()
