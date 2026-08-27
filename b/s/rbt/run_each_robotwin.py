#!/usr/bin/env python3
"""Sequential RoboTwin 2.0 per-task convert (LeRobot v3.0) + LingBot-VLA v2 fine-tune.

See b/d/rbt/run_ech_rbt_p012.md for design. Default config: b/s/rbt/config.yaml.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any

import yaml

THIS_FILE = Path(__file__).resolve()
# b/s/rbt/run_each_robotwin.py -> repo root is parents[3]
DEFAULT_REPO_ROOT = THIS_FILE.parents[3]
DEFAULT_CONFIG = THIS_FILE.parent / "config.yaml"


@dataclass
class TrainSchedule:
    """Per-task step plan derived from frame count and epoch budget."""

    task: str
    n_frames: int
    n_episodes: int | None
    n_gpus: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    global_batch_size: int
    num_epochs: int
    steps_per_epoch: int
    max_steps: int
    save_steps: int
    save_points: list[int]
    visual_steps: int

    def as_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        return data


def expand_path(value: str | Path) -> Path:
    return Path(os.path.expanduser(str(value))).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def format_path_template(value: str, **kwargs: str) -> str:
    return value.format(**kwargs)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_each_robotwin")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def detect_n_gpus(cuda_visible: str | None) -> int:
    if cuda_visible:
        devices = [x for x in cuda_visible.split(",") if x.strip() != ""]
        if devices:
            return len(devices)
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return 1
    return max(1, sum(1 for line in out.splitlines() if line.strip().startswith("GPU ")))


def compute_grad_acc(global_batch_size: int, micro_batch_size: int, n_gpus: int) -> int:
    denom = micro_batch_size * n_gpus
    if denom <= 0:
        raise ValueError("micro_batch_size and n_gpus must be positive")
    if global_batch_size % denom != 0:
        raise ValueError(
            f"global_batch_size={global_batch_size} is not divisible by "
            f"micro_batch_size({micro_batch_size}) * n_gpus({n_gpus}) = {denom}. "
            "Adjust micro_batch_size or GPU count."
        )
    return global_batch_size // denom


def compute_schedule(
    task: str,
    n_frames: int,
    global_batch_size: int,
    micro_batch_size: int,
    n_gpus: int,
    num_epochs: int,
    save_every_epoch_fraction: float,
    n_episodes: int | None = None,
) -> TrainSchedule:
    """Map dataset size + epoch budget onto trainer max_steps / save_steps.

    steps_per_epoch = floor(N_frames / B_global), matching TrainingArguments.compute_train_steps
    with drop_last=True and dp_size == world_size.
    """
    if n_frames <= 0:
        raise ValueError(f"{task}: n_frames must be positive, got {n_frames}")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if not (0.0 < save_every_epoch_fraction <= 1.0):
        raise ValueError("save_every_epoch_fraction must be in (0, 1]")

    grad_acc = compute_grad_acc(global_batch_size, micro_batch_size, n_gpus)
    steps_per_epoch = math.floor(n_frames / global_batch_size)
    if steps_per_epoch < 1:
        raise ValueError(
            f"{task}: n_frames={n_frames} < global_batch_size={global_batch_size}; "
            "cannot form one optimizer step per epoch."
        )
    max_steps = steps_per_epoch * num_epochs
    # Save every 1/4 of the total epoch budget (in optimizer steps).
    interval = max(1, int(round(steps_per_epoch * num_epochs * save_every_epoch_fraction)))
    # Prefer exact quarter of max_steps when fraction is 0.25.
    if abs(save_every_epoch_fraction - 0.25) < 1e-9:
        interval = max(1, max_steps // 4)
    save_points = list(range(interval, max_steps + 1, interval))
    if save_points[-1] != max_steps:
        save_points.append(max_steps)
    return TrainSchedule(
        task=task,
        n_frames=n_frames,
        n_episodes=n_episodes,
        n_gpus=n_gpus,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=grad_acc,
        global_batch_size=global_batch_size,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        max_steps=max_steps,
        save_steps=interval,
        save_points=save_points,
        visual_steps=interval,
    )


def read_dataset_meta(task_dir: Path) -> dict[str, Any]:
    info_path = task_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def codebase_major(version: str) -> int:
    # "v2.1" -> 2, "v3.0" -> 3
    text = version.lower().lstrip("v")
    return int(text.split(".", 1)[0])


def resolve_config(raw: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    paths = dict(raw.get("paths") or {})
    if not paths.get("repo_root"):
        paths["repo_root"] = str(repo_root)
    for key in ("repo_root", "venv", "hf_home", "data_root", "ckpt_root"):
        paths[key] = str(expand_path(paths[key]))
    hf_home = paths["hf_home"]
    model = dict(raw.get("model") or {})
    for key, value in list(model.items()):
        if isinstance(value, str):
            model[key] = str(expand_path(format_path_template(value, hf_home=hf_home)))
    cfg = dict(raw)
    cfg["paths"] = paths
    cfg["model"] = model
    cfg["tasks"] = list(raw.get("tasks") or [])
    cfg["train"] = dict(raw.get("train") or {})
    cfg["convert"] = dict(raw.get("convert") or {})
    cfg["runtime"] = dict(raw.get("runtime") or {})
    return cfg


def render_ft_yaml(template_path: Path, mapping: dict[str, str]) -> str:
    tmpl = Template(template_path.read_text(encoding="utf-8"))
    return tmpl.substitute(mapping)


def bool_yaml(value: bool) -> str:
    return "true" if value else "false"


def build_train_env(cfg: dict[str, Any], output_dir: Path, master_port: int) -> dict[str, str]:
    paths = cfg["paths"]
    runtime = cfg["runtime"]
    venv = Path(paths["venv"])
    hf_home = paths["hf_home"]
    npp = venv / "lib/python3.11/site-packages/nvidia/npp/lib"
    lib = venv / "lib"
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = str(venv / "bin") + os.pathsep + env.get("PATH", "")
    env["HF_HOME"] = hf_home
    extra_ld = [str(lib), str(npp)]
    old_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(extra_ld + ([old_ld] if old_ld else []))
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    env["TORCH_SHOW_CPP_STACKTRACES"] = "1"
    env["TORCH_DISABLE_ADDR2LINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CC"] = str(runtime.get("cc") or "/usr/bin/gcc")
    env["CXX"] = str(runtime.get("cxx") or "/usr/bin/g++")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(output_dir / "inductor_cache")
    env["CUDA_VISIBLE_DEVICES"] = str(runtime.get("cuda_visible_devices") or env.get("CUDA_VISIBLE_DEVICES", ""))
    env["MASTER_PORT"] = str(master_port)
    env["NNODES"] = "1"
    env["NODE_RANK"] = "0"
    return env


def convert_task(
    task: str,
    data_root: Path,
    python_bin: Path,
    skip_if_v3: bool,
    logger: logging.Logger,
    dry_run: bool,
) -> str:
    task_dir = data_root / task
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task dataset not found: {task_dir}")
    meta = read_dataset_meta(task_dir)
    version = str(meta.get("codebase_version", ""))
    logger.info("task=%s path=%s codebase_version=%s frames=%s", task, task_dir, version, meta.get("total_frames"))
    if skip_if_v3 and codebase_major(version) >= 3:
        logger.info("skip convert: already %s", version)
        return "skipped_v3"
    cmd = [
        str(python_bin),
        "-m",
        "lerobot.datasets.v30.convert_dataset_v21_to_v30",
        f"--repo-id={task}",
        f"--root={data_root}",
        "--push-to-hub=false",
    ]
    logger.info("convert cmd: %s", " ".join(cmd))
    if dry_run:
        return "dry_run"
    subprocess.run(cmd, check=True)
    new_meta = read_dataset_meta(task_dir)
    logger.info("converted %s -> %s frames=%s", task, new_meta.get("codebase_version"), new_meta.get("total_frames"))
    return "converted"


def write_run_files(
    output_dir: Path,
    yaml_text: str,
    schedule: TrainSchedule,
    stamp: str,
    extra: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / f"ft_{schedule.task}_{stamp}.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    meta = {
        "schedule": schedule.as_jsonable(),
        "stamp": stamp,
        **extra,
    }
    (output_dir / f"run_meta_{stamp}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return yaml_path


def launch_train(
    cfg: dict[str, Any],
    yaml_path: Path,
    output_dir: Path,
    stamp: str,
    env: dict[str, str],
    logger: logging.Logger,
    dry_run: bool,
) -> int:
    paths = cfg["paths"]
    repo = Path(paths["repo_root"])
    train_sh = repo / paths["train_sh"]
    stdout_log = output_dir / f"train_stdout_{stamp}.log"
    cmd = [
        "bash",
        str(train_sh),
        str(paths["train_entry"]),
        str(yaml_path),
        "--data.norm_stats_file",
        str(paths["norm_stats_file"]),
        "--train.use_compile",
        bool_yaml(bool(cfg["train"].get("use_compile", False))),
    ]
    logger.info("train cmd (cwd=%s): %s", repo, " ".join(cmd))
    logger.info("stdout log: %s", stdout_log)
    if dry_run:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "inductor_cache").mkdir(exist_ok=True)
    (output_dir / "runs").mkdir(exist_ok=True)
    with stdout_log.open("wb") as f:
        proc = subprocess.run(cmd, cwd=str(repo), env=env, stdout=f, stderr=subprocess.STDOUT)
    tee_src = repo / "log.txt"
    if tee_src.is_file():
        shutil.copy2(tee_src, output_dir / f"train_sh_tee_{stamp}.log")
    logger.info("train exit_code=%s", proc.returncode)
    return proc.returncode


def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Loop RoboTwin 2.0 tasks: convert to LeRobot v3 + fine-tune.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Pipeline YAML config.")
    p.add_argument("--tasks", nargs="+", default=None, help="Override task list.")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--ckpt-root", type=str, default=None)
    p.add_argument("--venv", type=str, default=None)
    p.add_argument("--hf-home", type=str, default=None)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--global-batch-size", type=int, default=None)
    p.add_argument("--micro-batch-size", type=int, default=None)
    p.add_argument("--save-every-epoch-fraction", type=float, default=None)
    p.add_argument("--convert-only", action="store_true")
    p.add_argument("--train-only", action="store_true", help="Skip convert (still checks v3).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--use-wandb", action="store_true")
    return p.parse_args()


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.tasks:
        cfg["tasks"] = args.tasks
    if args.data_root:
        cfg["paths"]["data_root"] = str(expand_path(args.data_root))
    if args.ckpt_root:
        cfg["paths"]["ckpt_root"] = str(expand_path(args.ckpt_root))
    if args.venv:
        cfg["paths"]["venv"] = str(expand_path(args.venv))
    if args.hf_home:
        cfg["paths"]["hf_home"] = str(expand_path(args.hf_home))
        # re-expand model paths that depend on hf_home
        hf_home = cfg["paths"]["hf_home"]
        raw_model = load_yaml(Path(args.config).resolve()).get("model") or {}
        for key, value in raw_model.items():
            if isinstance(value, str) and "{hf_home}" in value:
                cfg["model"][key] = str(expand_path(format_path_template(value, hf_home=hf_home)))
    if args.num_epochs is not None:
        cfg["train"]["num_epochs"] = args.num_epochs
    if args.global_batch_size is not None:
        cfg["train"]["global_batch_size"] = args.global_batch_size
    if args.micro_batch_size is not None:
        cfg["train"]["micro_batch_size"] = args.micro_batch_size
    if args.save_every_epoch_fraction is not None:
        cfg["train"]["save_every_epoch_fraction"] = args.save_every_epoch_fraction
    if args.continue_on_error:
        cfg["runtime"]["continue_on_error"] = True
    if args.use_wandb:
        cfg["train"]["use_wandb"] = True
    return cfg


def run_one_task(
    task: str,
    cfg: dict[str, Any],
    logger: logging.Logger,
    args: argparse.Namespace,
    master_port: int,
) -> None:
    paths = cfg["paths"]
    train_cfg = cfg["train"]
    data_root = Path(paths["data_root"])
    ckpt_root = Path(paths["ckpt_root"])
    venv = Path(paths["venv"])
    python_bin = venv / "bin" / "python"
    stamp = now_stamp()
    run_name = f"{task}_{stamp}"
    output_dir = ckpt_root / run_name
    wandb_name = run_name

    logger.info("==== task=%s stamp=%s output_dir=%s ====", task, stamp, output_dir)

    if not args.train_only:
        convert_task(
            task=task,
            data_root=data_root,
            python_bin=python_bin,
            skip_if_v3=bool(cfg["convert"].get("skip_if_v3", True)),
            logger=logger,
            dry_run=args.dry_run,
        )

    meta = read_dataset_meta(data_root / task)
    if not args.dry_run and codebase_major(str(meta.get("codebase_version", "v0"))) < 3:
        raise RuntimeError(f"{task} is still {meta.get('codebase_version')}; convert did not produce v3.0")

    cuda_visible = str(cfg["runtime"].get("cuda_visible_devices") or os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    n_gpus = int(train_cfg.get("n_gpus") or 0) or detect_n_gpus(cuda_visible)
    schedule = compute_schedule(
        task=task,
        n_frames=int(meta["total_frames"]),
        global_batch_size=int(train_cfg["global_batch_size"]),
        micro_batch_size=int(train_cfg["micro_batch_size"]),
        n_gpus=n_gpus,
        num_epochs=int(train_cfg["num_epochs"]),
        save_every_epoch_fraction=float(train_cfg.get("save_every_epoch_fraction", 0.25)),
        n_episodes=meta.get("total_episodes"),
    )
    logger.info(
        "schedule %s: frames=%s steps/epoch=%s epochs=%s max_steps=%s save_steps=%s save_points=%s "
        "micro=%s acc=%s gpus=%s gbs=%s",
        task,
        schedule.n_frames,
        schedule.steps_per_epoch,
        schedule.num_epochs,
        schedule.max_steps,
        schedule.save_steps,
        schedule.save_points,
        schedule.micro_batch_size,
        schedule.gradient_accumulation_steps,
        schedule.n_gpus,
        schedule.global_batch_size,
    )

    if args.convert_only:
        logger.info("convert-only: skip training for %s", task)
        return

    repo = Path(paths["repo_root"])
    template_path = Path(paths["ft_template"])
    if not template_path.is_absolute():
        template_path = repo / template_path
    robot_config_root = paths["robot_config_root"]
    mapping = {
        "MODEL_PATH": cfg["model"]["model_path"],
        "TOKENIZER_PATH": cfg["model"]["tokenizer_path"],
        "TRAIN_PATH": str((data_root / task).resolve()) + "/",
        "ROBOT_CONFIG_ROOT": robot_config_root,
        "OUTPUT_DIR": str(output_dir),
        "MOGE_PATH": cfg["model"]["moge_path"],
        "MORGBD_PATH": cfg["model"]["morgbd_path"],
        "VIDEO_CKPT_PATH": cfg["model"]["video_ckpt_path"],
        "VIDEO_CONFIG_PATH": cfg["model"]["video_config_path"],
        "MICRO_BATCH_SIZE": str(schedule.micro_batch_size),
        "GRAD_ACC": str(schedule.gradient_accumulation_steps),
        "GLOBAL_BATCH_SIZE": str(schedule.global_batch_size),
        "NUM_TRAIN_EPOCHS": str(schedule.num_epochs),
        "MAX_STEPS": str(schedule.max_steps),
        "SAVE_STEPS": str(schedule.save_steps),
        "SAVE_EPOCHS": "999999",
        "USE_COMPILE": bool_yaml(bool(train_cfg.get("use_compile", False))),
        "USE_WANDB": bool_yaml(bool(train_cfg.get("use_wandb", False))),
        "WANDB_NAME": wandb_name,
        "VISUAL_STEPS": str(schedule.visual_steps),
        "NUM_WORKERS": str(int(train_cfg.get("num_workers", 8))),
    }
    yaml_text = render_ft_yaml(template_path, mapping)
    yaml_path = write_run_files(
        output_dir,
        yaml_text,
        schedule,
        stamp,
        extra={
            "task": task,
            "data_root": str(data_root),
            "dataset_meta": {k: meta[k] for k in ("codebase_version", "total_frames", "total_episodes", "fps") if k in meta},
            "wandb_name": wandb_name,
        },
    )
    logger.info("wrote %s", yaml_path)

    env = build_train_env(cfg, output_dir, master_port)
    rc = launch_train(cfg, yaml_path, output_dir, stamp, env, logger, args.dry_run)
    if rc != 0:
        raise RuntimeError(f"training failed for {task} (exit {rc}); see {output_dir / f'train_stdout_{stamp}.log'}")


def main() -> int:
    args = parse_cli()
    repo_root = DEFAULT_REPO_ROOT
    raw = load_yaml(Path(args.config).resolve())
    cfg = resolve_config(raw, repo_root)
    cfg = apply_cli_overrides(cfg, args)

    ckpt_root = Path(cfg["paths"]["ckpt_root"])
    ckpt_root.mkdir(parents=True, exist_ok=True)
    batch_stamp = now_stamp()
    pipeline_log = ckpt_root / f"pipeline_{batch_stamp}.log"
    logger = setup_logger(pipeline_log)
    logger.info("pipeline start stamp=%s config=%s log=%s", batch_stamp, args.config, pipeline_log)
    logger.info("data_root=%s ckpt_root=%s venv=%s hf_home=%s", cfg["paths"]["data_root"], ckpt_root, cfg["paths"]["venv"], cfg["paths"]["hf_home"])
    logger.info("tasks=%s num_epochs=%s global_batch_size=%s", cfg["tasks"], cfg["train"]["num_epochs"], cfg["train"]["global_batch_size"])

    if not cfg["tasks"]:
        logger.error("empty task list")
        return 2

    master_port = int(cfg["runtime"].get("master_port") or 62500)
    failed: list[str] = []
    for i, task in enumerate(cfg["tasks"]):
        try:
            run_one_task(task, cfg, logger, args, master_port + i)
        except Exception:
            logger.exception("task failed: %s", task)
            failed.append(task)
            if not cfg["runtime"].get("continue_on_error"):
                return 1
    if failed:
        logger.error("failed tasks: %s", failed)
        return 1
    logger.info("pipeline finished ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
