# -*- coding: utf-8 -*-
"""Run DenseUAV training on Modal GPU.

Usage (from repo root):
  1. Upload data once: modal volume put denseuav-training data/train /path/to/local/DenseUAV_data/train
  2. Run: modal run baseline/train_modal.py [--run-name my_run]
  3. Download checkpoints: modal volume get denseuav-training checkpoints/<run_name>/net_119.pth ./

Requires: pip install modal, then modal token new (or modal setup).
"""
from __future__ import print_function, division

import os
import subprocess
import sys
from pathlib import Path

import modal

# Volume: persistent storage for data and checkpoints. Create once, then upload data.
VOLUME_NAME = "denseuav-training"
WORKSPACE = Path("/workspace")
BASELINE_DIR = WORKSPACE / "baseline"
DATA_VOLUME_MOUNT = "/data_vol"
DATA_DIR = Path(DATA_VOLUME_MOUNT) / "data" / "train"
CHECKPOINT_DIR = Path(DATA_VOLUME_MOUNT) / "checkpoints"

# Image: PyTorch (CUDA) + baseline code. Run from repo root so "baseline" is the folder to copy.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "PyYAML",
        "thop",
        "timm",
        "numpy",
        "opencv-python-headless",
        "matplotlib",
        "Pillow",
    )
    .add_local_dir(
        "baseline",
        remote_path=str(BASELINE_DIR),
        ignore=[".git", "__pycache__", "*.pyc", ".DS_Store", "*.mat", "*.json", "*.txt"],
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

app = modal.App("denseuav-training", image=image)


def _opts_to_args(opts_path):
    """Read opts.yaml and return list of CLI args for train.py (excluding data_dir/checkpoint_dir/name)."""
    import yaml
    with open(opts_path) as f:
        d = yaml.safe_load(f)
    # Keys that map to --flag; skip name, data_dir, use_gpu, nclasses (set at runtime)
    # in_planes is not a train.py CLI arg (model derives it); skip to avoid "unrecognized arguments"
    skip = {"name", "data_dir", "use_gpu", "nclasses", "in_planes"}
    args = []
    for k, v in d.items():
        if k in skip or v is None:
            continue
        if isinstance(v, bool):
            if v:
                args.extend([f"--{k}"])
        else:
            args.extend([f"--{k}", str(v)])
    return args


@app.function(
    volumes={DATA_VOLUME_MOUNT: volume},
    gpu="B200",
    timeout=86400,  # 24h max per run
    retries=modal.Retries(max_retries=2),
)
def train_on_modal(run_name: str):
    """Run baseline/train.py inside the container with data and checkpoints on the Volume."""
    opts_path = BASELINE_DIR / "opts.yaml"
    if not opts_path.exists():
        raise FileNotFoundError(f"opts.yaml not found at {opts_path}")

    extra_args = _opts_to_args(opts_path)
    cmd = (
        [sys.executable, "train.py",
         "--data_dir", str(DATA_DIR),
         "--checkpoint_dir", str(CHECKPOINT_DIR),
         "--name", run_name]
        + extra_args
    )
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print("Running:", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(BASELINE_DIR),
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if result.returncode != 0:
        raise RuntimeError(f"train.py exited with code {result.returncode}")
    # Persist writes to the Volume
    volume.commit()
    print(f"Checkpoints saved under Volume '{VOLUME_NAME}' at checkpoints/{run_name}/")


@app.local_entrypoint()
def main(run_name: str = "baseline_modal"):
    """Entrypoint: run training on Modal GPU. Use --run-name to set experiment name."""
    print(f"Starting training on Modal GPU (run_name={run_name}).")
    print(f"Data dir on Volume: {DATA_DIR}, Checkpoints: {CHECKPOINT_DIR}")
    # spawn + get() is recommended for long jobs (avoids 24h call expiry)
    train_on_modal.spawn(run_name).get()
    print("Done. Download checkpoints with:")
    print(f"  modal volume get {VOLUME_NAME} checkpoints/{run_name}/<file> ./")
