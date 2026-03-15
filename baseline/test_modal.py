# -*- coding: utf-8 -*-
"""Run DenseUAV feature extraction on Modal GPU and store results in Modal Dict.

Usage (from repo root):
  Test data and model/checkpoint must already be on the volume (same as train_modal).
  Set run_name, checkpoint / checkpoint_baseline_path, test_dir, batchsize in code or via CLI.

  Checkpoint can be:
  - On volume: checkpoints/<run_name>/<checkpoint> (leave checkpoint_baseline_path empty).
  - In workspace (bundled with baseline): set checkpoint_baseline_path e.g. "imacs/Swinv2S_256.pth".

  For 3 different architectures (Method 1 – no code change required): edit opts.yaml and
  run_name / checkpoint_baseline_path in main(), then run this script 3 times. After each change to opts.yaml,
  make a minor change in this file (e.g. add/remove a comment) so that Modal rebuilds the image and container
  with the updated opts. Alternatively, set opts_volume_path to load opts from the volume (no rebuild required).

  Results are stored in Modal Dict "denseuav-test-results" under key "{run_name}_mode_1".
  To read later: modal.Dict.from_name("denseuav-test-results")[key] gives the result dict
  (gallery_f, gallery_label, gallery_path, query_f, query_label, query_path).

Requires: pip install modal, then modal token new (or modal setup).
"""
from __future__ import print_function, division

import os
import sys
from argparse import Namespace
from pathlib import Path

import modal
import yaml

# Same volume and paths as train_modal
VOLUME_NAME = "denseuav-training"
WORKSPACE = Path("/workspace")
BASELINE_DIR = WORKSPACE / "baseline"
DATA_VOLUME_MOUNT = "/data_vol"
CHECKPOINT_DIR = Path(DATA_VOLUME_MOUNT) / "checkpoints"

# Image: PyTorch + baseline deps (no wandb needed for test). Include scipy for test.py import.
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
        "scipy",
    )
    .add_local_dir(
        "baseline",
        remote_path=str(BASELINE_DIR),
        ignore=[".git", "__pycache__", "*.pyc", ".DS_Store", "*.mat", "*.json", "*.txt"],
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("denseuav-test", image=image)

# Dict name for storing extraction results (persist, key = f"{run_name}_mode_1")
RESULTS_DICT_NAME = "denseuav-test-results"


def _build_opt_from_opts_and_overrides(opts_path, checkpoint_path, test_dir_path, run_name, batchsize):
    """Read opts.yaml into a namespace and override with test arguments (for run_extraction)."""
    with open(opts_path) as f:
        config = yaml.safe_load(f)
    opt = Namespace()
    for k, v in config.items():
        if v is None:
            continue
        setattr(opt, k, v)
    opt.checkpoint = str(checkpoint_path)
    opt.test_dir = str(test_dir_path)
    opt.name = run_name
    opt.batchsize = batchsize
    return opt


@app.function(
    volumes={DATA_VOLUME_MOUNT: volume},
    gpu="B200",
    timeout=3600,
    retries=modal.Retries(max_retries=2),
)
def extract_on_modal(
    run_name: str,
    checkpoint: str,
    test_dir: str,
    batchsize: int = 128,
    checkpoint_baseline_path: str | None = None,
    opts_volume_path: str | None = None,
):
    """Extract query/gallery embeddings (mode 1: drone->satellite) and store in Modal Dict.

    test_dir is the path inside the volume to the test root (e.g. 'data/test') with
    query_drone and gallery_satellite. Checkpoint: if checkpoint_baseline_path is set
    (e.g. 'imacs/Swinv2S_256.pth'), use that path under BASELINE_DIR (workspace);
    otherwise use volume path checkpoints/<run_name>/<checkpoint>.
    If opts_volume_path is set (e.g. 'opts_swin.yaml'), read opts from the volume at
    that path; otherwise use BASELINE_DIR/opts.yaml (from the image).
    """
    if opts_volume_path is not None:
        opts_path = Path(DATA_VOLUME_MOUNT) / opts_volume_path
    else:
        opts_path = BASELINE_DIR / "opts.yaml"
    if not opts_path.exists():
        raise FileNotFoundError(f"opts not found at {opts_path}")

    if checkpoint_baseline_path is not None:
        checkpoint_path = BASELINE_DIR / checkpoint_baseline_path
    else:
        checkpoint_path = Path(DATA_VOLUME_MOUNT) / "checkpoints" / run_name / checkpoint
    test_dir_path = Path(DATA_VOLUME_MOUNT) / test_dir
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    if not test_dir_path.exists():
        raise FileNotFoundError(f"Test dir not found at {test_dir_path}")

    opt = _build_opt_from_opts_and_overrides(
        opts_path, checkpoint_path, test_dir_path, run_name, batchsize
    )

    os.chdir(BASELINE_DIR)
    if str(BASELINE_DIR) not in sys.path:
        sys.path.insert(0, str(BASELINE_DIR))

    import test as test_module
    result = test_module.run_extraction(opt)

    result_dict = modal.Dict.from_name(RESULTS_DICT_NAME, create_if_missing=True)
    key = f"{run_name}_mode_1"
    result_dict[key] = result
    print(f"Stored result under Dict '{RESULTS_DICT_NAME}', key '{key}'.")


@app.local_entrypoint()
def main(
    # old code: defaults for Swinv2S_256 (run 1)
    # run_name: str = "Swinv2S_256",
    # checkpoint: str = "Swinv2S_256.pth",
    # checkpoint_baseline_path: str | None = "imacs/Swinv2S_256.pth",
    # old code: defaults for Convnext-T (run 2)
    # run_name: str = "Convnext_T",
    # checkpoint: str = "Convnext_T.pth",
    # checkpoint_baseline_path: str | None = "imacs/Convnext_T.pth",
    # new code: defaults for ViTS-224 (run 3); opts.yaml đã set backbone ViTS-224, head SingleBranch, in_planes 384, h/w 224
    run_name: str = "ViTS_224",
    checkpoint: str = "ViTS_224.pth",
    test_dir: str = "data/test",
    batchsize: int = 128,
    checkpoint_baseline_path: str | None = "imacs/ViTS_224.pth",
    opts_volume_path: str | None = None,
):
    """Entrypoint: run extraction on Modal and store in Dict. Change args in code or pass via CLI.
    For 3 architectures: run 3 times with different run_name, checkpoint_baseline_path (and
    optionally opts_volume_path if you put per-arch opts on the volume).
    """
    print(f"Extracting on Modal: run_name={run_name}, test_dir={test_dir}, batchsize={batchsize}")
    if opts_volume_path:
        print(f"Opts path (volume): {opts_volume_path}")
    else:
        print(f"Opts path (image): baseline/opts.yaml")
    if checkpoint_baseline_path:
        print(f"Checkpoint path (workspace): baseline/{checkpoint_baseline_path}")
    else:
        print(f"Checkpoint path (volume): checkpoints/{run_name}/{checkpoint}")
    print(f"Test data path on volume: {test_dir} (must contain query_drone, gallery_satellite)")
    extract_on_modal.remote(
        run_name, checkpoint, test_dir, batchsize, checkpoint_baseline_path, opts_volume_path
    )
    print(f"Done. Result stored in Modal Dict '{RESULTS_DICT_NAME}' under key '{run_name}_mode_1'.")
