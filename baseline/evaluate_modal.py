# -*- coding: utf-8 -*-
"""Run DenseUAV evaluation (CMC/mAP and/or SDM/MA) on Modal for faster run.

Usage (from repo root):

  CMC/mAP only (no volume needed; reads from Modal Dict):
    modal run baseline/evaluate_modal.py --result-key Swinv2S_256_mode_1
    modal run baseline/evaluate_modal.py --result-key ViTS_224_mode_1 --wandb-mode online

  SDM/MA (needs Dense_GPS_ALL.txt on volume; use --root-dir-on-volume to match upload path):
    modal volume put denseuav-training data/Dense_GPS_ALL.txt /path/to/local/Dense_GPS_ALL.txt
    modal run baseline/evaluate_modal.py --eval-type distance --result-key Swinv2S_256_mode_1 --root-dir-on-volume data

  Results: printed to stdout; optional W&B logging if --wandb-mode online.
  For W&B: create a Modal secret named wandb-secret with WANDB_API_KEY (same as train_modal).

Requires: pip install modal, modal token new (or modal setup).
"""
from __future__ import print_function, division

import os
import subprocess
import sys
from pathlib import Path

import modal

VOLUME_NAME = "denseuav-training"
WORKSPACE = Path("/workspace")
BASELINE_DIR = WORKSPACE / "baseline"
DATA_VOLUME_MOUNT = "/data_vol"
RESULTS_DICT_NAME = "denseuav-test-results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "tqdm",
        "wandb",
        "matplotlib",
    )
    .add_local_dir(
        "baseline",
        remote_path=str(BASELINE_DIR),
        ignore=[".git", "__pycache__", "*.pyc", ".DS_Store", "*.mat", "*.json", "*.txt"],
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("denseuav-evaluate", image=image)


@app.function(
    gpu="B200",
    timeout=1800,
    retries=modal.Retries(max_retries=1),
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def eval_gpu_on_modal(
    result_key: str,
    dict_name: str = RESULTS_DICT_NAME,
    wandb_project: str = "denseuav-eval",
    wandb_mode: str = "online",
):
    """Run CMC/mAP evaluation on Modal. Reads from Modal Dict (fast on Modal)."""
    os.chdir(BASELINE_DIR)
    if str(BASELINE_DIR) not in sys.path:
        sys.path.insert(0, str(BASELINE_DIR))

    import numpy as np
    import torch
    import evaluate_gpu

    d = modal.Dict.from_name(dict_name, create_if_missing=False)
    result = d[result_key]
    data = {
        "query_f": np.asarray(result["query_f"]),
        "gallery_f": np.asarray(result["gallery_f"]),
        "query_label": evaluate_gpu._normalize_labels(result["query_label"]),
        "gallery_label": evaluate_gpu._normalize_labels(result["gallery_label"]),
    }

    query_feature = torch.FloatTensor(data["query_f"])
    gallery_feature = torch.FloatTensor(data["gallery_f"])
    query_label = data["query_label"]
    gallery_label = data["gallery_label"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_feature = query_feature.to(device)
    gallery_feature = gallery_feature.to(device)

    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate_gpu.evaluate(
            query_feature[i], query_label[i], gallery_feature, gallery_label
        )
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float() / len(query_label)
    recall_1 = CMC[0].item() * 100
    recall_5 = CMC[4].item() * 100
    recall_10 = CMC[9].item() * 100
    recall_top1 = CMC[round(len(gallery_label) * 0.01)].item() * 100
    ap_pct = ap / len(query_label) * 100

    print(
        "Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f"
        % (recall_1, recall_5, recall_10, recall_top1, ap_pct)
    )

    if wandb_mode != "disabled":
        try:
            import wandb
            run = wandb.init(
                project=wandb_project,
                name=result_key,
                config={"result_key": result_key, "dict_name": dict_name},
                mode=wandb_mode,
            )
            run.log(
                {
                    "Recall@1": recall_1,
                    "Recall@5": recall_5,
                    "Recall@10": recall_10,
                    "Recall@top1": recall_top1,
                    "AP": ap_pct,
                },
                step=0,
            )
            run.finish()
        except Exception as e:
            print("WandB:", e)

    return {
        "Recall@1": recall_1,
        "Recall@5": recall_5,
        "Recall@10": recall_10,
        "Recall@top1": recall_top1,
        "AP": ap_pct,
    }


@app.function(
    volumes={DATA_VOLUME_MOUNT: volume},
    timeout=3600,
    retries=modal.Retries(max_retries=1),
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def eval_distance_on_modal(
    result_key: str,
    root_dir_on_volume: str,
    dict_name: str = RESULTS_DICT_NAME,
    wandb_project: str = "sdm_eval",
    wandb_mode: str = "online",
):
    """Run SDM/MA evaluation on Modal. Needs Dense_GPS_ALL.txt at volume root_dir_on_volume."""
    root_dir = Path(DATA_VOLUME_MOUNT) / root_dir_on_volume
    config_path = root_dir / "Dense_GPS_ALL.txt"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Dense_GPS_ALL.txt not found at {config_path}. "
            "Upload to the volume then pass --root-dir-on-volume to match (e.g. path/to/local or data/DenseUAV_data)."
        )

    cmd = [
        sys.executable,
        "evaluateDistance.py",
        "--result_key", result_key,
        "--root_dir", str(root_dir),
        "--dict_name", dict_name,
        "--wandb_project", wandb_project,
        "--wandb_mode", wandb_mode,
    ]
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
        raise RuntimeError(f"evaluateDistance.py exited with code {result.returncode}")
    return {"result_key": result_key, "status": "ok"}


@app.local_entrypoint()
def main(
    result_key: str,
    eval_type: str = "gpu",
    dict_name: str = RESULTS_DICT_NAME,
    root_dir_on_volume: str = "data",
    wandb_project: str = "sdm_eval",
    wandb_mode: str = "online",
):
    """Run evaluation on Modal. --eval-type gpu (CMC/mAP) or distance (SDM/MA)."""
    if eval_type == "gpu":
        out = eval_gpu_on_modal.remote(
            result_key, dict_name, wandb_project, wandb_mode
        )
        print("Metrics:", out)
    elif eval_type == "distance":
        out = eval_distance_on_modal.remote(
            result_key, root_dir_on_volume, dict_name, wandb_project, wandb_mode
        )
        print("Done:", out)
    else:
        sys.exit(f"Unknown eval_type: {eval_type}. Use gpu or distance.")
