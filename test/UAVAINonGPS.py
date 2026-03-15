import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    logo = mo.center(mo.image(src="resources/logo.png", alt="DenseUAV Logo", width=1000))
    logo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # DenseUAV Baseline

    C.B Nguyen, N.H Dang

    **Supervisor**: M.Sc. Nguyen Van Gia Thinh

    Welcome to the DenseUAV baseline analysis. This notebook provides comprehensive
    documentation, analysis, testing and improvement if possible on the UAV
    self-positioning without GPS by Deep Learning method.

    **Quick Navigation**
    - [Overview of Marimo](#overview-about-marimo-platform) - Introduces the platform used for testing, developing and presenting experiment results.
    - [Code analysis](#code-analysis-documentation) - Analyzes the code structure of the author's baseline.
    - [Experiment Processing](#experiment-processing) - Present our experiment processing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Overview about Marimo platform

    **Marimo** is a next-generation Python notebook that addresses many limitations of traditional Jupyter notebooks. Unlike Jupyter, Marimo is designed as a **reactive notebook** where cells automatically update when their dependencies change, creating a more reliable and interactive development environment.

    Key Advantages Over Traditional Notebooks

    1. **Reactive Execution**
    - Cells automatically re-run when upstream dependencies change
    - No more stale outputs or hidden state issues
    - Guarantees reproducible results

    2. **No Hidden State**
    - Deterministic execution order based on variable dependencies
    - Eliminates the common "it works in my notebook but not yours" problem
    - Variables are automatically tracked and managed

    3. **Interactive UI Elements**
    - Built-in widgets and interactive components
    - Real-time updates without manual cell execution
    - Rich HTML and multimedia support

    4. **Git-Friendly**
    - Notebooks are stored as clean Python files (`.py`)
    - Better version control and collaboration
    - No more messy JSON diffs

    5. **Modern Developer Experience**
    - Built-in formatting and linting
    - Code completion and error checking
    - Integrated package management
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Code Analysis & Documentation

    This document analyzes the author's official baseline from the [DenseUAV](https://github.com/Dmmm1997/DenseUAV) GitHub repository, located in the `baseline/` folder.

    ---

    ## Table of Contents

    1. [Overview](#overview)
    2. [Directory Structure](#directory-structure)
    3. [Model Architecture](#model-architecture)
    4. [Dataset & Data Loading](#dataset--data-loading)
    5. [Loss Functions](#loss-functions)
    6. [Training Pipeline](#training-pipeline)
    7. [Inference & Testing](#inference--testing)
    8. [Evaluation Metrics](#evaluation-metrics)
    9. [Configuration Reference (`opts.yaml`)](#configuration-reference-optsyaml)
    10. [Pretrained Checkpoint](#pretrained-checkpoint)
    11. [Benchmark Results](#benchmark-results)
    12. [Dependencies](#dependencies)
    13. [Quick Start](#quick-start)

    ---

    ## Overview

    The baseline implements **cross-view geo-localization** between UAV-view (drone) images and satellite-view images. Given a query drone image, the goal is to retrieve the most geographically similar satellite image (or vice versa) from a gallery, without using GPS.

    The system is trained on the **DenseUAV** dataset (2022 version, 2256 location classes) and evaluated using both standard retrieval metrics (CMC, mAP) and spatial accuracy metrics (SDM@K, MA@K).

    **Key design decisions:**

    - **Shared backbone**: both drone and satellite images pass through the same feature extractor
    - **Modular architecture**: backbones and heads are independently configurable via `opts.yaml`
    - **Multi-loss training**: combines classification loss, triplet loss, and KL divergence (mutual learning)
    - **Mixed precision**: autocast for faster CUDA training

    ---

    ## Directory Structure

    ```
    baseline/
    ├── train.py                    # Main training entry point
    ├── test.py                     # Feature extraction / inference
    ├── evaluate_gpu.py             # Compute CMC & mAP on GPU
    ├── evaluateDistance.py         # Compute SDM@K and MA@K (GPS-based)
    ├── opts.yaml                   # Default hyperparameter configuration
    ├── train_test_local.sh         # Convenience shell script (train → test → evaluate)
    │
    ├── net_119.pth                 # Pretrained checkpoint (epoch 119)
    ├── pytorch_result_1.mat        # Saved features from test.py
    ├── query_name.txt              # Query image file paths
    ├── gallery_name.txt            # Gallery image file paths
    ├── SDM@K(1,100).json           # Precomputed SDM results
    ├── MA@K(1,100)                 # Precomputed MA results
    │
    ├── models/
    │   ├── taskflow.py             # Top-level Model class (backbone + head composition)
    │   ├── model.py                # Legacy model implementations
    │   ├── __init__.py
    │   ├── Backbone/
    │   │   └── backbone.py         # Backbone factory (ViT, ResNet, Swin, etc.)
    │   └── Head/
    │       ├── head.py             # Head factory (dispatches to implementations below)
    │       ├── SingleBranch.py     # Single-branch CLS-token head
    │       ├── GeM.py              # Generalized Mean Pooling head
    │       ├── LPN.py              # Local Pattern Network (multi-part)
    │       ├── FSRA.py             # Feature Space Regularization Attention
    │       ├── NetVLAD.py          # NetVLAD aggregation head
    │       ├── NeXtVLAD.py         # NeXtVLAD aggregation head
    │       └── utils.py            # Shared utilities (ClassBlock, Pooling)
    │
    ├── datasets/
    │   ├── make_dataloader.py      # Dataset factory + augmentation pipeline
    │   ├── Dataloader_University.py# Core dataset class for DenseUAV pairs
    │   ├── queryDataset.py         # Query-time dataset (test transforms)
    │   └── autoaugment.py          # AutoAugment policy implementations
    │
    ├── losses/
    │   ├── loss.py                 # Main loss module (combines all losses)
    │   ├── cal_loss.py             # Legacy loss calculation helpers
    │   ├── TripletLoss.py          # Triplet loss variants
    │   ├── FocalLoss.py            # Focal loss
    │   ├── ArcfaceLoss.py          # ArcFace loss
    │   └── __Init__.py
    │
    ├── optimizers/
    │   └── make_optimizer.py       # Optimizer factory (SGD with per-group LR)
    │
    └── tool/
        ├── utils.py                # General utility functions
        ├── get_model_flops_params.py  # FLOPs / parameter counter
        ├── get_property.py         # Model property inspection
        ├── clear_model.py          # Checkpoint cleanup utilities
        └── mount_dist.sh           # Distributed training mount script
    ```

    ---

    ### Model Architecture

    #### Top-Level: `Model` (`models/taskflow.py`)

    ```
    Model
    ├── backbone   → make_backbone(opt)   # Shared feature extractor
    └── head       → make_head(opt)       # Aggregation + classifier
    ```

    Forward pass:

    ```python
    def forward(self, drone_image, satellite_image):
        drone_features    = self.backbone(drone_image)
        drone_res         = self.head(drone_features)      # (logits, feature)

        satellite_features = self.backbone(satellite_image)
        satellite_res      = self.head(satellite_features) # (logits, feature)

        return drone_res, satellite_res
    ```

    Either input can be `None` to run single-view inference.

    ---

    #### Backbones (`models/Backbone/backbone.py`)

    All backbones are loaded from `timm` and return a feature tensor plus an `output_channel` attribute consumed by the head.

    | Key                  | Architecture                | Output Channels |
    | -------------------- | --------------------------- | --------------- |
    | `ViTS-224`           | ViT-Small / patch16 / 224px | 384             |
    | `ViTS-384`           | ViT-Small / patch16 / 384px | 384             |
    | `ViTB-224`           | ViT-Base / patch16 / 224px  | 768             |
    | `DeitS-224`          | DeiT-Small / 224px          | 384             |
    | `DeitB-224`          | DeiT-Base / 224px           | 768             |
    | `ResNet50`           | ResNet-50                   | 2048            |
    | `EfficientNet-B2/B3` | EfficientNet                | varies          |
    | `VGG16`              | VGG-16                      | 512             |
    | `Swin-*`             | Swin Transformer variants   | varies          |
    | `ConvNeXt-*`         | ConvNeXt variants           | varies          |
    | `PVTv2-*`            | PVTv2 variants              | varies          |

    ---

    #### Head Architectures (`models/Head/`)

    `SingleBranch`

    - Uses the CLS token output (transformers) or global average pooling (CNNs)
    - Single FC classifier branch
    - Default head in `opts.yaml`

    `GeM` — Generalized Mean Pooling

    - Applies GeM pooling over spatial patch tokens: \( f = \left(\frac{1}{N}\sum_i x_i^p\right)^{1/p} \)
    - The pooling exponent `p` is a learnable parameter
    - Followed by a FC classifier

    `LPN` — Local Pattern Network

    - Divides spatial tokens into concentric non-overlapping regions
    - One global branch + `block` local branches (controlled by `--block`)
    - At inference, stacks all branch features into a single descriptor
    - Designed for geo-localization where local appearance varies by region

    `FSRA` — Feature Space Regularization Attention

    - Computes an attention heatmap over patch tokens
    - Selects and groups high-attention tokens into `block` sub-descriptors
    - Global classifier + one local classifier per block
    - Improves discrimination via part-based features

    `NetVLAD` / `NeXtVLAD`

    - VLAD-based aggregation over patch tokens
    - Learns cluster centers; each token is soft-assigned to clusters
    - Encodes residuals relative to cluster centers into a fixed-size descriptor

    ---

    ### Dataset & Data Loading

    #### Expected Directory Layout

    ```
    data_2022/
    ├── train/
    │   ├── satellite/
    │   │   └── {class_id}/  *.jpg      # One satellite image per location
    │   └── drone/
    │       └── {class_id}/  *.jpg      # Multiple drone images per location
    └── test/
        ├── query_drone/
        ├── query_satellite/
        ├── gallery_drone/
        └── gallery_satellite/
    ```

    - **2256 location classes** in training split
    - Each class has one canonical satellite patch and multiple drone images

    #### `Dataloader_University` (`datasets/Dataloader_University.py`)

    - Loads paired `(satellite_img, drone_img)` tensors with a shared integer label
    - `Sampler_University`: custom sampler that repeats samples per class (`sample_num` times) to balance class frequency
    - `train_collate_fn`: assembles batches with consistent satellite/drone grouping

    #### Augmentation Pipeline (`datasets/make_dataloader.py`)

    | View            | Augmentation                                                                                        |
    | --------------- | --------------------------------------------------------------------------------------------------- |
    | **UAV / Drone** | Random rotation (`rr: uav`), Random affine, Random erasing (`erasing_p=0.3`), optional color jitter |
    | **Satellite**   | `RotateAndCrop`, Random affine (`ra: satellite`), Random erasing (`re: satellite`)                  |
    | **Both**        | Resize to (h×w), Random horizontal flip, ImageNet normalization                                     |

    The view-specific augmentations are specified in `opts.yaml` via `rr`, `ra`, `re` flags.

    ---

    ### Loss Functions

    Training minimizes a **three-component loss**:

    ```
    L_total = L_cls + L_triplet + L_KL
    ```

    #### 1. Classification Loss (`cls_loss: CELoss`)

    Standard cross-entropy over `nclasses=2256` classes. Can be swapped for `FocalLoss` (α=0.25, γ=2).

    #### 2. Triplet / Feature Loss (`feature_loss: WeightedSoftTripletLoss`)

    Metric learning loss applied on the L2-normalized feature embeddings.

    | Option                    | Description                                                  |
    | ------------------------- | ------------------------------------------------------------ |
    | `WeightedSoftTripletLoss` | Soft triplet with exponential weighting (α=10) — **default** |
    | `HardMiningTripletLoss`   | Hard negative mining within the batch                        |
    | `TripletLoss`             | Standard triplet loss with fixed margin                      |

    #### 3. KL Divergence Loss (`kl_loss: KLLoss`)

    Mutual learning between drone and satellite branches — the KL divergence between the softmax output distributions encourages the two views to produce consistent predictions:

    ```
    L_KL = KL(P_drone || P_satellite) + KL(P_satellite || P_drone)
    ```

    ---

    ### Training Pipeline

    **Entry point:** `train.py`

    1. Load config from `opts.yaml` (overridable via CLI args)
    2. Build dataloaders (paired satellite+drone batches, size `batchsize=16`)
    3. Instantiate `Model(opt)` — loads from `load_from` checkpoint if specified
    4. Optimizer: **SGD** with backbone LR = `lr × 0.3`, head LR = `lr`
    5. LR schedule: `MultiStepLR` with milestones at epochs 70 and 110, γ=0.1
    6. Mixed precision via `torch.cuda.amp.autocast` (`autocast: true`)
    7. Warm-up epochs (if `warm_epoch > 0`) with linearly scaled LR
    8. Checkpoints saved every 10 epochs after epoch 110

    **Run training:**

    ```bash
    python train.py \
      --name my_experiment \
      --data_dir /path/to/data_2022/train \
      --backbone ViTS-224 \
      --head SingleBranch \
      --batchsize 16 \
      --num_epochs 120 \
      --lr 0.003
    ```

    Checkpoints are saved to `checkpoints/{name}/`.

    ---

    ### Inference & Testing

    **Entry point:** `test.py`

    1. Loads a trained checkpoint (`--checkpoint net_119.pth`)
    2. Iterates over query and gallery sets (drone or satellite)
    3. Supports **two retrieval modes**:
       - Mode 1: drone query → satellite gallery
       - Mode 2: satellite query → drone gallery
    4. **Flip augmentation**: features of the original and horizontally-flipped image are averaged
    5. Features are **L2-normalized**
    6. Outputs `pytorch_result_1.mat` containing query/gallery features and labels

    **Run testing:**

    ```bash
    cd checkpoints/my_experiment
    python test.py \
      --name my_experiment \
      --test_dir /path/to/data_2022/test \
      --checkpoint net_119.pth \
      --gpu_ids 0
    ```

    ---

    ### Evaluation Metrics

    #### Standard Retrieval: `evaluate_gpu.py`

    Operates on the `.mat` feature file from `test.py`.

    | Metric         | Description                                          |
    | -------------- | ---------------------------------------------------- |
    | **Rank-1**     | % of queries with correct match at position 1        |
    | **Rank-5**     | % of queries with correct match in top 5             |
    | **Rank-10**    | % of queries with correct match in top 10            |
    | **Rank-top1%** | % of queries with correct match in top 1% of gallery |
    | **mAP**        | Mean Average Precision across all queries            |

    Junk images (label = -1) are excluded from the ranking.

    ---

    #### Distance-Based: `evaluateDistance.py`

    Uses GPS coordinates from `Dense_GPS_ALL.txt` to measure physical proximity.

    SDM@K — Spatial Distance Metric

    Measures weighted average of geographic proximity of the top-K retrieved results:

    ```
    SDM@K = Σ_k [ w_k · exp(−d_k · M) ] / Σ_k w_k
    ```

    where `d_k` is the geographic distance (metres) to the k-th result and `w_k` is a rank-based weight. Higher is better, maximum = 1.0.

    MA@K — Meter Accuracy

    Fraction of queries for which the **top-1** retrieved result is within **K metres** of the true location (Haversine distance). Reported as a curve over K = 1…100 m.

    **Run evaluation:**

    ```bash
    python evaluate_gpu.py         # CMC / mAP
    python evaluateDistance.py     # SDM@K / MA@K
    ```

    ---

    ### Configuration Reference (`opts.yaml`)

    | Parameter        | Default                   | Description                          |
    | ---------------- | ------------------------- | ------------------------------------ |
    | `backbone`       | `ViTS-224`                | Backbone architecture key            |
    | `head`           | `SingleBranch`            | Head architecture key                |
    | `batchsize`      | `16`                      | Training batch size                  |
    | `h` / `w`        | `224` / `224`             | Input image resolution               |
    | `lr`             | `0.003`                   | Base learning rate (head)            |
    | `num_epochs`     | `120`                     | Total training epochs                |
    | `num_bottleneck` | `512`                     | Feature embedding dimension          |
    | `in_planes`      | `384`                     | Backbone output channels (auto-set)  |
    | `nclasses`       | `2256`                    | Number of location classes           |
    | `cls_loss`       | `CELoss`                  | Classification loss type             |
    | `feature_loss`   | `WeightedSoftTripletLoss` | Metric learning loss type            |
    | `kl_loss`        | `KLLoss`                  | Mutual learning loss type            |
    | `block`          | `1`                       | Number of parts (for LPN / FSRA)     |
    | `share`          | `true`                    | Share backbone weights between views |
    | `autocast`       | `true`                    | Enable mixed precision (AMP)         |
    | `droprate`       | `0.5`                     | Dropout rate in classifier           |
    | `erasing_p`      | `0.3`                     | Random erasing probability           |
    | `rr`             | `uav`                     | Random rotation target view          |
    | `ra`             | `satellite`               | Random affine target view            |
    | `re`             | `satellite`               | Random erasing target view           |
    | `cj`             | `no`                      | Color jitter toggle                  |
    | `sample_num`     | `1`                       | Repeat samples per class per epoch   |
    | `warm_epoch`     | `0`                       | LR warm-up epochs                    |
    | `DA`             | `false`                   | Domain adaptation toggle             |
    | `gpu_ids`        | `0`                       | GPU device index                     |
    | `num_worker`     | `8`                       | DataLoader worker count              |
    | `data_dir`       | _(path)_                  | Path to training data                |
    | `load_from`      | `no`                      | Path to checkpoint for fine-tuning   |

    ---

    ### Pretrained Checkpoint

    A pretrained model is included at `baseline/net_119.pth` (epoch 119).

    **Configuration used for this checkpoint** (from `opts.yaml`):

    - Backbone: `ViTS-224` (ViT-Small, 224×224)
    - Head: `SingleBranch`
    - Losses: `CELoss` + `WeightedSoftTripletLoss` (α=10) + `KLLoss`
    - Experiment name: `Loss_Experiment-CELoss-WeightedSoftTripletLoss_alpha10-KLLoss`

    ---

    ### Benchmark Results

    Results on the **DenseUAV test set** (drone → satellite retrieval), produced by the included checkpoint.

    #### SDM@K (Spatial Distance Metric)

    | K   | SDM@K |
    | --- | ----- |
    | 1   | 0.865 |
    | 5   | 0.804 |
    | 10  | 0.685 |
    | 20  | 0.509 |
    | 50  | 0.299 |
    | 100 | 0.187 |

    > SDM@1 = **0.865** indicates strong top-1 geographic proximity.

    #### MA@K (Meter Accuracy at K metres)

    | Threshold (m) | MA@K  |
    | ------------- | ----- |
    | 1 m           | 0.830 |
    | 10 m          | 0.830 |
    | 20 m          | 0.873 |
    | 23 m          | 0.905 |
    | 60 m          | 0.942 |
    | 100 m         | 0.958 |

    > **83.0%** of queries are localized within **1 metre** of ground truth at top-1.
    > **95.8%** are within **100 metres**.

    ---

    ### Dependencies

    **Core:**

    ```
    torch          # PyTorch (CUDA recommended)
    torchvision
    timm           # Pretrained models (ViT, Swin, ConvNeXt, etc.)
    numpy
    scipy
    Pillow
    opencv-python
    pyyaml
    tqdm
    matplotlib
    ```

    **Optional:**

    ```
    thop           # FLOPs/parameter counting (get_model_flops_params.py)
    resnest        # ResNeSt backbone support
    ```

    Install with:

    ```bash
    pip install torch torchvision timm numpy scipy Pillow opencv-python pyyaml tqdm matplotlib
    ```

    ---

    ### Quick Start

    ```bash
    # 1. Train
    python baseline/train.py \
      --name my_run \
      --data_dir /path/to/DenseUAV/data_2022/train

    # 2. Extract features
    cd checkpoints/my_run
    python ../../baseline/test.py \
      --name my_run \
      --test_dir /path/to/DenseUAV/data_2022/test \
      --checkpoint net_119.pth

    # 3. Evaluate
    python ../../baseline/evaluate_gpu.py          # CMC / mAP
    python ../../baseline/evaluateDistance.py      # SDM@K / MA@K
    ```

    Or use the convenience script:

    ```bash
    bash baseline/train_test_local.sh
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experiment planning
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Real-time check
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    phase1 = mo.center(mo.image(src="../test/uml/phase1.png", alt="DenseUAV Logo"))
    phase1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Performance Check
    """)
    return


@app.cell
def _(mo):
    phase2 = mo.center(mo.image(src="../test/uml/phase2.png", alt="DenseUAV Logo"))
    phase2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note for Code Fix
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Baseline mode 2 error
        - `dataloaders` (`baseline/test.py` line 117-127) requires 4 folders, including:
            - [x] `gallery_satellite`
            - [ ] `gallery_drone`
            - [ ] `query_satellite`
            - [x] `query_drone`
        - However, the dataset provided by the author only has 2 folders: `gallery_satellite` and `query_drone`. Currently missing 2 folders: `gallery_drone` and `query_satellite`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load embedding vector database
    """)
    return


@app.cell
def _():
    import scipy.io
    data = scipy.io.loadmat('../baseline/pytorch_result_1.mat')
    data
    return data, scipy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Satellite gallery
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Satellite gallery feature:
    """)
    return


@app.cell
def _(data):
    data['gallery_f']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Satellite gallery label:
    """)
    return


@app.cell
def _(data):
    data['gallery_label']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Query
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Query feature:
    """)
    return


@app.cell
def _(data):
    data['query_f']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Query label:
    """)
    return


@app.cell
def _(data):
    data['query_label']
    return


@app.cell
def _(data):
    # Check Junk label
    gl = data['gallery_label'][0]
    print((gl == -1).sum())   # Number of image have label -1
    return


@app.cell
def _(mo):
    mo.md(f"""
    # Evaluation metrics
    """)
    return


@app.cell(hide_code=True)
def _():
    import os, sys

    try:
        _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        _repo_root = os.path.abspath(os.path.join(os.getcwd(), ".." if os.path.basename(os.getcwd()) == "test" else "."))

    if _repo_root not in sys.path:
        sys.path.append(_repo_root)

    baseline_dir = os.path.join(_repo_root, "baseline")
    mat_file_path = os.path.join(baseline_dir, "pytorch_result_1.mat")

    # Change to baseline directory so the script can find pytorch_result_1.mat
    os.chdir(baseline_dir)
    return baseline_dir, os, sys


@app.cell(hide_code=True)
def _():
    from baseline import evaluate_gpu

    return (evaluate_gpu,)


@app.cell(hide_code=True)
def _():
    import torch
    import numpy as np

    return np, torch


@app.cell(hide_code=True)
def recall(evaluate_gpu, scipy, torch):
    # 1. Load features từ .mat
    result = scipy.io.loadmat("pytorch_result_1.mat")
    query_feature  = torch.FloatTensor(result["query_f"])          # shape: (Nq, D)
    query_label    = result["query_label"][0]                      # shape: (Nq,)
    gallery_feature = torch.FloatTensor(result["gallery_f"])       # shape: (Ng, D)
    gallery_label   = result["gallery_label"][0]                   # shape: (Ng,)

    # 2. Chọn device (GPU nếu có, không thì CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_feature   = query_feature.to(device)
    gallery_feature = gallery_feature.to(device)

    # 3. Vòng lặp đánh giá
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate_gpu.evaluate(
            query_feature[i],
            int(query_label[i]),
            gallery_feature,
            gallery_label
        )
        if CMC_tmp[0] == -1:
            continue
        CMC += CMC_tmp
        ap += ap_tmp

    CMC = CMC.float() / len(query_label)
    mAP = ap / len(query_label)
    return (CMC,)


@app.cell(hide_code=True)
def _(CMC, np):
    import matplotlib.pyplot as plt

    fig, _ax = plt.subplots(1, 1, figsize=(16, 6))

    # Plot 1: CMC Curve (Cumulative Matching Characteristic)
    cmc_np = CMC.cpu().numpy()
    ranks = np.arange(1, len(cmc_np) + 1)

    _ax.plot(ranks, cmc_np * 100, color="#2196F3", linewidth=1)
    _ax.set_title("CMC Curve (Cumulative Matching Characteristic)", fontsize=14, fontweight="bold")
    _ax.set_xlabel("Rank", fontsize=12)
    _ax.set_ylabel("Recognition Rate (%)", fontsize=12)
    _ax.set_xlim(1, min(100, len(cmc_np)))
    _ax.set_ylim(0, 105)
    _ax.grid(True, alpha=0.3)

    # Annotate key ranks
    for k in [1, 5, 10]:
        if k <= len(cmc_np):
            _ax.axhline(y=cmc_np[k - 1] * 100, color="gray", linestyle="--", alpha=0.4)
            _ax.scatter([k], [cmc_np[k - 1] * 100], color="#F44336", s=60, zorder=5)
            _ax.annotate(
                f"R@{k}: {cmc_np[k-1]*100:.2f}%",
                xy=(k, cmc_np[k - 1] * 100),
                xytext=(k + 5, cmc_np[k - 1] * 100 - 5),
                fontsize=10,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#F44336"),
                color="#F44336",
            )

    fig.tight_layout()
    plt.gca()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CMC Curve (Cumulative Matching Characteristic)

    The **CMC curve** shows how retrieval accuracy improves as we consider more candidates in the ranked list.

    - **X-axis (Rank):** The number of top-K retrieved gallery images considered.
    - **Y-axis (Recognition Rate %):** The percentage of queries for which the correct match appears within the top-K results.
    - **Key annotated points (R@1, R@5, R@10):** These are the standard recall benchmarks:
      - **R@1** — How often the *very first* retrieved satellite image is the correct one.
      - **R@5** — How often the correct match is among the top 5 results.
      - **R@10** — How often the correct match is among the top 10 results.
    - **Interpretation:** A curve that rises steeply and plateaus near 100% early indicates a strong model. The closer R@1 is to 100%, the better the model is at retrieving the exact correct satellite image on its first attempt.

    ## Evaluation Results

    | Metric | Value |
    |--------|-------|
    | **Recall@1** | {CMC[0] * 100:.2f}% |
    | **Recall@5** | {CMC[4] * 100:.2f}% |
    | **Recall@10** | {CMC[9] * 100:.2f}% |
    | **Recall@top1%** | {CMC[round(len(gallery_label) * 0.01)] * 100:.2f}% |
    | **mAP** | {mAP * 100:.2f}% |

    ## Summary

    | Plot | Scope | Purpose |
    |------|-------|---------|
    | **CMC Curve** | Aggregated over **all** queries | Shows overall retrieval performance across ranks |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Real-time testing
    """)
    return


@app.cell(hide_code=True)
def _(baseline_dir, sys):
    import importlib

    # Ensure baseline directory is on sys.path so that submodules like evaluate_gpu can be found
    if baseline_dir not in sys.path:
        sys.path.insert(0, baseline_dir)
    return


@app.cell
def _():
    from baseline.real_time_chibang import UAVnonGPS

    return (UAVnonGPS,)


@app.cell
def _(scipy):
    embedding_gallery = scipy.io.loadmat('../baseline/pytorch_result_real_time_chibang_1.mat')
    embedding_gallery
    return


@app.cell
def _(UAVnonGPS):
    uav = UAVnonGPS(
        EmbeddingDatabase="../baseline/pytorch_result_real_time_chibang_1.mat",
        model="net_119.pth",
        batchsize=128,
        mode=1,
    )
    return (uav,)


@app.cell
def _(plt):
    from PIL import Image

    _query_img = Image.open("/Users/chibangnguyen/ayai/UAV/denseUAV_baseline/test/real_time_query/2256_H80.JPG")

    fig_query, _ax_query = plt.subplots(1, 1, figsize=(8, 8))
    _ax_query.imshow(_query_img)
    _ax_query.set_title("Query Image: 2256_H80.JPG", fontsize=14, fontweight="bold")
    _ax_query.axis("off")
    fig_query.tight_layout()
    plt.gca()
    return (Image,)


@app.cell
def query_feature_test(baseline_dir, os, uav):
    _query_image_path = os.path.join(os.path.dirname(baseline_dir), "test", "real_time_query", "2256_H80.JPG")
    q_emb = uav.query_embedding(_query_image_path)
    q_emb
    return (q_emb,)


@app.cell
def _(q_emb, uav):
    ranking = uav.ranking_by_similarity(q_emb)

    for item in ranking[:10]:
        print(item["rank"], item["score"], item["gallery_path"], item["gallery_label"])
    return (ranking,)


@app.cell
def _(mo, os):
    import polars as pl

    def show_top_k_details(ranking, K=5, query_label=2256):
        _top_k_df = pl.DataFrame({
            "Rank": [item["rank"] for item in ranking[:K]],
            "Similarity Score": [round(item["score"], 6) for item in ranking[:K]],
            "Gallery Label": [int(item["gallery_label"]) for item in ranking[:K]],
            "Gallery Path": [os.path.basename(item["gallery_path"]) for item in ranking[:K]],
            "Match": ["✅" if int(item["gallery_label"]) == query_label else "❌" for item in ranking[:K]],
        })

        _top1_match = '✅ Correct' if int(ranking[0]['gallery_label']) == query_label else '❌ Wrong'

        return mo.vstack([
            mo.md(f"📊 Top-{K} Retrieval Details"),
            mo.md(
                f"**Query Label:** {query_label} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**Top-1 Score:** {ranking[0]['score']:.6f} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**Top-1 Match:** {_top1_match}"
            ),
            _top_k_df,
        ])

    return (show_top_k_details,)


@app.cell
def _(ranking, show_top_k_details):
    show_top_k_details(ranking, K=10, query_label=2256)
    return


@app.cell
def _(plt):
    def plot_top_k_scores(ranking, K=5, query_label=2256):
        _fig_scores, _ax_scores = plt.subplots(1, 1, figsize=(10, 4))

        _ranks = [f"Rank {item['rank']}\n(Label {int(item['gallery_label'])})" for item in ranking[:K]]
        _scores = [item["score"] for item in ranking[:K]]
        _colors = ["#4CAF50" if int(item["gallery_label"]) == query_label else "#F44336" for item in ranking[:K]]

        _bars = _ax_scores.bar(_ranks, _scores, color=_colors, edgecolor="white", linewidth=1.5, width=0.6)

        for _bar, _s in zip(_bars, _scores):
            _ax_scores.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.002,
                            f"{_s:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

        _ax_scores.set_title(f"Top-{K} Similarity Scores (Green = Correct Match, Red = Wrong)", fontsize=13, fontweight="bold")
        _ax_scores.set_ylabel("Cosine Similarity", fontsize=12)
        _ax_scores.set_ylim(0, max(_scores) * 1.12)
        _ax_scores.grid(axis="y", alpha=0.3)
        _fig_scores.tight_layout()
        return plt.gca()

    return (plot_top_k_scores,)


@app.cell
def _(plot_top_k_scores, ranking):
    plot_top_k_scores(ranking, K=10, query_label=2256)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## example 2
    """)
    return


@app.cell
def _(Image, plt):
    _query_img = Image.open("/Users/chibangnguyen/ayai/UAV/denseUAV_baseline/test/real_time_query/3032_H80.JPG")

    _fig_query, _ax_query = plt.subplots(1, 1, figsize=(8, 8))
    _ax_query.imshow(_query_img)
    _ax_query.set_title("Query Image: 3032_H80.JPG", fontsize=14, fontweight="bold")
    _ax_query.axis("off")
    _fig_query.tight_layout()
    plt.gca()
    return


@app.cell
def _(baseline_dir, os, uav):
    ranking2 = uav.ranking_by_similarity(uav.query_embedding(os.path.join(os.path.dirname(baseline_dir), "test", "real_time_query", "3032_H80.JPG")))
    return (ranking2,)


@app.cell
def _(ranking2, show_top_k_details):
    show_top_k_details(ranking2, K=10, query_label=3032)
    return


@app.cell
def _(plot_top_k_scores, ranking2):
    plot_top_k_scores(ranking2, K=10, query_label=3032)
    return


@app.cell(hide_code=True)
def _():
    ## Example 3
    return


@app.cell
def _(Image, plt):
    _query_img = Image.open("/Users/chibangnguyen/ayai/UAV/denseUAV_baseline/test/real_time_query/2811_H90.JPG")

    _fig_query, _ax_query = plt.subplots(1, 1, figsize=(8, 8))
    _ax_query.imshow(_query_img)
    _ax_query.set_title("Query Image: 2811_H90.JPG", fontsize=14, fontweight="bold")
    _ax_query.axis("off")
    _fig_query.tight_layout()
    plt.gca()
    return


@app.cell
def _(baseline_dir, os, uav):
    ranking3 = uav.ranking_by_similarity(uav.query_embedding(os.path.join(os.path.dirname(baseline_dir), "test", "real_time_query", "2811_H90.JPG")))
    return (ranking3,)


@app.cell
def _(ranking3, show_top_k_details):
    show_top_k_details(ranking3, K=10, query_label=2811)
    return


@app.cell
def _(plot_top_k_scores, ranking3):
    plot_top_k_scores(ranking3, K=10, query_label=2811)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training
    """)
    return


@app.cell
def _():
    import modal

    # Kết nối Dict và lấy đúng key (vd. một kiến trúc)
    _d = modal.Dict.from_name("denseuav-test-results", create_if_missing=False)
    _data = _d["Swinv2S_256_mode_1"]   # hoặc "Convnext_T_mode_1", ...

    _data

    # gallery_f = _data['gallery_f']
    # query_f = _data['query_f']
    # gallery_label = _data['gallery_label']
    # gallery_path = _data['gallery_path']
    # query_label = _data['query_label']
    # query_path = _data['query_path']
    return (modal,)


@app.cell
def _(modal):
    _d = modal.Dict.from_name("denseuav-test-results", create_if_missing=False)
    for _k in _d.keys():
        print(_k)
    return


if __name__ == "__main__":
    app.run()
