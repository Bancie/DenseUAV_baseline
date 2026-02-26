import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", layout_file="layouts/UAVAINonGPS.slides.json")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    logo = mo.center(mo.image(src="resources/logo.png", alt="DenseUAV Logo", width=1000))
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

    ## Quick Navigation
    - [Code analysis](#code-analysis-documentation) - Analyze code structure of author's baseline.
    - [Testing](#baseline-testing) - Some tests on author's baseline.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Code Analysis & Documentation

    This document analyzes the author's official baseline from the [DenseUAV](https://github.com/Dmmm1997/DenseUAV) GitHub repository, located in the `baseline/` folder.

    ---

    ### Table of Contents

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

    ### Overview

    The baseline implements **cross-view geo-localization** between UAV-view (drone) images and satellite-view images. Given a query drone image, the goal is to retrieve the most geographically similar satellite image (or vice versa) from a gallery, without using GPS.

    The system is trained on the **DenseUAV** dataset (2022 version, 2256 location classes) and evaluated using both standard retrieval metrics (CMC, mAP) and spatial accuracy metrics (SDM@K, MA@K).

    **Key design decisions:**

    - **Shared backbone**: both drone and satellite images pass through the same feature extractor
    - **Modular architecture**: backbones and heads are independently configurable via `opts.yaml`
    - **Multi-loss training**: combines classification loss, triplet loss, and KL divergence (mutual learning)
    - **Mixed precision**: autocast for faster CUDA training

    ---

    ### Directory Structure

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
    ## Baseline testing
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn

    return nn, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Architecture defined
    """)
    return


@app.cell(hide_code=True)
def _(nn):
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    arch = NeuralNetwork()
    arch
    return (arch,)


@app.cell(hide_code=True)
def _():
    ### Dataset
    return


@app.cell
def _():
    query_imagepath = "/Users/chibangnguyen/ayai/UAV/denseUAV_baseline/data/DenseUAV_data/test/query_drone/002256/H80.JPG"
    query_imagepath
    return (query_imagepath,)


@app.cell
def _(Image, query_imagepath):
    image = Image.open(query_imagepath)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Inference

    Inference on author's baseline pretrained model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Model defined
    """)
    return


@app.cell
def _():
    model_path = "../baseline/net_119.pth"
    model_path
    return (model_path,)


@app.cell(hide_code=True)
def _():
    #### Define task flow
    return


@app.cell
def _(model_path, torch):
    # Load the pretrained checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"Loaded checkpoint from: {model_path}")
    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    # Check if it's a state dict or full checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Found state_dict in checkpoint")
    else:
        state_dict = checkpoint
        print("Using checkpoint as state_dict directly")
    return (state_dict,)


@app.cell
def _(arch, state_dict):
    # Load the state dict into the architecture
    try:
        arch.load_state_dict(state_dict, strict=False)
        print("Successfully loaded state dict into architecture")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("This might be due to architecture mismatch between the checkpoint and current model")

    # Set model to evaluation mode for inference
    arch.eval()
    print("Model set to evaluation mode")
    return


@app.cell
def _(arch, torch):
    # Setup inference function without autograd
    def run_inference(model, input_tensor):
        """
        Run inference without computing gradients
        """
        with torch.no_grad():  # Disable gradient computation
            model.eval()  # Ensure model is in eval mode
            output = model(input_tensor)
            return output

    # Example usage - create dummy input tensor
    # Note: Adjust input shape based on actual model requirements
    dummy_input = torch.randn(1, 28*28)  # Batch size 1, flattened 28x28 image

    # Run inference
    with torch.no_grad():
        output = arch(dummy_input)
        print(f"Inference output shape: {output.shape}")
        print(f"Output: {output}")
    return


@app.cell
def _(torch):
    # For batch inference
    def batch_inference(model, data_loader):
        """
        Run inference on a batch of data without gradients
        """
        model.eval()
        predictions = []
    
        with torch.no_grad():
            for batch in data_loader:
                outputs = model(batch)
                predictions.append(outputs)
    
        return torch.cat(predictions, dim=0)

    print("Inference functions ready - model loaded and set for evaluation")
    return


@app.cell
def _():
    from PIL import Image

    return (Image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Class name define
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
