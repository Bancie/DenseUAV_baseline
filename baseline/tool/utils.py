"""Utility helpers shared across the DenseUAV baseline.

Provides logging, file management, model persistence, training helpers, and
metric computation utilities used by ``train.py``, ``test.py``, and the
evaluation scripts.

This module is not intended to be run directly; import individual symbols as
needed::

    from tool.utils import get_logger, save_network, load_network, set_seed
"""

import os
import torch
import yaml
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
from shutil import copyfile, copytree, rmtree
import logging
from models.taskflow import make_model
from thop import profile, clever_format


def get_logger(filename, verbosity=1, name=None):
    """Create (or retrieve) a logger that writes to both a file and stdout.

    Args:
        filename (str): Path to the log file.  The file is opened in write
            mode (``'w'``), so any previous content is truncated.
        verbosity (int, optional): Logging level selector.
            ``0`` → DEBUG, ``1`` → INFO (default), ``2`` → WARNING.
        name (str, optional): Logger name passed to
            ``logging.getLogger``.  ``None`` returns the root logger.

    Returns:
        logging.Logger: Configured logger instance with file and stream
            handlers attached.
    """
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def copy_file_or_tree(path, target_dir):
    """Copy a single file or an entire directory tree to a target directory.

    If ``path`` is a directory that already exists at the destination, the
    existing copy is removed before the new one is created.

    Args:
        path (str): Source file or directory path (relative or absolute).
        target_dir (str): Destination directory.  The copied item will be
            placed at ``os.path.join(target_dir, path)``.
    """
    target_path = os.path.join(target_dir, path)
    if os.path.isdir(path):
        if os.path.exists(target_path):
            rmtree(target_path)
        copytree(path, target_path)
    elif os.path.isfile(path):
        copyfile(path, target_path)


def copyfiles2checkpoints(opt):
    """Snapshot the current source tree into the run's checkpoint directory.

    Creates ``<checkpoint_dir>/<opt.name>/`` (if absent) and copies training
    scripts, dataset definitions, loss functions, model definitions, optimiser
    code, and tool utilities into it.  Also serialises the full ``opt``
    namespace to ``opts.yaml`` for reproducibility.

    Args:
        opt (argparse.Namespace): Parsed training configuration.  Must contain
            an attribute ``name`` (str) that identifies the current run.
            May contain ``checkpoint_dir`` (str) for configurable root; default ``checkpoints``.
    """
    # new: use opt.checkpoint_dir so Modal can write to Volume
    checkpoint_root = getattr(opt, 'checkpoint_dir', 'checkpoints')
    dir_name = os.path.join(checkpoint_root, opt.name)
    if not os.path.isdir(checkpoint_root):
        os.mkdir(checkpoint_root)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    # record every run
    copy_file_or_tree('train.py', dir_name)
    copy_file_or_tree('test.py', dir_name)
    copy_file_or_tree('evaluate_gpu.py', dir_name)
    copy_file_or_tree('evaluateDistance.py', dir_name)
    copy_file_or_tree('datasets', dir_name)
    copy_file_or_tree('losses', dir_name)
    copy_file_or_tree('models', dir_name)
    copy_file_or_tree('optimizers', dir_name)
    copy_file_or_tree('tool', dir_name)
    copy_file_or_tree('train_test_local.sh', dir_name)

    # save opts
    with open('%s/opts.yaml' % dir_name, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)


def make_weights_for_balanced_classes(images, nclasses):
    """Compute per-sample weights for balanced class sampling.

    Each sample's weight is inversely proportional to its class frequency,
    so that a ``WeightedRandomSampler`` produces roughly uniform class
    distributions in each mini-batch.

    Args:
        images (list[tuple[str, int]]): List of ``(path, class_index)`` pairs
            as returned by ``torchvision.datasets.ImageFolder.imgs``.
        nclasses (int): Total number of classes.

    Returns:
        list[float]: Per-sample weights of the same length as ``images``.
    """
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # count the image number in every class
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def get_model_list(dirname, key):
    """Return the path to the latest ``.pth`` checkpoint matching a key.

    Args:
        dirname (str): Directory to search for checkpoint files.
        key (str): Substring that the checkpoint filename must contain
            (e.g. ``'net'``).

    Returns:
        str | None: Absolute path of the lexicographically last matching
            ``.pth`` file, or ``None`` if the directory does not exist or
            contains no matching files.
    """
    if os.path.exists(dirname) is False:
        print('no dir: %s' % dirname)
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

######################################################################
# Save model
# ---------------------------


def save_network(network, dirname, epoch_label, checkpoint_root=None):
    """Serialise model weights to ``<checkpoint_root>/<dirname>/net_<epoch>.pth``.

    The model is temporarily moved to CPU for saving and then moved back to
    GPU (if available) afterwards.

    Args:
        network (torch.nn.Module): The model whose ``state_dict`` will be
            saved.
        dirname (str): Run name; the checkpoint is saved under
            ``<checkpoint_root>/<dirname>/``.
        epoch_label (int | str): Used to construct the filename.  An integer
            value produces ``net_XXX.pth`` (zero-padded to 3 digits); a string
            value produces ``net_<epoch_label>.pth``.
        checkpoint_root (str | None): Root directory for checkpoints; if None,
            defaults to ``./checkpoints`` (for backward compatibility and Modal Volume).
    """
    # new: configurable checkpoint root for Modal
    root = checkpoint_root if checkpoint_root is not None else './checkpoints'
    out_dir = os.path.join(root, dirname)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth' % epoch_label
    else:
        save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(root, dirname, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


class UnNormalize(object):
    """Reverse ImageNet-style normalisation on a batch of images.

    Applies the inverse of ``transforms.Normalize(mean, std)`` in-place to
    each channel of the input tensor.

    Args:
        mean (sequence): Per-channel mean used during normalisation,
            e.g. ``[0.485, 0.456, 0.406]``.
        std (sequence): Per-channel standard deviation used during
            normalisation, e.g. ``[0.229, 0.224, 0.225]``.

    Example:
        >>> unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        >>> img_restored = unnorm(img_normalised)
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """Un-normalise a batch tensor channel-by-channel.

        Args:
            tensor (torch.Tensor): Normalised image tensor of shape
                ``(B, C, H, W)``.

        Returns:
            torch.Tensor: Un-normalised tensor of the same shape,
                modified in-place.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def check_box(images, boxes):
    """Visualise bounding boxes overlaid on a batch of images using Matplotlib.

    Boxes are expected in feature-map coordinates and are rescaled to pixel
    space (feature stride 16, target image size 255).

    Args:
        images (torch.Tensor): Batch of images of shape ``(B, C, H, W)``,
            possibly normalised.
        boxes (torch.Tensor): Bounding box tensor of shape ``(B, 4)``
            in feature-map coordinates ``[x1, y1, x2, y2]``.
    """
    # Unorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # images = Unorm(images)*255
    images = images.permute(0, 2, 3, 1).cpu().detach().numpy()
    boxes = (boxes.cpu().detach().numpy()/16*255).astype(np.int)
    for img, box in zip(images, boxes):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(img)
        rect = plt.Rectangle(box[0:2], box[2]-box[0], box[3]-box[1])
        ax.add_patch(rect)
        plt.show()


######################################################################
#  Load model for resume
# ---------------------------
def load_network(opt):
    """Reconstruct a model and load weights from a checkpoint file.

    Builds the model architecture using ``make_model(opt)`` and then loads
    the state dict from ``opt.checkpoint`` (interpreted as a bare filename
    resolved relative to the current working directory).

    Args:
        opt (argparse.Namespace): Configuration namespace.  Must contain:
            - ``checkpoint`` (str): Filename of the ``.pth`` weights file.
            - All fields required by ``make_model`` (``backbone``, ``head``,
              etc.).

    Returns:
        torch.nn.Module: Model with weights loaded, still on CPU.
    """
    save_filename = opt.checkpoint
    model = make_model(opt)
    print('Load the model from %s' % save_filename)
    network = model
    network.load_state_dict(torch.load(save_filename))
    return network


def toogle_grad(model, requires_grad):
    """Enable or disable gradient computation for all parameters of a model.

    Args:
        model (torch.nn.Module): Model whose parameters will be modified.
        requires_grad (bool): If ``True``, gradients will be computed for all
            parameters; if ``False``, all parameters are frozen.
    """
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def update_average(model_tgt, model_src, beta):
    """Update a target model as an exponential moving average of a source model.

    Implements the EMA update::

        theta_tgt = beta * theta_tgt + (1 - beta) * theta_src

    Both models' gradients are temporarily disabled during the update.

    Args:
        model_tgt (torch.nn.Module): Target (EMA) model whose parameters are
            updated in-place.
        model_src (torch.nn.Module): Source model providing the new parameter
            values.
        beta (float): Smoothing coefficient in ``[0, 1)``.  Values close to 1
            give slow updates (e.g. ``0.999``).
    """
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

    toogle_grad(model_src, True)


def get_preds(outputs, outputs2):
    """Extract predicted class indices from model logit outputs.

    Handles both a single logit tensor and a list of logit tensors (e.g. for
    multi-branch / multi-scale heads).

    Args:
        outputs (torch.Tensor | list[torch.Tensor]): Logits for the first
            view (satellite).  Shape ``(N, C)`` per tensor.
        outputs2 (torch.Tensor | list[torch.Tensor]): Logits for the second
            view (drone).  Shape ``(N, C)`` per tensor.

    Returns:
        tuple:
            - preds (torch.Tensor | list[torch.Tensor]): Predicted class
              indices for the first view.
            - preds2 (torch.Tensor | list[torch.Tensor]): Predicted class
              indices for the second view.
    """
    if isinstance(outputs, list):
        preds = []
        preds2 = []
        for out, out2 in zip(outputs, outputs2):
            preds.append(torch.max(out.data, 1)[1])
            preds2.append(torch.max(out2.data, 1)[1])
    else:
        _, preds = torch.max(outputs.data, 1)
        _, preds2 = torch.max(outputs2.data, 1)
    return preds, preds2


def calc_flops_params(model,
                      input_size_drone,
                      input_size_satellite,
                      ):
    """Profile a dual-branch model for MACs and parameter count using ``thop``.

    Generates random input tensors on GPU, runs a single forward pass with
    ``thop.profile``, and formats the results with ``thop.clever_format``.

    Args:
        model (torch.nn.Module): The dual-branch model to profile.  Must
            accept two positional tensor arguments ``(drone_input,
            satellite_input)``.
        input_size_drone (tuple[int, ...]): Shape of the drone input tensor,
            e.g. ``(1, 3, 256, 256)``.
        input_size_satellite (tuple[int, ...]): Shape of the satellite input
            tensor, e.g. ``(1, 3, 256, 256)``.

    Returns:
        tuple[str, str]: Human-readable ``(macs, params)`` strings, e.g.
            ``('12.345G', '86.570M')``.
    """
    inputs_drone = torch.randn(input_size_drone).cuda()
    inputs_satellite = torch.randn(input_size_satellite).cuda()
    total_ops, total_params = profile(
        model, (inputs_drone, inputs_satellite,), verbose=False)
    macs, params = clever_format([total_ops, total_params], "%.3f")
    return macs, params


def set_seed(seed):
    """Fix all random seeds for fully reproducible training runs.

    Sets seeds for PyTorch (CPU and all GPUs), NumPy, and Python's built-in
    ``random`` module.  Also configures cuDNN to operate in deterministic mode.

    Args:
        seed (int): The seed value to use for all RNGs.

    Note:
        Setting ``cudnn.deterministic = True`` may reduce throughput.
        ``cudnn.benchmark`` is set to ``False`` to prevent non-deterministic
        algorithm selection.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    random.seed(seed)
