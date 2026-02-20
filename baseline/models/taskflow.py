"""Top-level model combining a shared backbone and a task-specific head.

This module provides :class:`Model`, the primary entry point for the
DenseUAV geo-localization baseline.  A single backbone processes both
drone-view and satellite-view images, and a shared head produces
classification logits together with a compact embedding vector.
"""

import torch.nn as nn
from .Backbone.backbone import make_backbone
from .Head.head import make_head
import os
import torch


class Model(nn.Module):
    """Dual-view geo-localization model with a shared backbone and head.

    Both the drone branch and the satellite branch share identical backbone
    and head weights.  Either input can be ``None`` to skip the corresponding
    forward pass (useful for asymmetric training or single-view inference).

    Attributes:
        backbone (nn.Module): Shared feature extractor for both views.
        head (nn.Module): Aggregation and classification head.
        opt: Option namespace used to construct the model.
    """

    def __init__(self, opt):
        """Initializes Model from an option namespace.

        Args:
            opt: Argument namespace containing at least:
                - backbone (str): Backbone architecture key (e.g. ``"ViTS-224"``).
                - head (str): Head architecture key (e.g. ``"FSRA"``).
                - load_from (str): Path to a checkpoint file (may be empty or
                  non-existent, in which case no weights are loaded).
                - Additional head-specific fields populated on *opt* after
                  backbone construction (e.g. ``in_planes`` is set here).
        """
        super().__init__()
        self.backbone = make_backbone(opt)
        opt.in_planes = self.backbone.output_channel
        self.head = make_head(opt)
        self.opt = opt

    def forward(self, drone_image, satellite_image):
        """Extracts features and computes logits for drone and satellite views.

        Args:
            drone_image (torch.Tensor or None): Batch of drone-view images with
                shape ``(N, 3, H, W)``, or ``None`` to skip the drone branch.
            satellite_image (torch.Tensor or None): Batch of satellite-view
                images with shape ``(N, 3, H, W)``, or ``None`` to skip the
                satellite branch.

        Returns:
            tuple[list or None, list or None]: A 2-tuple
            ``(drone_res, satellite_res)``.  Each element is either ``None``
            (when the corresponding input was ``None``) or the output of the
            head, typically ``[logits, feature]`` where *logits* has shape
            ``(N, num_classes)`` and *feature* has shape
            ``(N, num_bottleneck)``.
        """
        if drone_image is None:
            drone_res = None
        else:
            drone_features = self.backbone(drone_image)
            drone_res = self.head(drone_features)
        if satellite_image is None:
            satellite_res = None
        else:
            satellite_features = self.backbone(satellite_image)
            satellite_res = self.head(satellite_features)
        
        return drone_res,satellite_res
    
    def load_params(self, load_from):
        """Loads matching parameters from a checkpoint into this model.

        Parameters are matched by name and tensor shape.  Only matching
        entries overwrite the current state; all other parameters remain
        unchanged, enabling partial fine-tuning from heterogeneous checkpoints.

        Args:
            load_from (str): File-system path to a ``torch.save``-compatible
                checkpoint dictionary mapping parameter names to tensors.
        """
        pretran_model = torch.load(load_from)
        model2_dict = self.state_dict()
        state_dict = {k: v for k, v in pretran_model.items() if k in model2_dict.keys() and v.size() == model2_dict[k].size()}
        model2_dict.update(state_dict)
        self.load_state_dict(model2_dict)



def make_model(opt):
    """Constructs a :class:`Model` and optionally loads pre-trained weights.

    Args:
        opt: Argument namespace forwarded to :class:`Model`.  If
            ``opt.load_from`` points to an existing file the weights are loaded
            via :meth:`Model.load_params`.

    Returns:
        Model: Ready-to-use model instance.
    """
    model = Model(opt)
    if os.path.exists(opt.load_from):
        model.load_params(opt.load_from)
    return model
