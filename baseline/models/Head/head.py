"""Head factory that dispatches to specialized aggregation modules.

Supported head keys (set via ``opt.head``):
- ``"SingleBranch"``  – CLS-token classifier for ViT-family backbones.
- ``"SingleBranchCNN"`` – Global average-pooling classifier for CNN backbones.
- ``"SingleBranchSwin"`` – Adaptive pooling classifier for Swin backbones.
- ``"NetVLAD"``       – VLAD aggregation over patch tokens.
- ``"FSRA"``          – Attention heatmap-based local+global classification.
- ``"LPN"``           – Local Pattern Network with concentric spatial regions.
- ``"GeM"``           – Generalized Mean Pooling over patch tokens.
"""

import torch.nn as nn
from .SingleBranch import SingleBranch, SingleBranchCNN, SingleBranchSwin
from .FSRA import FSRA
from .LPN import LPN
from .GeM import GeM
from .NetVLAD import NetVLAD

def make_head(opt):
    """Constructs a :class:`Head` from an option namespace.

    Args:
        opt: Argument namespace forwarded to :class:`Head`.

    Returns:
        Head: Initialized head module ready for use.
    """
    return Head(opt)


class Head(nn.Module):
    """Dispatcher module that wraps one of the supported aggregation heads.

    Acts as a thin adapter so the rest of the pipeline only interacts with
    a single :class:`Head` interface regardless of the chosen aggregation
    strategy.

    Attributes:
        head (nn.Module): The concrete head instance created by
            :meth:`init_head`.
        opt: Option namespace used to configure the head.
    """

    def __init__(self, opt) -> None:
        """Initializes Head by constructing the requested aggregation module.

        Args:
            opt: Argument namespace with at least:
                - head (str): Aggregation head key (see module docstring).
                - in_planes (int): Number of input channels from the backbone.
                - nclasses (int): Number of identity classes for the classifier.
                - droprate (float): Dropout probability.
                - num_bottleneck (int): Bottleneck dimension for
                  :class:`~utils.ClassBlock`.
                - block (int): Number of local blocks / VLAD clusters
                  (head-dependent).
        """
        super().__init__()
        self.head = self.init_head(opt)
        self.opt = opt

    def init_head(self, opt):
        """Instantiates the concrete head module specified by ``opt.head``.

        Args:
            opt: Argument namespace (see :meth:`__init__` for required fields).

        Returns:
            nn.Module: One of :class:`~SingleBranch.SingleBranch`,
            :class:`~SingleBranch.SingleBranchCNN`,
            :class:`~SingleBranch.SingleBranchSwin`,
            :class:`~NetVLAD.NetVLAD`, :class:`~FSRA.FSRA`,
            :class:`~LPN.LPN`, or :class:`~GeM.GeM`.

        Raises:
            NameError: If ``opt.head`` is not one of the recognised keys.
        """
        head = opt.head
        if head == "SingleBranch":
            head_model = SingleBranch(opt)
        elif head == "SingleBranchCNN":
            head_model = SingleBranchCNN(opt)
        elif head == "SingleBranchSwin":
            head_model = SingleBranchSwin(opt)
        elif head == "NetVLAD":
            head_model = NetVLAD(opt)
        elif head == "FSRA":
            head_model = FSRA(opt)
        elif head == "LPN":
            head_model = LPN(opt)
        elif head == "GeM":
            head_model = GeM(opt)
        else:
            raise NameError("{} not in the head list!!!".format(head))
        return head_model

    def forward(self, features):
        """Passes backbone features through the wrapped head.

        Args:
            features (torch.Tensor): Feature tensor produced by the backbone.
                Shape depends on the backbone family (see
                :meth:`~Backbone.backbone.Backbone.forward`).

        Returns:
            list: Head output, typically ``[logits, feature]``.  During
            training *logits* is a tensor of shape ``(N, num_classes)`` and
            *feature* has shape ``(N, num_bottleneck)``.  During evaluation
            *feature* may be stacked across local branches.
        """
        features = self.head(features)
        return features
