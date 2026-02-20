"""Single-branch classification heads for different backbone families.

Three variants are provided:

- :class:`SingleBranch` – uses the CLS token from ViT / DeiT backbones.
- :class:`SingleBranchCNN` – applies global average pooling to CNN spatial
  feature maps.
- :class:`SingleBranchSwin` – applies adaptive 1-D pooling to the token
  sequence produced by Swin / PVTv2 backbones.

Each variant returns ``[logits, feature]`` during training and
``feature`` only during evaluation (behaviour delegated to
:class:`~utils.ClassBlock`).
"""

import torch.nn as nn
from .utils import ClassBlock, Pooling


class SingleBranch(nn.Module):
    """Global classifier that uses the CLS token from ViT-family backbones.

    Attributes:
        opt: Option namespace used to build the module.
        classifier (ClassBlock): Linear bottleneck + BN + classification head.
    """

    def __init__(self, opt) -> None:
        """Initializes SingleBranch.

        Args:
            opt: Argument namespace with at least:
                - in_planes (int): Feature dimension of the CLS token (e.g.
                  384 for ViT-S, 768 for ViT-B).
                - nclasses (int): Number of identity classes.
                - droprate (float): Dropout probability inside
                  :class:`~utils.ClassBlock`.
                - num_bottleneck (int): Bottleneck dimension of the linear
                  projection before classification.
        """
        super().__init__()
        self.opt = opt
        self.classifier = ClassBlock(
            opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck=opt.num_bottleneck)

    def forward(self, features):
        """Classifies the CLS token.

        Args:
            features (torch.Tensor): Token tensor from a ViT / DeiT backbone
                with shape ``(N, num_patches + 1, C)``.  Index 0 along dim 1
                is the CLS token.

        Returns:
            list[torch.Tensor, torch.Tensor]: ``[cls, feature]`` where *cls*
            has shape ``(N, num_classes)`` and *feature* has shape
            ``(N, num_bottleneck)``.
        """
        global_feature = features[:, 0]
        cls, feature = self.classifier(global_feature)
        return [cls, feature]


class SingleBranchCNN(nn.Module):
    """Global classifier that applies adaptive average pooling to CNN features.

    Attributes:
        opt: Option namespace used to build the module.
        pool (nn.AdaptiveAvgPool2d): Reduces spatial dimensions to ``(1, 1)``.
        classifier (ClassBlock): Linear bottleneck + BN + classification head.
    """

    def __init__(self, opt) -> None:
        """Initializes SingleBranchCNN.

        Args:
            opt: Argument namespace with at least:
                - in_planes (int): Number of channels in the CNN feature map
                  (e.g. 2048 for ResNet-50).
                - nclasses (int): Number of identity classes.
                - droprate (float): Dropout probability.
                - num_bottleneck (int): Bottleneck projection dimension.
        """
        super().__init__()
        self.opt = opt
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = ClassBlock(
            opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck=opt.num_bottleneck)

    def forward(self, features):
        """Pools CNN feature map and classifies the result.

        Args:
            features (torch.Tensor): CNN spatial feature map with shape
                ``(N, C, H', W')``.

        Returns:
            list[torch.Tensor, torch.Tensor]: ``[cls, feature]`` where *cls*
            has shape ``(N, num_classes)`` and *feature* has shape
            ``(N, num_bottleneck)``.
        """
        global_feature = self.pool(features).reshape(features.shape[0], -1)
        cls, feature = self.classifier(global_feature)
        return [cls, feature]


class SingleBranchSwin(nn.Module):
    """Global classifier that pools the token sequence from Swin-family backbones.

    Uses :class:`nn.AdaptiveAvgPool1d` to collapse the token dimension,
    treating it as the temporal/sequence axis.

    Attributes:
        opt: Option namespace used to build the module.
        pool (nn.AdaptiveAvgPool1d): Collapses the token sequence to length 1.
        classifier (ClassBlock): Linear bottleneck + BN + classification head.
    """

    def __init__(self, opt) -> None:
        """Initializes SingleBranchSwin.

        Args:
            opt: Argument namespace with at least:
                - in_planes (int): Feature dimension per token (e.g. 768 for
                  Swin-B).
                - nclasses (int): Number of identity classes.
                - droprate (float): Dropout probability.
                - num_bottleneck (int): Bottleneck projection dimension.
        """
        super().__init__()
        self.opt = opt
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = ClassBlock(
            opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck=opt.num_bottleneck)

    def forward(self, features):
        """Pools Swin token sequence and classifies the result.

        Args:
            features (torch.Tensor): Token sequence from a Swin / PVTv2
                backbone with shape ``(N, num_tokens, C)``.

        Returns:
            list[torch.Tensor, torch.Tensor]: ``[cls, feature]`` where *cls*
            has shape ``(N, num_classes)`` and *feature* has shape
            ``(N, num_bottleneck)``.
        """
        global_feature = self.pool(features.transpose(
            2, 1)).reshape(features.shape[0], -1)
        cls, feature = self.classifier(global_feature)
        return [cls, feature]
