"""Generalized Mean Pooling head for ViT-family backbones.

GeM pooling aggregates patch tokens (excluding the CLS token) using a
learnable per-channel exponent *p*, providing a smooth interpolation
between average pooling (p→1) and max pooling (p→∞).
"""

import torch.nn as nn
from .utils import ClassBlock, Pooling, vector2image



class GeM(nn.Module):
    """Head that applies Generalized Mean Pooling over ViT patch tokens.

    The CLS token (index 0) is discarded; the remaining patch tokens are
    transposed to ``(N, C, num_patches)`` and fed to a :class:`~utils.Pooling`
    layer configured with the ``"gem"`` strategy before classification.

    Attributes:
        opt: Option namespace used to build the module.
        classifier (ClassBlock): Linear bottleneck + BN + classification head.
        pool (Pooling): GeM pooling layer operating on the patch-token
            sequence.
    """

    def __init__(self, opt) -> None:
        """Initializes the GeM head.

        Args:
            opt: Argument namespace with at least:
                - in_planes (int): Feature dimension per token (backbone
                  output channels, e.g. 384 for ViT-S).
                - nclasses (int): Number of identity classes.
                - droprate (float): Dropout probability in
                  :class:`~utils.ClassBlock`.
                - num_bottleneck (int): Bottleneck projection dimension.
                - h (int): Input image height (used to infer the number of
                  patches as ``(h // 16) * (w // 16)``).
                - w (int): Input image width.
        """
        super().__init__()
        self.opt = opt
        self.classifier = ClassBlock(
            opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck=opt.num_bottleneck)
        self.pool = Pooling(opt.h//16*opt.w//16, "gem")

    def forward(self, features):
        """Pools patch tokens with GeM and classifies the result.

        The CLS token at position 0 is excluded; all remaining patch tokens
        are used for pooling.

        Args:
            features (torch.Tensor): Token tensor from a ViT backbone with
                shape ``(N, num_patches + 1, C)`` where index 0 is the CLS
                token.

        Returns:
            list[torch.Tensor, torch.Tensor]: ``[cls, feature]`` where *cls*
            has shape ``(N, num_classes)`` and *feature* has shape
            ``(N, num_bottleneck)``.
        """
        local_feature = features[:, 1:]
        local_feature = local_feature.transpose(1,2).contiguous()
        # local_feature = vector2image(local_feature,dim = 2)
        global_feature = self.pool(local_feature)
        cls, feature = self.classifier(global_feature)
        return [cls, feature]
