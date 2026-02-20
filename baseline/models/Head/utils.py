"""Shared building blocks for all head modules.

This module provides:

- :func:`weights_init_kaiming` – Kaiming initialization for Conv/Linear/BN.
- :func:`weights_init_classifier` – Near-zero normal initialization for the
  final classification layer.
- :class:`ClassBlock` – Bottleneck + BN (+ optional ReLU/Dropout) +
  classification layer.
- :class:`Gem_heat` – Soft-weighted GeM pooling using a learnable attention
  vector over channels.
- :class:`GeM` – Channel-wise Generalized Mean Pooling with a learnable
  per-channel exponent.
- :class:`Pooling` – Unified pooling dispatcher (avg, max, avg+max, gem).
- :func:`vector2image` – Reshapes a 1-D token sequence into a 2-D spatial
  grid.
"""

from torch import nn
import torch
from torch.nn import functional as F
import numpy as np


def weights_init_kaiming(m):
    """Applies Kaiming normal initialization to Conv, Linear, and BN layers.

    Intended to be passed to :meth:`nn.Module.apply`.

    * **Linear**: fan-out Kaiming normal for weights; zeros for bias.
    * **Conv**: fan-in Kaiming normal for weights; zeros for bias (if present).
    * **BatchNorm**: constant 1.0 for weight; zeros for bias.

    Args:
        m (nn.Module): A single module (leaf node) to initialize in-place.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    """Initializes classification Linear layers with a near-zero normal.

    Near-zero weight initialization (std=0.001) keeps initial class scores
    close to uniform, stabilizing early training.  Intended to be passed to
    :meth:`nn.Module.apply`.

    Args:
        m (nn.Module): A single module (leaf node) to initialize in-place.
            Only :class:`nn.Linear` layers are affected.
    """
    # classname = m.__class__.__name__
    # if classname.find('Linear') != -1:
    #     nn.init.normal_(m.weight, std=0.001)
    #     if m.bias:
    #         nn.init.constant_(m.bias, 0.0)
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    """Configurable bottleneck classification block.

    The block consists of an optional linear projection followed by batch
    normalization, an optional LeakyReLU, an optional dropout layer, and a
    final linear classification layer.

    Architecture::

        [Linear(input_dim → num_bottleneck)] → BN → [LeakyReLU] →
        [Dropout] → Linear(num_bottleneck → class_num)

    Unlike the legacy version in :mod:`model`, this variant **always** returns
    ``(logits, feature)`` regardless of the training flag.

    Attributes:
        return_f (bool): Kept for interface compatibility; always returns
            features in this implementation.
        add_block (nn.Sequential): Feature-extraction sub-network
            (projection + normalization + optional activation/dropout).
        classifier (nn.Sequential): Single linear classification layer.
    """

    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        """Initializes ClassBlock.

        Args:
            input_dim (int): Dimensionality of the input feature vector.
            class_num (int): Number of output classes.
            droprate (float): Dropout probability.  A value of 0 disables
                dropout.
            relu (bool): If ``True``, insert a LeakyReLU(0.1) after BN.
            bnorm (bool): If ``True``, insert a BatchNorm1d layer.
            num_bottleneck (int): Width of the intermediate bottleneck.
                Ignored when ``linear=False``.
            linear (bool): If ``True``, prepend a
                ``Linear(input_dim, num_bottleneck)`` projection.  If
                ``False``, the bottleneck width is set to ``input_dim``.
            return_f (bool): Retained for API compatibility; has no effect
                in this implementation (features are always returned).
        """
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        """Extracts bottleneck features and computes class logits.

        Args:
            x (torch.Tensor): Input feature vector of shape
                ``(N, input_dim)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(cls_, feature_)`` where
            *cls_* has shape ``(N, class_num)`` and *feature_* has shape
            ``(N, num_bottleneck)``.
        """
        feature_ = self.add_block(x)
        cls_ = self.classifier(feature_)
        return cls_, feature_


class Gem_heat(nn.Module):
    """Soft-weighted GeM pooling using a learnable channel-attention vector.

    Instead of a fixed exponent, a learnable weight vector of shape
    ``(dim,)`` is softmax-normalised and used as channel-wise attention
    coefficients applied to the input sequence via a matrix product.

    Attributes:
        p (nn.Parameter): Learnable attention weight vector of shape
            ``(dim,)``.  Softmax is applied during the forward pass.
        eps (float): Numerical stability constant (currently unused in the
            forward computation but kept for interface parity with
            :class:`GeM`).
    """

    def __init__(self, dim=768, p=3, eps=1e-6):
        """Initializes Gem_heat.

        Args:
            dim (int): Number of input channels (length of the attention
                weight vector).
            p (float): Initial value for every element of the attention
                weight vector before softmax.
            eps (float): Numerical stability constant (reserved).
        """
        super(Gem_heat, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p)  # initial p
        self.eps = eps

    def forward(self, x):
        """Applies soft-weighted pooling to the input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, L, C)`` where *L*
                is the sequence length and *C* equals ``dim``.

        Returns:
            torch.Tensor: Pooled descriptor of shape ``(N, L)``.
        """
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        """Computes the soft-weighted channel aggregation.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, L, C)``.
            p (torch.Tensor or float): Attention weight vector of shape
                ``(C,)``; softmax is applied internally.
            eps (float): Unused; kept for interface parity.

        Returns:
            torch.Tensor: Aggregated tensor of shape ``(N, L)``.
        """
        # x = torch.transpose(x, 1, -1)
        p = F.softmax(p).unsqueeze(-1)
        x = torch.matmul(x, p)
        # x = torch.transpose(x, 1, -1)
        # x = F.avg_pool1d(x, x.size(-1))
        x = x.view(x.size(0), x.size(1))
        # x = x.pow(1. / p)
        return x


class GeM(nn.Module):
    # channel-wise GeM zhedong zheng
    """Channel-wise Generalized Mean Pooling with learnable exponent.

    Each channel has its own learnable exponent ``p_c``.  The pooling
    operation for channel *c* is::

        GeM_c(x) = (avg(x_c ^ p_c)) ^ (1 / p_c)

    This is a differentiable generalization of average pooling (p=1) toward
    max pooling (p→∞).

    Attributes:
        p (nn.Parameter): Per-channel learnable exponent vector of shape
            ``(dim,)``.  Initialized to *p*.
        eps (float): Small constant added before the power operation for
            numerical stability.
    """

    def __init__(self, dim=2048, p=1, eps=1e-6):
        """Initializes GeM.

        Args:
            dim (int): Number of input channels (length of the learnable
                exponent vector).
            p (float): Initial value for all exponent elements.
            eps (float): Numerical stability constant added to activations
                before exponentiation.
        """
        super(GeM,  self).__init__()
        self.p = nn.Parameter(torch.ones(dim)*p)  # initial p
        self.eps = eps

    def forward(self, x):
        """Applies channel-wise GeM pooling to a 2-D feature map.

        Args:
            x (torch.Tensor): Input feature map of shape ``(N, C, H, W)``.

        Returns:
            torch.Tensor: Pooled descriptor of shape ``(N, C)``.
        """
        x = torch.transpose(x, 1, -1)
        x = (x+self.eps).pow(self.p)
        x = torch.transpose(x, 1, -1)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.view(x.size(0), x.size(1)).contiguous()
        x = x.pow(1./self.p)
        return x



class Pooling(nn.Module):
    """Unified pooling dispatcher supporting multiple pooling strategies.

    Wraps average pooling, max pooling, their concatenation, and GeM
    pooling behind a single interface so that downstream modules can
    select a strategy via a string argument.

    Attributes:
        pool (str): The selected pooling strategy.
        avgpool2 (nn.AdaptiveAvgPool2d): Average pool (present for
            ``"avg"`` and ``"avg+max"``).
        maxpool2 (nn.AdaptiveMaxPool2d): Max pool (present for ``"max"``
            and ``"avg+max"``).
        gem2 (Gem_heat): Soft-weighted GeM pool (present for ``"gem"``).
    """

    def __init__(self, dim, pool="avg"):
        """Initializes Pooling.

        Args:
            dim (int): Channel dimension passed to :class:`Gem_heat` when
                ``pool="gem"``.  Ignored for the other strategies.
            pool (str): Pooling strategy.  One of:

                - ``"avg"``     – global adaptive average pooling.
                - ``"max"``     – global adaptive max pooling.
                - ``"avg+max"`` – concatenation of average and max pool
                  outputs along the channel dimension.
                - ``"gem"``     – soft-weighted Generalized Mean Pooling
                  via :class:`Gem_heat`.
        """
        super(Pooling, self).__init__()
        self.pool = pool
        if pool == 'avg+max':
            self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            self.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'avg':
            self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            self.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'gem':
            self.gem2 = Gem_heat(dim=dim)

    def forward(self, x):
        """Applies the selected pooling strategy.

        Args:
            x (torch.Tensor): For ``"avg"``, ``"max"``, and ``"avg+max"``:
                a spatial feature map of shape ``(N, C, H, W)``.
                For ``"gem"``: a token sequence of shape ``(N, C, L)``
                (as expected by :class:`Gem_heat`).

        Returns:
            torch.Tensor: Pooled tensor.  Shape is ``(N, C, 1, 1)`` for
            spatial strategies, ``(N, 2*C, 1, 1)`` for ``"avg+max"``,
            and ``(N, L)`` for ``"gem"``.
        """
        if self.pool == 'avg+max':
            x1 = self.avgpool2(x)
            x2 = self.maxpool2(x)
            x = torch.cat((x1, x2), dim=1)
        elif self.pool == 'avg':
            x = self.avgpool2(x)
        elif self.pool == 'max':
            x = self.maxpool2(x)
        elif self.pool == 'gem':
            x = self.gem2(x)
        return x


def vector2image(x, dim=1):
    """Reshapes a flat token sequence into a square 2-D spatial grid.

    Assumes the sequence length or channel count is a perfect square.

    Args:
        x (torch.Tensor): Input tensor of shape ``(B, N, C)``.
        dim (int): Axis to reshape into a square grid:

            - ``1``: reshape the *N* dimension into
              ``(sqrt(N), sqrt(N))``, yielding ``(B, sqrt(N), sqrt(N), C)``.
            - ``2``: reshape the *C* dimension into
              ``(sqrt(C), sqrt(C))``, yielding ``(B, N, sqrt(C), sqrt(C))``.

    Returns:
        torch.Tensor: Reshaped tensor (see *dim* for exact shape).

    Raises:
        TypeError: If *dim* is not 1 or 2.
    """
    B, N, C = x.shape
    if dim == 1:
        return x.reshape(B, int(np.sqrt(N)), int(np.sqrt(N)), C)
    if dim == 2:
        return x.reshape(B, N, int(np.sqrt(C)), int(np.sqrt(C)))
    else:
        raise TypeError("dim is not correct!!")
