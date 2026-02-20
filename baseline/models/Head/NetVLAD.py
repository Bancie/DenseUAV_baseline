"""NetVLAD aggregation head for ViT-family backbones.

NetVLAD (Arandjelovic et al., 2016) aggregates local descriptors into a
fixed-size representation by computing soft-assigned residuals to a set of
learned cluster centres.  This module adapts the original formulation to
operate on patch tokens produced by Vision Transformer backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ClassBlock, Pooling, vector2image


class NetVLAD(nn.Module):
    """Top-level NetVLAD head combining token reshaping with VLAD aggregation.

    Patch tokens (excluding the CLS token) are transposed and reshaped into
    a 2-D spatial grid before being passed to :class:`NetVLAD_block`.  The
    resulting VLAD descriptor is classified by a :class:`~utils.ClassBlock`.

    Attributes:
        opt: Option namespace used to build the module.
        classifier (ClassBlock): Linear bottleneck + BN + classification head
            operating on the flattened VLAD descriptor.
        netvlad (NetVLAD_block): Core VLAD aggregation module.
    """

    def __init__(self, opt) -> None:
        """Initializes the NetVLAD head.

        Args:
            opt: Argument namespace with at least:
                - in_planes (int): Feature dimension per patch token.
                - nclasses (int): Number of identity classes.
                - droprate (float): Dropout probability.
                - num_bottleneck (int): Bottleneck projection dimension for
                  :class:`~utils.ClassBlock`.
                - block (int): Number of VLAD clusters
                  (``num_clusters`` in :class:`NetVLAD_block`).  The VLAD
                  descriptor has dimension ``in_planes * block``.
        """
        super(NetVLAD, self).__init__()
        self.opt = opt
        self.classifier = ClassBlock(
            int(opt.in_planes*opt.block), opt.nclasses, opt.droprate, num_bottleneck=opt.num_bottleneck)
        self.netvlad = NetVLAD_block(
            num_clusters=opt.block, dim=opt.in_planes, alpha=100.0, normalize_input=True)

    def forward(self, features):
        """Aggregates patch tokens via VLAD and classifies the result.

        Args:
            features (torch.Tensor): Token tensor from a ViT backbone with
                shape ``(N, num_patches + 1, C)``.  The CLS token at index 0
                is discarded.

        Returns:
            list[torch.Tensor, torch.Tensor]: ``[cls, feature]`` where *cls*
            has shape ``(N, num_classes)`` and *feature* has shape
            ``(N, num_bottleneck)``.
        """
        local_feature = features[:, 1:]
        local_feature = local_feature.transpose(1, 2)

        local_feature = vector2image(local_feature, dim=2)
        local_features = self.netvlad(local_feature)

        cls, feature = self.classifier(local_features)
        return [cls, feature]


class NetVLAD_block(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """Initializes NetVLAD_block with cluster centres and a soft-assignment
        convolution.

        Args:
            num_clusters (int): Number of VLAD cluster centres *K*.
            dim (int): Descriptor (channel) dimension *D*.
            alpha (float): Sharpness parameter for initializing the
                soft-assignment weights.  Larger values produce harder
                cluster assignments at initialization.
            normalize_input (bool): If ``True``, each input descriptor is
                L2-normalised before computing soft assignments.
        """
        super(NetVLAD_block, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(
            torch.rand(num_clusters, dim))  # 聚类中心，参见注释1
        self._init_params()

    def _init_params(self):
        """Initializes conv weights and biases from the cluster centroids.

        Sets the 1×1 convolution so that its output approximates the
        soft-assignment score:
        ``score_k(x) = 2·alpha·c_k^T·x − alpha·||c_k||²``
        which is derived from the squared Euclidean distance between *x*
        and centroid *c_k* scaled by *alpha*.
        """
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        """Computes the VLAD descriptor for a batch of local feature maps.

        Args:
            x (torch.Tensor): Local feature map of shape ``(N, C, H, W)``
                where ``H * W`` equals the number of local descriptors and
                ``C`` is the descriptor dimension.

        Returns:
            torch.Tensor: L2-normalised VLAD descriptor of shape
            ``(N, num_clusters * C)``.
        """
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim，使用L2归一化特征维度

        # soft-assignment
        # (N, C, H, W)->(N, num_clusters, H, W)->(N, num_clusters, H * W)
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        # (N, num_clusters, H * W)  # 参见注释3
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)  # (N, C, H, W) -> (N, C, H * W)

        # calculate residuals to each clusters
        # 减号前面前记为a，后面记为b, residual = a - b
        # a: (N, C, H * W) -> (num_clusters, N, C, H * W) -> (N, num_clusters, C, H * W)
        # b: (num_clusters, C) -> (H * W, num_clusters, C) -> (num_clusters, C, H * W)
        # residual: (N, num_clusters, C, H * W) 参见注释2
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -
                                  1, -1).permute(1, 2, 0).unsqueeze(0)

        # soft_assign: (N, num_clusters, H * W) -> (N, num_clusters, 1, H * W)
        # (N, num_clusters, C, H * W) * (N, num_clusters, 1, H * W)
        residual *= soft_assign.unsqueeze(2)
        # (N, num_clusters, C, H * W) -> (N, num_clusters, C)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        # flatten；vald: (N, num_clusters, C) -> (N, num_clusters * C)
        vlad = vlad.view(x.size(0), -1)
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
