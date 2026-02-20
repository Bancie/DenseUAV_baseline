"""NeXtVLAD aggregation head for ViT-family backbones.

NeXtVLAD (Lin et al., 2018) extends NetVLAD by first expanding and then
grouping descriptors before VLAD aggregation, significantly reducing the
parameter count while retaining expressive power.  This module adapts the
formulation to operate on patch tokens from Vision Transformer backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ClassBlock, Pooling, vector2image


class NeXtVLAD(nn.Module):
    """Top-level NeXtVLAD head combining token reshaping with grouped VLAD
    aggregation.

    Patch tokens (excluding the CLS token) are transposed and reshaped into
    a 2-D spatial grid before being passed to :class:`NeXtVLAD_block`.  The
    resulting descriptor is classified by a :class:`~utils.ClassBlock`.

    Attributes:
        opt: Option namespace used to build the module.
        classifier (ClassBlock): Linear bottleneck + BN + classification head
            operating on the NeXtVLAD descriptor.
        netvlad (NeXtVLAD_block): Core grouped VLAD aggregation module.
    """

    def __init__(self, opt) -> None:
        """Initializes the NeXtVLAD head.

        Args:
            opt: Argument namespace with at least:
                - in_planes (int): Feature dimension per patch token.
                - nclasses (int): Number of identity classes.
                - droprate (float): Dropout probability.
                - num_bottleneck (int): Bottleneck projection dimension for
                  :class:`~utils.ClassBlock`.
                - block (int): Number of VLAD clusters passed to
                  :class:`NeXtVLAD_block` as ``num_clusters``.  The output
                  descriptor dimension is ``in_planes * block``.
        """
        super(NeXtVLAD, self).__init__()
        self.opt = opt
        self.classifier = ClassBlock(
            int(opt.in_planes*opt.block), opt.nclasses, opt.droprate, num_bottleneck=opt.num_bottleneck)
        self.netvlad = NeXtVLAD_block(
            num_clusters=opt.block, dim=opt.in_planes)

    def forward(self, features):
        """Aggregates patch tokens via NeXtVLAD and classifies the result.

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
    


class NeXtVLAD_block(nn.Module):
    """NeXtVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=1024, lamb=2, groups=8, max_frames=300):
        """Initializes NeXtVLAD_block.

        Args:
            num_clusters (int): Number of VLAD cluster centres *K*.
            dim (int): Input descriptor dimension *D*.
            lamb (int): Expansion factor λ.  The descriptor is first
                projected to ``λ * D`` before grouping.
            groups (int): Number of feature groups *G*.  The expanded
                descriptor is split into *G* groups of size
                ``λ * D // G`` each.
            max_frames (int): Maximum sequence length (used to size the
                :class:`nn.BatchNorm1d` layer applied before soft
                assignment).
        """
        super(NeXtVLAD_block, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.K = num_clusters
        self.G = groups
        self.group_size = int((lamb * dim) // self.G)
        # expansion FC
        self.fc0 = nn.Linear(dim, lamb * dim)
        # soft assignment FC (the cluster weights)
        self.fc_gk = nn.Linear(lamb * dim, self.G * self.K)
        # attention over groups FC
        self.fc_g = nn.Linear(lamb * dim, self.G)
        self.cluster_weights2 = nn.Parameter(torch.rand(1, self.group_size, self.K))

        self.bn0 = nn.BatchNorm1d(max_frames)
        self.bn1 = nn.BatchNorm1d(1)

    def forward(self, x, mask=None):
        """Computes the NeXtVLAD descriptor for a batch of local feature maps.

        The forward pass follows the NeXtVLAD pipeline:

        1. **Expand** each descriptor from *D* to λ*D* via a linear layer.
        2. **Group** the expanded descriptor into *G* groups of size λ*D/G*.
        3. **Soft-assign** each group to *K* clusters.
        4. **Attend** across groups with a sigmoid gating.
        5. **Aggregate** residuals to cluster centres.
        6. **Normalise** and flatten.

        Args:
            x (torch.Tensor): Local feature map of shape ``(N, C, H, W)``
                produced by :func:`~utils.vector2image`, where ``H * W``
                equals ``M`` (the number of local descriptors) and ``C``
                equals the token dimension.  Internally reshaped to
                ``(N, M, C)`` before processing.
            mask (torch.Tensor or None): Optional attention mask of shape
                ``(N, M)`` to suppress contributions from padded positions.

        Returns:
            torch.Tensor: NeXtVLAD descriptor of shape
            ``(N, K * group_size)``.
        """
        #         print(f"x: {x.shape}")

        _, M, N = x.shape
        # expansion FC: B x M x N -> B x M x λN
        x_dot = self.fc0(x)

        # reshape into groups: B x M x λN -> B x M x G x (λN/G)
        x_tilde = x_dot.reshape(-1, M, self.G, self.group_size)

        # residuals across groups and clusters: B x M x λN -> B x M x (G*K)
        WgkX = self.fc_gk(x_dot)
        WgkX = self.bn0(WgkX)

        # residuals reshape across clusters: B x M x (G*K) -> B x (M*G) x K
        WgkX = WgkX.reshape(-1, M * self.G, self.K)

        # softmax over assignment: B x (M*G) x K -> B x (M*G) x K
        alpha_gk = F.softmax(WgkX, dim=-1)

        # attention across groups: B x M x λN -> B x M x G
        alpha_g = torch.sigmoid(self.fc_g(x_dot))
        if mask is not None:
            alpha_g = torch.mul(alpha_g, mask.unsqueeze(2))

        # reshape across time: B x M x G -> B x (M*G) x 1
        alpha_g = alpha_g.reshape(-1, M * self.G, 1)

        # apply attention: B x (M*G) x K (X) B x (M*G) x 1 -> B x (M*G) x K
        activation = torch.mul(alpha_gk, alpha_g)

        # sum over time and group: B x (M*G) x K -> B x 1 x K
        a_sum = torch.sum(activation, -2, keepdim=True)

        # calculate group centers: B x 1 x K (X) 1 x (λN/G) x K -> B x (λN/G) x K
        a = torch.mul(a_sum, self.cluster_weights2)

        # permute: B x (M*G) x K -> B x K x (M*G)
        activation = activation.permute(0, 2, 1)

        # reshape: B x M x G x (λN/G) -> B x (M*G) x (λN/G)
        reshaped_x_tilde = x_tilde.reshape(-1, M * self.G, self.group_size)

        # cluster activation: B x K x (M*G) (X) B x (M*G) x (λN/G) -> B x K x (λN/G)
        vlad = torch.matmul(activation, reshaped_x_tilde)
        # print(f"vlad: {vlad.shape}")

        # permute: B x K x (λN/G) (X) B x (λN/G) x K
        vlad = vlad.permute(0, 2, 1)
        # distance to centers: B x (λN/G) x K (-) B x (λN/G) x K
        vlad = torch.sub(vlad, a)
        # normalize: B x (λN/G) x K
        vlad = F.normalize(vlad, 1)
        # reshape: B x (λN/G) x K -> B x 1 x (K * (λN/G))
        vlad = vlad.reshape(-1, 1, self.K * self.group_size)
        vlad = self.bn1(vlad)
        # reshape:  B x 1 x (K * (λN/G)) -> B x (K * (λN/G))
        vlad = vlad.reshape(-1, self.K * self.group_size)

        return vlad
