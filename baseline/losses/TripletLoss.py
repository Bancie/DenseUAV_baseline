import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalise a tensor to unit length along the specified axis.

    Args:
        x (torch.Tensor): Input tensor of arbitrary shape.
        axis (int, optional): Dimension along which to normalise.
            Defaults to ``-1`` (last dimension).

    Returns:
        torch.Tensor: L2-normalised tensor of the same shape as ``x``.
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-6)
    return x


def euclidean_dist(x, y):
    """Compute the pairwise Euclidean distance matrix between two sets of vectors.

    Args:
        x (torch.Tensor): Feature matrix of shape ``[m, d]``.
        y (torch.Tensor): Feature matrix of shape ``[n, d]``.

    Returns:
        torch.Tensor: Distance matrix of shape ``[m, n]`` where entry
        ``[i, j]`` is the Euclidean distance between ``x[i]`` and ``y[j]``.
        Values are clamped to ``1e-6`` before taking the square root for
        numerical stability.
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-6).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """Compute the pairwise cosine distance matrix between two sets of vectors.

    The cosine distance is defined as ``(1 - cosine_similarity) / 2``, so
    it lies in ``[0, 1]``.

    Args:
        x (torch.Tensor): Feature matrix of shape ``[m, d]``.
        y (torch.Tensor): Feature matrix of shape ``[n, d]``.

    Returns:
        torch.Tensor: Distance matrix of shape ``[m, n]`` where entry
        ``[i, j]`` is the cosine distance between ``x[i]`` and ``y[j]``.
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def example_mining(dist_mat, labels, return_inds=False):
    """Mine the hardest positive and a mean-distance negative for each anchor.

    For each anchor sample selects the hardest positive (maximum distance
    among same-class samples) and computes the *mean* distance to all
    negative (different-class) samples rather than the hardest negative.

    Args:
        dist_mat (torch.Tensor): Pairwise distance matrix of shape ``[N, N]``.
        labels (torch.Tensor): Class labels of shape ``[N]``.
        return_inds (bool, optional): Whether to also return the indices of
            selected positives and negatives.  Defaults to ``False``.

    Returns:
        tuple:
            * **dist_ap** (torch.Tensor): Distances to hardest positives,
              shape ``[N]``.
            * **dist_an** (torch.Tensor): Mean distances to negatives,
              shape ``[N]``.
            * **p_inds** (torch.Tensor, optional): Indices of selected hard
              positives, shape ``[N]``.  Only returned when
              ``return_inds=True``.
            * **n_inds** (torch.Tensor, optional): Indices of selected hard
              negatives, shape ``[N]``.  Only returned when
              ``return_inds=True``.

    Note:
        Only considers the case where all classes have the same number of
        samples, allowing all anchors to be processed in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, _ = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an = torch.mean(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


def hard_example_mining(dist_mat, labels, return_inds=False):
    """Mine the hardest positive and hardest negative for each anchor.

    For each anchor selects the hardest positive (maximum distance among
    same-class samples) and the hardest negative (minimum distance among
    different-class samples).

    Args:
        dist_mat (torch.Tensor): Pairwise distance matrix of shape ``[N, N]``.
        labels (torch.Tensor): Class labels of shape ``[N]``.
        return_inds (bool, optional): Whether to also return the indices of
            the selected hard positives and negatives.  Defaults to ``False``.

    Returns:
        tuple:
            * **dist_ap** (torch.Tensor): Distances to hardest positives,
              shape ``[N]``.
            * **dist_an** (torch.Tensor): Distances to hardest negatives,
              shape ``[N]``.
            * **p_inds** (torch.Tensor, optional): Global indices of selected
              hard positives, shape ``[N]``.  Only returned when
              ``return_inds=True``.
            * **n_inds** (torch.Tensor, optional): Global indices of selected
              hard negatives, shape ``[N]``.  Only returned when
              ``return_inds=True``.

    Note:
        Only considers the case where all classes have the same number of
        samples, allowing all anchors to be processed in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class HardMiningTripletLoss(object):
    """Triplet loss using *harder* example mining.

    Extends the standard hard-mining triplet loss by scaling the
    anchor-positive and anchor-negative distances by ``(1 + hard_factor)``
    and ``(1 - hard_factor)`` respectively, making the positives appear
    farther and the negatives appear closer before computing the margin
    ranking loss.

    Args:
        margin (float | None): Margin for ``nn.MarginRankingLoss``.  When
            ``None``, ``nn.SoftMarginLoss`` is used instead.
        hard_factor (float, optional): Scaling offset applied to distances
            before the ranking loss.  ``0.0`` recovers standard hard mining.
            Defaults to ``0.0``.
        normalize_feature (bool, optional): Whether to L2-normalise feature
            embeddings before computing distances.  Defaults to ``False``.
    """

    def __init__(self, margin=None, hard_factor=0.0, normalize_feature=False):
        self.margin = margin
        self.hard_factor = hard_factor
        self.normalize_feature = normalize_feature
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels):
        """Compute the harder-mining triplet loss.

        Args:
            global_feat (torch.Tensor): Feature matrix of shape
                ``(batch_size, feat_dim)``.
            labels (torch.Tensor): Ground-truth class indices of shape
                ``(batch_size,)``.

        Returns:
            torch.Tensor: Scalar triplet loss value.
        """
        if self.normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss


class TripletLoss(object):
    """Triplet loss using mean-negative example mining.

    Similar to ``HardMiningTripletLoss`` but uses ``example_mining`` which
    selects the hardest positive and the *mean* negative distance rather than
    the hardest negative, resulting in a smoother gradient signal.

    Args:
        margin (float | None): Margin for ``nn.MarginRankingLoss``.  When
            ``None``, ``nn.SoftMarginLoss`` is used instead.
        hard_factor (float, optional): Distance scaling offset.
            Defaults to ``0.0``.
        normalize_feature (bool, optional): Whether to L2-normalise features
            before computing distances.  Defaults to ``False``.
    """

    def __init__(self, margin=None, hard_factor=0.0, normalize_feature=False):
        self.margin = margin
        self.hard_factor = hard_factor
        self.normalize_feature = normalize_feature
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels):
        """Compute the mean-negative triplet loss.

        Args:
            global_feat (torch.Tensor): Feature matrix of shape
                ``(batch_size, feat_dim)``.
            labels (torch.Tensor): Ground-truth class indices of shape
                ``(batch_size,)``.

        Returns:
            torch.Tensor: Scalar triplet loss value.
        """
        if self.normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss


class Tripletloss(nn.Module):
    """Cross-view triplet loss with hard positive/negative mining.

    Mines hard positives and negatives *across views* only: for a sample in
    the first half of the batch (satellite) the positive and negative are
    drawn from the second half (drone) and vice versa.  This enforces
    cross-view discriminability rather than within-view discriminability.

    Reference:
        Hermans et al. *In Defense of the Triplet Loss for Person
        Re-Identification*. arXiv:1703.07737.

    Code imported from
    https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float, optional): Margin for ``nn.MarginRankingLoss``.
            Defaults to ``0.3``.
    """

    def __init__(self, margin=0.3):
        super(Tripletloss, self).__init__()
        self.margin = margin
        # MarginRankingLoss computes ReLU(ap - y*an + margin);
        # ap is the anchor-positive distance, an is the anchor-negative distance.
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """Compute the cross-view triplet loss.

        The input batch is assumed to be arranged as
        ``[satellite_samples | drone_samples]`` of equal size.  Hard positives
        and negatives are mined from the *opposite* half for each sample.

        Args:
            inputs (torch.Tensor): Feature matrix of shape
                ``(batch_size, feat_dim)`` where ``batch_size`` is even.
            targets (torch.Tensor): Ground-truth class indices of shape
                ``(batch_size,)``.

        Returns:
            torch.Tensor: Scalar triplet loss value.
        """

        n = inputs.size(0)

        inputs = normalize(inputs, axis=-1)
        dist = euclidean_dist(inputs, inputs)
        # For each anchor, find the hardest positive and negative in the opposite view
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            if i < n/2:
                dist_ap.append(dist[i][int(n/2):n][mask[i][int(n/2):n]].max().unsqueeze(0))
                dist_an.append(dist[i][int(n/2):n][(mask[i] == 0)[int(n/2):n]].min().unsqueeze(0))
            else:
                dist_ap.append(dist[i][0:int(n/2)][mask[i][0:int(n/2)]].max().unsqueeze(0))
                dist_an.append(dist[i][0:int(n/2)][(mask[i] == 0)[0:int(n/2)]].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)

        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class WeightedSoftTripletLoss(nn.Module):
    """Soft triplet loss with log-sum-exp weighting.

    Computes a smooth triplet loss using the formula:

    .. math::

        \\mathcal{L} = \\frac{1}{N}\\sum_{i=1}^{N}
        \\log\\!\\left(1 + e^{\\alpha(d_{ap,i} - d_{an,i})}\\right)

    where ``α`` (``self.alpha``) is a temperature parameter that controls
    the sharpness of the soft-max weighting.  Hard example mining is used to
    select ``d_ap`` and ``d_an`` before applying the smooth loss, combining
    the benefits of hard mining with a differentiable, margin-free objective.

    Args:
        None.  The temperature parameter ``alpha`` is fixed at ``10``.
    """

    def __init__(self):
        super(WeightedSoftTripletLoss, self).__init__()
        self.alpha = 10

    def forward(self, inputs, targets):
        """Compute the weighted soft triplet loss.

        Args:
            inputs (torch.Tensor): Feature matrix of shape
                ``(batch_size, feat_dim)``.
            targets (torch.Tensor): Ground-truth class indices of shape
                ``(batch_size,)``.

        Returns:
            torch.Tensor: Scalar loss value.
        """

        n = inputs.size(0)

        inputs = normalize(inputs, axis=-1)

        dist = euclidean_dist(inputs, inputs)
        dist_ap, dist_an = hard_example_mining(dist, targets)
        loss = torch.log(1+torch.exp(self.alpha*(dist_ap-dist_an))).mean()
        return loss
