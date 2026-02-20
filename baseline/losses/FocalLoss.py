from torch import nn
import torch
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Focal Loss for dense prediction tasks.

    Implements the focal loss proposed in:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    The loss down-weights well-classified examples and focuses training on
    hard negatives using a modulating factor ``(1 - p_t) ** gamma``:

    .. math::

        FL(p_t) = -\\alpha_t \\cdot (1 - p_t)^{\\gamma} \\cdot \\log(p_t)

    Args:
        alpha (float | list[float]): Class weighting coefficient(s).

            * **float** – The first class receives weight ``alpha`` and all
              remaining classes receive ``1 - alpha``.  Commonly used in
              object detection to suppress the background class (e.g.
              ``alpha=0.25`` in RetinaNet).  Must satisfy ``alpha < 1``.
            * **list[float]** – Per-class weights of length ``num_classes``.

        gamma (float): Focusing parameter that reduces the loss contribution
            of easy examples.  ``gamma=0`` recovers cross-entropy.  Set to
            ``2`` in the original paper.
        num_classes (int): Total number of classes.
        size_average (bool, optional): If ``True`` the loss is averaged over
            the batch; otherwise it is summed.  Defaults to ``True``.
    """

    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)  # alpha vector: [alpha, 1-alpha, 1-alpha, ...]

        self.gamma = gamma

    def forward(self, preds, labels):
        """Compute the focal loss between predictions and ground-truth labels.

        Args:
            preds (torch.Tensor): Raw (un-normalised) class scores.  Accepts
                shapes ``[B, C]`` (classification) or ``[B, N, C]``
                (detection with ``N`` proposals), where ``B`` is the batch
                size, ``N`` the number of detections, and ``C`` the number
                of classes.
            labels (torch.Tensor): Ground-truth class indices.  Accepts
                shapes ``[B]`` or ``[B, N]`` matching ``preds``.

        Returns:
            torch.Tensor: Scalar focal loss, averaged or summed over the
            batch depending on ``self.size_average``.
        """
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)     # softmax

        # Gather the predicted probability for the ground-truth class (nll_loss part)
        preds_softmax = preds_softmax.gather(1, labels.reshape(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.reshape(-1, 1))
        alpha = self.alpha.gather(0, labels.reshape(-1))
        # (1 - p_t) ** gamma  is the focal modulation factor
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
