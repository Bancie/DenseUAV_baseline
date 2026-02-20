import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch import nn
from .TripletLoss import Tripletloss, WeightedSoftTripletLoss


def cal_cls_loss(outputs, labels, loss_func):
    """Compute the average classification loss over one or more output heads.

    When ``outputs`` is a list (multi-head classifier) the loss is computed
    independently for each head and averaged.  When it is a single tensor
    the loss is computed directly.

    Args:
        outputs (torch.Tensor | list[torch.Tensor]): Logits from one or
            more classifier heads, each with shape ``(B, num_classes)``.
        labels (torch.Tensor): Ground-truth integer class indices, shape
            ``(B,)``.
        loss_func (callable): Classification loss function, e.g.
            ``nn.CrossEntropyLoss()``.

    Returns:
        torch.Tensor: Scalar classification loss averaged over heads.
    """
    loss = 0
    if isinstance(outputs, list):
        for i in outputs:
            loss += loss_func(i, labels)
        loss = loss/len(outputs)
    else:
        loss = loss_func(outputs, labels)
    return loss


def cal_kl_loss(outputs, outputs2, loss_func):
    """Compute KL divergence for mutual learning between two network branches.

    Uses ``outputs`` as the predicted log-probability distribution
    (``log_softmax``) and ``outputs2`` as the target probability distribution
    (``softmax``, treated as a constant via ``Variable``).  When both are
    lists the loss is averaged over all corresponding head pairs.

    Args:
        outputs (torch.Tensor | list[torch.Tensor]): Logits from branch 1
            (e.g. satellite), used as the predicted distribution.
        outputs2 (torch.Tensor | list[torch.Tensor]): Logits from branch 2
            (e.g. drone), used as the target distribution (treated as
            constant / detached).
        loss_func (nn.KLDivLoss): KL divergence loss module configured with
            ``reduction='batchmean'``.

    Returns:
        torch.Tensor: Scalar KL divergence loss averaged over heads.
    """
    loss = 0
    if isinstance(outputs, list):
        for i in range(len(outputs)):
            loss += loss_func(F.log_softmax(outputs[i], dim=1),
                              F.softmax(Variable(outputs2[i]), dim=1))
        loss = loss/len(outputs)
    else:
        loss = loss_func(F.log_softmax(outputs, dim=1),
                         F.softmax(Variable(outputs2), dim=1))
    return loss


def cal_triplet_loss(outputs, outputs2, labels, loss_func, split_num=8):
    """Compute triplet loss on concatenated embeddings from two branches.

    Concatenates the feature tensors from both branches along the batch
    dimension and applies the triplet loss on the combined batch.  When
    ``outputs`` is a list the loss is averaged over all feature heads.

    Args:
        outputs (torch.Tensor | list[torch.Tensor]): Feature embeddings from
            branch 1 (e.g. satellite), shape ``(B, D)`` or list thereof.
        outputs2 (torch.Tensor | list[torch.Tensor]): Feature embeddings from
            branch 2 (e.g. drone), shape ``(B, D)`` or list thereof.
        labels (torch.Tensor): Class labels shared by both branches, shape
            ``(B,)``.  Labels are duplicated when concatenating.
        loss_func (callable): Triplet loss callable that accepts
            ``(features, labels)``.
        split_num (int, optional): Kept for API compatibility; currently
            unused.  Defaults to ``8``.

    Returns:
        torch.Tensor: Scalar triplet loss averaged over feature heads.
    """
    if isinstance(outputs, list):
        loss = 0
        for i in range(len(outputs)):
            out_concat = torch.cat((outputs[i], outputs2[i]), dim=0)
            labels_concat = torch.cat((labels, labels), dim=0)
            loss += loss_func(out_concat, labels_concat)
        loss = loss/len(outputs)
    else:
        out_concat = torch.cat((outputs, outputs2), dim=0)
        labels_concat = torch.cat((labels, labels), dim=0)
        loss = loss_func(out_concat, labels_concat)
    return loss


def cal_loss(opt, outputs, outputs2, labels, labels3):
    """Compute the combined training loss for a single optimisation step.

    Aggregates up to three loss components depending on ``opt`` flags:

    1. **Classification loss** (always active) – cross-entropy applied
       independently to satellite and drone classifier outputs.
    2. **Triplet loss** (controlled by ``opt.triplet_loss``) – either
       ``WeightedSoftTripletLoss`` (when ``opt.WSTR`` is ``True``) or the
       standard ``Tripletloss`` with ``margin=0.3``.
    3. **KL divergence loss** (controlled by ``opt.kl_loss``) – mutual
       learning signal between satellite and drone logits.

    Args:
        opt: Configuration namespace with the following attributes:

            * ``triplet_loss`` (bool): Enable triplet loss.
            * ``WSTR`` (bool): Use ``WeightedSoftTripletLoss`` when
              ``True``, otherwise use ``Tripletloss``.
            * ``kl_loss`` (bool): Enable KL divergence loss.
            * ``batchsize`` (int): Batch size.
            * ``sample_num`` (int): Number of samples per class per batch.

        outputs (tuple): ``(cls1, feature1)`` from the satellite branch,
            where ``cls1`` is the classification logit(s) and ``feature1``
            is the feature embedding(s).
        outputs2 (tuple): ``(cls2, feature2)`` from the drone branch with
            the same structure as ``outputs``.
        labels (torch.Tensor): Class labels for the satellite branch, shape
            ``(B,)``.
        labels3 (torch.Tensor): Class labels for the drone branch, shape
            ``(B,)``.

    Returns:
        tuple:
            * **loss** (torch.Tensor): Scalar total loss.
            * **cls_loss** (torch.Tensor): Classification loss component.
            * **f_triplet_loss** (torch.Tensor): Triplet loss component
              (zero tensor if disabled).
            * **kl_loss** (torch.Tensor): KL divergence loss component
              (zero tensor if disabled).
    """
    cls1, feature1 = outputs
    cls2, feature2 = outputs2
    loss = 0.0
    criterion = nn.CrossEntropyLoss()
    loss_kl = nn.KLDivLoss(reduction='batchmean')
    if opt.WSTR:
        triplet_loss = WeightedSoftTripletLoss()
    else:
        triplet_loss = Tripletloss(margin=0.3)
    # Triplet loss
    f_triplet_loss = torch.tensor((0))
    if opt.triplet_loss:
        split_num = opt.batchsize//opt.sample_num
        f_triplet_loss = cal_triplet_loss(
            feature1, feature2, labels, triplet_loss, split_num)
        loss += f_triplet_loss

    # Classification loss
    cls_loss = cal_cls_loss(cls1, labels, criterion) + \
        cal_cls_loss(cls2, labels3, criterion)
    loss += cls_loss
    # KL divergence loss for mutual learning
    kl_loss = torch.tensor((0))
    if opt.kl_loss:
        kl_loss = cal_kl_loss(cls1, cls2, loss_kl)
        loss += kl_loss

    return loss, cls_loss, f_triplet_loss, kl_loss
