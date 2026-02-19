import torch
from torch import nn
from .TripletLoss import Tripletloss, WeightedSoftTripletLoss, HardMiningTripletLoss, TripletLoss
from .FocalLoss import FocalLoss
import torch.nn.functional as F
from torch.autograd import Variable


class Loss(nn.Module):
    """Combined training loss for cross-view geo-localisation.

    Aggregates three optional loss components:

    1. **Classification loss** – cross-entropy (CELoss) or focal loss
       (FocalLoss) applied independently to satellite and drone predictions.
    2. **Triplet / feature loss** – one of several triplet loss variants
       (``TripletLoss``, ``HardMiningTripletLoss``, ``Tripletloss``,
       ``WeightedSoftTripletLoss``) applied to concatenated satellite + drone
       feature embeddings.
    3. **KL divergence loss** – mutual learning signal computed between the
       satellite and drone class logits.

    Args:
        opt: Configuration namespace with the following attributes:

            * ``cls_loss`` (str): ``"CELoss"``, ``"FocalLoss"``, or any
              other value to disable classification loss.
            * ``feature_loss`` (str): ``"TripletLoss"``,
              ``"HardMiningTripletLoss"``, ``"Tripletloss"``,
              ``"WeightedSoftTripletLoss"``, or any other value to disable.
            * ``kl_loss`` (str): ``"KLLoss"`` to enable KL divergence, or
              any other value to disable.
            * ``nclasses`` (int): Number of geo-location classes; used by
              ``FocalLoss``.
            * ``batchsize`` (int): Batch size; used to compute ``split_num``
              for the triplet loss.
            * ``sample_num`` (int): Number of samples per class in a batch.
    """

    def __init__(self, opt) -> None:
        super(Loss, self).__init__()
        self.opt = opt
        # Classification loss
        if opt.cls_loss == "CELoss":
            self.cls_loss = nn.CrossEntropyLoss()
        elif opt.cls_loss == "FocalLoss":
            self.cls_loss = FocalLoss(alpha=0.25, gamma=2, num_classes=opt.nclasses)
        else:
            self.cls_loss = None

        # Feature (contrastive/triplet) loss
        if opt.feature_loss == "TripletLoss":
            self.feature_loss = TripletLoss(margin=0.3, normalize_feature=True)
        elif opt.feature_loss == "HardMiningTripletLoss":
            self.feature_loss = HardMiningTripletLoss(margin=0.3, normalize_feature=True)
        elif opt.feature_loss == "Tripletloss":
            self.feature_loss = Tripletloss(margin=0.3)
        elif opt.feature_loss == "WeightedSoftTripletLoss":
            self.feature_loss = WeightedSoftTripletLoss()
        else:
            self.feature_loss = None

        # KL divergence loss for mutual learning between views
        if opt.kl_loss == "KLLoss":
            self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        else:
            self.kl_loss = None

    def forward(self, outputs, outputs2, labels, labels2):
        """Compute the combined loss for one training step.

        Args:
            outputs (tuple): ``(cls1, feature1)`` from the satellite branch,
                where ``cls1`` is the logit tensor (or list of tensors) and
                ``feature1`` is the embedding (or list of embeddings).
            outputs2 (tuple): ``(cls2, feature2)`` from the drone branch,
                with the same structure as ``outputs``.
            labels (torch.Tensor): Integer class labels for the satellite
                samples, shape ``(B,)``.
            labels2 (torch.Tensor): Integer class labels for the drone
                samples, shape ``(B,)``.

        Returns:
            tuple:
                * **loss** (torch.Tensor): Scalar total loss.
                * **res_cls_loss** (torch.Tensor): Classification loss
                  component (zero tensor if disabled).
                * **res_triplet_loss** (torch.Tensor): Triplet loss component
                  (zero tensor if disabled).
                * **res_kl_loss** (torch.Tensor): KL divergence loss component
                  (zero tensor if disabled).
        """
        cls1, feature1 = outputs
        cls2, feature2 = outputs2
        loss = 0

        # Classification loss
        res_cls_loss = torch.tensor((0))
        if self.cls_loss is not None:
            res_cls_loss = self.calc_cls_loss(cls1, labels, self.cls_loss) + \
                self.calc_cls_loss(cls2, labels2, self.cls_loss)
            loss += res_cls_loss

        # Triplet loss
        res_triplet_loss = torch.tensor((0))
        if self.feature_loss is not None:
            split_num = self.opt.batchsize//self.opt.sample_num
            res_triplet_loss = self.calc_triplet_loss(
                feature1, feature2, labels, self.feature_loss, split_num)
            loss += res_triplet_loss

        # KL divergence loss for mutual learning between views
        res_kl_loss = torch.tensor((0))
        if self.kl_loss is not None:
            res_kl_loss = self.calc_kl_loss(cls1, cls2, self.kl_loss)
            loss += res_kl_loss

        return loss, res_cls_loss, res_triplet_loss, res_kl_loss

    def calc_cls_loss(self, outputs, labels, loss_func):
        """Compute average classification loss over a list of output heads.

        Supports multi-head classifiers: when ``outputs`` is a list the loss
        is summed over all heads and divided by the number of heads.

        Args:
            outputs (torch.Tensor | list[torch.Tensor]): Logits from one or
                more classifier heads, each with shape ``(B, num_classes)``.
            labels (torch.Tensor): Ground-truth class indices, shape ``(B,)``.
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

    def calc_kl_loss(self, outputs, outputs2, loss_func):
        """Compute KL divergence loss for mutual learning between two views.

        Treats ``outputs`` as the log-probability distribution and
        ``outputs2`` as the target probability distribution (detached via
        ``Variable``).  When both outputs are lists the loss is averaged over
        all corresponding head pairs.

        Args:
            outputs (torch.Tensor | list[torch.Tensor]): Logits from view 1
                (satellite), used as the predicted log-probabilities.
            outputs2 (torch.Tensor | list[torch.Tensor]): Logits from view 2
                (drone), used as the target probabilities (treated as
                constant).
            loss_func (nn.KLDivLoss): KL divergence loss module.

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

    def calc_triplet_loss(self, outputs, outputs2, labels, loss_func, split_num=8):
        """Compute triplet loss on concatenated satellite and drone embeddings.

        Concatenates the satellite and drone feature tensors along the batch
        dimension to create a single batch that contains samples from both
        views with shared labels, then computes the triplet loss.  When
        ``outputs`` is a list the loss is averaged over all feature heads.

        Args:
            outputs (torch.Tensor | list[torch.Tensor]): Feature embeddings
                from the satellite branch, shape ``(B, D)`` or list thereof.
            outputs2 (torch.Tensor | list[torch.Tensor]): Feature embeddings
                from the drone branch, shape ``(B, D)`` or list thereof.
            labels (torch.Tensor): Class labels for the satellite samples,
                shape ``(B,)``.  The same labels are used for the drone
                samples (same-class pairing is assumed).
            loss_func (callable): Triplet loss callable.
            split_num (int, optional): Unused split parameter kept for API
                compatibility.  Defaults to ``8``.

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
