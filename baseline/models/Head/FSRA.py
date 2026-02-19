"""Feature-Slicing and Re-Aggregation (FSRA) head for ViT-family backbones.

FSRA ranks patch tokens by their mean activation (heatmap) in descending
order and then splits the ranked sequence into ``block`` equal groups.  Each
group is mean-pooled into a local descriptor and classified independently.
The CLS token is classified by a separate global branch.

During training both global and local ``(logits, feature)`` pairs are
returned as lists; during evaluation the feature vectors are stacked along
the last dimension.
"""

import torch
import torch.nn as nn
from .utils import ClassBlock


class FSRA(nn.Module):
    """Attention-heatmap-guided local+global classification head.

    Attributes:
        opt: Option namespace used to build the module.
        class_name (str): Attribute-name prefix for local classifiers
            (``"classifier_heat"``).
        block (int): Number of local feature groups.
        classifier1 (ClassBlock): Global classifier for the CLS token.
        classifier_heat<i> (ClassBlock): Local classifier for the *i*-th
            token group (attributes set dynamically for ``i`` in
            ``1..block``).
    """

    def __init__(self, opt) -> None:
        """Initializes FSRA.

        Args:
            opt: Argument namespace with at least:
                - in_planes (int): Feature dimension per token.
                - nclasses (int): Number of identity classes.
                - droprate (float): Dropout probability.
                - block (int): Number of local token groups.  When
                  ``block == 1`` only the global branch is active.
        """
        super().__init__()

        self.opt = opt
        num_classes = opt.nclasses
        droprate = opt.droprate
        in_planes = opt.in_planes
        self.class_name = "classifier_heat"
        self.block = opt.block
        # global classifier
        self.classifier1 = ClassBlock(in_planes, num_classes, droprate)
        # local classifier
        for i in range(self.block):
            name = self.class_name + str(i+1)
            setattr(self, name, ClassBlock(in_planes, num_classes, droprate))

    def forward(self, features):
        """Produces global and local logits/features from ViT token output.

        When ``block == 1`` only the global branch executes and a flat
        ``(global_cls, global_feature)`` pair is returned.

        Args:
            features (torch.Tensor): Token tensor from a ViT backbone with
                shape ``(N, num_patches + 1, C)``.  Index 0 is the CLS token.

        Returns:
            list: ``[total_cls, total_features]`` where:

            * During **training** – *total_cls* is a list of
              ``block + 1`` tensors each of shape ``(N, num_classes)``
              (global first, then local); *total_features* is a list of
              ``block + 1`` tensors each of shape ``(N, in_planes)``.
            * During **evaluation** – *total_features* is stacked into a
              tensor of shape ``(N, in_planes, block + 1)``.
        """
        global_cls, global_feature = self.classifier1(features[:, 0])
        # tranformer_feature = torch.mean(features,dim=1)
        # tranformer_feature = self.classifier1(tranformer_feature)
        if self.block == 1:
            return global_cls, global_feature

        part_features = features[:, 1:]

        heat_result = self.get_heartmap_pool(part_features)
        cls_list, features_list = self.part_classifier(
            self.block, heat_result, cls_name=self.class_name)

        total_cls = [global_cls] + cls_list
        total_features = [global_feature] + features_list
        if not self.training:
            total_features = torch.stack(total_features,dim=-1)
        return [total_cls, total_features]

    def get_heartmap_pool(self, part_features, add_global=False, otherbranch=False):
        """Ranks patch tokens by activation magnitude and splits into groups.

        Each patch token is scored by its mean value across the channel
        dimension (a cheap proxy for saliency).  Tokens are sorted in
        descending order and split into ``block`` equal-sized groups.
        Each group is mean-pooled to produce one local descriptor.

        Args:
            part_features (torch.Tensor): Patch token tensor of shape
                ``(N, num_patches, C)`` (CLS token excluded).
            add_global (bool): If ``True``, add the global mean feature
                (broadcast over blocks) to each local descriptor.
            otherbranch (bool): If ``True``, also return the mean of
                local descriptors from groups 1..block as a second
                output (used for auxiliary losses).

        Returns:
            torch.Tensor or tuple[torch.Tensor, torch.Tensor]:
            - When ``otherbranch=False`` (default): local descriptor
              tensor of shape ``(N, C, block)``.
            - When ``otherbranch=True``: a tuple
              ``(part_features_, otherbranch_)`` where
              *part_features_* has shape ``(N, C, block)`` and
              *otherbranch_* has shape ``(N, C)``.
        """
        heatmap = torch.mean(part_features, dim=-1)
        size = part_features.size(1)
        arg = torch.argsort(heatmap, dim=1, descending=True)
        x_sort = [part_features[i, arg[i], :]
                  for i in range(part_features.size(0))]
        x_sort = torch.stack(x_sort, dim=0)

        split_each = size / self.block
        split_list = [int(split_each) for i in range(self.block - 1)]
        split_list.append(size - sum(split_list))
        split_x = x_sort.split(split_list, dim=1)

        split_list = [torch.mean(split, dim=1) for split in split_x]
        part_featuers_ = torch.stack(split_list, dim=2)
        if add_global:
            global_feat = torch.mean(part_features, dim=1).view(
                part_features.size(0), -1, 1).expand(-1, -1, self.block)
            part_featuers_ = part_featuers_ + global_feat
        if otherbranch:
            otherbranch_ = torch.mean(
                torch.stack(split_list[1:], dim=2), dim=-1)
            return part_featuers_, otherbranch_
        return part_featuers_

    def part_classifier(self, block, x, cls_name='classifier_lpn'):
        """Applies a dedicated classifier to each local descriptor group.

        Args:
            block (int): Number of local groups.
            x (torch.Tensor): Local descriptor tensor of shape
                ``(N, C, block)``.
            cls_name (str): Prefix of the dynamically set classifier
                attribute names.  The *i*-th classifier is accessed as
                ``self.<cls_name><i+1>``.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]:
            ``(cls_list, features_list)`` where each list contains
            *block* tensors.  *cls_list[i]* has shape
            ``(N, num_classes)`` and *features_list[i]* has shape
            ``(N, in_planes)``.
        """
        part = {}
        cls_list, features_list = [], []
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = cls_name + str(i+1)
            c = getattr(self, name)
            res = c(part[i])
            cls_list.append(res[0])
            features_list.append(res[1])
        return cls_list, features_list
