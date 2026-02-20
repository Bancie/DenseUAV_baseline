"""Local Pattern Network (LPN) head for ViT-family backbones.

LPN splits the 2-D spatial arrangement of patch tokens into concentric
annular regions.  Each region is pooled independently and classified by its
own :class:`~utils.ClassBlock`.  A global branch also classifies the CLS
token.  During training both global and local logits / features are returned
as lists; during evaluation all feature vectors are stacked along the last
dimension for compact retrieval descriptors.
"""

import torch
import torch.nn as nn
import numpy as np
from .utils import ClassBlock
from torch.nn import functional as F
import math 


class LPN(nn.Module):
    """Local Pattern Network head that combines a global branch with
    concentric spatial region classifiers.

    The patch tokens are reshaped into a 2-D grid
    ``(N, C, sqrt(P), sqrt(P))`` and then split into ``block`` concentric
    ring regions via :meth:`get_part_pool`.  Each region and the CLS token
    are classified by dedicated :class:`~utils.ClassBlock` instances.

    Attributes:
        block (int): Number of concentric local regions.
        global_classifier (ClassBlock): Classifier for the CLS token.
        classifier_lpn_<i> (ClassBlock): Classifier for the *i*-th local
            region (attributes set dynamically for ``i`` in
            ``1..block``).
        opt: Option namespace used to build the module.
    """

    def __init__(self, opt):
        """Initializes LPN.

        Args:
            opt: Argument namespace with at least:
                - in_planes (int): Feature dimension per token.
                - nclasses (int): Number of identity classes.
                - droprate (float): Dropout probability.
                - num_bottleneck (int): Bottleneck projection dimension.
                - block (int): Number of concentric local regions to extract.
        """
        super().__init__()
        
        self.block = opt.block
        self.global_classifier = ClassBlock(opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck= opt.num_bottleneck)
        for i in range(opt.block):
            name = 'classifier_lpn_' + str(i+1)
            setattr(self, name, ClassBlock(opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck= opt.num_bottleneck))
        self.opt = opt

    def forward(self, features):
        """Extracts global and local features from ViT token output.

        Args:
            features (torch.Tensor): Token tensor from a ViT backbone with
                shape ``(N, num_patches + 1, C)``.  Index 0 is the CLS token.

        Returns:
            list: ``[total_cls, total_features]`` where:

            * During **training** – *total_cls* is a list of
              ``block + 1`` tensors each of shape ``(N, num_classes)``
              (global first, then local); *total_features* is a list of
              ``block + 1`` tensors each of shape ``(N, num_bottleneck)``.
            * During **evaluation** – *total_cls* is as above;
              *total_features* is a single stacked tensor of shape
              ``(N, num_bottleneck, block + 1)``.
        """
        cls_token = features[:, 0]
        image_tokens = features[:, 1:]
        # 全局特征
        global_cls, global_feature = self.global_classifier(cls_token)
        # LPN特征
        image_tokens = image_tokens.reshape(image_tokens.size(0),int(np.sqrt(image_tokens.size(1))),int(np.sqrt(image_tokens.size(1))),image_tokens.size(2))
        image_tokens = image_tokens.permute(0,3,1,2)
        LPN_result = self.get_part_pool(image_tokens).squeeze()
        LPN_cls_features = self.part_classifier(LPN_result)
        LPN_cls = []
        LPN_features = []
        for f in LPN_cls_features:
            LPN_cls.append(f[0])
            LPN_features.append(f[1])
        total_cls = [global_cls]+LPN_cls
        total_features = [global_feature]+LPN_features
        if not self.training:
            total_features = torch.stack(total_features,dim=-1)
        return [total_cls,total_features]
    

    def get_part_pool(self, x, pool='avg', no_overlap=True):
        """Pools concentric ring regions of a 2-D feature map.

        The feature map is conceptually divided into ``block`` concentric
        rectangular rings centred at the spatial midpoint.  Each ring is
        pooled to a ``(1, 1)`` descriptor via adaptive average (or max)
        pooling.  When ``no_overlap`` is ``True`` the content of the
        inner ring is subtracted from the outer crop before pooling, so
        each descriptor captures only the annular region.

        Args:
            x (torch.Tensor): Spatial feature map of shape
                ``(N, C, H, W)``.
            pool (str): Pooling strategy – ``"avg"`` (default) or
                ``"max"``.
            no_overlap (bool): If ``True``, remove inner-ring content
                from each outer crop to obtain non-overlapping annular
                descriptors.

        Returns:
            torch.Tensor: Concatenated pooled regions of shape
            ``(N, C, block, 1)`` (concatenated along dim 2 before
            being squeezed by the caller).
        """
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H / 2), int(W / 2)
        per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H + (self.block - c_h) * 2, W + (self.block - c_w) * 2
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H / 2), int(W / 2)
            per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                xmin = c_h - i * per_h
                xmax = c_h + i * per_h
                ymin = c_w - i * per_w
                ymax = c_w + i * per_w
                x_curr = x[:, :, xmin:xmax, ymin:ymax]
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    x_pad = F.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.append(avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    pad_h = c_h - (i - 1) * per_h
                    pad_w = c_w - (i - 1) * per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2) + 2 * pad_h == H:
                        x_pad = F.pad(x_pre, (pad_h, pad_h, pad_w, pad_w), "constant", 0)
                    else:
                        ep = H - (x_pre.size(2) + 2 * pad_h)
                        x_pad = F.pad(x_pre, (pad_h + ep, pad_h, pad_w + ep, pad_w), "constant", 0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.append(avgpool)
        return torch.cat(result, dim=2)

    def part_classifier(self, x, cls_name='classifier_lpn_'):
        """Applies a dedicated classifier to each local pooled region.

        Args:
            x (torch.Tensor): Pooled region tensor of shape
                ``(N, C, block)`` where the last dimension indexes the
                concentric regions.
            cls_name (str): Prefix of the dynamically set classifier
                attribute names.  The *i*-th classifier is accessed as
                ``self.<cls_name><i>`` for ``i`` in ``1..block``.

        Returns:
            list[list[torch.Tensor, torch.Tensor]]: A list of ``block``
            elements; each element is ``[cls, feature]`` returned by the
            corresponding :class:`~utils.ClassBlock`.
        """
        part = {}
        predict = {}
        for i in range(self.block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = cls_name + str(i + 1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        return y
