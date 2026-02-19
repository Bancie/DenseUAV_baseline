"""Legacy two-view geo-localization model (ResNet / Transformer backbone).

This module contains the original DenseUAV model classes that pre-date the
modular backbone/head refactor.  New experiments should use
:mod:`taskflow` instead.  The classes here are retained for reproducibility
and backward-compatible checkpoint loading.
"""

import argparse
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
# from .transformer.modeling import VisionTransformer,CONFIGS
# from models.FPN import FPN
import math
# from tool.roi_align import RoIAlign
from resnest.torch import resnest50,resnest101
from torch.nn import AvgPool2d
import copy
from .TransReId import make_transformer_model
import timm

######################################################################
class GeM(nn.Module):
    # channel-wise GeM zhedong zheng
    """Channel-wise Generalized Mean Pooling with a learnable exponent.

    Each channel has an independent learnable exponent parameter *p_c*.
    The pooling for channel *c* is::

        GeM_c(x) = (avg((x_c + eps) ^ p_c)) ^ (1 / p_c)

    Attributes:
        p (nn.Parameter): Per-channel exponent vector of shape ``(dim,)``.
        eps (float): Small constant for numerical stability.
    """

    def __init__(self, dim = 2048, p=3, eps=1e-6):
        """Initializes GeM.

        Args:
            dim (int): Number of input channels.
            p (float): Initial value for all exponent elements.
            eps (float): Numerical stability constant added before
                exponentiation.
        """
        super(GeM,  self).__init__()
        self.p = nn.Parameter(torch.ones(dim)*p) #initial p
        self.eps = eps

    def forward(self, x):
        """Applies channel-wise GeM pooling.

        Args:
            x (torch.Tensor): Input feature map of shape ``(N, C, H, W)``.

        Returns:
            torch.Tensor: Pooled descriptor of shape ``(N, C)``.
        """
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        """Core GeM computation.

        Args:
            x (torch.Tensor): Input feature map of shape ``(N, C, H, W)``.
            p (torch.Tensor or float): Per-channel exponent vector or scalar.
            eps (float): Numerical stability constant.

        Returns:
            torch.Tensor: Pooled descriptor of shape ``(N, C)``.
        """
        x = torch.transpose(x, 1, -1)
        x = (x+eps).pow(p)
        x = torch.transpose(x, 1, -1)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.view(x.size(0), x.size(1))
        x = x.pow(1./p)
        return x

    def __repr__(self):
        """Returns a string representation showing the first exponent value and eps."""
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def weights_init_kaiming(m):
    """Applies Kaiming normal initialization to Conv, Linear, and BN layers.

    Intended to be passed to :meth:`nn.Module.apply`.

    * **Conv**: fan-in Kaiming normal for weights.
    * **Linear**: fan-out Kaiming normal for weights; constant 0 for bias.
    * **BatchNorm1d**: normal(1.0, 0.02) for weight; constant 0 for bias.

    Args:
        m (nn.Module): A single leaf module to initialize in-place.
    """
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    """Initializes classification Linear layers with a near-zero normal.

    Near-zero weight initialization (std=0.001) keeps initial class scores
    close to uniform, stabilizing early training.  Intended to be passed to
    :meth:`nn.Module.apply`.

    Args:
        m (nn.Module): A single leaf module to initialize in-place.
            Only :class:`nn.Linear` layers are affected.
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    """Sets all ReLU layers to in-place mode to reduce memory usage.

    Intended to be passed to :meth:`nn.Module.apply`.

    Args:
        m (nn.Module): A single leaf module to modify in-place.
    """
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    """Configurable bottleneck classification block (legacy version).

    Architecture::

        [Linear(input_dim → num_bottleneck)] → BN → [LeakyReLU] →
        [Dropout] → Linear(num_bottleneck → class_num)

    When ``return_f=True`` and the model is in training mode the block
    returns ``(logits, feature)``.  During evaluation only the bottleneck
    feature is returned (no logits).

    Attributes:
        return_f (bool): Whether to return the bottleneck feature during
            training.
        add_block (nn.Sequential): Feature-extraction sub-network.
        classifier (nn.Sequential): Single linear classification layer.
    """

    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        """Initializes ClassBlock.

        Args:
            input_dim (int): Dimensionality of the input feature vector.
            class_num (int): Number of output classes.
            droprate (float): Dropout probability (0 disables dropout).
            relu (bool): Insert a LeakyReLU(0.1) after BN if ``True``.
            bnorm (bool): Insert a BatchNorm1d layer if ``True``.
            num_bottleneck (int): Width of the intermediate bottleneck.
                Ignored when ``linear=False``.
            linear (bool): If ``True``, prepend a linear projection from
                ``input_dim`` to ``num_bottleneck``.  If ``False``,
                ``num_bottleneck`` is set to ``input_dim``.
            return_f (bool): If ``True``, return ``(logits, feature)``
                during training; return only ``feature`` during evaluation.
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
        if droprate>0:
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
        """Extracts bottleneck features and optionally computes class logits.

        Args:
            x (torch.Tensor): Input feature vector of shape
                ``(N, input_dim)``.

        Returns:
            torch.Tensor or tuple[torch.Tensor, torch.Tensor]:

            * **Training + return_f=True**: ``(logits, feature)`` where
              *logits* has shape ``(N, class_num)`` and *feature* has
              shape ``(N, num_bottleneck)``.
            * **Training + return_f=False**: *logits* of shape
              ``(N, class_num)``.
            * **Evaluation**: *feature* of shape ``(N, num_bottleneck)``
              (classification is skipped).
        """
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x,f
            else:
                x = self.classifier(x)
                return x
        else:
            return x

class Pooling(nn.Module):
    """Unified pooling dispatcher for CNN feature maps (legacy version).

    Attributes:
        pool (str): Selected pooling strategy.
        avgpool2 (nn.AdaptiveAvgPool2d): Present for ``"avg"`` /
            ``"avg+max"``.
        maxpool2 (nn.AdaptiveMaxPool2d): Present for ``"max"`` /
            ``"avg+max"``.
        gem2 (GeM): Present for ``"gem"``.
    """

    def __init__(self,pool="avg"):
        """Initializes Pooling.

        Args:
            pool (str): Strategy – one of ``"avg"``, ``"max"``,
                ``"avg+max"``, or ``"gem"``.
        """
        super(Pooling, self).__init__()
        self.pool = pool
        if pool == 'avg+max':
            self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            self.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
            # self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool == 'avg':
            self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            # self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool == 'max':
            self.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'gem':
            self.gem2 = GeM(dim=2048)

    def forward(self,x):
        """Applies the selected pooling to a CNN spatial feature map.

        Args:
            x (torch.Tensor): Spatial feature map of shape
                ``(N, C, H, W)``.

        Returns:
            torch.Tensor: Pooled tensor of shape ``(N, C, 1, 1)`` for
            ``"avg"`` / ``"max"``; ``(N, 2*C, 1, 1)`` for
            ``"avg+max"``; ``(N, C)`` for ``"gem"``.
        """
        if self.pool == 'avg+max':
            x1 = self.avgpool2(x)
            x2 = self.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
        elif self.pool == 'avg':
            x = self.avgpool2(x)
        elif self.pool == 'max':
            x = self.maxpool2(x)
        elif self.pool == 'gem':
            x = self.gem2(x)
        return x


class block_attention(nn.Module):
    """Intra-block soft attention that re-weights features by their softmax
    score along the part dimension.

    Attributes:
        (no learnable parameters)
    """

    def __init__(self):
        """Initializes block_attention (no learnable parameters)."""
        super(block_attention,self).__init__()


    def forward(self,x):
        """Applies softmax attention along dim 2 (the part/block axis).

        Args:
            x (torch.Tensor): Part feature tensor of shape
                ``(N, C, num_parts)``.

        Returns:
            torch.Tensor: Attention-weighted features of shape
            ``(N, C, num_parts)``.
        """
        weights = F.softmax(x, dim=2)
        x = x * weights

        return x

class MSBA(nn.Module):
    """Multi-Scale Block Attention model (legacy, experimental).

    Extracts concentric ring features from a CNN feature map and applies
    block-level soft attention before classification.

    Attributes:
        pool (str): Pooling strategy.
        block (int): Number of concentric region blocks.
        attention (block_attention): Soft-attention over block features.
        classifier (ClassBlock): Joint global classifier.
        classifier<i> (ClassBlock): Per-block local classifiers (dynamic
            attributes for *i* in ``0..block+1``).
    """

    def __init__(self, class_num, droprate=0.5, init_model=None, pool='avg', block=6):
        """Initializes MSBA.

        Args:
            class_num (int): Number of identity classes.
            droprate (float): Dropout probability for all ClassBlock
                instances.
            init_model: If not ``None``, the underlying feature extractor
                and pooling layer are borrowed from this model instance.
            pool (str): Pooling strategy (``"avg"``, ``"max"``, or
                ``"avg+max"``).
            block (int): Number of concentric ring regions.
        """
        super(MSBA, self).__init__()
        self.pool = pool
        self.block = block
        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool

        self.attention = block_attention()
        self.classifier = ClassBlock(3072,class_num,droprate)
        for i in range(self.block+2):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate))

    def forward(self, x):
        """Extracts multi-scale block features and returns per-block logits.

        Args:
            x (torch.Tensor): CNN spatial feature map of shape
                ``(N, C, H, W)``.

        Returns:
            list[torch.Tensor] or torch.Tensor: During training, a list of
            ``block + 2`` logit tensors each of shape ``(N, class_num)``.
            During evaluation, a stacked tensor of shape
            ``(N, class_num, block + 2)``.
        """
        if self.pool == 'avg+max':
            x1 = self.get_part_pool(x, pool='avg')
            x2 = self.get_part_pool(x, pool='max')
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1), -1)
        elif self.pool == 'avg':
            x = self.get_part_pool(x1)
            x = x.view(x.size(0), x.size(1), -1)
            x1 = torch.nn.AdaptiveMaxPool2d((1,1))(x1)
            x1 = x1.view(x1.size(0),x1.size(1),-1)[:,:,0]
            x2 = torch.nn.AdaptiveMaxPool2d((1, 1))(x2)
            x2 = x2.view(x2.size(0), x2.size(1), -1)[:, :, 0]
        elif self.pool == 'max':
            x = self.get_part_pool(x, pool='max')
            x = x.view(x.size(0), x.size(1), -1)
        y = self.part_classifier(x)
        x22 = torch.cat([x1, x2], dim=1)
        y22 = self.classifier(x22)
        return y


    def get_part_pool(self, x, pool='avg', no_overlap=True):
        """Extracts pooled concentric ring regions from a feature map.

        Args:
            x (torch.Tensor): Spatial feature map of shape
                ``(N, C, H, W)``.
            pool (str): ``"avg"`` or ``"max"``.
            no_overlap (bool): If ``True``, subtract inner crop from
                outer crop to obtain non-overlapping ring descriptors.

        Returns:
            torch.Tensor: Concatenated pooled regions of shape
            ``(N, C, block + 2, 1)``.
        """
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1))
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.block),W/(2*self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.block-c_h)*2, W+(self.block-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H,new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.block),W/(2*self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    x_pad = F.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.append(avgpool)
                if i == 3:
                    x_12 = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    x_12 = pooling(x_12)
                    result.append(x_12)
                    x_123 = x[:, :, (c_h - i * per_h):(c_h + i * per_h),(c_w - i * per_w):(c_w + i * per_w)]
                    x_123 = pooling(x_123)
                    result.append(x_123)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.append(avgpool)
        return torch.cat(result, dim=2)

    def part_classifier(self, x):
        """Applies per-block classifiers after block-level attention.

        Args:
            x (torch.Tensor): Part feature tensor of shape
                ``(N, C, block + 2)``.

        Returns:
            list[torch.Tensor] or torch.Tensor: During training, a list of
            ``block + 2`` logit tensors each of shape ``(N, class_num)``.
            During evaluation, a stacked tensor of shape
            ``(N, class_num, block + 2)``.
        """
        part = {}
        predict = {}
        x = self.attention(x)
        for i in range(self.block+2):
            part[i] = x[:,:,i].view(x.size(0),-1)
            # print(part[i].shape)
            # exit(0)
            # part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self, name) 
            predict[i] = c(part[i])
        y = []
        for i in range(self.block+2):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y


class refine_ft_net(nn.Module):
    """ResNet-50 fine-tuning network with optional LPN and FPN branches.

    Supports standard global pooling classification as well as Local Pattern
    Network (LPN) partitioning with an optional Dynamic Index (DI) module for
    adaptive region refinement.

    Attributes:
        DI (bool): Whether the Dynamic Index branch is active.
        return_box (bool): If ``True``, return bounding-box predictions
            instead of features.
        return_f (bool): Whether to return bottleneck features alongside
            logits during training.
        neck (nn.Module or None): Feature Pyramid Network neck (if enabled).
        classifier (ClassBlock): Global classifier on top of C5 features.
        classifier_1 (ClassBlock): Auxiliary classifier on top of C4
            features (when FPN is disabled).
        LPN (bool): Whether LPN partitioning is enabled.
        pool_type (str): Pooling strategy used by LPN.
        block (int): Number of LPN blocks.
        classifier_lpn_<i> (ClassBlock): Per-block LPN classifiers (dynamic
            attributes for ``i`` in ``0..block-1``).
        global_classifier (ClassBlock): Global branch classifier for LPN.
        pool (Pooling): Pooling module.
        model (nn.Module): Underlying ResNet-50 (or ResNeSt-50) backbone.
    """

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg',fpn=False,LPN=False,block=2,return_box=False,return_f=False):
        """Initializes refine_ft_net.

        Args:
            class_num (int): Number of identity classes.
            droprate (float): Dropout probability for all ClassBlock
                instances.
            stride (int): Stride for the last ResNet layer-4 block.  Use
                ``1`` for a higher-resolution feature map.
            init_model: If not ``None``, the backbone and pooling layer are
                borrowed from this model instance instead of creating a new
                one.
            pool (str): Pooling strategy (``"avg"``, ``"max"``,
                ``"avg+max"``, or ``"gem"``).
            fpn (bool): If ``True``, attach an FPN neck and classify the
                concatenated multi-scale features.
            LPN (bool): If ``True``, enable Local Pattern Network
                partitioning.
            block (int): Number of LPN spatial blocks.
            return_box (bool): If ``True``, return predicted bounding boxes
                from the DI sub-module rather than features.
            return_f (bool): If ``True``, return bottleneck features
                alongside class logits during training.
        """
        super(refine_ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        resnest = False
        if resnest:
            model_ft = resnest50(pretrained=True)
        self.DI= False
        self.return_box = return_box
        self.return_f = return_f

        # avg pooling to global pooling
        if stride == 1:
            # resnest
            if resnest:
                model_ft.layer4[0].downsample[0].stride = (1, 1)
                model_ft.layer4[0].avd_layer.stride = 1
                model_ft.layer4[0].downsample[0]=AvgPool2d(kernel_size=3, stride=(1, 1), padding=1)
            else:
            # resnet
                model_ft.layer4[0].downsample[0].stride = (1,1)
                model_ft.layer4[0].conv2.stride = (1,1)


        if fpn:
            self.neck = FPN()
            self.neck.apply(weights_init_kaiming)
            self.classifier = ClassBlock(1024,class_num,droprate,return_f=self.return_f)
        else:
            self.neck = None
            self.classifier = ClassBlock(2048, class_num, droprate,return_f=self.return_f)
            self.classifier_1 = ClassBlock(1024, class_num, droprate,return_f=self.return_f)

        self.LPN=LPN
        if self.LPN:
            self.pool_type = pool
            self.block = block
            for i in range(self.block):
                name = 'classifier_lpn_' + str(i)
                setattr(self, name, ClassBlock(2048, class_num, droprate,return_f=self.return_f))

            # 动态调整框
            if self.DI:
                for i in range(self.block-1):
                    name_DI = "DI_lpn_" + str(i)
                    setattr(self, name_DI, nn.Sequential(ClassBlock(2048, 2, droprate, relu=False,return_f=self.return_f)))

            # 全局分支
            self.global_classifier = ClassBlock(2048,class_num,droprate,return_f=self.return_f)

        self.pool = Pooling(pool)
        self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block


    def forward(self, x):
        """Runs the ResNet backbone and optional LPN / FPN branches.

        Args:
            x (torch.Tensor): Input image batch of shape ``(N, 3, H, W)``.

        Returns:
            torch.Tensor or list or tuple: Output depends on configuration:

            * **LPN + training + return_f**: ``(cls_list, feature_list)``
              each containing ``block + 1`` tensors.
            * **LPN + training**: list of ``block + 1`` classification
              tensors.
            * **LPN + evaluation**: stacked feature tensor of shape
              ``(N, 2048, block + 1)``.
            * **Global (no LPN)**: single classification output from
              :class:`ClassBlock` applied to the C5 feature.
        """
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        c1 = self.model.maxpool(x)
        c2 = self.model.layer1(c1)
        c3 = self.model.layer2(c2)
        c4 = self.model.layer3(c3)
        c5 = self.model.layer4(c4)

        if self.LPN:
            if self.DI:
                DI_list = []
                for i in range(self.block-1):
                    DI_name = "DI_lpn_"+str(i)
                    DI_c = getattr(self,DI_name)
                    DI_list.append(DI_c(self.pool(c5).view(c5.size(0),c5.size(1))))
                x = self.get_2part_pool(c5, self.pool_type, DI_list=DI_list)
                if self.return_box:
                    return x
            else:
                # x = self.get_2part_pool(c5, self.pool_type, DI_list=None)
                x = self.get_part_pool(c5)
            x = x.view(x.size(0), x.size(1), -1)
            y = self.part_classifier(x)


            #增加全局分支
            global_feature = self.pool(c5).view(c5.size(0),c5.size(1))
            global_feature = self.global_classifier(global_feature)
            if self.training:
                y.append(global_feature)

                if self.return_f:
                    cls,features = [],[]
                    for i in y:
                        cls.append(i[0])
                        features.append(i[1])
                    return cls,features

            else:
                global_feature = global_feature.view(global_feature.size(0),-1,1)
                y = torch.cat((y,global_feature),dim=2)

            return y

        if self.neck:
            features = [c2, c3, c4, c5]
            c2,c3,c4,c5 = self.neck(features)
            y = torch.cat([self.pool(p).view(c2.size(0),c2.size(1)) for p in [c2,c3,c4,c5]],dim=1)
            y = self.classifier(y)
            # y = self.pool(c4).view(c4.size(0),c4.size(1))
        else:
            # three_view_c3_c4_2loss##################################
            # y1 = self.pool(c4).view(c4.size(0), c4.size(1))
            # y1 = self.classifier_1(y1)
            # y2 = self.pool(c5).view(c5.size(0), c5.size(1))
            # y2 = self.classifier(y2)
            # if self.training:
            #     y = [y1, y2]
            # else:
            #     y = torch.cat((y1,y2),dim=1)

            #three_view_c5_base #########################################
            y = self.pool(c5).view(c5.size(0),c5.size(1))
            y = self.classifier(y)
        return y
        # return y


    def get_2part_pool(self, x, pool='avg',DI_list = None):
        """Extracts two-region pooled descriptors with optional DI refinement.

        Args:
            x (torch.Tensor): Spatial feature map of shape
                ``(N, C, H, W)``.
            pool (str): ``"avg"`` or ``"max"``.
            DI_list (list or None): Per-block dynamic index offsets
                predicted by the DI sub-module.  When not ``None`` the crop
                boundaries are shifted by the predicted offsets.

        Returns:
            torch.Tensor: Concatenated pooled regions along dim 2.
        """
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H / 2), int(W / 2)
        per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        roi_align = RoIAlign(1,1)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                xmin = c_h - i * per_h
                xmax = c_h + i * per_h
                ymin = c_w - i * per_w
                ymax = c_w + i * per_w
                if DI_list:
                    DI = DI_list[i-1]
                    tanh = nn.Tanh()
                    DI = tanh(DI)
                    xmin = xmin+DI[:,0].cpu().detach()*per_h/2
                    xmax = xmax+DI[:,1].cpu().detach()*per_h/2
                    ymin = ymin+DI[:,0].cpu().detach()*per_w/2
                    ymax = ymax+DI[:,1].cpu().detach()*per_w/2
                    box = torch.stack([xmin, ymin, xmax, ymax], dim=1).type(torch.int)
                    index = torch.range(0,x.size(0)-1,dtype=torch.int)
                    crops = roi_align(x.type(torch.FloatTensor),box.type(torch.FloatTensor).cpu(),index).cuda()
                    # crops = crops.view(crops.size(0),crops.size(1))
                else:
                    crops = x[:, :, int(xmin):int(xmax), int(ymin):int(ymax)]
                if self.return_box:
                    assert DI_list!=True, "box不存在！！！"
                    return box

                avgpool = pooling(crops)
                result.append(avgpool)
            else:
                # xmin = xmin.numpy(),xmax.numpy(),ymin.numpy(),ymax.numpy()
                if DI_list:
                    x_res = []
                    for b in range(x.size(0)):
                        x_pre = x[b, :, math.ceil(xmin[b]):math.floor(xmax[b]), math.ceil(ymin[b]):math.floor(ymax[b])]
                        x_pad = F.pad(x_pre, (math.ceil(ymin[b]), H-math.floor(ymax[b]),math.ceil(xmin[b]), W-math.floor(xmax[b])), "constant", 0)
                        x_curr = x[b] - x_pad
                        x_res.append(x_curr)
                    x_res = torch.stack(x_res,dim=0)
                else:
                    x_pre = x[::, :, int(xmin):int(xmax), int(ymin):int(ymax)]
                    x_pad = F.pad(x_pre, (int(ymin), H - int(ymax), int(xmin), W - int(xmax)), "constant", 0)
                    x_curr = x - x_pad
                    x_res = x_curr
                avgpool = pooling(x_res)
                result.append(avgpool)
        return torch.cat(result, dim=2)


    def get_part_pool(self, x, pool='avg', no_overlap=True):
        """Extracts pooled concentric ring regions from a ResNet feature map.

        Args:
            x (torch.Tensor): Spatial feature map of shape
                ``(N, C, H, W)``.
            pool (str): ``"avg"`` (default) or ``"max"``.
            no_overlap (bool): If ``True``, subtract the inner ring content
                from each outer crop to obtain non-overlapping annular
                descriptors.

        Returns:
            torch.Tensor: Concatenated pooled regions along dim 2 with shape
            ``(N, C, block, 1)``.
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

    def part_classifier(self, x):
        """Applies per-block LPN classifiers.

        Args:
            x (torch.Tensor): Part feature tensor of shape
                ``(N, C, block)``.

        Returns:
            list[torch.Tensor] or torch.Tensor: During training, a list of
            ``block`` outputs from the per-block ClassBlock instances.
            During evaluation, a stacked tensor of shape
            ``(N, class_num, block)`` (or the raw ClassBlock output if
            ``return_f=True``).
        """
        part = {}
        predict = {}
        for i in range(self.block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier_lpn_' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y



class two_view_net(nn.Module):
    """Dual-view network with optional weight-sharing (legacy version).

    Wraps two :class:`refine_ft_net` (or Transformer-based) sub-models, one
    per view.  When ``share_weight=True`` the same sub-model is used for
    both views.

    Attributes:
        model_1 (nn.Module): Sub-model for the first view (drone).
        model_2 (nn.Module): Sub-model for the second view (satellite).
            Only present when ``share_weight=False``.
        share_weight (bool): Whether the two views share weights.
    """

    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False, Transformer=False, LPN=True, block=4, return_box=False, return_f=False,MSBA=False,deit=False):
        """Initializes two_view_net.

        Args:
            class_num (int): Number of identity classes.
            droprate (float): Dropout probability.
            stride (int): Stride for the last ResNet layer-4 block.
            pool (str): Pooling strategy.
            share_weight (bool): If ``True``, both views use ``model_1``.
            Transformer (bool): If ``True``, use a Transformer-based
                backbone via :func:`.TransReId.make_transformer_model`.
            LPN (bool): Enable Local Pattern Network partitioning.
            block (int): Number of LPN spatial blocks.
            return_box (bool): Return bounding boxes (DI branch).
            return_f (bool): Return bottleneck features during training.
            MSBA (bool): Use Multi-Scale Block Attention model.
            deit (bool): Use DeiT distillation token in the Transformer
                branch.
        """
        super(two_view_net, self).__init__()
        if Transformer:
            self.model_1 = make_transformer_model(num_class=class_num,block=block,return_f=return_f,deit=deit)
        elif MSBA:
            self.model_1 = MSBA_net(class_num,droprate=droprate,stride=stride)
        else:
            self.model_1 = refine_ft_net(class_num,droprate=droprate, stride=stride, pool=pool, LPN=LPN, block=block,
                                         return_box=return_box,return_f=return_f)

        self.share_weight = share_weight
        if not share_weight:
            if Transformer:
                self.model_2 = make_transformer_model(num_class=class_num, view_num=0, jvm=False,return_f=return_f)
            else:
                self.model_2 = refine_ft_net(class_num,droprate=droprate, stride=stride, pool=pool, LPN=LPN, block=block,
                                             return_box=return_box,return_f=return_f)

    def forward(self, x1, x2):
        """Runs both view branches and returns their outputs.

        Args:
            x1 (torch.Tensor or None): Drone-view image batch of shape
                ``(N, 3, H, W)``, or ``None`` to skip.
            x2 (torch.Tensor or None): Satellite-view image batch of shape
                ``(N, 3, H, W)``, or ``None`` to skip.

        Returns:
            tuple: ``(y1, y2)`` where each element is the output of the
            corresponding sub-model, or ``None`` if the input was ``None``.
        """
        if x1 is None:
            y1 = None
        else:
            y1 = self.model_1(x1)

        if x2 is None:
            y2 = None
        else:
            if self.share_weight:
                y2 = self.model_1(x2)
            else:
                y2 = self.model_2(x2)

        return y1, y2


def make_model(opt):
    """Constructs a :class:`two_view_net` from an option namespace.

    Args:
        opt: Argument namespace with at least:
            - nclasses (int): Number of identity classes.
            - droprate (float): Dropout probability.
            - stride (int): Last ResNet block stride.
            - pool (str): Pooling strategy.
            - share (bool): Whether to share weights between views.
            - transformer (bool): Use Transformer backbone.
            - LPN (bool): Enable LPN.
            - block (int): Number of LPN blocks.
            - triplet_loss (bool): Return bottleneck features
              (``return_f``).
            - MSBA (bool): Use MSBA model.
            - deit (bool): Use DeiT distillation token.

    Returns:
        two_view_net: Initialized dual-view model.
    """
    model = two_view_net(opt.nclasses, droprate=opt.droprate, stride=opt.stride, pool=opt.pool,
                             share_weight=opt.share,Transformer=opt.transformer,LPN=opt.LPN,block=opt.block,return_f=opt.triplet_loss,MSBA=opt.MSBA,deit=opt.deit)

    return model


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = two_view_net(751, droprate=0.5, VGG16=True)
    #net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 256))
    output,output = net(input,input)
    print('net output size:')
    print(output.shape)
