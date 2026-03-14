"""Backbone factory for the DenseUAV geo-localization baseline.

All backbone models are loaded from the ``timm`` library with pretrained
ImageNet weights.  The :class:`Backbone` wrapper exposes a unified
``forward_features`` interface and stores the output channel count in
:attr:`~Backbone.output_channel` so downstream heads can be sized
automatically.
"""

import torch.nn as nn
import timm


def make_backbone(opt):
    """Constructs a :class:`Backbone` from an option namespace.

    Args:
        opt: Argument namespace with at least:
            - backbone (str): Architecture key (e.g. ``"ViTS-224"``).
            - h (int): Input image height in pixels.
            - w (int): Input image width in pixels.

    Returns:
        Backbone: Initialized backbone with
        :attr:`~Backbone.output_channel` set to the number of output
        feature channels.
    """
    backbone_model = Backbone(opt)
    return backbone_model


class Backbone(nn.Module):
    """Thin wrapper around a ``timm`` backbone that exposes feature maps.

    The wrapper calls ``forward_features`` on the underlying model so that
    heads receive raw spatial or token feature maps rather than classification
    logits.

    Attributes:
        opt: Option namespace used to build this backbone.
        img_size (tuple[int, int]): ``(height, width)`` of the expected input.
        backbone (nn.Module): Underlying ``timm`` model instance.
        output_channel (int): Number of channels in the feature tensor
            returned by :meth:`forward`.
    """

    def __init__(self, opt):
        """Initializes Backbone from an option namespace.

        Args:
            opt: Argument namespace with at least:
                - backbone (str): One of the supported architecture keys (see
                  :meth:`init_backbone` for the full list).
                - h (int): Input image height.
                - w (int): Input image width.
        """
        super().__init__()
        self.opt = opt
        self.img_size = (opt.h,opt.w)
        self.backbone,self.output_channel = self.init_backbone(opt.backbone)
        

    def init_backbone(self, backbone):
        """Creates a pretrained ``timm`` model and returns its output channels.

        Supported architecture keys and their ``output_channel`` values:

        +---------------------+----------------+
        | Key                 | output_channel |
        +=====================+================+
        | ``ViTS-224``        | 384            |
        +---------------------+----------------+
        | ``ViTS-384``        | 384            |
        +---------------------+----------------+
        | ``DeitS-224``       | 384            |
        +---------------------+----------------+
        | ``DeitB-224``       | 384            |
        +---------------------+----------------+
        | ``ViTB-224``        | 768            |
        +---------------------+----------------+
        | ``SwinB-224``       | 768            |
        +---------------------+----------------+
        | ``Swinv2S-256``     | 768            |
        +---------------------+----------------+
        | ``Swinv2T-256``     | 768            |
        +---------------------+----------------+
        | ``Convnext-T``      | 768            |
        +---------------------+----------------+
        | ``Pvtv2b2``         | 512            |
        +---------------------+----------------+
        | ``EfficientNet-B2`` | 1408           |
        +---------------------+----------------+
        | ``EfficientNet-B3`` | 1536           |
        +---------------------+----------------+
        | ``resnet50``        | 2048           |
        +---------------------+----------------+
        | ``vgg16``           | 512            |
        +---------------------+----------------+

        Args:
            backbone (str): Architecture key from the table above.

        Returns:
            tuple[nn.Module, int]: ``(backbone_model, output_channel)`` where
            *backbone_model* is a pretrained ``timm`` model instance and
            *output_channel* is the number of feature channels it produces.

        Raises:
            NameError: If *backbone* is not in the supported key list.
        """
        if backbone=="resnet50":
            backbone_model = timm.create_model('resnet50', pretrained=True)
            output_channel = 2048
        elif backbone=="ViTS-224":
            backbone_model = timm.create_model("vit_small_patch16_224", pretrained=True, img_size=self.img_size)
            output_channel = 384
        elif backbone=="ViTS-384":
            backbone_model = timm.create_model("vit_small_patch16_384", pretrained=True)
            output_channel = 384
        elif backbone=="DeitS-224":
            backbone_model = timm.create_model("deit_small_distilled_patch16_224", pretrained=True)
            output_channel = 384
        elif backbone=="DeitB-224":
            backbone_model = timm.create_model("deit_base_distilled_patch16_224", pretrained=True)
            output_channel = 384
        elif backbone=="Pvtv2b2":
            backbone_model = timm.create_model("pvt_v2_b2", pretrained=True)
            output_channel = 512
        elif backbone=="ViTB-224":
            backbone_model = timm.create_model("vit_base_patch16_224", pretrained=True)
            output_channel = 768
        elif backbone=="SwinB-224":
            backbone_model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)
            output_channel = 768
        elif backbone=="Swinv2S-256":
            backbone_model = timm.create_model("swinv2_small_window8_256", pretrained=True)
            output_channel = 768
        elif backbone=="Swinv2T-256":
            backbone_model = timm.create_model("swinv2_tiny_window16_256", pretrained=True)
            output_channel = 768
        elif backbone=="Convnext-T":
            backbone_model = timm.create_model("convnext_tiny", pretrained=True)
            output_channel = 768
        elif backbone=="EfficientNet-B2":
            backbone_model = timm.create_model("efficientnet_b2", pretrained=True)
            output_channel = 1408
        elif backbone=="EfficientNet-B3":
            backbone_model = timm.create_model("efficientnet_b3", pretrained=True)
            output_channel = 1536
        elif backbone=="vgg16":
            backbone_model = timm.create_model("vgg16", pretrained=True)
            output_channel = 512
        else:
            raise NameError("{} not in the backbone list!!!".format(backbone))
        return backbone_model,output_channel

    def forward(self, image):
        """Extracts feature maps from the input image batch.

        Delegates to ``backbone.forward_features``, which returns:

        * **ViT / DeiT**: token tensor of shape
          ``(N, num_patches + 1, C)`` where index 0 is the CLS token.
        * **CNN (ResNet, EfficientNet, VGG)**: spatial tensor of shape
          ``(N, C, H', W')``.
        * **Swin / PVTv2**: sequence tensor of shape
          ``(N, num_tokens, C)``.

        Args:
            image (torch.Tensor): Input image batch of shape
                ``(N, 3, H, W)``.

        Returns:
            torch.Tensor: Backbone feature tensor; exact shape depends on the
            chosen architecture (see above).
        """
        features = self.backbone.forward_features(image)
        # new code: timm Swinv2 returns (N, H, W, C); convert to (N, C, H, W) for CNN-style heads
        if getattr(self.opt, "backbone", None) in ("Swinv2S-256", "Swinv2T-256") and features.dim() == 4 and features.shape[-1] == self.output_channel:
            features = features.permute(0, 3, 1, 2)
        return features



    
