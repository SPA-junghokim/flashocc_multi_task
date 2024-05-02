from mmdet.models.backbones import ResNet
from .resnet import CustomResNet
from .swin import SwinTransformer
from .tri_res import CustomTriRes
from .tri_res_v2 import CustomTriResV2
from .tri_res_v3 import CustomTriResV3
from .tri_res_v4 import CustomTriResV4
from .tri_res_v5 import CustomTriResV5
from .tri_res_v6 import CustomTriResV6
from .repvgg import RepVGG
from .inceptionnext import MetaNeXt
from .convnext import ConvNeXt
from .resnet_inceptblock import CustomResNet_inc
from .resnet_convblock import CustomResNet_conv

__all__ = ['ResNet', 'CustomResNet', 'SwinTransformer', 'RepVGG']
