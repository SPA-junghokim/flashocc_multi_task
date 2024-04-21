from mmdet.models.backbones import ResNet
from .resnet import CustomResNet
from .swin import SwinTransformer
from .tri_res import CustomTriRes
from .tri_res_v2 import CustomTriResV2
from .tri_res_v3 import CustomTriResV3
from .tri_res_v4 import CustomTriResV4
from .tri_res_v5 import CustomTriResV5
from .tri_res_v6 import CustomTriResV6

__all__ = ['ResNet', 'CustomResNet', 'SwinTransformer']
