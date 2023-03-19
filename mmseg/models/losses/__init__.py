# Copyright (c) OpenMMLab. All rights reserved.
from .accuracy import Accuracy, accuracy
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .dice_loss import DiceLoss
from .focal_loss import FocalLoss
from .lovasz_loss import LovaszLoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss
from .indice_cross_entropy_loss import IndiceCrossEntropyLoss
from .indice_pixel_cross_entropy_loss import IndicePixelCrossEntropyLoss
from .mse_loss import MSELoss
from .mse_loss_depth import MSELossDepth
from .smoth_l1_loss import SmothL1Loss

__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'reduce_loss',
    'weight_reduce_loss', 'weighted_loss', 'LovaszLoss', 'DiceLoss',
    'FocalLoss',
    'IndiceCrossEntropyLoss', 'IndicePixelCrossEntropyLoss',
    'MSELoss', 'MSELossDepth', 'SmothL1Loss'
]
