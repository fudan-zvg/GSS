
import torch.nn as nn
from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList,load_state_dict)

class NoneBackbone(BaseModule):

    def __init__(self):
        super(NoneBackbone, self).__init__()
    def forward(self, inputs):
        return tuple(inputs)

    def train(self, mode=True):
        super(NoneBackbone, self).train(mode)
        pass
