from .builder import DATASETS
from .pipelines import LoadImageFromFile
from .custom import CustomDataset

@DATASETS.register_module()
class MixRain(CustomDataset):

    def __init__(self, **kwargs):
        super(MixRain, self).__init__(**kwargs)
        self.gt_seg_map_loader = LoadImageFromFile()
