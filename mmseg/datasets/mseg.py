import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose, LoadAnnotations
from .custom import CustomDataset
from .pipelines import ToUniversalLabel

@DATASETS.register_module()
class MSegDataset(CustomDataset):

    CLASSES =['backpack', 'umbrella', 'bag', 'tie', 'suitcase', 'case', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal_other', 'microwave', 'radiator', 'oven', 'toaster', 'storage_tank', 'conveyor_belt', 'sink', 'refrigerator', 'washer_dryer', 'fan', 'dishwasher', 'toilet', 'bathtub', 'shower', 'tunnel', 'bridge', 'pier_wharf', 'tent', 'building', 'ceiling', 'laptop', 'keyboard', 'mouse', 'remote', 'cell phone', 'television', 'floor', 'stage', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'fruit_other', 'food_other', 'chair_other', 'armchair', 'swivel_chair', 'stool', 'seat', 'couch', 'trash_can', 'potted_plant', 'nightstand', 'bed', 'table', 'pool_table', 'barrel', 'desk', 'ottoman', 'wardrobe', 'crib', 'basket', 'chest_of_drawers', 'bookshelf', 'counter_other', 'bathroom_counter', 'kitchen_island', 'door', 'light_other', 'lamp', 'sconce', 'chandelier', 'mirror', 'whiteboard', 'shelf', 'stairs', 'escalator', 'cabinet', 'fireplace', 'stove', 'arcade_machine', 'gravel', 'platform', 'playingfield', 'railroad', 'road', 'snow', 'sidewalk_pavement', 'runway', 'terrain', 'book', 'box', 'clock', 'vase', 'scissors', 'plaything_other', 'teddy_bear', 'hair_dryer', 'toothbrush', 'painting', 'poster', 'bulletin_board', 'bottle', 'cup', 'wine_glass', 'knife', 'fork', 'spoon', 'bowl', 'tray', 'range_hood', 'plate', 'person', 'rider_other', 'bicyclist', 'motorcyclist', 'paper', 'streetlight', 'road_barrier', 'mailbox', 'cctv_camera', 'junction_box', 'traffic_sign', 'traffic_light', 'fire_hydrant', 'parking_meter', 'bench', 'bike_rack', 'billboard', 'sky', 'pole', 'fence', 'railing_banister', 'guard_rail', 'mountain_hill', 'rock', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 'net', 'base', 'sculpture', 'column', 'fountain', 'awning', 'apparel', 'banner', 'flag', 'blanket', 'curtain_other', 'shower_curtain', 'pillow', 'towel', 'rug_floormat', 'vegetation', 'bicycle', 'car', 'autorickshaw', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'trailer', 'boat_ship', 'slow_wheeled_object', 'river_lake', 'sea', 'water_other', 'swimming_pool', 'waterfall', 'wall', 'window', 'window_blind', 'unlabeled']

    PALETTE = [[2, 7, 214], [9, 16, 234], [3, 52, 5], [13, 65, 50], [9, 52, 79], [5, 64, 106], [5, 67, 136], [3, 65, 180], [9, 62, 215], [14, 59, 242], [6, 117, 8], [3, 104, 42], [11, 115, 76], [2, 113, 115], [4, 114, 134], [5, 108, 167], [2, 110, 203], [6, 118, 241], [14, 168, 13], [13, 156, 40], [5, 158, 82], [9, 158, 101], [5, 155, 137], [9, 166, 169], [14, 160, 212], [2, 165, 243], [12, 210, 13], [9, 215, 41], [7, 214, 78], [12, 210, 108], [11, 208, 138], [0, 216, 177], [0, 206, 200], [14, 210, 234], [58, 10, 9], [55, 15, 50], [66, 7, 82], [70, 1, 114], [59, 4, 134], [63, 16, 174], [68, 11, 205], [58, 16, 245], [61, 54, 6], [57, 52, 46], [67, 64, 69], [60, 60, 104], [61, 62, 149], [58, 66, 167], [65, 59, 203], [60, 67, 247], [66, 118, 14], [55, 117, 46], [55, 109, 74], [67, 109, 113], [68, 108, 137], [64, 108, 181], [65, 105, 213], [55, 116, 243], [69, 167, 9], [65, 162, 40], [59, 166, 77], [60, 161, 101], [62, 158, 140], [63, 160, 178], [70, 154, 207], [65, 158, 247], [61, 213, 4], [62, 215, 44], [66, 218, 79], [57, 217, 108], [63, 212, 136], [60, 215, 179], [55, 205, 200], [56, 210, 239], [112, 4, 14], [115, 6, 35], [123, 5, 82], [119, 6, 109], [116, 5, 135], [121, 2, 181], [111, 5, 212], [118, 6, 238], [116, 66, 5], [122, 53, 43], [116, 59, 69], [110, 53, 112], [112, 55, 146], [117, 61, 175], [120, 61, 211], [111, 57, 248], [124, 118, 11], [122, 106, 36], [123, 105, 83], [110, 106, 108], [116, 115, 137], [116, 104, 171], [111, 108, 209], [119, 105, 234], [125, 167, 9], [122, 166, 48], [119, 169, 72], [118, 163, 101], [115, 161, 137], [116, 162, 167], [118, 162, 211], [123, 166, 236], [124, 218, 2], [118, 209, 37], [124, 215, 83], [111, 211, 116], [115, 219, 139], [124, 205, 176], [115, 219, 211], [114, 207, 248], [171, 13, 15], [179, 10, 45], [172, 10, 79], [171, 14, 116], [177, 6, 135], [173, 4, 182], [180, 1, 215], [171, 8, 235], [171, 66, 5], [179, 57, 35], [168, 66, 70], [171, 62, 107], [172, 56, 138], [169, 57, 178], [178, 67, 205], [180, 67, 236], [178, 115, 10], [173, 117, 41], [179, 107, 79], [177, 104, 115], [174, 104, 137], [166, 115, 169], [167, 109, 212], [175, 116, 235], [172, 168, 15], [176, 158, 36], [169, 159, 68], [165, 158, 115], [167, 154, 147], [176, 154, 172], [165, 154, 206], [171, 154, 247], [176, 216, 4], [166, 205, 50], [169, 216, 83], [180, 216, 106], [176, 207, 140], [171, 205, 179], [173, 219, 215], [180, 220, 246], [234, 3, 10], [229, 14, 46], [224, 15, 69], [227, 7, 103], [230, 7, 148], [235, 13, 169], [221, 2, 213], [229, 7, 235], [234, 62, 10], [235, 53, 38], [226, 65, 68], [227, 60, 105], [229, 56, 145], [231, 54, 168], [230, 52, 215], [232, 57, 240], [228, 103, 10], [221, 111, 47], [229, 103, 70], [229, 118, 108], [234, 107, 138], [230, 109, 177], [229, 113, 209], [234, 103, 239], [231, 168, 17], [234, 154, 43], [227, 162, 83], [227, 164, 112], [222, 156, 146], [233, 155, 174], [220, 158, 213], [221, 163, 236], [228, 211, 4], [232, 220, 40], [233, 213, 78], [233, 220, 113], [233, 210, 142], [223, 217, 173], [225, 207, 207], [226, 220, 243], [0, 0, 0]]

    TEST_SPLIT_PALETTE = {
        'camvid-11': [[128, 0, 0], [128, 128, 0], [128, 128, 128], [64, 0, 128],
                      [192, 128, 128], [128, 64, 128], [64, 64, 0], [64, 64, 128],
                      [192, 192, 128], [0, 0, 192], [0, 128, 192]],
        'kitti-19': [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                     [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                     [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                     [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                     [0, 0, 230], [119, 11, 32]],
        'pascal-context-60': [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                       [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                       [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                       [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                       [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                       [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                       [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                       [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                       [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                       [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                       [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                       [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                       [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                       [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                       [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]],
        'scannet-20': [[8, 255, 51], [255, 6, 82], [255, 6, 51], [7, 255, 224],
                       [224, 255, 8], [6, 51, 255], [140, 140, 140], [20, 255, 0],
                       [255,224, 0], [255, 82, 0], [255, 61, 6], [255, 184, 184],
                       [255, 112, 0], [140, 140, 140], [153, 0, 255], [255, 204, 0],
                       [0, 204, 255], [0, 41, 255], [184, 255, 0], [92, 0, 255]],
        'voc2012': [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                   [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
                   [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
                   [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
                   [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]],
        'wilddash-19': [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                        [0, 80, 100], [0, 0, 230], [119, 11, 32]],
    }
    def __init__(self,
                 img_prefix=None,
                 seg_map_prefix=None,
                 dataset_name=None,
                 **kwargs):
        super(MSegDataset, self).__init__(**kwargs)
        self.dataset_name = dataset_name
        self.img_prefix = img_prefix
        self.seg_map_prefix = seg_map_prefix
        # load annotations
        self.img_infos = self.mseg_load_annotations(self.img_dir, self.img_prefix, self.img_suffix,
                                               self.ann_dir, self.seg_map_prefix,
                                               self.seg_map_suffix, self.split)
        self.to_universal_label = ToUniversalLabel(dataset_name=self.dataset_name, use_naive_taxonomy=False)
        if self.dataset_name in self.to_universal_label.tax_converter.test_datasets or True:
            self.__class__.CLASSES = self.to_universal_label.tax_converter.dataset_classnames[self.dataset_name]
            self.__class__.PALETTE = self.__class__.PALETTE[:len(self.__class__.CLASSES)]
            self.CLASSES, self.PALETTE = self.__class__.CLASSES, self.__class__.PALETTE

    def mseg_load_annotations(self, img_dir, img_prefix, img_suffix, ann_dir, seg_map_prefix, seg_map_suffix,
                         split):
        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                if img_prefix is not None:
                    img_info = dict(filename=img_prefix + img_name + img_suffix)
                else:
                    img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name + seg_map_suffix
                    if seg_map_prefix is not None:
                        seg_map = seg_map_prefix + img_name + seg_map_suffix
                    if self.dataset_name == 'scannet-20':
                        seg_map = img_name.replace('color', 'label20') + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)

                img_infos.append(img_info)
        else:
            for img in self.file_client.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=img_suffix,
                    recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    if seg_map_prefix is not None and img_prefix is not None:
                        seg_map = seg_map.replace(img_prefix, seg_map_prefix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])
        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_gt_seg_map_by_idx(self, index):
        """Get one ground truth segmentation map for evaluation."""
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        if self.dataset_name in self.to_universal_label.tax_converter.train_datasets:
            self.to_universal_label(results)
        return results['gt_semantic_seg']

    def pre_pipeline(self, results):
        super().pre_pipeline(results)
        results['dataset_name'] = self.dataset_name
        results['dataset_type'] = 'mseg'