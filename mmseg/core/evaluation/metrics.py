# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import mmcv
import numpy as np
import torch

def f_score(precision, recall, beta=1):
    """calculate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        label = torch.from_numpy(label)

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255
    # from torchvision.utils import save_image, make_grid
    # print('cjq debug', pred_label)
    # print('label', label)
    # import pdb;pdb.set_trace()
    # pred = encode_to_segmap(pred_label.unsqueeze(0).unsqueeze(0), palette=PALETTE, num_classes=19)
    # gt = encode_to_segmap(label.unsqueeze(0).unsqueeze(0), palette=PALETTE, num_classes=19).float()
    # deta = torch.abs(pred - gt)
    # deta[deta > 0] = 1
    # save_image(torch.cat([pred / 255.0, gt  / 255.0, deta], dim=0),
    #            'work_dirs/mseg_debug/show_miou/debug_eval_rec_new' + str(torch.tensor(pred_label != label).sum().item()) + '.png')
    # print('cjq debug', pred_label.max(), label.max())
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    
    # print('cjq debug', mask)
    # print('cjq debug', torch.tensor(pred_label != label).sum(), '\n\n')
    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    for result, gt_seg_map in zip(results, gt_seg_maps):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                result, gt_seg_map, num_classes, ignore_index,
                label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]:
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <IoU> ndarray: Per category IoU, shape (num_classes, ).
    """
    iou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return iou_result


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <Dice> ndarray: Per category dice, shape (num_classes, ).
    """

    dice_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return dice_result


def mean_fscore(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                nan_to_num=None,
                label_map=dict(),
                reduce_zero_label=False,
                beta=1):
    """Calculate Mean F-Score (mFscore)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.


     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Fscore> ndarray: Per category recall, shape (num_classes, ).
            <Precision> ndarray: Per category precision, shape (num_classes, ).
            <Recall> ndarray: Per category f-score, shape (num_classes, ).
    """
    fscore_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mFscore'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
        beta=beta)
    return fscore_result


def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str] | Iterables): list of ground
            truth segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            results, gt_seg_maps, num_classes, ignore_index, label_map,
            reduce_zero_label)
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics


def pre_eval_to_metrics(pre_eval_results,
                        metrics=['mIoU'],
                        nan_to_num=None,
                        beta=1):
    """Convert pre-eval results to metrics.

    Args:
        pre_eval_results (list[tuple[torch.Tensor]]): per image eval results
            for computing evaluation metric
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """

    # convert list of tuples to tuple of lists, e.g.
    # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
    # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
    # print('cjq debug pre_eval_to_metrics:1', pre_eval_results)
    pre_eval_results = tuple(zip(*pre_eval_results))
    # print('cjq debug pre_eval_to_metrics:2', pre_eval_results)
    # print('cjq debug pre_eval_to_metrics:3', len(pre_eval_results))
    assert len(pre_eval_results) == 4

    total_area_intersect = sum(pre_eval_results[0])
    total_area_union = sum(pre_eval_results[1])
    total_area_pred_label = sum(pre_eval_results[2])
    total_area_label = sum(pre_eval_results[3])

    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label, metrics, nan_to_num,
                                        beta)

    return ret_metrics


def total_area_to_metrics(total_area_intersect,
                          total_area_union,
                          total_area_pred_label,
                          total_area_label,
                          metrics=['mIoU'],
                          nan_to_num=None,
                          beta=1):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (ndarray): The intersection of prediction and
            ground truth histogram on all classes.
        total_area_union (ndarray): The union of prediction and ground truth
            histogram on all classes.
        total_area_pred_label (ndarray): The prediction histogram on all
            classes.
        total_area_label (ndarray): The ground truth histogram on all classes.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc
        elif metric == 'mFscore':
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall

    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics

# PALETTE = torch.tensor([[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
#            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
#            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
#            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
#            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
#            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
#            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
#            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
#            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
#            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
#            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
#            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
#            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
#            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
#            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
#            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
#            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
#            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
#            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
#            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
#            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
#            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
#            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
#            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
#            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
#            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
#            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
#            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
#            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
#            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
#            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
#            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
#            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
#            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
#            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
#            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
#            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
#            [102, 255, 0], [92, 0, 255], [255, 255, 255]])

PALETTE = torch.tensor([[222, 222, 145], [18, 30, 7], [8, 23, 47], 
                        [30, 6, 96], [1, 13, 164], [12, 28, 191], 
                        [25, 52, 32], [29, 48, 52], [15, 51, 95], 
                        [25, 56, 167], [25, 42, 210], [27, 81, 31], 
                        [9, 88, 54], [27, 92, 113], [11, 99, 151], 
                        [26, 110, 183], [24, 130, 26], [4, 122, 75], 
                        [3, 132, 98], [26, 147, 167]])

def encode_to_segmap(indice, palette, num_classes):
    indice[indice > num_classes] = num_classes
    return palette[indice.long()].squeeze(1).permute(0, 3, 1, 2)
