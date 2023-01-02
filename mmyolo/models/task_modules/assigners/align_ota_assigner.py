# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn.functional as F
from mmdet.models.task_modules.assigners import AssignResult, SimOTAAssigner
from mmdet.utils import ConfigType
from mmengine.structures import InstanceData

from mmyolo.registry import TASK_UTILS

INF = 1000000.0
EPS = 1.0e-7


@TASK_UTILS.register_module()
class AlignOTAAssigner(SimOTAAssigner):
    """Computes matching between predictions and ground truth.

    Args:
        center_radius (int | float, optional): Ground truth center size
            to judge whether a prior is in center. Default 2.5.
        candidate_topk (int, optional): The candidate top-k which used to
            get top-k ious to calculate dynamic-k. Default 10.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 3.0.
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
    """

    def __init__(self,
                 center_radius: float = 2.5,
                 candidate_topk: int = 10,
                 iou_weight: float = 3.0,
                 cls_weight: float = 1.0,
                 iou_calculator: ConfigType = dict(type='BboxOverlaps2D')):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign gt to priors using SimOTA. It will switch to CPU mode when
        GPU is out of memory.
        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
        Returns:
            assign_result (obj:`AssignResult`): The assigned result.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        num_gt = gt_bboxes.size(0)

        decoded_bboxes = pred_instances.bboxes
        pred_scores = pred_instances.scores
        priors = pred_instances.priors
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ),
                                                   0,
                                                   dtype=torch.long)
        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
            priors, gt_bboxes)
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = decoded_bboxes.new_full((num_bboxes, ),
                                                          -1,
                                                          dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + EPS)

        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64),
                      pred_scores.shape[-1]).float().unsqueeze(0).repeat(
                          num_valid, 1, 1))

        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores
        # disable AMP autocast and calculate BCE with FP32 to avoid overflow
        with torch.cuda.amp.autocast(enabled=False):
            cls_cost = (
                F.binary_cross_entropy(
                    valid_pred_scores.to(dtype=torch.float32),
                    soft_label,
                    reduction='none',
                ) *
                scale_factor.abs().pow(2.0).to(dtype=valid_pred_scores.dtype))

        cls_cost = cls_cost.sum(dim=-1)
        cost_matrix = (
            cls_cost * self.cls_weight + iou_cost * self.iou_weight +
            (~is_in_boxes_and_center) * INF)
        matched_pred_ious, matched_gt_inds = \
            self.dynamic_k_matching(
                cost_matrix, pairwise_ious, num_gt, valid_mask)

        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes, ),
                                                 -INF,
                                                 dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
