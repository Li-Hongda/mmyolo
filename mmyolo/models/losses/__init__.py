# Copyright (c) OpenMMLab. All rights reserved.
from .iou_loss import IoULoss, bbox_overlaps
from .quality_focal_loss import QualityFocalLoss

__all__ = ['IoULoss', 'bbox_overlaps', 'QualityFocalLoss']
