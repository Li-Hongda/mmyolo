# Copyright (c) OpenMMLab. All rights reserved.
from .distill_loss import CWDLoss, MimicLoss
from .iou_loss import IoULoss, bbox_overlaps

__all__ = ['IoULoss', 'bbox_overlaps', 'CWDLoss', 'MimicLoss']
