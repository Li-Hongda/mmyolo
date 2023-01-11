# Copyright (c) OpenMMLab. All rights reserved.
from .yolo_detector import YOLODetector
from .yolo_distill_detector import KnowledgeDistillationYOLODetector

__all__ = ['YOLODetector', 'KnowledgeDistillationYOLODetector']
