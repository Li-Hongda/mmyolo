# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmyolo.datasets.transforms import ScaleAwareAutoAugmentation
from mmyolo.registry import HOOKS

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class DamoSetAttributeHook(Hook):
    """Get the current hook during training."""

    def before_run(self, runner) -> None:
        for t in runner.train_dataloader.dataset.pipeline.transforms:
            if isinstance(t, ScaleAwareAutoAugmentation):
                transform = t
                break
        transform.num_workers = runner.train_dataloader.num_workers
        transform.batch_size = runner.train_dataloader.batch_size // \
            runner.world_size
        transform.max_iters = runner.max_iters
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        if hasattr(model, 'teacher_model'):
            align_cfg = dict(
                student_channel=model.neck.out_channels,
                teacher_channel=model.teacher_model.neck.out_channels)
            model.bbox_head.build_align_module(align_cfg)
        model.bbox_head.num_data = len(runner.train_dataloader)

    def before_train_iter(self,
                          runner: Runner,
                          batch_idx: int,
                          data_batch: DATA_BATCH = None):
        """Get current iter for data augmentation or loss calculation."""
        iter = runner.train_loop.iter
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        model.bbox_head.iter = iter
