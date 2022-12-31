import torch
import torch.nn as nn
import torch.nn.functional as F
from mmyolo.registry import MODELS, TASK_UTILS
import copy
import numpy as np
from mmcv.cnn import Scale
from ..task_modules.assigners.align_ota_assigner import AlignOTAAssigner
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.models.utils import multi_apply, filter_scores_and_topk
from mmdet.utils import reduce_mean
from mmcv.cnn import ConvModule
from mmdet.models.losses import (DistributionFocalLoss, GIoULoss)
from ..losses import QualityFocalLoss
from typing import List, Optional, Tuple
from torch import Tensor
import torchvision
from mmengine.config import ConfigDict
from mmengine.model import (BaseModule, bias_init_with_prob, constant_init,
                            normal_init)
from .yolov5_head import YOLOv5Head
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.utils import (InstanceList, OptMultiConfig, OptConfigType, ConfigType)

@MODELS.register_module()
class ZEROHeadModule(BaseModule):
    def __init__(self,
                 num_classes=80,
                 in_channels=[128,256,512],
                 stacked_convs=0,
                 feat_channels=256,
                 reg_max=16,
                 strides=[8, 16, 32],                
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.stacked_convs = stacked_convs
        self.featmap_strides = strides
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        if stacked_convs == 0:
            feat_channels = in_channels
        if isinstance(feat_channels, list):
            self.feat_channels = feat_channels
        else:
            self.feat_channels = [feat_channels] * len(self.featmap_strides)
        # add 1 for keep consistance with former models
        self.cls_out_channels = num_classes + 1
        self.reg_max = reg_max
        self.integral = Integral(self.reg_max)
        self._init_layers()

    def _build_not_shared_convs(self, in_channel, feat_channels):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = feat_channels if i > 0 else in_channel
            kernel_size = 3 if i > 0 else 1
            cls_convs.append(
                ConvModule(chn,
                          feat_channels,
                          kernel_size,
                          stride=1,
                          conv_cfg=self.conv_cfg,                          
                          norm_cfg=self.norm_cfg,
                          act_cfg=self.act_cfg))
            reg_convs.append(
                ConvModule(chn,
                          feat_channels,
                          kernel_size,
                          stride=1,
                          conv_cfg=self.conv_cfg,
                          norm_cfg=self.norm_cfg,
                          act_cfg=self.act_cfg))

        return cls_convs, reg_convs

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for i in range(len(self.featmap_strides)):
            cls_convs, reg_convs = self._build_not_shared_convs(
                self.in_channels[i], self.feat_channels[i])
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

        self.gfl_cls = nn.ModuleList([
            nn.Conv2d(self.feat_channels[i],
                      self.cls_out_channels,
                      3,
                      padding=1) for i in range(len(self.featmap_strides))
        ])

        self.gfl_reg = nn.ModuleList([
            nn.Conv2d(self.feat_channels[i],
                      4 * (self.reg_max + 1),
                      3,
                      padding=1) for i in range(len(self.featmap_strides))
        ])

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.featmap_strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for cls_conv in self.cls_convs:
            for m in cls_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)                    
        for reg_conv in self.reg_convs:
            for m in reg_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        for i in range(len(self.featmap_strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)

    def forward(self, xin, labels=None, imgs=None, aux_targets=None):
        b, c, h, w = xin[0].shape
        # forward for bboxes and classification prediction
        cls_scores, bbox_preds, bbox_before_softmax = multi_apply(
            self.forward_single,
            xin,
            self.cls_convs,
            self.reg_convs,
            self.gfl_cls,
            self.gfl_reg,
            self.scales,
        )

        if self.training:
            return cls_scores, bbox_preds, bbox_before_softmax
        else:
            return cls_scores, bbox_preds

    def forward_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg, scale):
        """Forward feature of a single scale level.
        """
        cls_feat = x
        reg_feat = x

        for cls_conv, reg_conv in zip(cls_convs, reg_convs):
            cls_feat = cls_conv(cls_feat)
            reg_feat = reg_conv(reg_feat)

        bbox_pred = scale(gfl_reg(reg_feat)).float()
        N, C, H, W = bbox_pred.size()
        ##
        bbox_before_softmax = bbox_pred.reshape(N, 4, self.reg_max + 1, H,
                                                W)
        bbox_before_softmax = bbox_before_softmax.flatten(
            start_dim=3).permute(0, 3, 1, 2)
        bbox_pred = F.softmax(bbox_pred.reshape(N, 4, self.reg_max + 1, H, W),
                              dim=2)

        cls_score = gfl_cls(cls_feat).sigmoid()

        # cls_score = cls_score.flatten(start_dim=2).permute(
        #     0, 2, 1)  # N, h*w, self.num_classes+1
        # bbox_pred = bbox_pred.flatten(start_dim=3).permute(
        #     0, 3, 1, 2)  # N, h*w, 4, self.reg_max+1
        return cls_score, bbox_pred, bbox_before_softmax


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    """
    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        """
        b, hw, _, _ = x.size()
        x = x.reshape(b * hw * 4, self.reg_max + 1)
        y = self.project.type_as(x).unsqueeze(1)
        x = torch.matmul(x, y).reshape(b, hw, 4)
        return x


@MODELS.register_module()
class ZEROHead(YOLOv5Head):
    """Ref to Generalized Focal Loss V2: Learning Reliable Localization Quality
    Estimation for Dense Object Detection.
    """
    def __init__(self,
                 head_module: nn.Module,
                 prior_generator: ConfigType = dict(
                    type='mmdet.MlvlPointGenerator', offset=0, strides=[8, 16,
                                                                    32]),    
                 bbox_coder: ConfigType = dict(type='mmdet.DistancePointBBoxCoder'),
                 loss_cls: ConfigType = dict(
                    type='mmdet.QualityFocalLoss',
                    use_sigmoid=True,
                    beta=2.0,
                    loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                    type='mmdet.GIoULoss', loss_weight=2.0),
                 loss_obj: ConfigType = dict(
                    type='mmdet.DistributionFocalLoss',
                    use_sigmoid=True,
                    reduction='sum',
                    loss_weight=1.0),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            head_module=head_module,
            prior_generator=prior_generator,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_obj=loss_obj,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg.assigner)
            if self.train_cfg.get('sampler', None) is not None:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg.sampler, default_args=dict(context=self))
            else:
                self.sampler = PseudoSampler(context=self)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        return self.head_module(x)

    def loss(self, xin, labels=None, imgs=None, aux_targets=None):

        # prepare labels during training
        b, c, h, w = xin[0].shape
        if labels is not None:
            gt_bbox_list = []
            gt_cls_list = []
            for label in labels:
                gt_bbox_list.append(label.bbox)
                gt_cls_list.append((label.get_field('labels') -
                                    1).long())  # labels starts from 1

        # prepare priors for label assignment and bbox decode
        mlvl_priors_list = [
            self.get_single_level_center_priors(xin[i].shape[0],
                                                xin[i].shape[-2:],
                                                stride,
                                                dtype=torch.float32,
                                                device=xin[0].device)
            for i, stride in enumerate(self.featmap_strides)
        ]
        mlvl_priors = torch.cat(mlvl_priors_list, dim=1)

        # forward for bboxes and classification prediction
        cls_scores, bbox_preds, bbox_before_softmax = multi_apply(
            self.forward_single,
            xin,
            self.cls_convs,
            self.reg_convs,
            self.gfl_cls,
            self.gfl_reg,
            self.scales,
        )
        cls_scores = torch.cat(cls_scores, dim=1)
        bbox_preds = torch.cat(bbox_preds, dim=1)
        bbox_before_softmax = torch.cat(bbox_before_softmax, dim=1)

        # calculating losses
        loss = self._loss(
            cls_scores,
            bbox_preds,
            bbox_before_softmax,
            gt_bbox_list,
            gt_cls_list,
            mlvl_priors,
        )
        return loss


    def loss_by_feat(self, **kwargs) -> dict:
        pass


    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None,                    
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        # prepare priors for label assignment and bbox decode
        assert len(cls_scores) == len(bbox_preds)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        num_imgs = len(batch_img_metas)
        multi_label = cfg.get('multi_label', False)
        multi_label &= self.num_classes > 1
        cfg.multi_label = multi_label

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride = True)
            # self.mlvl_priors = [
            #     self.get_single_level_center_priors(cls_scores[i].shape[0],
            #                                         featmap_sizes[i],
            #                                         stride,
            #                                         dtype=torch.float32,
            #                                         device=bbox_preds[i].device)
            #     for i, stride in enumerate(self.featmap_strides)
            # ]
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)[None,...]

        # mlvl_strides = [
        #     flatten_priors.new_full(
        #         (featmap_size.numel() * self.num_base_priors, ), stride) for
        #     featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        # ]
        # flatten_stride = torch.cat(mlvl_strides)
        flatten_cls_scores = [
            cls_score.flatten(start_dim=2).permute(0, 2, 1)
            for cls_score in cls_scores
        ]  # N, h*w, self.num_classes+1
        flatten_bbox_preds = [
            bbox_pred.flatten(start_dim=3).permute(0, 3, 1, 2) 
            for bbox_pred in bbox_preds
        ] # N, h*w, 4, self.reg_max+1


        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)[:, :, :self.num_classes]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_bbox_preds = self.head_module.integral(flatten_bbox_preds) * flatten_priors[..., 2, None]
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[..., :2].expand(num_imgs,-1,2), flatten_bbox_preds)
        # batch bbox decode
        
        # bbox_preds = distance2bbox(flatten_priors[..., :2], bbox_preds)
        ## TODO:discuss with teacher about the postprocess postion
        results_list = []
        for (bboxes, scores, img_meta) in zip(flatten_decoded_bboxes, flatten_cls_scores, batch_img_metas):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(
                scores=scores, labels=labels, bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(
                results=results,
                cfg=cfg,
                rescale=False,
                with_nms=with_nms,
                img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list


    def get_single_level_center_priors(self, batch_size, featmap_size, stride,
                                       dtype, device):

        h, w = featmap_size
        x_range = (torch.arange(0, int(w), dtype=dtype,
                                device=device)) * stride
        y_range = (torch.arange(0, int(h), dtype=dtype,
                                device=device)) * stride

        x = x_range.repeat(h, 1)
        y = y_range.unsqueeze(-1).repeat(1, w)

        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0], ), stride)
        priors = torch.stack([x, y, strides, strides], dim=-1)

        return priors.unsqueeze(0).repeat(batch_size, 1, 1)

    def _loss(
        self,
        cls_scores,
        bbox_preds,
        bbox_before_softmax,
        gt_bboxes,
        gt_labels,
        mlvl_center_priors,
        gt_bboxes_ignore=None,
    ):
        """Compute losses of the head.
        """
        device = cls_scores[0].device

        # get decoded bboxes for label assignment
        dis_preds = self.integral(bbox_preds) * mlvl_center_priors[..., 2,
                                                                   None]
        decoded_bboxes = distance2bbox(mlvl_center_priors[..., :2], dis_preds)
        cls_reg_targets = self.get_targets(cls_scores,
                                           decoded_bboxes,
                                           gt_bboxes,
                                           mlvl_center_priors,
                                           gt_labels_list=gt_labels)

        if cls_reg_targets is None:
            return None

        (labels_list, label_scores_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, dfl_targets_list, num_pos) = cls_reg_targets

        num_total_pos = max(
            reduce_mean(torch.tensor(num_pos).type(
                torch.float).to(device)).item(), 1.0)

        labels = torch.cat(labels_list, dim=0)
        label_scores = torch.cat(label_scores_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        dfl_targets = torch.cat(dfl_targets_list, dim=0)

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # bbox_preds = bbox_preds.reshape(-1, 4 * (self.reg_max + 1))
        bbox_before_softmax = bbox_before_softmax.reshape(
            -1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)

        loss_qfl = self.loss_cls(cls_scores, (labels, label_scores),
                                 avg_factor=num_total_pos)

        pos_inds = torch.nonzero((labels >= 0) & (labels < self.num_classes),
                                 as_tuple=False).squeeze(1)

        if len(pos_inds) > 0:
            weight_targets = cls_scores.detach()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            norm_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)
            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=1.0 * norm_factor,
            )
            loss_dfl = self.loss_dfl(
                bbox_before_softmax[pos_inds].reshape(-1, self.reg_max + 1),
                dfl_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * norm_factor,
            )

        else:
            loss_bbox = bbox_preds.sum() * 0.0
            loss_dfl = bbox_preds.sum() * 0.0

        total_loss = loss_qfl + loss_bbox + loss_dfl

        return dict(
            total_loss=total_loss,
            loss_cls=loss_qfl,
            loss_bbox=loss_bbox,
            loss_dfl=loss_dfl,
        )

    def get_targets(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    mlvl_center_priors,
                    gt_labels_list=None,
                    unmap_outputs=True):
        """Get targets for GFL head.
        """
        num_imgs = mlvl_center_priors.shape[0]

        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        (all_labels, all_label_scores, all_label_weights, all_bbox_targets,
         all_bbox_weights, all_dfl_targets, all_pos_num) = multi_apply(
             self.get_target_single,
             mlvl_center_priors,
             cls_scores,
             bbox_preds,
             gt_bboxes_list,
             gt_labels_list,
         )
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        all_pos_num = sum(all_pos_num)

        return (all_labels, all_label_scores, all_label_weights,
                all_bbox_targets, all_bbox_weights, all_dfl_targets,
                all_pos_num)

    def get_target_single(self,
                          center_priors,
                          cls_scores,
                          bbox_preds,
                          gt_bboxes,
                          gt_labels,
                          unmap_outputs=True,
                          gt_bboxes_ignore=None):
        """Compute regression, classification targets for anchors in a single
        image.
        """
        # assign gt and sample anchors

        num_valid_center = center_priors.shape[0]

        labels = center_priors.new_full((num_valid_center, ),
                                        self.num_classes,
                                        dtype=torch.long)
        label_weights = center_priors.new_zeros(num_valid_center,
                                                dtype=torch.float)
        label_scores = center_priors.new_zeros(num_valid_center,
                                               dtype=torch.float)

        bbox_targets = torch.zeros_like(center_priors)
        bbox_weights = torch.zeros_like(center_priors)
        dfl_targets = torch.zeros_like(center_priors)

        if gt_labels.size(0) == 0:

            return (labels, label_scores, label_weights, bbox_targets,
                    bbox_weights, dfl_targets, 0)

        assign_result = self.assigner.assign(cls_scores.detach(),
                                             center_priors,
                                             bbox_preds.detach(), gt_bboxes,
                                             gt_labels)

        pos_inds, neg_inds, pos_bbox_targets, pos_assign_gt_inds = self.sample(
            assign_result, gt_bboxes)
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            labels[pos_inds] = gt_labels[pos_assign_gt_inds]
            label_scores[pos_inds] = pos_ious
            label_weights[pos_inds] = 1.0

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            dfl_targets[pos_inds, :] = (bbox2distance(
                center_priors[pos_inds, :2] / center_priors[pos_inds, None, 2],
                pos_bbox_targets / center_priors[pos_inds, None, 2],
                self.reg_max))
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        # map up to original set of anchors

        return (labels, label_scores, label_weights, bbox_targets,
                bbox_weights, dfl_targets, pos_inds.size(0))

    def sample(self, assign_result, gt_bboxes):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0,
                                 as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0,
                                 as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]

        return pos_inds, neg_inds, pos_gt_bboxes, 