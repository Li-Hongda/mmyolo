# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.losses import mse_loss, weighted_loss

from mmyolo.registry import MODELS


class MimicLoss(nn.Module):

    def forward(
        self,
        student_feat: Tuple[torch.Tensor],
        teacher_feat: Tuple[torch.Tensor],
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[Union[str, bool]] = None
    ) -> Tuple[Union[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward function.

        Args:
            student_feat (tuple): The student model prediction with
                shape (N, C, H, W) in list.
            teacher_feat (tuple): The teacher model prediction with
                shape (N, C, H, W) in list.
            weight (Tensor, optional): Element-wise weights.
            avg_factor (float, optional): Average factor when computing the
                mean of losses.
            reduction_override (str, bool, optional): Same as built-in losses
                of PyTorch. Defaults to None.
        Returns:
            torch.Tensor: The calculated loss
        """
        assert len(student_feat) == len(teacher_feat)
        losses = []

        for (s, t) in zip(student_feat, teacher_feat):
            assert s.shape == t.shape
            losses.append(mse_loss(s, t))
        loss = sum(losses)
        return loss


class MGDLoss(nn.Module):

    def __init__(self,
                 channels_s,
                 channels_t,
                 alpha_mgd=0.00002,
                 lambda_mgd=0.65):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.generation = [
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3,
                          padding=1)).to(device) for channel in channels_t
        ]

    def forward(
        self,
        student_feat: Tuple[torch.Tensor],
        teacher_feat: Tuple[torch.Tensor],
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[Union[str, bool]] = None
    ) -> Tuple[Union[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward function.

        Args:
            student_feat (tuple): The student model prediction with
                shape (N, C, H, W) in list.
            teacher_feat (tuple): The teacher model prediction with
                shape (N, C, H, W) in list.
            weight (Tensor, optional): Element-wise weights.
            avg_factor (float, optional): Average factor when computing the
                mean of losses.
            reduction_override (str, bool, optional): Same as built-in losses
                of PyTorch. Defaults to None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert len(student_feat) == len(teacher_feat)
        losses = []

        for idx, (s, t) in enumerate(zip(student_feat, teacher_feat)):
            assert s.shape == t.shape
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N

        return dis_loss


@weighted_loss
def cwd_loss(student_feat: torch.Tensor,
             teacher_feat: torch.Tensor,
             T: float = 1.0):
    assert student_feat.shape == teacher_feat.shape
    N, C, H, W = student_feat.shape
    # normalize in channel diemension
    softmax_pred_T = F.softmax(teacher_feat.view(-1, W * H) / T, dim=1)

    logsoftmax = torch.nn.LogSoftmax(dim=1)
    cost = torch.sum(
        softmax_pred_T * logsoftmax(teacher_feat.view(-1, W * H) / T) -
        softmax_pred_T * logsoftmax(student_feat.view(-1, W * H) / T)) * (
            T**2)
    return cost / (C * N)


@MODELS.register_module()
class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.

    <https://arxiv.org/abs/2011.13256>`_.
    """

    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        student_feat: Tuple[torch.Tensor],
        teacher_feat: Tuple[torch.Tensor],
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[Union[str, bool]] = None
    ) -> Tuple[Union[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward function.

        Args:
            student_feat (tuple): The student model prediction with
                shape (N, C, H, W) in list.
            teacher_feat (tuple): The teacher model prediction with
                shape (N, C, H, W) in list.
            weight (Tensor, optional): Element-wise weights.
            avg_factor (float, optional): Average factor when computing the
                mean of losses.
            reduction_override (str, bool, optional): Same as built-in losses
                of PyTorch. Defaults to None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert len(student_feat) == len(teacher_feat)
        losses = []

        for (s, t) in zip(student_feat, teacher_feat):
            cost = cwd_loss(s, t)
            losses.append(cost)
        loss = sum(losses) * weight

        return loss
