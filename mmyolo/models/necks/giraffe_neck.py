# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
# from mmdet.models.backbones.csp_darknet import CSPLayer
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule

from mmyolo.registry import MODELS
from ..layers import RepVGGBlock

# from .base_yolo_neck import BaseYOLONeck


class BasicBlock_3x3_Reverse(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 shortcut: bool = True):
        super().__init__()
        assert in_channels == out_channels
        hidden_channels = int(in_channels * expansion)
        self.conv1 = ConvModule(
            hidden_channels,
            out_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = RepVGGBlock(
            in_channels,
            hidden_channels,
            3,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            use_bn_first=False)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        if self.shortcut:
            return x + y
        else:
            return y


class CSPStage(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 num_blocks: int = 1,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None,
                 block_fn='BasicBlock_3x3_Reverse'):
        super().__init__(init_cfg=init_cfg)
        first_channel = int(out_channels // 2)
        mid_channel = int(out_channels - first_channel)
        self.conv_short = ConvModule(
            in_channels,
            first_channel,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv_main = ConvModule(
            in_channels,
            mid_channel,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.convs = nn.Sequential()

        next_ch_in = mid_channel
        for i in range(num_blocks):
            if block_fn == 'BasicBlock_3x3_Reverse':
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse(
                        next_ch_in,
                        mid_channel,
                        expansion,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        shortcut=True))
            else:
                raise NotImplementedError
            next_ch_in = mid_channel
        self.conv_final = ConvModule(
            mid_channel * num_blocks + first_channel,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        y1 = self.conv_short(x)
        y2 = self.conv_main(x)

        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        y = self.conv_final(y)
        return y


@MODELS.register_module()
class GiraffeNeckv2(BaseModule):

    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        expansion=0.75,
        in_features=[2, 3, 4],
        num_csp_blocks: int = 3,
        freeze_all: bool = False,
        use_depthwise: bool = False,
        spp=False,
        upsample_cfg: ConfigType = dict(scale_factor=2, mode='nearest'),
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='ReLU', inplace=True),
        init_cfg: OptMultiConfig = None,
        block_name='BasicBlock_3x3_Reverse',
    ):
        super().__init__(init_cfg=init_cfg)
        self.expansion = expansion
        self.num_csp_blocks = round(num_csp_blocks * deepen_factor)
        self.upsample_cfg = upsample_cfg
        self.in_features = in_features
        self.conv_cfg = conv_cfg
        self.conv = DepthwiseSeparableConvModule \
            if use_depthwise else ConvModule
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.freeze_all = freeze_all

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self._init_layers()

    def _init_layers(self):
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.bu_conv13 = self.conv(
            self.in_channels[1],
            self.in_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.merge_3 = CSPStage(
            self.in_channels[1] + self.in_channels[2],
            self.in_channels[2],
            self.expansion,
            round(3 * self.deepen_factor),
            act_cfg=self.act_cfg)
        self.bu_conv24 = self.conv(
            self.in_channels[0],
            self.in_channels[0],
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.merge_4 = CSPStage(
            self.in_channels[0] + self.in_channels[1] + self.in_channels[2],
            self.in_channels[1],
            self.expansion,
            round(3 * self.deepen_factor),
            act_cfg=self.act_cfg)

        self.merge_5 = CSPStage(
            self.in_channels[1] + self.in_channels[0],
            self.out_channels[0],
            self.expansion,
            round(3 * self.deepen_factor),
            act_cfg=self.act_cfg)
        self.bu_conv57 = self.conv(
            self.out_channels[0],
            self.out_channels[0],
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.merge_7 = CSPStage(
            self.out_channels[0] + self.in_channels[1],
            self.out_channels[1],
            self.expansion,
            round(3 * self.deepen_factor),
            act_cfg=self.act_cfg)
        self.bu_conv46 = self.conv(
            self.in_channels[1],
            self.in_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bu_conv76 = self.conv(
            self.out_channels[1],
            self.out_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.merge_6 = CSPStage(
            self.in_channels[1] + self.out_channels[1] + self.in_channels[2],
            self.out_channels[2],
            self.expansion,
            round(3 * self.deepen_factor),
            act_cfg=self.act_cfg)

        # self.reduce_layers = nn.ModuleList()
        # for idx in range(len(in_channels)):
        #     self.reduce_layers.append(self.build_reduce_layer(idx))

        # # build top-down blocks
        # self.upsample_layers = nn.ModuleList()
        # self.extra_downsample_layers = nn.ModuleList()
        # self.top_down_layers = nn.ModuleList()
        # for idx in range(len(in_channels) - 1, 0, -1):
        #     self.upsample_layers.append(self.build_upsample_layer(idx))
        #     self.extra_downsample_layers.append(self.build_downsample_layer(idx))
        # for idx in range(len(in_channels), 0, -1):
        #     self.top_down_layers.append(self.build_top_down_layer(idx))

        # # build bottom-up blocks
        # self.downsample_layers = nn.ModuleList()
        # self.bottom_up_layers = nn.ModuleList()
        # for idx in range(len(in_channels)):
        #     self.downsample_layers.append(self.build_downsample_layer(idx))
        #     self.bottom_up_layers.append(self.build_bottom_up_layer(idx))

        # self.out_layers = nn.ModuleList()
        # for idx in range(len(in_channels)):
        #     self.out_layers.append(self.build_out_layer(idx))

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        layer = nn.Identity()

        return layer

    def build_upsample_layer(self, *args, **kwargs) -> nn.Module:
        """build upsample layer."""
        return nn.Upsample(**self.upsample_cfg)

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """

        if idx == 0:
            self.merge_3 = CSPStage(
                self.in_channels[1] + self.in_channels[2],
                self.in_channels[2],
                self.expansion,
                round(3 * self.deepen_factor),
                act_cfg=self.act_cfg)
        if idx == 1:
            self.merge_4 = CSPStage(
                self.in_channels[0] + self.in_channels[1] +
                self.in_channels[2],
                self.in_channels[1],
                self.expansion,
                round(3 * self.deepen_factor),
                act_cfg=self.act_cfg)
        if idx == 2:
            self.merge_5 = CSPStage(
                self.in_channels[1] + self.in_channels[0],
                self.out_channels[0],
                self.expansion,
                round(3 * self.deepen_factor),
                act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        self.merge_7 = CSPStage(
            self.out_channels[0] + self.in_channels[1],
            self.out_channels[1],
            self.expansion,
            round(3 * self.deepen_factor),
            act_cfg=self.act_cfg)

        self.merge_6 = CSPStage(
            self.in_channels[1] + self.out_channels[1] + self.in_channels[2],
            self.out_channels[2],
            self.expansion,
            round(3 * self.deepen_factor),
            act_cfg=self.act_cfg)

    def build_extra_downsample_layer(self, idx: int) -> nn.Module:
        """build extra downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        layer = self.conv(
            self.in_channels[idx],
            self.in_channels[idx],
            3,
            2,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        return layer

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        if idx == 0:
            # bu_conv57
            layer = self.conv(
                self.out_channels[0],
                self.out_channels[0],
                3,
                2,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            # bu_conv46 && bu_conv76
            layer = self.conv(
                self.in_channels[1],
                self.in_channels[1],
                3,
                2,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        return layer

    def build_out_layer(self, idx: int) -> nn.Module:
        """build out layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The out layer.
        """
        return ConvModule(
            self.in_channels[idx],
            self.out_channels,
            1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, out_features):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        [x2, x1, x0] = out_features

        # node x3
        x13 = self.bu_conv13(x1)
        x3 = torch.cat([x0, x13], 1)
        x3 = self.merge_3(x3)

        # node x4
        x34 = self.upsample(x3)
        x24 = self.bu_conv24(x2)
        x4 = torch.cat([x1, x24, x34], 1)
        x4 = self.merge_4(x4)

        # node x5
        x45 = self.upsample(x4)
        x5 = torch.cat([x2, x45], 1)
        x5 = self.merge_5(x5)

        # node x8
        # x8 = x5

        # node x7
        x57 = self.bu_conv57(x5)
        x7 = torch.cat([x4, x57], 1)
        x7 = self.merge_7(x7)

        # node x6
        x46 = self.bu_conv46(x4)
        x76 = self.bu_conv76(x7)
        x6 = torch.cat([x3, x46, x76], 1)
        x6 = self.merge_6(x6)

        outputs = (x5, x7, x6)
        return outputs
