# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.backbones.csp_darknet import Focus, SPPBottleneck
from mmdet.utils import ConfigType, OptMultiConfig
from mmengine.model import BaseModule

from mmyolo.registry import MODELS
from ..layers import RepVGGBlock
from .base_backbone import BaseBackbone


class ResConvBlock(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 btn_channels,
                 kernel_size=1,
                 stride=1,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 reparam=False,
                 arch='S'):
        super().__init__()
        self.stride = stride
        if arch in ['T', 'S']:
            self.conv1 = ConvModule(
                in_channels,
                btn_channels,
                kernel_size=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.conv1 = ConvModule(
                in_channels,
                btn_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                norm_cfg=norm_cfg,
                act_cfg=None)

        if not reparam:
            self.conv2 = ConvModule(
                btn_channels,
                out_channels,
                kernel_size,
                stride,
                norm_cfg=norm_cfg,
                act_cfg=None)
        else:
            self.conv2 = RepVGGBlock(
                btn_channels,
                out_channels,
                kernel_size,
                stride,
                norm_cfg=norm_cfg,
                act_cfg=None,
                use_bn_first=False)

        self.activation_function = MODELS.build(act_cfg)

        if in_channels != out_channels and stride != 2:
            self.residual_proj = ConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                norm_cfg=norm_cfg,
                act_cfg=None)
        else:
            self.residual_proj = None

    def forward(self, x):
        if self.residual_proj is not None:
            reslink = self.residual_proj(x)
        else:
            reslink = x
        x = self.conv1(x)
        x = self.activation_function(x)
        x = self.conv2(x)
        if self.stride != 2:
            x = x + reslink
        x = self.activation_function(x)
        return x


class SuperResStem(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 btn_channels: int,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1,
                 num_blocks: int = 1,
                 use_spp: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 reparam: bool = False,
                 arch: str = 'S'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.block_list = nn.ModuleList()
        for block_id in range(num_blocks):
            if block_id == 0:
                in_channels = self.in_channels
                out_channels = self.out_channels
                this_stride = stride
                this_kernel_size = kernel_size
            else:
                in_channels = self.out_channels
                out_channels = self.out_channels
                this_stride = 1
                this_kernel_size = kernel_size
            the_block = ResConvBlock(
                in_channels,
                out_channels,
                btn_channels,
                this_kernel_size,
                this_stride,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                reparam=reparam,
                arch=arch)
            self.block_list.append(the_block)
            if block_id == 0 and use_spp:
                self.block_list.append(
                    SPPBottleneck(out_channels, out_channels))

    def forward(self, x):
        output = x
        for block in self.block_list:
            output = block(output)
        return output


class CSPWrapper(BaseModule):

    def __init__(self,
                 convstem: nn.Module,
                 use_spp: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True)):
        super().__init__()
        self.use_spp = use_spp
        if isinstance(convstem, tuple):
            in_channels = convstem[0].in_channels
            out_channels = convstem[-1].out_channels
            hidden_channels = convstem[0].out_channels // 2
            _convstem = nn.ModuleList()
            for modulelist in convstem:
                for layer in modulelist.block_list:
                    _convstem.append(layer)
        else:
            in_channels = convstem.in_channels
            out_channels = convstem.out_channels
            hidden_channels = out_channels // 2
            _convstem = convstem.block_list

        self.convstem = nn.ModuleList()
        for layer in _convstem:
            self.convstem.append(layer)

        self.downsampler = ConvModule(
            in_channels,
            hidden_channels * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg)
        if self.use_spp:
            self.spp = SPPBottleneck(
                hidden_channels * 2, hidden_channels * 2, act_cfg=act_cfg)
        if len(self.convstem) > 0:
            self.conv_main = ConvModule(
                hidden_channels * 2,
                hidden_channels,
                kernel_size=1,
                stride=1,
                norm_cfg=norm_cfg)
            self.conv_short = ConvModule(
                hidden_channels * 2,
                out_channels // 2,
                kernel_size=1,
                stride=1,
                norm_cfg=norm_cfg)
            self.conv_final = ConvModule(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                norm_cfg=norm_cfg)

    def forward(self, x):
        x = self.downsampler(x)
        if self.use_spp:
            x = self.spp(x)
        if len(self.convstem) > 0:
            shortcut = self.conv_short(x)
            x = self.conv_main(x)
            for block in self.convstem:
                x = block(x)
            x = torch.cat((x, shortcut), dim=1)
            x = self.conv_final(x)
        return x


class CSPStem(BaseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 btn_channels: int,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1,
                 num_blocks: int = 1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 reparam: bool = False,
                 arch: str = 'S'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.block_list = nn.ModuleList()
        if self.stride == 2:
            self.num_blocks = num_blocks - 1
        else:
            self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.arch = arch
        out_channels = out_channels // 2

        self.block_list = nn.ModuleList()
        for block_id in range(self.num_blocks):
            if self.stride == 1 and block_id == 0:
                in_channels = in_channels // 2
            else:
                in_channels = out_channels
            the_block = ResConvBlock(
                in_channels,
                out_channels,
                btn_channels,
                kernel_size,
                norm_cfg=norm_cfg,
                act_cfg=self.act_cfg,
                reparam=reparam,
                arch=arch)
            self.block_list.append(the_block)

    def forward(self, x):
        output = x
        for block in self.block_list:
            output = block(output)
        return output


@MODELS.register_module()
class TinyNAS(BaseBackbone):
    # in_channels, out_channels, btn_channels, kernel_size, \
    # stride, num_blocks, use_spp
    arch_settings = {
        'T': [[24, 64, 24, 3, 2, 2, False], [64, 96, 64, 3, 2, 2, False],
              [96, 192, 96, 3, 2, 2, False], [192, 192, 152, 3, 1, 2, False],
              [192, 384, 192, 3, 2, 1, True]],
        'S': [[32, 128, 24, 3, 2, 1, False], [128, 128, 88, 3, 2, 5, False],
              [128, 256, 128, 3, 2, 3, False], [256, 256, 120, 3, 1, 2, False],
              [256, 512, 144, 3, 2, 1, True]],
        'M': [[32, 128, 64, 3, 2, 2, None], [128, 128, 64, 3, 2, 4, None],
              [128, 256, 256, 3, 2, 4, [256, 256, 256, 3, 1, 4]],
              [256, 512, 256, 3, 2, 3, None]],
    }

    def __init__(self,
                 arch: str = 'S',
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 4, 5),
                 frozen_stages: int = -1,
                 reparam: bool = True,
                 plugins: Union[dict, List[dict]] = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        assert arch in self.arch_settings.keys()
        self.arch = arch
        self.reparam = reparam
        super().__init__(
            self.arch_settings[arch],
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        stem = Focus(
            3,
            int(self.arch_setting[0][0] * self.widen_factor),
            kernel_size=3,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        return stem

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        if self.arch in ['S', 'T']:
            in_channels, out_channels, btn_channels, kernel_size, \
                stride, num_blocks, use_spp = setting
            # if the_block_class == 'SuperResConvK1KX':
            spp = use_spp if stage_idx == len(
                self.arch_settings[self.arch]) - 1 else False
            stage = []
            the_block = SuperResStem(
                in_channels,
                out_channels,
                btn_channels,
                kernel_size,
                stride,
                num_blocks,
                spp,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                reparam=self.reparam,
                arch=self.arch)
        elif self.arch == 'M':
            in_channels, out_channels, btn_channels, kernel_size, \
                stride, num_blocks, extra_cfg = setting
            stage = []
            use_spp = True if stage_idx == len(
                self.arch_settings[self.arch]) - 1 else False
            if extra_cfg is not None:
                in_c, out_c, btn_c, k_size, s, n_blocks = extra_cfg
                block_1 = CSPStem(
                    in_channels,
                    out_channels,
                    btn_channels,
                    kernel_size,
                    stride,
                    num_blocks,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    reparam=self.reparam,
                    arch=self.arch)
                block_2 = CSPStem(
                    in_c,
                    out_c,
                    btn_c,
                    k_size,
                    s,
                    n_blocks,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    reparam=self.reparam,
                    arch=self.arch)
                the_block = CSPWrapper((block_1, block_2),
                                       use_spp=use_spp,
                                       norm_cfg=self.norm_cfg,
                                       act_cfg=self.act_cfg)
            else:
                block = CSPStem(
                    in_channels,
                    out_channels,
                    btn_channels,
                    kernel_size,
                    stride,
                    num_blocks,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    reparam=self.reparam,
                    arch=self.arch)
                the_block = CSPWrapper(
                    block,
                    use_spp=use_spp,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
        stage.append(the_block)
        return stage
