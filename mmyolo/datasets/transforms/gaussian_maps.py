# This file mainly comes from
# https://github.com/dvlab-research/SA-AutoAug/blob/master/FCOS/fcos_core/augmentations/box_level_augs/gaussian_maps.py
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import numpy as np

# import torch

# def _gaussian_map(img, boxes, scale_splits=None, scale_ratios=None):
#     g_maps = torch.zeros(*img.shape[:-1])
#     height, width = img.shape[0], img.shape[1]

#     x_range = torch.arange(0, height, 1)
#     y_range = torch.arange(0, width, 1)
#     xx, yy = torch.meshgrid(x_range, y_range)
#     pos = torch.empty(xx.shape + (2, ))
#     pos[:, :, 0] = xx
#     pos[:, :, 1] = yy

#     for j, box in enumerate(boxes):
#         y1, x1, y2, x2 = box
#         x, y, h, w = x1, y1, x2 - x1, y2 - y1
#         mean_torch = torch.tensor([x + h // 2, y + w // 2])
#         if scale_ratios is None:
#             scale_ratio = 1.0
#         else:
#             ratio_list = [0.2, 0.4, 0.6, 0.8, 1.0, 2, 4, 6, 8, 10]
#             if h * w < scale_splits[0]:
#                 scale_ratio = ratio_list[scale_ratios[0]] * \
#                 scale_splits[0] / (h * w)
#             elif h * w < scale_splits[1]:
#                 scale_ratio = ratio_list[scale_ratios[1]] * (
#                     scale_splits[0] + scale_splits[1]) / 2.0 / (h * w)
#             elif h * w < scale_splits[2]:
#                 scale_ratio = ratio_list[scale_ratios[2]] * \
#                     scale_splits[2] / (h * w)
#             else:
#                 scale_ratio = ratio_list[scale_ratios[2]]

#         r_var = (scale_ratio * height * width / (2 * np.pi))**0.5
#         var_x = torch.tensor([(h / height) * r_var],
#                              dtype=torch.float32)
#         var_y = torch.tensor([(w / width) * r_var],
#                              dtype=torch.float32)
#         g_map = torch.exp(-(((xx.float() - mean_torch[0])**2 /
#                              (2.0 * var_x**2) +
#                              (yy.float() - mean_torch[1])**2 /
#                              (2.0 * var_y**2))))
#         g_maps += g_map
#     g_maps = g_maps.numpy()
#     g_maps = np.stack((g_maps, g_maps, g_maps),axis=-1)
#     return g_maps


def _gaussian_map(img, boxes, scale_splits=None, scale_ratios=None):
    g_maps = np.zeros(img.shape[:-1], dtype=np.float32)
    height, width = img.shape[0], img.shape[1]
    x_range = np.arange(0, width, 1, dtype=np.float32)
    y_range = np.arange(0, height, 1, dtype=np.float32)
    yy, xx = np.meshgrid(x_range, y_range)

    for box in boxes:
        x1, y1, x2, y2 = box
        x, y, h, w = x1, y1, y2 - y1, x2 - x1
        mean = np.array([y + h // 2, x + w // 2], dtype=np.float32)
        if scale_ratios is None:
            scale_ratio = 1.0
        else:
            ratio_list = [0.2, 0.4, 0.6, 0.8, 1.0, 2, 4, 6, 8, 10]
            if h * w < scale_splits[0]:
                scale_ratio = ratio_list[scale_ratios[0]] * scale_splits[0] / (
                    h * w)
            elif h * w < scale_splits[1]:
                scale_ratio = ratio_list[scale_ratios[1]] * (
                    scale_splits[0] + scale_splits[1]) / 2.0 / (
                        h * w)
            elif h * w < scale_splits[2]:
                scale_ratio = ratio_list[scale_ratios[2]] * scale_splits[2] / (
                    h * w)
            else:
                scale_ratio = ratio_list[scale_ratios[2]]

        r_var = (scale_ratio * height * width / (2 * np.pi))**0.5
        var_y = np.array([(h / height) * r_var], dtype=np.float32)
        var_x = np.array([(w / width) * r_var], dtype=np.float32)
        g_map = np.exp(-((yy - mean[1])**2 / (2.0 * var_x**2) +
                         (xx - mean[0])**2 / (2.0 * var_y**2)))
        g_maps += g_map
    g_maps = np.stack((g_maps, g_maps, g_maps), axis=-1)
    return g_maps
