# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional

import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmdet.datasets.transforms.augment_wrappers import _MAX_LEVEL, level_to_mag
from mmdet.structures.bbox import HorizontalBoxes

from mmyolo.registry import TRANSFORMS
from .gaussian_maps import _gaussian_map


@TRANSFORMS.register_module()
class ColorTransform(BaseTransform):
    """Base class for color transformations. All color transformations need to
    inherit from this base class. ``ColorTransform`` unifies the class
    attributes and class functions of color transformations (Color, Brightness,
    Contrast, Sharpness, Solarize, SolarizeAdd, Equalize, AutoContrast, Invert,
    and Posterize), and only distort color channels, without impacting the
    locations of the instances.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing the geometric
            transformation and should be in range [0, 1]. Defaults to 1.0.
        level (int, optional): The level should be in range [0, _MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for color transformation.
            Defaults to 0.1.
        max_mag (float): The maximum magnitude for color transformation.
            Defaults to 1.9.
    """

    def __init__(self,
                 prob: float = 1.0,
                 bbox_prob: float = 0.3,
                 level: Optional[int] = None,
                 min_mag: float = 0.1,
                 max_mag: float = 1.9,
                 scale_splits: List[int] = [2048, 10240, 51200],
                 scale_ratio: List[int] = [7, 5, 1]) -> None:
        assert 0 <= prob <= 1.0, f'The probability of the transformation ' \
                                 f'should be in range [0,1], got {prob}.'
        assert level is None or isinstance(level, int), \
            f'The level should be None or type int, got {type(level)}.'
        assert level is None or 0 <= level <= _MAX_LEVEL, \
            f'The level should be in range [0,{_MAX_LEVEL}], got {level}.'
        assert isinstance(min_mag, float), \
            f'min_mag should be type float, got {type(min_mag)}.'
        assert isinstance(max_mag, float), \
            f'max_mag should be type float, got {type(max_mag)}.'
        assert min_mag <= max_mag, \
            f'min_mag should smaller than max_mag, ' \
            f'got min_mag={min_mag} and max_mag={max_mag}'
        self.prob = prob
        self.bbox_prob = bbox_prob
        self.level = level
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.scale_splits = scale_splits
        self.scale_ratio = scale_ratio

    def _transform_img(self, results: dict, mag: float) -> None:
        """Transform the image."""
        pass

    # @cache_randomness
    def _random_disable(self, iter_ratio: float):
        """Randomly disable the transform."""
        return np.random.rand() > (self.prob * iter_ratio)

    def _box_sample_prob(self, bbox: HorizontalBoxes) -> float:
        ratios = np.array(self.scale_ratio)
        ratios = ratios / ratios.sum()
        area = bbox.areas.item()
        if area == 0:
            return 0
        if area < self.scale_splits[0]:
            scale_ratio = ratios[0]
        elif area < self.scale_splits[1]:
            scale_ratio = ratios[1]
        else:
            scale_ratio = ratios[2]
        return self.bbox_prob * scale_ratio

    @cache_randomness
    def _get_mag(self):
        """Get the magnitude of the transform."""
        return level_to_mag(self.level, self.min_mag, self.max_mag)

    def transform(self, results: dict) -> dict:
        """Transform function for images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Transformed results.
        """

        if self._random_disable(results['iter_ratio']):
            return results
        mag = self._get_mag()
        results['cropped_gt_bboxes'] = []
        mag = self._get_mag()
        for bbox in results['gt_bboxes']:
            prob = self._box_sample_prob(bbox)
            if np.random.rand() >= prob:
                continue
            results['cropped_gt_bboxes'].append(bbox.tensor.squeeze())
        results['aug_img'] = self._transform_img(results['img'], mag)
        # if len(results.get('cropped_gt_bboxes')) > 0:
        results['img'] = self._gaussian_transform(results)
        return results

    def _gaussian_transform(self, results: dict) -> torch.tensor:
        cp_img = copy.deepcopy(results['img'])
        cp_aug_img = copy.deepcopy(results['aug_img'])
        g_maps = _gaussian_map(cp_img, results['cropped_gt_bboxes'],
                               self.scale_splits, self.scale_ratio)
        g_maps = g_maps.clip(min=0, max=1.0)
        out_img = cp_img * (1 - g_maps) + cp_aug_img * g_maps
        return out_img

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'level={self.level}, '
        repr_str += f'min_mag={self.min_mag}, '
        repr_str += f'max_mag={self.max_mag})'
        return repr_str


@TRANSFORMS.register_module()
class Color(ColorTransform):
    """Adjust the color balance of the image, in a manner similar to the
    controls on a colour TV set. A magnitude=0 gives a black & white image,
    whereas magnitude=1 gives the original image. The bboxes, masks and
    segmentations are not modified.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Color transformation.
            Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for Color transformation.
            Defaults to 0.1.
        max_mag (float): The maximum magnitude for Color transformation.
            Defaults to 1.9.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.1,
                 max_mag: float = 1.9) -> None:
        assert 0. <= min_mag <= 2.0, \
            f'min_mag for Color should be in range [0,2], got {min_mag}.'
        assert 0. <= max_mag <= 2.0, \
            f'max_mag for Color should be in range [0,2], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, img: np.ndarray, mag: float) -> np.ndarray:
        """Apply Color transformation to image."""
        # NOTE defaultly the image should be BGR format
        img = mmcv.adjust_color(img, mag).astype(img.dtype)
        return img


@TRANSFORMS.register_module()
class Brightness(ColorTransform):
    """Adjust the brightness of the image. A magnitude=0 gives a black image,
    whereas magnitude=1 gives the original image. The bboxes, masks and
    segmentations are not modified.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Brightness transformation.
            Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for Brightness transformation.
            Defaults to 0.1.
        max_mag (float): The maximum magnitude for Brightness transformation.
            Defaults to 1.9.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.1,
                 max_mag: float = 1.9) -> None:
        assert 0. <= min_mag <= 2.0, \
            f'min_mag for Brightness should be in range [0,2], got {min_mag}.'
        assert 0. <= max_mag <= 2.0, \
            f'max_mag for Brightness should be in range [0,2], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, img: np.ndarray, mag: float) -> np.ndarray:
        """Adjust the brightness of image."""
        img = mmcv.adjust_brightness(img, mag).astype(img.dtype)
        return img


@TRANSFORMS.register_module()
class Sharpness(ColorTransform):
    """Adjust images sharpness. A positive magnitude would enhance the
    sharpness and a negative magnitude would make the image blurry. A
    magnitude=0 gives the origin img.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Sharpness transformation.
            Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for Sharpness transformation.
            Defaults to 0.1.
        max_mag (float): The maximum magnitude for Sharpness transformation.
            Defaults to 1.9.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.1,
                 max_mag: float = 1.9) -> None:
        assert 0. <= min_mag <= 2.0, \
            f'min_mag for Sharpness should be in range [0,2], got {min_mag}.'
        assert 0. <= max_mag <= 2.0, \
            f'max_mag for Sharpness should be in range [0,2], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, img: np.ndarray, mag: float) -> np.ndarray:
        """Adjust the image sharpness."""
        img = mmcv.adjust_sharpness(img, mag).astype(img.dtype)
        return img


@TRANSFORMS.register_module()
class Solarize(ColorTransform):
    """Solarize images (Invert all pixels above a threshold value of
    magnitude.).

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing Solarize transformation.
            Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for Solarize transformation.
            Defaults to 0.0.
        max_mag (float): The maximum magnitude for Solarize transformation.
            Defaults to 256.0.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 256.0) -> None:
        assert 0. <= min_mag <= 256.0, f'min_mag for Solarize should be ' \
                                       f'in range [0, 256], got {min_mag}.'
        assert 0. <= max_mag <= 256.0, f'max_mag for Solarize should be ' \
                                       f'in range [0, 256], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, img: np.ndarray, mag: float) -> np.ndarray:
        """Invert all pixel values above magnitude."""
        img = mmcv.solarize(img, mag).astype(img.dtype)
        return img


@TRANSFORMS.register_module()
class SolarizeAdd(ColorTransform):
    """SolarizeAdd images. For each pixel in the image that is less than 128,
    add an additional amount to it decided by the magnitude.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        prob (float): The probability for performing SolarizeAdd
            transformation. Defaults to 1.0.
        level (int, optional): Should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for SolarizeAdd transformation.
            Defaults to 0.0.
        max_mag (float): The maximum magnitude for SolarizeAdd transformation.
            Defaults to 110.0.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 109.5) -> None:
        assert 0. <= min_mag <= 110.0, f'min_mag for SolarizeAdd should be ' \
                                       f'in range [0, 110], got {min_mag}.'
        assert 0. <= max_mag <= 110.0, f'max_mag for SolarizeAdd should be ' \
                                       f'in range [0, 110], got {max_mag}.'
        super().__init__(
            prob=prob, level=level, min_mag=min_mag, max_mag=max_mag)

    def _transform_img(self, img: np.ndarray, mag: float) -> np.ndarray:
        """SolarizeAdd the image."""
        img_solarized = np.where(img < 128, np.minimum(img + mag, 255), img)
        img_solarized = img_solarized.astype(img.dtype)
        return img_solarized
