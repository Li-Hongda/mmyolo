# Copyright (c) OpenMMLab. All rights reserved.

import copy
from typing import List, Optional, Union

import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmdet.datasets.transforms.augment_wrappers import _MAX_LEVEL, level_to_mag
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type

from mmyolo.registry import TRANSFORMS
from .gaussian_maps import _gaussian_map


@TRANSFORMS.register_module()
class BoxLevelGeomTransform(BaseTransform):
    """Base class for box-level geometric transformations. All box-level
    geometric transformations need to inherit from this base class.
    ``BoxLevelGeomTransform`` unifies the class attributes and class functions
    of geometric transformations (ShearX, ShearY, Rotate, TranslateX, and
    TranslateY)

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)

    Modified Keys:

    - img
    - gt_bboxes

    Added Keys:

    - homography_matrix

    Args:
        prob (float): The probability for performing the geometric
            transformation and should be in range [0, 1]. Defaults to 1.0.
        level (int, optional): The level should be in range [0, _MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum magnitude for geometric transformation.
            Defaults to 0.0.
        max_mag (float): The maximum magnitude for geometric transformation.
            Defaults to 1.0.
        reversal_prob (float): The probability that reverses the geometric
            transformation magnitude. Should be in range [0,1].
            Defaults to 0.5.
        img_border_value (int | float | tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 128.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 prob: float = 1.0,
                 bbox_prob: float = 0.3,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 1.0,
                 reversal_prob: float = 0.5,
                 img_border_value: Union[int, float, tuple] = 128,
                 interpolation: str = 'bilinear',
                 scale_splits: List[int] = [2048, 10240, 51200],
                 scale_ratio: List[int] = [3, 3, 3]) -> None:
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
        assert isinstance(reversal_prob, float), \
            f'reversal_prob should be type float, got {type(max_mag)}.'
        assert 0 <= reversal_prob <= 1.0, \
            f'The reversal probability of the transformation magnitude ' \
            f'should be type float, got {type(reversal_prob)}.'
        if isinstance(img_border_value, (float, int)):
            img_border_value = tuple([float(img_border_value)] * 3)
        elif isinstance(img_border_value, tuple):
            assert len(img_border_value) == 3, \
                f'img_border_value as tuple must have 3 elements, ' \
                f'got {len(img_border_value)}.'
            img_border_value = tuple([float(val) for val in img_border_value])
        else:
            raise ValueError(
                'img_border_value must be float or tuple with 3 elements.')
        assert np.all([0 <= val <= 255 for val in img_border_value]), 'all ' \
            'elements of img_border_value should between range [0,255].' \
            f'got {img_border_value}.'
        self.prob = prob
        self.bbox_prob = bbox_prob
        self.level = level
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.reversal_prob = reversal_prob
        self.img_border_value = img_border_value
        self.interpolation = interpolation
        self.scale_splits = scale_splits
        self.scale_ratio = scale_ratio

    # @cache_randomness
    def _random_disable(self, iter_ratio: float):
        """Randomly disable the transform."""
        return np.random.rand() > (self.prob * iter_ratio)

    def _transform_img(self, img: np.ndarray, mag: float) -> np.ndarray:
        """Transform the image."""
        return img

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

    def _transform_bboxes(self, bbox: HorizontalBoxes, results: dict,
                          mag: float) -> None:
        """Transform the bboxes."""
        # results['gt_bboxes'].project_(self.homography_matrix)
        # results['gt_bboxes'].clip_(results['img_shape'])
        # return results['gt_bboxes']
        self.add_gt = False

    @cache_randomness
    def _get_mag(self):
        """Get the magnitude of the transform."""
        mag = level_to_mag(self.level, self.min_mag, self.max_mag)
        return -mag if np.random.rand() > self.reversal_prob else mag

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function for images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Transformed results.
        """
        if self._random_disable(results['iter_ratio']):
            return results
        results['cropped_img'] = []
        results['cropped_gt_bboxes'] = []
        results['cropped_gt_labels'] = []
        results['cropped_gt_ignore_flags'] = []
        mag = self._get_mag()
        for idx, bbox in enumerate(results['gt_bboxes']):
            prob = self._box_sample_prob(bbox)
            if np.random.rand() >= prob:
                continue
            x1, y1, x2, y2 = bbox.tensor.long().squeeze()
            cropped_img = results['img'][y1:y2, x1:x2, :]
            if cropped_img.shape[0] * cropped_img.shape[1] <= 0:
                continue
            cropped_img = self._transform_img(cropped_img, mag)
            results['cropped_img'].append(cropped_img)
            self._transform_bboxes(bbox, results, mag)
            results['cropped_gt_bboxes'].append(bbox.tensor)
            results['cropped_gt_labels'].append(
                results['gt_bboxes_labels'][idx])
            results['cropped_gt_ignore_flags'].append(
                results['gt_ignore_flags'][idx])
        if len(results.get('cropped_gt_bboxes')) > 0:
            results['img'] = self._gaussian_transform(results)
            if self.add_gt:
                results['gt_bboxes'].tensor = torch.cat(
                    (results['gt_bboxes'].tensor,
                     torch.cat(results['cropped_gt_bboxes'])))
                results['gt_bboxes_labels'] = np.concatenate(
                    (results['gt_bboxes_labels'],
                     np.stack(results['cropped_gt_labels'])))
                results['gt_ignore_flags'] = np.concatenate(
                    (results['gt_ignore_flags'],
                     np.stack(results['cropped_gt_ignore_flags'])))
        return results

    def _gaussian_transform(self, results: dict) -> np.ndarray:
        cp_img = copy.deepcopy(results['img'])
        for i, box in enumerate(results['cropped_gt_bboxes']):
            x1, y1, x2, y2 = results['cropped_gt_bboxes'][i].long().squeeze()

            y_crop = copy.deepcopy(cp_img[y1:y2, x1:x2, :])
            x_crop = results['cropped_img'][i][:y_crop.shape[0], :y_crop.
                                               shape[1], :]

            if y_crop.shape[1] * y_crop.shape[0] == 0:
                continue

            g_maps = _gaussian_map(x_crop,
                                   [[0, 0, y_crop.shape[1], y_crop.shape[0]]])
            h, w = x_crop.shape[:2]
            cp_img[y1:y1 + h, x1:x1 +
                   w, :] = g_maps * x_crop + (1 - g_maps) * y_crop[:h, :w, :]
        return cp_img

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'level={self.level}, '
        repr_str += f'min_mag={self.min_mag}, '
        repr_str += f'max_mag={self.max_mag}, '
        repr_str += f'reversal_prob={self.reversal_prob}, '
        repr_str += f'img_border_value={self.img_border_value}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class BoxLevelShearX(BoxLevelGeomTransform):
    """Shear the images, bboxes horizontally.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)

    Modified Keys:

    - img
    - gt_bboxes

    Added Keys:

    - homography_matrix

    Args:
        prob (float): The probability for performing Shear and should be in
            range [0, 1]. Defaults to 1.0.
        level (int, optional): The level should be in range [0, _MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum angle for the horizontal shear.
            Defaults to 0.0.
        max_mag (float): The maximum angle for the horizontal shear.
            Defaults to 30.0.
        reversal_prob (float): The probability that reverses the horizontal
            shear magnitude. Should be in range [0,1]. Defaults to 0.5.
        img_border_value (int | float | tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 128.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 15.0,
                 reversal_prob: float = 0.5,
                 img_border_value: Union[int, float, tuple] = 128,
                 interpolation: str = 'bilinear') -> None:
        assert 0. <= min_mag <= 90., \
            f'min_mag angle for ShearX should be ' \
            f'in range [0, 90], got {min_mag}.'
        assert 0. <= max_mag <= 90., \
            f'max_mag angle for ShearX should be ' \
            f'in range [0, 90], got {max_mag}.'
        super().__init__(
            prob=prob,
            level=level,
            min_mag=min_mag,
            max_mag=max_mag,
            reversal_prob=reversal_prob,
            img_border_value=img_border_value,
            interpolation=interpolation)

    @cache_randomness
    def _get_mag(self):
        """Get the magnitude of the transform."""
        mag = level_to_mag(self.level, self.min_mag, self.max_mag)
        mag = np.tan(mag * np.pi / 180)
        return -mag if np.random.rand() > self.reversal_prob else mag

    def _transform_img(self, img: np.ndarray, mag: float) -> np.ndarray:
        """Shear the image horizontally."""
        img = mmcv.imshear(
            img,
            mag,
            direction='horizontal',
            border_value=self.img_border_value,
            interpolation=self.interpolation)
        return img


@TRANSFORMS.register_module()
class BoxLevelShearY(BoxLevelGeomTransform):
    """Shear the images, bboxes vertically.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)

    Modified Keys:

    - img
    - gt_bboxes

    Added Keys:

    - homography_matrix

    Args:
        prob (float): The probability for performing ShearY and should be in
            range [0, 1]. Defaults to 1.0.
        level (int, optional): The level should be in range [0,_MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum angle for the vertical shear.
            Defaults to 0.0.
        max_mag (float): The maximum angle for the vertical shear.
            Defaults to 30.0.
        reversal_prob (float): The probability that reverses the vertical
            shear magnitude. Should be in range [0,1]. Defaults to 0.5.
        img_border_value (int | float | tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 128.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 15.0,
                 reversal_prob: float = 0.5,
                 img_border_value: Union[int, float, tuple] = 128,
                 interpolation: str = 'bilinear') -> None:
        assert 0. <= min_mag <= 90., \
            f'min_mag angle for ShearY should be ' \
            f'in range [0, 90], got {min_mag}.'
        assert 0. <= max_mag <= 90., \
            f'max_mag angle for ShearY should be ' \
            f'in range [0, 90], got {max_mag}.'
        super().__init__(
            prob=prob,
            level=level,
            min_mag=min_mag,
            max_mag=max_mag,
            reversal_prob=reversal_prob,
            img_border_value=img_border_value,
            interpolation=interpolation)

    @cache_randomness
    def _get_mag(self):
        """Get the magnitude of the transform."""
        mag = level_to_mag(self.level, self.min_mag, self.max_mag)
        mag = np.tan(mag * np.pi / 180)
        return -mag if np.random.rand() > self.reversal_prob else mag

    def _transform_img(self, img: np.ndarray, mag: float) -> np.ndarray:
        """Shear the image vertically."""
        img = mmcv.imshear(
            img,
            mag,
            direction='vertical',
            border_value=self.img_border_value,
            interpolation=self.interpolation)
        return img


@TRANSFORMS.register_module()
class BoxLevelRotate(BoxLevelGeomTransform):
    """Rotate the images, bboxes.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)

    Modified Keys:

    - img
    - gt_bboxes

    Added Keys:

    - homography_matrix

    Args:
        prob (float): The probability for perform transformation and
            should be in range 0 to 1. Defaults to 1.0.
        level (int, optional): The level should be in range [0, _MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The maximum angle for rotation.
            Defaults to 0.0.
        max_mag (float): The maximum angle for rotation.
            Defaults to 30.0.
        reversal_prob (float): The probability that reverses the rotation
            magnitude. Should be in range [0,1]. Defaults to 0.5.
        img_border_value (int | float | tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 128.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 30.0,
                 reversal_prob: float = 0.5,
                 img_border_value: Union[int, float, tuple] = 128,
                 interpolation: str = 'bilinear') -> None:
        assert 0. <= min_mag <= 180., \
            f'min_mag for Rotate should be in range [0,180], got {min_mag}.'
        assert 0. <= max_mag <= 180., \
            f'max_mag for Rotate should be in range [0,180], got {max_mag}.'
        super().__init__(
            prob=prob,
            level=level,
            min_mag=min_mag,
            max_mag=max_mag,
            reversal_prob=reversal_prob,
            img_border_value=img_border_value,
            interpolation=interpolation)

    def _transform_img(self, img: np.ndarray, mag: float) -> np.ndarray:
        """Rotate the image."""
        img = mmcv.imrotate(
            img,
            mag,
            border_value=self.img_border_value,
            interpolation=self.interpolation)
        return img


@TRANSFORMS.register_module()
class BoxLevelTranslateX(BoxLevelGeomTransform):
    """Translate the images, bboxes horizontally.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)

    Modified Keys:

    - img
    - gt_bboxes

    Added Keys:

    - homography_matrix

    Args:
        prob (float): The probability for perform transformation and
            should be in range 0 to 1. Defaults to 1.0.
        level (int, optional): The level should be in range [0, _MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum pixel's offset ratio for horizontal
            translation. Defaults to 0.0.
        max_mag (float): The maximum pixel's offset ratio for horizontal
            translation. Defaults to 0.1.
        reversal_prob (float): The probability that reverses the horizontal
            translation magnitude. Should be in range [0,1]. Defaults to 0.5.
        img_border_value (int | float | tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 128.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 120.0,
                 reversal_prob: float = 0.5,
                 img_border_value: Union[int, float, tuple] = 128,
                 interpolation: str = 'bilinear') -> None:
        super().__init__(
            prob=prob,
            level=level,
            min_mag=min_mag,
            max_mag=max_mag,
            reversal_prob=reversal_prob,
            img_border_value=img_border_value,
            interpolation=interpolation)

    def _transform_bboxes(self, bbox: HorizontalBoxes, results: dict,
                          mag: float) -> torch.tensor:
        self.add_gt = True
        offset = (int(mag), 0)
        bbox.translate_(offset)
        bbox.clip_(results['img_shape'])


@TRANSFORMS.register_module()
class BoxLevelTranslateY(BoxLevelGeomTransform):
    """Translate the images, bboxes vertically.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)

    Modified Keys:

    - img
    - gt_bboxes

    Added Keys:

    - homography_matrix

    Args:
        prob (float): The probability for perform transformation and
            should be in range 0 to 1. Defaults to 1.0.
        level (int, optional): The level should be in range [0, _MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        min_mag (float): The minimum pixel's offset ratio for vertical
            translation. Defaults to 0.0.
        max_mag (float): The maximum pixel's offset ratio for vertical
            translation. Defaults to 0.1.
        reversal_prob (float): The probability that reverses the vertical
            translation magnitude. Should be in range [0,1]. Defaults to 0.5.
        img_border_value (int | float | tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 128.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 min_mag: float = 0.0,
                 max_mag: float = 120.0,
                 reversal_prob: float = 0.5,
                 img_border_value: Union[int, float, tuple] = 128,
                 interpolation: str = 'bilinear') -> None:
        super().__init__(
            prob=prob,
            level=level,
            min_mag=min_mag,
            max_mag=max_mag,
            reversal_prob=reversal_prob,
            img_border_value=img_border_value,
            interpolation=interpolation)

    def _transform_bboxes(self, bbox: HorizontalBoxes, results: dict,
                          mag: float) -> torch.tensor:
        self.add_gt = True
        offset = (0, int(mag))
        bbox.translate_(offset)
        bbox.clip_(results['img_shape'])


@TRANSFORMS.register_module()
class BoxLevelHorizontalFlip(BoxLevelGeomTransform):
    """Flip the images, bboxes horizontally.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)

    Modified Keys:

    - img
    - gt_bboxes

    Added Keys:

    - homography_matrix

    Args:
        prob (float): The probability for performing flip and should be in
            range [0, 1]. Defaults to 1.0.
        level (int, optional): The level should be in range [0, _MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        reversal_prob (float): The probability that reverses the horizontal
            flip magnitude. Should be in range [0,1]. Defaults to 0.5.
        img_border_value (int | float | tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 128.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 reversal_prob: float = 0.5,
                 img_border_value: Union[int, float, tuple] = 128,
                 interpolation: str = 'bilinear') -> None:
        super().__init__(
            prob=prob,
            level=level,
            reversal_prob=reversal_prob,
            img_border_value=img_border_value,
            interpolation=interpolation)

    def _transform_img(self, img: np.ndarray, mag: float) -> np.ndarray:
        """Flip the image horizontally."""
        img = mmcv.imflip(img, direction='horizontal')
        return img


@TRANSFORMS.register_module()
class BoxLevelVerticalFlip(BoxLevelGeomTransform):
    """Flip the images, bboxes vertically.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)

    Modified Keys:

    - img
    - gt_bboxes

    Added Keys:

    - homography_matrix

    Args:
        prob (float): The probability for performing flip and should be in
            range [0, 1]. Defaults to 1.0.
        level (int, optional): The level should be in range [0, _MAX_LEVEL].
            If level is None, it will generate from [0, _MAX_LEVEL] randomly.
            Defaults to None.
        reversal_prob (float): The probability that reverses the vertical
            flip magnitude. Should be in range [0,1]. Defaults to 0.5.
        img_border_value (int | float | tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 128.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 prob: float = 1.0,
                 level: Optional[int] = None,
                 reversal_prob: float = 0.5,
                 img_border_value: Union[int, float, tuple] = 128,
                 interpolation: str = 'bilinear') -> None:
        super().__init__(
            prob=prob,
            level=level,
            reversal_prob=reversal_prob,
            img_border_value=img_border_value,
            interpolation=interpolation)

    def _transform_img(self, img: np.ndarray, mag: float) -> np.ndarray:
        """Flip the image vertically."""
        img = mmcv.imflip(img, direction='vertical')
        return img
