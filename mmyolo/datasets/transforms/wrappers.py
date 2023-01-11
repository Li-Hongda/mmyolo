# Copyright (c) OpenMMLab. All rights reserved.

from typing import Callable, Dict, List, Optional, Sequence, Union

import mmengine
import numpy as np
# from mmcv.transforms import Compose, RandomChoice
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine.config import ConfigDict

from mmyolo.registry import TRANSFORMS

# Define type of transform or transform config
Transform = Union[Dict, Callable[[Dict], Dict]]

# Import nullcontext if python>=3.7, otherwise use a simple alternative
# implementation.
try:
    from contextlib import nullcontext  # type: ignore
except ImportError:
    from contextlib import contextmanager

    @contextmanager  # type: ignore
    def nullcontext(resource=None):
        try:
            yield resource
        finally:
            pass


_MAX_LEVEL = 10

SCALE_AWARE_AUTOAUG_POLICIES = [
    [('Color', 0.4, 2), ('BoxLevelTranslateX', 0.4, 4)],
    [('Brightness', 0.2, 4), ('BoxLevelRotate', 0.4, 2)],
    [('Sharpness', 0.4, 2), ('BoxLevelShearX', 0.2, 6)],
    [('SolarizeAdd', 0.2, 2), ('BoxLevelHorizontalFlip', 0.5, 1)],
    [('Color', 0.0, 8), ('BoxLevelTranslateY', 0.2, 8)],
]


class Compose(BaseTransform):
    """Compose multiple transforms sequentially.

    Args:
        transforms (list[dict | callable]): Sequence of transform object or
            config dict to be composed.

    Examples:
        >>> pipeline = [
        >>>     dict(type='Compose',
        >>>         transforms=[
        >>>             dict(type='LoadImageFromFile'),
        >>>             dict(type='Normalize')
        >>>         ]
        >>>     )
        >>> ]
    """

    def __init__(self, transforms: Union[Transform, Sequence[Transform]]):
        super().__init__()

        if not isinstance(transforms, Sequence):
            transforms = [transforms]
        self.transforms: List = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

    def __iter__(self):
        """Allow easy iteration over the transform sequence."""
        return iter(self.transforms)

    def transform(self, results: Dict) -> Optional[Dict]:
        """Call function to apply transforms sequentially.

        Args:
            results (dict): A result dict contains the results to transform.

        Returns:
            dict or None: Transformed results.
        """
        for t in self.transforms:
            results = t(results)  # type: ignore
            if results is None:
                return None
        return results

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string


@TRANSFORMS.register_module()
class RandomChoice(BaseTransform):
    """Process data with a randomly chosen transform from given candidates.

    Args:
        transforms (list[list]): A list of transform candidates, each is a
            sequence of transforms.
        prob (list[float], optional): The probabilities associated
            with each pipeline. The length should be equal to the pipeline
            number and the sum should be 1. If not given, a uniform
            distribution will be assumed.

    Examples:
        >>> # config
        >>> pipeline = [
        >>>     dict(type='RandomChoice',
        >>>         transforms=[
        >>>             [dict(type='RandomHorizontalFlip')],  # subpipeline 1
        >>>             [dict(type='RandomRotate')],  # subpipeline 2
        >>>         ]
        >>>     )
        >>> ]
    """

    def __init__(self,
                 transforms: List[Union[Transform, List[Transform]]],
                 prob: Optional[List[float]] = None):

        super().__init__()

        if prob is not None:
            assert mmengine.is_seq_of(prob, float)
            assert len(transforms) == len(prob), \
                '``transforms`` and ``prob`` must have same lengths. ' \
                f'Got {len(transforms)} vs {len(prob)}.'
            assert sum(prob) == 1

        self.prob = prob
        self.transforms = [Compose(transforms) for transforms in transforms]

    def __iter__(self):
        return iter(self.transforms)

    @cache_randomness
    def random_pipeline_index(self) -> int:
        """Return a random transform index."""
        indices = np.arange(len(self.transforms))
        return np.random.choice(indices, p=self.prob)

    def transform(self, results: Dict) -> Optional[Dict]:
        """Randomly choose a transform to apply."""
        idx = self.random_pipeline_index()
        return self.transforms[idx](results)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(transforms = {self.transforms}'
        repr_str += f'prob = {self.prob})'
        return repr_str


def policies():
    """Autoaugment policies that was used in AutoAugment Paper."""
    policies = list()
    for policy_args in SCALE_AWARE_AUTOAUG_POLICIES:
        policy = list()
        for args in policy_args:
            policy.append(dict(type=args[0], prob=args[1], level=args[2]))
        policies.append(policy)
    return policies


@TRANSFORMS.register_module()
class ScaleAwareAutoAugmentation(RandomChoice):

    def __init__(self,
                 policies: List[List[Union[dict, ConfigDict]]] = policies(),
                 prob: Optional[List[float]] = None) -> None:
        assert isinstance(policies, list) and len(policies) > 0, \
            'Policies must be a non-empty list.'
        super().__init__(transforms=policies, prob=prob)
        self.policies = policies
        self.count = 0

    def transform(self, results: Dict) -> Optional[Dict]:
        """Randomly choose a transform to apply."""
        idx = self.random_pipeline_index()
        iteration = self.count // self.batch_size * self.num_workers
        results['iter_ratio'] = iteration / self.max_iters
        self.count += 1
        return self.transforms[idx](results)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(policies={self.policies}, ' \
               f'prob={self.prob})'
