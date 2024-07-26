from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any
from warnings import warn

import numpy as np
import torch

from monai.transforms import RandCoarseTransform
from monai.config.type_definitions import NdarrayOrTensor
from monai.config import DtypeLike, KeysCollection, SequenceStr
from collections.abc import Hashable, Mapping, Sequence
from monai.data.meta_obj import get_track_meta
from monai.transforms.transform import RandomizableTransform, MapTransform
from monai.utils.module import min_version, optional_import
from monai.utils.type_conversion import (
    convert_to_tensor,
)

skimage, _ = optional_import("skimage", "0.19.0", min_version)


class RandCoarseDropout(RandCoarseTransform):
    """
    Randomly coarse dropout regions in the image, then fill in the rectangular regions with specified value.
    Or keep the rectangular regions and fill in the other areas with specified value.
    Refer to papers: https://arxiv.org/abs/1708.04552, https://arxiv.org/pdf/1604.07379
    And other implementation: https://albumentations.ai/docs/api_reference/augmentations/transforms/
    #albumentations.augmentations.transforms.CoarseDropout.

    Args:
        holes: number of regions to dropout, if `max_holes` is not None, use this arg as the minimum number to
            randomly select the expected number of regions.
        spatial_size: spatial size of the regions to dropout, if `max_spatial_size` is not None, use this arg
            as the minimum spatial size to randomly select size for every region.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of input img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        dropout_holes: if `True`, dropout the regions of holes and fill value, if `False`, keep the holes and
            dropout the outside and fill value. default to `True`.
        fill_value: target value to fill the dropout regions, if providing a number, will use it as constant
            value to fill all the regions. if providing a tuple for the `min` and `max`, will randomly select
            value for every pixel / voxel from the range `[min, max)`. if None, will compute the `min` and `max`
            value of input image then randomly select value to fill, default to None.
        max_holes: if not None, define the maximum number to randomly select the expected number of regions.
        max_spatial_size: if not None, define the maximum spatial size to randomly select size for every region.
            if some components of the `max_spatial_size` are non-positive values, the transform will use the
            corresponding components of input img size. For example, `max_spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        prob: probability of applying the transform.

    """

    def __init__(
        self,
        holes: int,
        spatial_size: Sequence[int] | int,
        dropout_holes: bool = True,
        fill_value: tuple[float, float] | float | None = None,
        max_holes: int | None = None,
        max_spatial_size: Sequence[int] | int | None = None,
        prob: float = 0.1,
    ) -> None:
        super().__init__(
            holes=holes,
            spatial_size=spatial_size,
            max_holes=max_holes,
            max_spatial_size=max_spatial_size,
            prob=prob,
        )
        self.dropout_holes = dropout_holes
        if isinstance(fill_value, (tuple, list)):
            if len(fill_value) != 2:
                raise ValueError(
                    "fill value should contain 2 numbers if providing the `min` and `max`."
                )
        self.fill_value = fill_value

    def _transform_holes(self, img: np.ndarray):
        """
        Fill the randomly selected `self.hole_coords` in input images.
        Please note that we usually only use `self.R` in `randomize()` method, here is a special case.

        """
        fill_value = (
            (img.min(), img.max()) if self.fill_value is None else self.fill_value
        )

        if img.shape[1] > 300:
            print("Calculating the hole coords for the label data ooooopppppp")
            # multiply hole coords by 2
            self.new_hole_coords = [
                (
                    slice(None, None, None),
                    slice(
                        s.start * 2 + 1 if s.start + 1 is not None else None,
                        s.stop * 2 if s.stop is not None else None,
                        s.step,
                    ),
                    slice(
                        t.start * 2 + 1 if t.start is not None else None,
                        t.stop * 2 if t.stop is not None else None,
                        t.step,
                    ),
                    slice(
                        u.start * 2 + 1 if u.start is not None else None,
                        u.stop * 2 if u.stop is not None else None,
                        u.step,
                    ),
                )
                for _, s, t, u in self.hole_coords
            ]
            self.hole_coords = self.new_hole_coords

        if self.dropout_holes:
            for h in self.hole_coords:
                if isinstance(fill_value, (tuple, list)):
                    img[h] = self.R.uniform(
                        fill_value[0], fill_value[1], size=img[h].shape
                    )
                else:
                    img[h] = fill_value
            ret = img
        else:
            if isinstance(fill_value, (tuple, list)):
                ret = self.R.uniform(
                    fill_value[0], fill_value[1], size=img.shape
                ).astype(img.dtype, copy=False)
            else:
                ret = np.full_like(img, fill_value)
            for h in self.hole_coords:
                ret[h] = img[h]
        return ret


class RandCoarseDropoutd(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RandCoarseDropout`.
    Expect all the data specified by `keys` have same spatial shape and will randomly dropout the same regions
    for every key, if want to dropout differently for every key, please use this transform separately.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        holes: number of regions to dropout, if `max_holes` is not None, use this arg as the minimum number to
            randomly select the expected number of regions.
        spatial_size: spatial size of the regions to dropout, if `max_spatial_size` is not None, use this arg
            as the minimum spatial size to randomly select size for every region.
            if some components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of input img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        dropout_holes: if `True`, dropout the regions of holes and fill value, if `False`, keep the holes and
            dropout the outside and fill value. default to `True`.
        fill_value: target value to fill the dropout regions, if providing a number, will use it as constant
            value to fill all the regions. if providing a tuple for the `min` and `max`, will randomly select
            value for every pixel / voxel from the range `[min, max)`. if None, will compute the `min` and `max`
            value of input image then randomly select value to fill, default to None.
        max_holes: if not None, define the maximum number to randomly select the expected number of regions.
        max_spatial_size: if not None, define the maximum spatial size to randomly select size for every region.
            if some components of the `max_spatial_size` are non-positive values, the transform will use the
            corresponding components of input img size. For example, `max_spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        prob: probability of applying the transform.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = RandCoarseDropout.backend

    def __init__(
        self,
        keys: KeysCollection,
        holes: int,
        spatial_size: Sequence[int] | int,
        dropout_holes: bool = True,
        fill_value: tuple[float, float] | float | None = None,
        max_holes: int | None = None,
        max_spatial_size: Sequence[int] | int | None = None,
        prob: float = 0.1,
        allow_missing_keys: bool = False,
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob=prob)
        self.dropper = RandCoarseDropout(
            holes=holes,
            spatial_size=spatial_size,
            dropout_holes=dropout_holes,
            fill_value=fill_value,
            max_holes=max_holes,
            max_spatial_size=max_spatial_size,
            prob=1.0,
        )

    def set_random_state(
        self, seed: int | None = None, state: np.random.RandomState | None = None
    ) -> RandCoarseDropoutd:
        super().set_random_state(seed, state)
        self.dropper.set_random_state(seed, state)
        return self

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        # expect all the specified keys have same spatial shape and share same random holes
        first_key: Hashable = self.first_key(d)
        if first_key == ():
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        self.dropper.randomize(d[first_key].shape[1:])
        for key in self.key_iterator(d):
            d[key] = self.dropper(img=d[key], randomize=False)

        return d
