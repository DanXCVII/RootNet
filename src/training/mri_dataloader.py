import sys
import os

script_dir = os.path.dirname(__file__)  # Get the directory of the current script
utils_dir = os.path.join(script_dir, "..")  # Path to the utils directory
sys.path.append(os.path.abspath(utils_dir))

from utils import (
    RandCropByPosNegLabeldWithResAdjust,
    LinearWarmupScheduler,
    CompositeScheduler,
    RandCoarseDropoutd,
    RandAffined,
)

# from crop_transform import RandCropByPosNegLabeldWithResAdjust
# from schedulers import LinearWarmupScheduler, CompositeScheduler

import json
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from monai.data import (
    DataLoader,
    CacheDataset,
    list_data_collate,
)
from monai.transforms import (
    Compose,
    ScaleIntensityd,
    LoadImaged,
    EnsureChannelFirstd,
    SpatialPadd,
    RandBiasFieldd,
    ThresholdIntensityd,
    Orientationd,
    ScaleIntensityRanged,
    Resized,
    RandFlipd,
    # RandAdjustContrastd, # not suitable because it shrinks the root diameter (RandCoarseDropout, HistogramNormalize)
    RandHistogramShiftd,
    RandRotate90d,
    RandShiftIntensityd,
)

from data import CreateJsonFileConfig

import monai

"""
Description:    This script contains the dataloader for the training of MyUpscaleSwinUNETR etc.
                It loads the data form the data folder and performs transformations like
                cropping, flipping, shifting, etc.
Usage:  Can be used as a DataLoader for a PyTorch Lightning Trainer.
"""

class MRIDataLoader(pl.LightningDataModule):
    def __init__(self, relative_data_path, batch_size, upscale, samples_per_volume, patch_size, allow_missing_keys=False):
        super().__init__()

        self.batch_size = batch_size
        self.upscale = upscale
        self.samples_per_volume = samples_per_volume
        self.patch_size = patch_size

        print("creating data_config.json")
        myCreateJsonFileConfig = CreateJsonFileConfig(relative_data_path)
        myCreateJsonFileConfig.create_config()

        current_directory = os.getcwd()
        self.config_file = os.path.join(current_directory, "data_config.json")

        self.cache_transform_list = [
            LoadImaged(
                keys=["image", "label"], 
                allow_missing_keys=True
            ),
            EnsureChannelFirstd(
                keys=["image", "label"], 
                allow_missing_keys=True,
            ),
            SpatialPadd(
                keys=["image"], 
                spatial_size=[237, 237, 201], 
                method="symmetric", 
                mode="constant", 
                constant_values=0,
            ),
            SpatialPadd(keys=["label"], spatial_size=[474, 474, 402], 
                method="symmetric", 
                mode="constant", 
                constant_values=0,
            ),
            Orientationd(
                keys=["image", "label"], 
                axcodes="RAS", 
                allow_missing_keys=True,
            ),
            ScaleIntensityd(
                keys=["image"],
                minv=0,
                maxv=1,
            ),
            # NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ]
        if not self.upscale:
            self.cache_transform_list.append(
                Resized(
                    keys=["image"],
                    spatial_size=self.patch_size,
                    mode="trilinear",
                )
            )
        self.cache_transforms = Compose(self.cache_transform_list)

    def setup(self, stage=None):
        with open(self.config_file, "r") as file:
            # Load the JSON data into a dictionary
            data_config = json.load(file)                

            train_transform_list = []
            train_transform_list.extend(self.cache_transform_list)
            train_transform_list.insert(
                1,
                RandAffined(
                    keys=["image", "label"],
                    prob=0.1,
                    rotate_range=((-0.785, 0.785), (-0.785, 0.785), (-0.785, 0.785)),
                    scale_range=((-0.2, 0.5), (-0.2, 0.5), (-0.2, 0.5)),
                    mode=("bilinear", "nearest"),
                ),
            )
            train_transform_list.extend(
                [
                    RandBiasFieldd(
                        keys=["image"],
                        coeff_range=(0.0, 0.1),
                        prob=0.1,
                    ),
                    RandCoarseDropoutd(
                        keys=["image", "label"],
                        holes=6,
                        fill_value=0,
                        spatial_size=(15, 15, 20),
                        prob=0.1,
                    ),
                    RandCropByPosNegLabeldWithResAdjust(
                        image_key="image",
                        label_key="label",
                        spatial_size=self.patch_size
                        if self.upscale
                        else tuple(x // 2 for x in self.patch_size),
                        pos=1,
                        neg=1,
                        num_samples=self.samples_per_volume,
                        image_threshold=0,
                    ),
                    RandHistogramShiftd(
                        keys=["image"],
                        num_control_points=10,
                        prob=0.1,
                    ),
                    ScaleIntensityd(
                        keys=["image"],
                        minv=0,
                        maxv=1,
                    ),
                    # CropForegroundd(keys=["image", "label"], source_key="image"),
                    RandFlipd(
                        keys=["image", "label"],
                        spatial_axis=[0],
                        prob=0.5,
                    ),
                    RandFlipd(
                        keys=["image", "label"],
                        spatial_axis=[1],
                        prob=0.5,
                    ),
                    RandFlipd(
                        keys=["image", "label"],
                        spatial_axis=[2], # z-axis
                        prob=0.10,
                    ),
                    RandRotate90d(
                        keys=["image", "label"],
                        prob=0.50,
                        max_k=3,
                    ),
                    RandShiftIntensityd(
                        keys=["image"],
                        offsets=0.10,
                        prob=0.50,
                    ),
                ],
            )
            train_transforms = Compose(train_transform_list)
            self.train_ds = CacheDataset(
                data=data_config["training"],
                transform=train_transforms,
                cache_num=24,  # 24
                cache_rate=1.0,
                num_workers=8,  # 8,
            )
            self.val_ds = CacheDataset(
                data=data_config["validation"],
                transform=self.cache_transforms,
                cache_num=6,  # 6
                cache_rate=1.0,
                num_workers=8,  # 8,
            )
            self.test_ds = CacheDataset(
                data=data_config["test"],
                transform=self.cache_transforms,
                cache_num=6,  # 6
                cache_rate=1.0,
                num_workers=8,  # 8,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,  # 8
            pin_memory=True,
            collate_fn=list_data_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,  # 4
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,  # 4
            pin_memory=True,
        )
