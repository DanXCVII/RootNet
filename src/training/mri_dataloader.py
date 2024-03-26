import sys

sys.path.append("..")

from utils import (
    RandCropByPosNegLabeldWithResAdjust,
    MyTransformDataset,
    LinearWarmupScheduler,
    CompositeScheduler,
)

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
    Orientationd,
    ScaleIntensityRanged,
    Resized,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
)

from data import CreateJsonFileConfig

"""
Description:    This script contains the dataloader for the training of MySwinUNETR etc.
                It loads the data form the data folder and performs transformations like
                cropping, flipping, shifting, etc.
Usage:  Can be used as a DataLoader for a PyTorch Lightning Trainer.
"""

class MRIDataLoader(pl.LightningDataModule):
    def __init__(self, batch_size, upscale, samples_per_volume, img_shape):
        super().__init__()

        self.batch_size = batch_size
        self.upscale = upscale
        self.samples_per_volume = samples_per_volume
        self.img_shape = img_shape

        print("creating data_config.json")
        myCreateJsonFileConfig = CreateJsonFileConfig()
        myCreateJsonFileConfig.create_config()

        self.config_file = "data_config.json"

    def setup(self, stage=None):
        with open(self.config_file, "r") as file:
            # Load the JSON data into a dictionary
            data_config = json.load(file)

            cache_transform_list = [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityd(
                    keys=["image"],
                    minv=0,
                    maxv=1,
                ),
                # ScaleIntensityRanged(
                #     keys=["image"],
                #     a_min=0,
                #     a_max=30000,
                #     b_min=0.0,
                #     b_max=1.0,
                #     clip=True,
                # ),
                # NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
                RandCropByPosNegLabeldWithResAdjust(
                    image_key="image",
                    label_key="label",
                    spatial_size=self.img_shape
                    if self.upscale
                    else tuple(x // 2 for x in self.img_shape),
                    pos=1,
                    neg=1,
                    num_samples=self.samples_per_volume,
                    image_threshold=0,
                ),
            ]
            if not self.upscale:
                print("model img_shape: ", self.img_shape)
                cache_transform_list.append(
                    Resized(
                        keys=["image"],
                        spatial_size=self.img_shape,
                        mode="trilinear",
                    )
                )
            cache_transforms = Compose(cache_transform_list)

            self.train_transforms = Compose(
                [
                    # CropForegroundd(keys=["image", "label"], source_key="image"),
                    RandFlipd(
                        keys=["image", "label"],
                        spatial_axis=[0],
                        prob=0.10,
                    ),
                    RandFlipd(
                        keys=["image", "label"],
                        spatial_axis=[1],
                        prob=0.10,
                    ),
                    RandFlipd(
                        keys=["image", "label"],
                        spatial_axis=[2],
                        prob=0.10,
                    ),
                    RandRotate90d(
                        keys=["image", "label"],
                        prob=0.10,
                        max_k=3,
                    ),
                    RandShiftIntensityd(
                        keys=["image"],
                        offsets=0.10,
                        prob=0.50,
                    ),
                ]
            )
            self.train_ds = CacheDataset(
                data=data_config["training"],
                transform=cache_transforms,
                cache_num=24,  # 24
                cache_rate=1.0,
                num_workers=8,  # 8,
            )
            self.val_ds = CacheDataset(
                data=data_config["validation"],
                transform=cache_transforms,
                cache_num=6,  # 6
                cache_rate=1.0,
                num_workers=8,  # 8,
            )
            self.test_ds = CacheDataset(
                data=data_config["test"],
                transform=cache_transforms,
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
