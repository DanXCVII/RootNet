# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Union

import sys
import os

sys.path.append("..")
# sys.path.append("../models")


from data import CreateJsonFileConfig
from utils import RandCropByPosNegLabeldWithResAdjust
from models import MyUNETR

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from monai.utils import set_determinism
from monai.metrics import DiceMetric
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
)


from monai.data import (
    DataLoader,
    CacheDataset,
    list_data_collate,
)

import torch
import numpy as np
import nibabel as nib
import random

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import json

torch.set_float32_matmul_precision("medium")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_determinism(seed=seed)

seed = 42
seed_everything(seed)


class MyUNETRWrapper(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_shape: Tuple[int, int, int] = (128, 128, 128),
        feature_size: int = 16,
        batch_size: int = 4,
        max_epochs: int = 50,
        check_val: int = 5,
    ) -> None:
        """
        Pytorch Lightning Module which sets up the training of the UNETR (with superresolution) for the data provided in RootNet/data

        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_shape: dimension of input image.
            feature_size: dimension of network feature size.
        """

        super().__init__()

        self._model = MyUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_shape=img_shape,
            feature_size=feature_size,
        )

        # Loss function and metrics
        self.loss_function = DiceCELoss(softmax=False, include_background=True) #, weight=torch.tensor([1.0, 2.0]))
        self.dice_metric = DiceMetric(
            include_background=True,
            reduction="mean",
            get_not_nans=False,
        )  # TODO: include_background default is False

        # Training parameters
        self.max_epochs = max_epochs
        self.check_val = check_val
        self.batch_size = batch_size

        # Variables to store training and validation metrics
        self.metric_values = []
        self.epoch_loss_values = []
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []
        self.best_val_dice = 0
        self.best_val_epoch = 0

    def forward(self, x_in):
        output = self._model(x_in)

        return output

    def prepare_data(self):
        print("creating data_config.json")
        myCreateJsonFileConfig = CreateJsonFileConfig()
        myCreateJsonFileConfig.create_config()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            with open("data_config.json", "r") as file:
                # Load the JSON data into a dictionary
                data_config = json.load(file)

                print("batch_size transform", self.batch_size)
                train_transforms = Compose(
                    [
                        LoadImaged(keys=["image", "label"]),
                        EnsureChannelFirstd(keys=["image", "label"]),
                        Orientationd(keys=["image", "label"], axcodes="RAS"),
                        # Spacingd(
                        #     keys=["image", "label"], # different transform for image and label
                        #     pixdim=(0.27, 0.27, 0.27), # maybe make z twice as big because the slices are always much bigger
                        #     mode=("bilinear", "nearest"),
                        # ),
                        ScaleIntensityRanged(
                            keys=["image"],
                            a_min=0,
                            a_max=30000,
                            b_min=0.0,
                            b_max=1.0,
                            clip=True,
                        ),
                        # CropForegroundd(keys=["image", "label"], source_key="image"),
                        RandCropByPosNegLabeldWithResAdjust(
                            image_key="image",
                            label_key="label",
                            spatial_size=(128, 128, 128),
                            pos=1,
                            neg=1,
                            num_samples=4,  # TODO: set parameter
                            image_threshold=0,
                        ),
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
                val_transforms = Compose(
                    [
                        LoadImaged(keys=["image", "label"]),
                        EnsureChannelFirstd(keys=["image", "label"]),
                        ScaleIntensityRanged(
                            keys=["image"],
                            a_min=0,
                            a_max=30000,
                            b_min=0.0,
                            b_max=1.0,
                            clip=True,
                        ),
                    ]
                )
                test_transforms = Compose(
                    [
                        LoadImaged(keys=["image", "label"]),
                        EnsureChannelFirstd(keys=["image", "label"]),
                        ScaleIntensityRanged(
                            keys=["image"],
                            a_min=0,
                            a_max=30000,
                            b_min=0.0,
                            b_max=1.0,
                            clip=True,
                        ),
                    ]
                )
                self.train_ds = CacheDataset(
                    data=data_config["training"],
                    transform=train_transforms,
                    cache_num=24,  # 24
                    cache_rate=1.0,
                    num_workers=8,  # 8,
                )
                self.val_ds = CacheDataset(
                    data=data_config["validation"],
                    transform=val_transforms,
                    cache_num=6,  # 6
                    cache_rate=1.0,
                    num_workers=8,  # 8,
                )
                self.test_ds = CacheDataset(
                    data=data_config["test"],
                    transform=test_transforms,
                    cache_num=6,  # 6
                    cache_rate=1.0,
                    num_workers=8,  # 8,
                )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,  # 8
            pin_memory=True,
            collate_fn=list_data_collate,
        )

        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,  # 4
            pin_memory=True,
        )
        
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,  # 4
            pin_memory=True,
        )

        return test_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        output = self.forward(images)

        loss = self.loss_function(output, labels)

        loss_dict = {"loss": loss}

        self.training_step_outputs.append(loss_dict)

        return loss_dict
    
    def on_train_epoch_end(self, unused=None, outputs=None): # on_train_epoch_end in newest pytorch version

        if self.training_step_outputs:
            avg_loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()

            # Only perform the following operations on the main process
            if self.trainer.is_global_zero:
                # Store or log the average loss
                self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())
                self.log("Train/avg_epoch_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False) #, sync_dist=True)

                # Clear training step outputs
                self.training_step_outputs.clear()

    def _save_as_nifti(self, mri_grid, filename):
        """
        saves the mri grid as a nifti file

        Args:
        - mri_grid: 3d (numpy) array of the root container with dimensions (nx, ny, nz)
        - filename: path to the nifti file
        """
        # since all the mris have swapped z and x coordinate, the dimensions have to be swapped as well
        affine_transformation = np.array(
            [
                [1, 0.0, 0.0, 0.0],
                [0.0, 0.27, 0.0, 0.0],
                [0.0, 0.0, 0.27, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # create a nifti file
        img = nib.Nifti1Image(mri_grid, affine_transformation)
        # save the nifti file
        nib.save(img, filename)

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (128, 128, 128)
        sw_batch_size = 4

        outputs = sliding_window_inference(
            images,
            roi_size,
            sw_batch_size,
            self.forward,
        )

        loss = self.loss_function(outputs, labels)
        binary_prediction = (outputs >= 0.5).int()

        # # Convert to NumPy array
        # if trainer.is_global_zero and batch_idx % 3 == 0:
        #     self._save_as_nifti(
        #         batch["label"][0][0].cpu().numpy(),
        #         f"example_model_predictions/label-{batch_idx}.nii.gz",
        #     )
        #     self._save_as_nifti(
        #         outputs[0][0].cpu().numpy(),
        #         f"example_model_predictions/output{batch_idx}-{batch_idx}.nii.gz",
        #     )
        #     self._save_as_nifti(
        #         binary_prediction[0][0].cpu().numpy(),
        #         f"example_model_predictions/binary_output-{batch_idx}.nii.gz",
        #     )

        # decollated_outputs = decollate_batch(binary_prediction)
        # decollated_labels = decollate_batch(labels)

        dice = self.dice_metric(y_pred=binary_prediction, y=labels).mean()

        # self.log("Validation/val_loss", loss, on_step=False, on_epoch=True, prog_bar=False) # , sync_dist=True)
        # self.log("Validation/val_dice", dice, on_step=False, on_epoch=True, prog_bar=False) # , sync_dist=True)

        if self.trainer.is_global_zero:
            print("\ndice shape", dice.shape)
            print(f"Batch {batch_idx}: dice metric", dice)

        val_loss_dict = {"val_loss": loss, "val_number": binary_prediction.shape[0]}

        self.validation_step_outputs.append(val_loss_dict)

        return val_loss_dict

    def on_validation_epoch_end(self, outputs=None):
        # 1. Gather Individual Metrics
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]

        mean_val_loss = val_loss / num_items if num_items > 0 else 0.0
        mean_val_dice = self.dice_metric.aggregate().item()

        tensorboard_logs = {
            "avg_val_dice": mean_val_dice,
            "avg_val_loss": mean_val_loss,
        }

        if self.trainer.is_global_zero:
            if mean_val_dice > self.best_val_dice:
                self.best_val_dice = mean_val_dice
                self.best_val_epoch = self.current_epoch
            print(
                f"\ncurrent epoch: {self.current_epoch} "
                f"\ncurrent mean dice: {mean_val_dice:.4f}"
                f"\nbest mean dice: {self.best_val_dice:.4f} "
                f"\nat epoch: {self.best_val_epoch}"
            )

        self.metric_values.append(mean_val_dice)

        self.log(
            "Validation/avg_val_dice",
            mean_val_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            # sync_dist=True
        )
        self.log(
            "Validation/avg_val_loss",
            mean_val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            # sync_dist=True
        )

        # profiler.describe()

        # Clear validation step outputs and reset metrics for all processes
        self.validation_step_outputs.clear()
        self.dice_metric.reset()

        # Return the logs only from the main process
        return {"log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (128, 128, 128)
        sw_batch_size = 4

        outputs = sliding_window_inference(
            images,
            roi_size,
            sw_batch_size,
            self.forward,
        )

        loss = self.loss_function(outputs, labels)

        binary_prediction = (outputs >= 0.5).int()

        dice = self.dice_metric(y_pred=binary_prediction, y=labels)

        self.log("Test/test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("Test/test_dice", dice, on_step=False, on_epoch=True, prog_bar=False)

        test_dict = {"test_loss": loss, "test_number": binary_prediction.shape[0]}

        self.test_step_outputs.append(test_dict)

        return test_dict


    def on_test_epoch_end(self):
        # 1. Gather Individual Metrics
        test_loss, num_items = 0, 0
        for output in self.test_step_outputs:
            test_loss += output["test_loss"].sum().item()
            num_items += output["test_number"]

        mean_test_loss = test_loss / num_items if num_items > 0 else 0.0
        mean_test_dice = self.dice_metric.aggregate().item()

        self.metric_values.append(mean_test_dice)

        tensorboard_logs = {
            "avg_val_dice": mean_test_dice,
            "avg_val_loss": mean_test_loss,
        }

        self.log(
            "Test/avg_epoch_test_dice",
            mean_test_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        self.log(
            "Test/avg_epoch_test_loss",
            mean_test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        # Clear validation step outputs and reset metrics for all processes
        self.test_step_outputs.clear()
        self.dice_metric.reset()

        # Return the logs only from the main
        return {"log": tensorboard_logs} 

class MyUNETRSetup:
    def __init__(self, model_params):
        """
        Pipeline for training the UNETR (with superresolution) for the data provided in RootNet/data

        Args:
        - model_params: dictionary containing the parameters for the MyUNETRWrapper class
        """
        self.model = MyUNETRWrapper(**model_params)
        self._setup_checkpointing()
        self._setup_logger()
        # self.setup_profiler()

    def _setup_checkpointing(self):
        self.checkpoint_callback = ModelCheckpoint(
            dirpath="../../runs",
            filename="best_metric_model",
            save_top_k=1,
            monitor="Validation/avg_val_loss",
            mode="min",
            every_n_epochs=5,
        )

    def _setup_logger(self):
        self.tb_logger = TensorBoardLogger("tb_logs", name="my_model")

    def _get_trainer(self, checkpoint_path=None):
        nnodes = os.getenv("SLURM_NNODES", 1)
        trainer = pl.Trainer(
            accelerator="gpu",
            strategy=pl.strategies.DDPStrategy(find_unused_parameters=False),
            devices=4,
            num_nodes=int(nnodes),
            max_epochs=self.model.max_epochs,
            check_val_every_n_epoch=self.model.check_val,
            logger=self.tb_logger,
            callbacks=[self.checkpoint_callback],
            # sync_batchnorm=True,
            precision=16,
            log_every_n_steps=42,
            default_root_dir=".tb_logs/runs",
            # resume_from_checkpoint=checkpoint_path,
            # profiler=self.profiler,
        )

        return trainer
    
    # def _setup_profiler(self):
    #     self.profiler = PyTorchProfiler(
    #         profiled_functions=["forward", "training_step"],
    #         record_shapes=True,
    #         profile_memory=True,
    #         use_cuda=True,
    #     )

    def train(self, model_checkpoint=None):
        trainer = self._get_trainer(checkpoint_path=model_checkpoint)
        trainer.fit(self.model, ckpt_path=model_checkpoint)

    def test(self, ckpt_path=None):
        trainer = self._get_trainer(checkpoint_path=ckpt_path)

        trainer.test(self.model, ckpt_path=ckpt_path)

    def print_model_stats(self):
        num_parameters = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {num_parameters}")

        # Uncomment to calculate FLOPs and params
        # sample_input = torch.randn(1, 1, 128, 128, 128)
        # flops, params = profile(self.model, inputs=(sample_input,))
        # print(f"flops: {flops}")
        # print(f"params: {params}")

# Example Usage:
model_params = {
    'in_channels': 1,
    'out_channels': 1,
    'img_shape': (128, 128, 128),
    'feature_size': 16,
    'batch_size': 1,
    'max_epochs': 200,
    'check_val': 10
}

training_pipeline = MyUNETRSetup(model_params)
training_pipeline.train(model_checkpoint="../../runs/best_metric_model-v6.ckpt")

# training_pipeline.test("../../runs/best_metric_model-v2.ckpt")
# training_pipeline.print_model_stats()
