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
from enum import Enum

import sys
import os

sys.path.append("..")
# sys.path.append("../models")


from data import CreateJsonFileConfig
from utils import (
    RandCropByPosNegLabeldWithResAdjust,
    MyTransformDataset,
    LinearWarmupScheduler,
    CompositeScheduler,
)
from mri_dataloader import MRIDataLoader
from models import MyUNETR, MySwinUNETR, UNet, UNETR

from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from monai.utils import set_determinism
from monai.metrics import DiceMetric


from monai.data import (
    DataLoader,
    CacheDataset,
    decollate_batch,
    list_data_collate,
)
from monai.transforms import (
    AsDiscrete,
)

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
import numpy as np
import nibabel as nib
import random

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import json

torch.set_float32_matmul_precision("medium")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_determinism(seed=seed)


seed = 40
seed_everything(seed)


class MyUNETRWrapper(pl.LightningModule):
    def __init__(
        self,
        model: str,
        model_params: dict,
        learning_rate: float = 0.0001,
        img_shape: Tuple[int, int, int] = (96, 96, 96),
    ) -> None:
        """
        Pytorch Lightning Module which sets up the training of the UNETR (with superresolution) for the data provided in RootNet/data

        Args:
        - learning_rate: learning rate for the optimizer
        - model: the model to train
        """

        super().__init__()
        self.save_hyperparameters()

        if model is None:
            raise ValueError("model cannot be None")
        elif model == ModelType.MYUNETR.name:
            self.model = MyUNETR(**model_params, img_shape=img_shape)
        elif model == ModelType.UNETR.name:
            self.model = UNETR(**model_params, img_shape=img_shape)
        elif model == ModelType.UNET.name:
            self.model = UNet(**model_params, img_shape=img_shape)
        elif model == ModelType.SWINUNETR.name:
            self.model = MySwinUNETR(**model_params, img_shape=img_shape)

        # Loss function and metrics
        self.loss_function = DiceCELoss(
            sigmoid=True,
            include_background=False,
            to_onehot_y=True,  # set true for softmax loss
        )
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False,
        )  # TODO: include_background default is False
        self.img_shape = img_shape

        # Training parameters
        self.learning_rate = learning_rate

        # Variables to store training and validation metrics
        self.metric_values = []
        self.epoch_loss_values = []
        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []
        self.best_val_dice = 0
        self.best_val_epoch = 0

        self.post_pred = AsDiscrete(argmax=True, to_onehot=2)
        self.post_label = AsDiscrete(to_onehot=2)

    def forward(self, x_in):
        output = self.model(x_in)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

        # warmup_epochs = 10  # Number of epochs for warmup
        # total_epochs = 800  # Total training epochs
        # target_lr = 0.0001    # Target LR after warmup

        # Warmup Scheduler
        # warmup_scheduler = LinearWarmupScheduler(optimizer, warmup_epochs, target_lr)

        # Cosine Annealing Scheduler
        # annealing_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=0)

        # Exposure Scheduler
        # exponential_scheduler = ExponentialLR(optimizer, gamma=0.95)

        return optimizer

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": exponential_scheduler,
        #         "interval": "epoch"
        #     }
        # }

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]

        output = self.forward(images)

        loss = self.loss_function(output, labels)

        # if torch.isnan(loss).any():
        #     nan_count = torch.isnan(output).sum().item()
        #     zero_count = (output == 0).sum().item()
        #     non_zero_count = (output != 0).sum().item()

        #     print(f"nan count: {nan_count}")
        #     print(f"zero count: {zero_count}")
        #     print(f"non zero count: {non_zero_count}")

        loss_dict = {"loss": loss}

        self.training_step_outputs.append(loss_dict)

        return loss_dict

    def on_train_epoch_end(
        self, unused=None, outputs=None
    ):  # on_train_epoch_end in newest pytorch version
        if self.training_step_outputs:
            avg_loss = torch.stack(
                [x["loss"] for x in self.training_step_outputs]
            ).mean()

            # Only perform the following operations on the main process
            if self.trainer.is_global_zero:
                # Store or log the average loss
                self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())
                self.log(
                    "Train/avg_epoch_loss",
                    avg_loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )  # , sync_dist=True)

                self.log(
                    "learning_rate",
                    self.trainer.optimizers[0].param_groups[0]["lr"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

                # Clear training step outputs
                self.training_step_outputs.clear()

    def on_after_backward(self):
        # Log the gradients of each parameter
        if self.trainer.global_step % 3 == 0:
            for name, param in self.named_parameters():
                self.log(
                    f"grad/{name}",
                    param.grad.norm(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,  # TODO: maybe set to false if it causes an issue
                )

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = self.img_shape
        sw_batch_size = 4

        outputs = sliding_window_inference(
            images,
            roi_size,
            sw_batch_size,
            self.forward,
        )

        loss = self.loss_function(outputs, labels)

        # for Sigmoid loss
        # out = torch.sigmoid(outputs)
        # binary_prediction = (out >= 0.5).int()

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

        # for softmax loss
        decollated_outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        decollated_labels = [self.post_label(i) for i in decollate_batch(labels)]

        dice = self.dice_metric(y_pred=decollated_outputs, y=decollated_labels)

        if self.trainer.is_global_zero:
            print("\ndice shape", dice.shape)
            print(f"Batch {batch_idx}: dice metric", dice)

        val_loss_dict = {"val_loss": loss, "val_number": outputs.shape[0]}

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
        roi_size = self.img_shape
        sw_batch_size = 4

        outputs = sliding_window_inference(
            images,
            roi_size,
            sw_batch_size,
            self.forward,
        )

        loss = self.loss_function(outputs, labels)

        # for softmax
        decollated_outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        decollated_labels = [self.post_label(i) for i in decollate_batch(labels)]

        dice = self.dice_metric(y_pred=decollated_outputs, y=decollated_labels)

        # for sigmoid
        # binary_prediction = (outputs >= 0.5).int()

        # dice = self.dice_metric(y_pred=binary_prediction, y=labels)

        self.log("Test/test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("Test/test_dice", dice, on_step=False, on_epoch=True, prog_bar=False)

        test_dict = {"test_loss": loss, "test_number": decollated_outputs.shape[0]}

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
            logger=True,
        )
        self.log(
            "Test/avg_epoch_test_loss",
            mean_test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Clear validation step outputs and reset metrics for all processes
        self.test_step_outputs.clear()
        self.dice_metric.reset()

        # Return the logs only from the main
        return {"log": tensorboard_logs}

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


# create an enum for the different models
class ModelType(Enum):
    UNETR = 1
    UNET = 2
    SWINUNETR = 3
    MYUNETR = 4
