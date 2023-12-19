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
from utils import (
    RandCropByPosNegLabeldWithResAdjust,
    MyTransformDataset,
    LinearWarmupScheduler,
    CompositeScheduler,
)
from models import MyUNETR, MySwinUNETR, UNet, UNETR

from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from monai.utils import set_determinism
from monai.metrics import DiceMetric
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    AsDiscrete,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    NormalizeIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    Resized,
    RandRotate90d,
)

from monai.data import (
    DataLoader,
    CacheDataset,
    decollate_batch,
    list_data_collate,
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
        batch_size: int = 4,
        upscale: bool = True,
        learning_rate: float = 0.0001,
        model: pl.LightningModule = None,
        samples_per_volume: int = 3,
        img_shape: Tuple[int, int, int] = (96, 96, 96),
    ) -> None:
        """
        Pytorch Lightning Module which sets up the training of the UNETR (with superresolution) for the data provided in RootNet/data

        Args:
        - batch_size: batch size for training
        - upscale: whether to upscale the images to twice the input size if the model has a superresolution head
        - learning_rate: learning rate for the optimizer
        - model: the model to train
        - samples_per_volume: number of samples to extract from each volume
        """

        super().__init__()

        self.model = model

        # Loss function and metrics
        self.loss_function = DiceCELoss(
            sigmoid=True,
            include_background=False,
            to_onehot_y=True # set true for softmax loss
        )
        self.dice_metric = DiceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False,
        )  # TODO: include_background default is False
        self.upscale = upscale
        self.samples_per_volume = samples_per_volume
        self.img_shape = img_shape

        # Training parameters
        self.batch_size = batch_size
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

    def prepare_data(self):
        print("creating data_config.json")
        myCreateJsonFileConfig = CreateJsonFileConfig()
        myCreateJsonFileConfig.create_config()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            with open("data_config.json", "r") as file:
                # Load the JSON data into a dictionary
                data_config = json.load(file)

                cache_transform_list = [
                    LoadImaged(keys=["image", "label"]),
                    EnsureChannelFirstd(keys=["image", "label"]),
                    Orientationd(keys=["image", "label"], axcodes="RAS"),
                    ScaleIntensityRanged(
                        keys=["image"],
                        a_min=0,
                        a_max=30000,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
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
        dataset = MyTransformDataset(self.train_ds, transform=self.train_transforms)
        train_loader = DataLoader(
            dataset,
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
                    # sync_dist=True
                )

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


class MyUNETRSetup:
    def __init__(
        self,
        train_params,
        model_name,
        checkpoint_path,
        best_model_checkpoint,
        max_epochs=500,
        check_val=10,
        store_model_epoch=30,
    ):
        """
        Pipeline for training the UNETR (with superresolution) for the data provided in RootNet/data

        Args:
        - model_params: dictionary containing the parameters for the MyUNETRWrapper class
        - model_name: name of the model for creating the checkpoint dir and tensorboard logs
        - checkpoint_path: path to the checkpoint dir
        - best_model_checkpoint: name of the best model checkpoint
        - max_epochs: maximum number of epochs to train
        - check_val: number of epochs after which to perform validation
        - store_model_epoch: number of epochs after which to store the model, stop and restart training
        """
        self.max_epochs = max_epochs
        self.check_val = check_val
        self.store_model_epoch = store_model_epoch
        self.checkpoint_path = checkpoint_path
        self.best_model_checkpoint = best_model_checkpoint
        self.train_params = train_params

        self.model_name = model_name
        self._setup_checkpointing()
        self._setup_logger()
        # self.setup_profiler()

    def _setup_checkpointing(self):
        self.checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_path,
            filename=self.best_model_checkpoint,
            save_top_k=1,
            monitor="Validation/avg_val_loss",
            mode="min",
            every_n_epochs=1,
        )

    def _setup_logger(self):
        self.tb_logger = (
            TensorBoardLogger("tb_logs", name=self.model_name),
        )  # name="my_model")

    def _get_trainer(self, max_epochs, check_val):
        nnodes = os.getenv("SLURM_NNODES", 1)
        trainer = pl.Trainer(
            accelerator="gpu",
            strategy=pl.strategies.DDPStrategy(find_unused_parameters=False),
            devices=4,
            num_nodes=int(nnodes),
            max_epochs=max_epochs,
            check_val_every_n_epoch=check_val,
            logger=self.tb_logger,
            callbacks=[self.checkpoint_callback],
            gradient_clip_val=5,
            # sync_batchnorm=True,
            precision=16,
            log_every_n_steps=42,
            default_root_dir=".tb_logs/runs",
            # resume_from_checkpoint=checkpoint_path,
            # profiler=self.profiler,
        )
        print(f"trainer.max_epochs: {trainer.max_epochs}")

        return trainer

    # def _setup_profiler(self):
    #     self.profiler = PyTorchProfiler(
    #         profiled_functions=["forward", "training_step"],
    #         record_shapes=True,
    #         profile_memory=True,
    #         use_cuda=True,
    #     )

    def _get_continue_train_epoch(self, checkpoint_path):
        continue_training_epoch = 0
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            continue_training_epoch = checkpoint["epoch"]

        return continue_training_epoch

    def train(self, model_checkpoint=None):
        """
        Trains the model for store_model_epoch epochs until max_epochs is reached
        This is because a (Pin memory error arises otherwise)

        Args:
        - model_checkpoint: path to the model checkpoint to continue training from
        """
        continue_training_epoch = self._get_continue_train_epoch(model_checkpoint)
        print(f"continue_training_epoch: {continue_training_epoch}")

        for i in range(0, max_epochs, self.store_model_epoch):
            if i > continue_training_epoch:
                max_epochs_adjusted = i
                print(f"max_epochs_adjusted: {max_epochs_adjusted}")

                trainer = self._get_trainer(
                    max_epochs=max_epochs_adjusted,
                    check_val=check_val,
                )

                self.model = MyUNETRWrapper(**train_params)
                if i <= self.store_model_epoch:
                    trainer.fit(self.model)
                else:
                    trainer.fit(self.model, ckpt_path=model_checkpoint)
                trainer.save_checkpoint(model_checkpoint)

    def test(self, ckpt_path=None):
        trainer = self._get_trainer(checkpoint_path=ckpt_path)

        trainer.test(self.model, ckpt_path=ckpt_path)


# Example Usage:
img_shape = (96, 96, 96)

# model = UNet()

# model = MySwinUNETR(img_shape=img_shape, in_channels=1, out_channels=2)

model = MyUNETR(
    in_channels=1,
    out_channels=2,
    feature_size=16,
    img_shape=(96, 96, 96),
)

# model = UNETR()

train_params = {
    "batch_size": 3,
    "upscale": True,
    "learning_rate": 0.0001,
    "model": model,
    "samples_per_volume": 2,
    "img_shape": img_shape,
}

model_name = f"MyUNETR_lr_{train_params['learning_rate']}_simpleUp_softmax"
checkpoint_path = f"../../runs/{model_name}"
checkpoint_file = "latest_model"
best_checkpoint_file = "best_metric_model"

max_epochs = 550
check_val = 10
store_model_epoch = 100

training_pipeline = MyUNETRSetup(
    train_params,
    model_name,
    checkpoint_path,
    best_checkpoint_file,
    max_epochs=max_epochs,
    check_val=check_val,
    store_model_epoch=store_model_epoch,
)

training_pipeline.train(
    model_checkpoint=f"{checkpoint_path}/{checkpoint_file}.ckpt",
)

############## TEST ##############

# training_pipeline.test("../../runs/best_metric_model-v2.ckpt")
# training_pipeline.print_model_stats()
