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

###################### Loss imports ######################
from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.utils import (
    LossReduction,
)
from monai.metrics import ConfusionMatrixMetric

###################### Model imports ######################

from typing import Tuple, Union
from enum import Enum

import sys
import os

sys.path.append("..")
# sys.path.append("../models")


from utils import (
    Visualizations,
    MRIoperations,
)
from mri_dataloader import MRIDataLoader
from models import MyUNETR, MyUpscaleSwinUNETR, UNet, UNETR, SwinUNETR

from monai.losses import DiceCELoss, DiceFocalLoss  # , DiceLoss
from monai.inferers import sliding_window_inference
from monai.utils import set_determinism
from monai.metrics import DiceMetric


from monai.data import (
    decollate_batch,
)
from monai.transforms import (
    AsDiscrete,
)

import torch
import numpy as np
import random

import pytorch_lightning as pl


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


# create an enum for the different models
class ModelType(Enum):
    UNETR = 1
    UNET = 2
    UPSCALESWINUNETR = 3
    MYUNETR = 4
    SWINUNETR = 5


class OutputActivation(Enum):
    SIGMOID = 1
    SOFTMAX = 2


class MyUNETRWrapper(pl.LightningModule):
    def __init__(
        self,
        model: str,
        model_params: dict,
        learning_rate: float = 0.0001,
        img_shape: Tuple[int, int, int] = (96, 96, 96),
        output_activation="SIGMOID",
    ) -> None:
        """
        Pytorch Lightning Module which sets up the training of the UNETR (with superresolution) for the data provided in RootNet/data

        Args:
        - learning_rate: learning rate for the optimizer
        - model: the model to train
        """

        super().__init__()
        self.save_hyperparameters()

        patch_shape = img_shape

        if model is None:
            raise ValueError("model cannot be None")
        elif model == ModelType.MYUNETR.name:
            self.model = MyUNETR(**model_params, img_shape=patch_shape)
        elif model == ModelType.UNETR.name:
            self.model = UNETR(**model_params, img_shape=patch_shape)
        elif model == ModelType.UNET.name:
            self.model = UNet(**model_params, img_shape=patch_shape)
        elif model == ModelType.UPSCALESWINUNETR.name:
            self.model = MyUpscaleSwinUNETR(**model_params, img_shape=patch_shape)
        elif model == ModelType.SWINUNETR.name:
            self.model = SwinUNETR(**model_params, img_shape=patch_shape)
        else:
            raise ValueError("Model must be of type ModelType.")

        # Loss function and metrics
        self.loss_function = DiceLoss(
            sigmoid=(
                True if output_activation == OutputActivation.SIGMOID.name else False
            ),
            softmax=(
                True if output_activation == OutputActivation.SOFTMAX.name else False
            ),
            include_background=True,
            to_onehot_y=True,  # set true for softmax loss
            weight=torch.tensor([1, 1.1]).cuda(),
        )
        self.root_confusion_matrix_metrics = ConfusionMatrixMetric(
            include_background=True,
            metric_name=("precision", "recall", "f1 score"),
            reduction="mean",
            get_not_nans=False,
        )
        self.background_confusion_matrix_metrics = ConfusionMatrixMetric(
            include_background=True,
            metric_name=("precision", "recall", "f1 score"),
            reduction="mean",
            get_not_nans=False,
        )
        self.dice_metric = DiceMetric(
            include_background=True,
            reduction="mean",
            get_not_nans=False,
        )  # TODO: include_background default is False
        self.patch_shape = patch_shape

        # Training parameters
        self.learning_rate = learning_rate
        self.output_activation = output_activation

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

        loss_dict = {"loss": loss}

        self.training_step_outputs.append(loss_dict)

        return loss_dict

    def on_train_epoch_end(self, unused=None, outputs=None):
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

    # def on_after_backward(self):
    #     # Log the gradients of each parameter
    #     if self.trainer.global_step % 3 == 0:
    #         for name, param in self.named_parameters():
    #             self.log(
    #                 f"grad/{name}",
    #                 param.grad.norm(),
    #                 on_step=False,
    #                 on_epoch=True,
    #                 prog_bar=True,
    #                 logger=True,
    #                 sync_dist=True,  # TODO: maybe set to false if it causes an issue
    #             )

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = self.patch_shape
        sw_batch_size = 4

        labels = labels[
            :, :, : images.shape[2] * 2, : images.shape[3] * 2, : images.shape[4] * 2
        ]

        outputs = sliding_window_inference(
            images,
            roi_size,
            sw_batch_size,
            self.forward,
        )

        loss = self.loss_function(outputs, labels)

        # for Sigmoid loss
        if self.output_activation == OutputActivation.SIGMOID.name:
            out = torch.sigmoid(outputs)
            binary_output = (out >= 0.5).int()

            final_label = torch.stack(
                [self.post_label(i) for i in decollate_batch(labels)]
            )

        elif self.output_activation == OutputActivation.SOFTMAX.name:
            binary_output = [self.post_pred(i) for i in decollate_batch(outputs)]
            final_label = [self.post_label(i) for i in decollate_batch(labels)]

        else:
            raise ValueError("Output activation must be of type OutputActivation.")

        dice = self.dice_metric(y_pred=binary_output, y=final_label)
        root_cmm = self.root_confusion_matrix_metrics(
            y_pred=binary_output[:, 1, :, :, :].unsqueeze(1),
            y=final_label[:, 1, :, :, :].unsqueeze(1),
        )
        background_cmm = self.background_confusion_matrix_metrics(
            y_pred=binary_output[:, 0, :, :, :].unsqueeze(1),
            y=final_label[:, 0, :, :, :].unsqueeze(1),
        )

        if self.trainer.is_global_zero:
            print("\ndice shape", dice.shape)
            print(f"Batch {batch_idx}: dice metric", dice)

        # For inspection: save the predictions as nifti files
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

        val_output = {
            "val_loss": loss,
            "val_number": outputs.shape[0],
            "confusion_matrix_metrics": root_cmm,
            "background_confusion_matrix_metrics": background_cmm,
        }

        self.validation_step_outputs.append(val_output)

        return val_output

    def on_validation_epoch_end(self, outputs=None):
        # 1. Gather Individual Metrics
        mean_val_loss = torch.stack(
            [x["val_loss"] for x in self.validation_step_outputs]
        )
        mean_val_loss = mean_val_loss.mean()
        mean_val_dice = self.dice_metric.aggregate().item()

        # Stack the tensors along a new dimension (e.g., dimension 0)
        root_cmm = self.root_confusion_matrix_metrics.aggregate()
        background_cmm = self.background_confusion_matrix_metrics.aggregate()

        tensorboard_logs = {
            "Validation/avg_val_dice": mean_val_dice,
            "Validation/avg_val_loss": mean_val_loss,
            "Validation/root_avg_precision": root_cmm[0],
            "Validation/root_avg_recall": root_cmm[1],
            "Validation/root_avg_f1": root_cmm[2],
            "Validation/background_avg_precision": background_cmm[0],
            "Validation/background_avg_recall": background_cmm[1],
            "Validation/background_avg_f1": background_cmm[2],
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

        # Loop through the dictionary and log each metric
        for metric_name, metric_value in tensorboard_logs.items():
            self.log(
                metric_name,
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                # sync_dist=True  # Uncomment if you are training in a distributed setting and want to synchronize metrics
            )

        # profiler.describe()

        # Clear validation step outputs and reset metrics for all processes
        self.validation_step_outputs.clear()
        self.dice_metric.reset()

        # Return the logs only from the main process
        return {"log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = self.patch_shape
        sw_batch_size = 4

        labels = labels[
            :, :, : images.shape[2] * 2, : images.shape[3] * 2, : images.shape[4] * 2
        ]

        outputs = sliding_window_inference(
            images,
            roi_size,
            sw_batch_size,
            self.forward,
        )

        loss = self.loss_function(outputs, labels)

        # for Sigmoid loss
        if self.output_activation == OutputActivation.SIGMOID.name:
            out = torch.sigmoid(outputs)
            binary_output = (out >= 0.5).int()

            final_label = torch.stack(
                [self.post_label(i) for i in decollate_batch(labels)]
            )

        elif self.output_activation == OutputActivation.SOFTMAX.name:
            binary_output = torch.stack(
                [self.post_pred(i) for i in decollate_batch(outputs)]
            )
            final_label = torch.stack(
                [self.post_label(i) for i in decollate_batch(labels)]
            )

        else:
            raise ValueError("Output activation must be of type OutputActivation.")

        print("outputs datatype", type(outputs))

        dice = self.dice_metric(y_pred=binary_output, y=final_label)
        test = binary_output[:, 1, :, :, :].unsqueeze(1)

        root_cmm = self.root_confusion_matrix_metrics(
            y_pred=binary_output[:, 1, :, :, :].unsqueeze(1),
            y=final_label[:, 1, :, :, :].unsqueeze(1),
        )
        background_cmm = self.background_confusion_matrix_metrics(
            y_pred=binary_output[:, 0, :, :, :].unsqueeze(1),
            y=final_label[:, 0, :, :, :].unsqueeze(1),
        )

        # if the ouput has 2 channels, which means that the Softmax activation function was used,
        # then the last channel is the probability of the root
        label_idx = outputs.shape[1] - 1

        model_dir = os.path.dirname(f"./test_model_output/{batch_idx}/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        visualizer = Visualizations()
        for slice in [0.3, 0.5, 0.6, 0.7]:
            visualizer.plot_row(
                batch["image"][0, 0, :, :, :].cpu(),
                outputs[0, label_idx, :, :, :].cpu(),
                binary_output[0, label_idx, :, :, :].cpu(),
                slice_frac=slice,
                filename=f"test_model_output/{batch_idx}/prediction_{slice}",
                label=final_label[0, label_idx, :, :, :].cpu(),
            )

        # For inspection: save the predictions as nifti files
        # Convert to NumPy array
        mri_ops = MRIoperations()
        mri_ops.save_mri(
            f"test_model_output/{batch_idx}/label.nii.gz",
            batch["label"][0][0].cpu().numpy(),
        )
        mri_ops.save_mri(
            f"test_model_output/{batch_idx}/output.nii.gz", outputs[0][0].cpu().numpy()
        )
        mri_ops.save_mri(
            f"test_model_output/{batch_idx}/binary_output.nii.gz",
            binary_output[0][0].cpu().numpy(),
        )

        test_output = {
            "test_loss": loss,
            "test_number": outputs.shape[0],
            "confusion_matrix_metrics": root_cmm,
            "background_confusion_matrix_metrics": background_cmm,
        }

        self.test_step_outputs.append(test_output)

        return test_output

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


class DiceLoss(_Loss):
    """
    Compute average Dice loss between two tensors. It can support both multi-classes and multi-labels tasks.
    The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target` (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The `smooth_nr` and `smooth_dr` parameters are values added to the intersection and union components of
    the inter-over-union calculation to smooth results respectively, these values should be small.

    The original paper: Milletari, F. et. al. (2016) V-Net: Fully Convolutional Neural Networks forVolumetric
    Medical Image Segmentation, 3DV, 2016.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Callable | None = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: LossReduction | str = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        weight: Sequence[float] | float | int | torch.Tensor | None = None,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            jaccard: compute Jaccard Index (soft IoU) instead of dice or not.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

            smooth_nr: a small constant added to the numerator to avoid zero.
            smooth_dr: a small constant added to the denominator to avoid nan.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a Dice loss value is computed independently from each item in the batch
                before any `reduction`.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes. If not ``include_background``,
                the number of classes should not include the background category class 0).
                The value/values should be no less than 0. Defaults to None.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super().__init__(reduction=LossReduction(reduction).value)
        if other_act is not None and not callable(other_act):
            raise TypeError(
                f"other_act must be None or callable but is {type(other_act).__name__}."
            )
        if int(sigmoid) + int(softmax) + int(other_act is not None) > 1:
            raise ValueError(
                "Incompatible values: more than 1 of [sigmoid=True, softmax=True, other_act is not None]."
            )
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other_act = other_act
        self.squared_pred = squared_pred
        self.jaccard = jaccard
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.batch = batch
        self.weight = weight
        self.register_buffer("class_weight", torch.ones(2))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> from monai.losses.dice import *  # NOQA
            >>> import torch
            >>> from monai.losses.dice import DiceLoss
            >>> B, C, H, W = 7, 5, 3, 2
            >>> input = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = DiceLoss(reduction='none')
            >>> loss = self(input, target)
            >>> assert np.broadcast_shapes(loss.shape, input.shape) == input.shape
        """
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        if self.other_act is not None:
            input = self.other_act(input)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn(
                    "single channel prediction, `include_background=False` ignored."
                )
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(
                f"ground truth has different shape ({target.shape}) from input ({input.shape})"
            )

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis

        intersection = torch.sum(target * input, dim=reduce_axis)

        if self.squared_pred:
            ground_o = torch.sum(target**2, dim=reduce_axis)
            pred_o = torch.sum(input**2, dim=reduce_axis)
        else:
            ground_o = torch.sum(target, dim=reduce_axis)
            pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (
            denominator + self.smooth_dr
        )

        if self.weight is not None and target.shape[1] != 1:
            # make sure the lengths of weights are equal to the number of classes
            num_of_classes = target.shape[1]
            if isinstance(self.weight, (float, int)):
                self.class_weight = torch.as_tensor([self.weight] * num_of_classes)
            else:
                self.class_weight = torch.as_tensor(self.weight)
                if self.class_weight.shape[0] != num_of_classes:
                    raise ValueError(
                        """the length of the `weight` sequence should be the same as the number of classes.
                        If `include_background=False`, the weight should not include
                        the background category class 0."""
                    )
            if self.class_weight.min() < 0:
                raise ValueError(
                    "the value/values of the `weight` should be no less than 0."
                )
            # apply class_weight to loss

            self.class_weight = self.class_weight.to(f)

            f = f * self.class_weight

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(
                f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
            )

        return f
