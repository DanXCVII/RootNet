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
import torch.nn.functional as F

from monai.networks import one_hot
from monai.utils import (
    LossReduction,
)
from monai.metrics import ConfusionMatrixMetric
from monai.metrics import HausdorffDistanceMetric
from monai.metrics import SurfaceDistanceMetric
from monai.metrics import compute_hausdorff_distance
from monai.metrics import get_confusion_matrix  
from torchmetrics.classification import PrecisionRecallCurve, BinaryPrecisionRecallCurve
import time

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
    ChainedScheduler,
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

import matplotlib.pyplot as plt


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


class MyPlSetup(pl.LightningModule):
    def __init__(
        self,
        model: str,
        model_params: dict,
        learning_rate: float = 0.0001,
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        class_weight=[1, 1],
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

        if model is None:
            raise ValueError("model cannot be None")
        elif model == ModelType.MYUNETR.name:
            self.model = MyUNETR(**model_params, patch_size=patch_size)
        elif model == ModelType.UNETR.name:
            self.model = UNETR(**model_params, patch_size=patch_size)
        elif model == ModelType.UNET.name:
            self.model = UNet(**model_params, patch_size=patch_size)
        elif model == ModelType.UPSCALESWINUNETR.name:
            self.model = MyUpscaleSwinUNETR(**model_params, patch_size=patch_size)
        elif model == ModelType.SWINUNETR.name:
            self.model = SwinUNETR(**model_params, patch_size=patch_size)
        else:
            raise ValueError("Model must be of type ModelType.")

        # Loss function and metrics
        include_background = False
        self.loss_function = DiceLoss(
            sigmoid=(
                True if output_activation == OutputActivation.SIGMOID.name else False
            ),
            softmax=(
                True if output_activation == OutputActivation.SOFTMAX.name else False
            ),
            include_background=include_background,
            to_onehot_y=True,  # set true for softmax loss
            weight=torch.tensor(class_weight).cuda(),
        )
        
        self.hausdorff_distance_metric = HausdorffDistanceMetric(
            include_background=include_background,
            reduction="mean",
            get_not_nans=False,
            percentile=95,
        )
        self.surface_distance_metric = SurfaceDistanceMetric(
            include_background=include_background,
            reduction="mean",
            get_not_nans=False,
            symmetric=True,
        )
        self.root_confusion_matrix_metrics = ConfusionMatrixMetric(
            include_background=include_background,
            metric_name=("precision", "recall", "f1 score"),
            reduction="mean",
            get_not_nans=False,
        )
        self.background_confusion_matrix_metrics = ConfusionMatrixMetric(
            include_background=include_background,
            metric_name=("precision", "recall", "f1 score"),
            reduction="mean",
            get_not_nans=False,
        )
        self.root_confusion_matrix_metrics_agg = ConfusionMatrixMetric(
            include_background=include_background,
            metric_name=("precision", "recall", "f1 score"),
            reduction="mean",
            get_not_nans=False,
        )
        self.background_confusion_matrix_metrics_agg = ConfusionMatrixMetric(
            include_background=include_background,
            metric_name=("precision", "recall", "f1 score"),
            reduction="mean",
            get_not_nans=False,
        )
        self.dice_metric = DiceMetric(
            include_background=include_background,
            reduction="mean",
            get_not_nans=False,
        )  # TODO: include_background default is False
        self.precision_recall_curve = PrecisionRecallCurve(num_classes=1)
        self.patch_size = patch_size
        self.root_f1s_arr = []

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

        warmup_cosine_scheduler = ChainedScheduler(
            optimizer,
            T_0 = 6,
            T_mul = 1,
            eta_min = 1e-5,
            gamma = 0.9,
            max_lr = self.learning_rate,
            warmup_steps= 5,
        )

        # return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warmup_cosine_scheduler,
                "interval": "epoch"
            }
        }

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
                    "Train/learning_rate",
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

    def _get_root_probability_binary(
        self,
        logits,
        labels,
    ):
        if self.output_activation == OutputActivation.SIGMOID.name:
            post_pred = torch.sigmoid(logits)
            binary_output = (post_pred >= 0.5).int()

            final_label = torch.stack(
                [self.post_label(i) for i in decollate_batch(labels)]
            )
            root_idx = 0

            # the bakcground probability is the inverse of the root probability
            background_pred = 1 - binary_output[:, 0, :, :, :].unsqueeze(1)

        elif self.output_activation == OutputActivation.SOFTMAX.name:
            post_pred = torch.softmax(logits, dim=1)
            binary_output = torch.stack(
                [self.post_pred(i) for i in decollate_batch(post_pred)]
            )
            final_label = torch.stack(
                [self.post_label(i) for i in decollate_batch(labels)]
            )

            # the background probability lies in the first channel for Softmax
            background_pred = binary_output[:, 0, :, :, :].unsqueeze(1)

            root_idx = 1

        else:
            raise ValueError("Output activation must be of type OutputActivation.")

        return post_pred, binary_output, final_label, background_pred, root_idx

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        sw_batch_size = 4

        labels = labels[
            :, :, : images.shape[2] * 2, : images.shape[3] * 2, : images.shape[4] * 2
        ]

        logits = sliding_window_inference(
            images,
            self.patch_size,
            sw_batch_size,
            self.forward,
        )

        loss = self.loss_function(logits, labels)

        _, binary_output, final_label, background_pred, root_idx = (
            self._get_root_probability_binary(logits, labels)
        )

        dice = self.dice_metric(y_pred=binary_output, y=final_label)

        root_cmm = self.root_confusion_matrix_metrics(
            y_pred=binary_output[:, root_idx, :, :, :].unsqueeze(1),
            y=final_label[:, root_idx, :, :, :].unsqueeze(1),
        )

        background_cmm = self.background_confusion_matrix_metrics(
            y_pred=background_pred,
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
            "val_number": logits.shape[0],
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
            # "Validation/root_avg_precision": root_cmm[0],
            # "Validation/root_avg_recall": root_cmm[1],
            # "Validation/root_avg_f1": root_cmm[2],
            # "Validation/background_avg_precision": background_cmm[0],
            # "Validation/background_avg_recall": background_cmm[1],
            # "Validation/background_avg_f1": background_cmm[2],
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
        self.root_confusion_matrix_metrics.reset()
        self.background_confusion_matrix_metrics.reset()

        # Return the logs only from the main process
        return {"log": tensorboard_logs}

    def _save_mri(self, mri_path, name, mri_data):
        mri_ops = MRIoperations()

        mri_shape = mri_data.shape

        mri_full_path = f"{mri_path}/{name}_{mri_shape[0]}x{mri_shape[1]}x{mri_shape[2]}.nii.gz"

        mri_ops.save_mri(mri_full_path, mri_data)

    def test_step(self, batch, batch_idx):
        print("test_step")
        images, labels = batch["image"], batch["label"]
        sw_batch_size = 2

        labels = labels[
            :,
            :,
            : images.shape[2] * 2,
            : images.shape[3] * 2,
            : images.shape[4] * 2,
        ]

        batch_image_np = batch["image"].cpu().numpy()
        print("starting test")
        print(self.output_activation)
        logits = sliding_window_inference(
            images,
            self.patch_size,
            sw_batch_size,
            self.forward,
        )

        # loss = self.loss_function(logits, labels)

        _, binary_output, final_label, background_pred, root_idx = (
            self._get_root_probability_binary(logits, labels)
        )
        binary_pred = binary_output[:, root_idx, :, :, :].unsqueeze(1)
        label_filtered = final_label[:, 0, :, :, :].unsqueeze(1)

        # uncomment, if evaluation for thresholding is needed
        # iterate over values between 0.998 and 0.9995 with step 0.0002
        thresh = True
        root_cmms = []
        best_percentile = 0
        
        # start timer
        time_start = time.time()
        if thresh:            
            start = 0.997
            rg = 28
            step_size = 0.0001

            binary_pred_arr = []
            binary_output_arr = []

            for i in range(rg):
                percentile = torch.quantile(batch["image"], start + i * step_size)
                mask = (batch["image"] > percentile).float()
                mask_up = F.interpolate(mask, size=(mask.shape[2]*2, mask.shape[3]*2, mask.shape[4]*2), mode='trilinear', align_corners=False)
                mask = mask_up

                mask_up = (mask_up > 0).float()

                inverse_mask = 1 - mask
                mask_2 = torch.cat((inverse_mask, mask), dim=1)

                binary_pred = mask
                if self.output_activation == OutputActivation.SIGMOID.name:
                    binary_output = mask
                else:
                    binary_output = mask_2

                binary_pred_arr.append(binary_pred.cpu().numpy())
                binary_output_arr.append(binary_output.cpu().numpy())

                root_cmm = self.root_confusion_matrix_metrics(
                    y_pred=binary_output[:, root_idx, :, :, :].unsqueeze(1),
                    y=final_label[:, root_idx, :, :, :].unsqueeze(1),
                )
                root_cmms.append(self.root_confusion_matrix_metrics.aggregate())
                self.root_confusion_matrix_metrics.reset()

            # get the index of the percentile with the highest f1 score+
            root_f1s = np.array([array[2].cpu().numpy() for array in root_cmms])
            self.root_f1s_arr.append(root_f1s)
            best_percentile = np.argmax(root_f1s)

            x_values = [start + i * step_size for i in range(rg)]

            # save the f1 sorces and x_values in a csv file
            np.savetxt(f'./test_model_output/dice_scores_{batch_idx}.csv', root_f1s, delimiter=',')
            np.savetxt(f'./test_model_output/x_values_{batch_idx}.csv', x_values, delimiter=',')
            self.plot_f1_scores(x_values, f'./test_model_output/f1_scores_{batch_idx}.png')


            binary_pred = torch.from_numpy(binary_pred_arr[best_percentile]).to('cuda')
            binary_output = torch.from_numpy(binary_output_arr[best_percentile]).to('cuda')

        # end timer
        print(f"Time taken: {time.time() - time_start}")

        # hd = self.hausdorff_distance_metric(y_pred=binary_pred, y=label_filtered)
        sd = self.surface_distance_metric(y_pred=binary_pred, y=label_filtered)
        
        # print(f"Hausdorff Distance: {hd}")
        print(f"Surface Distance: {sd}")

        dice = self.dice_metric(y_pred=binary_output, y=final_label)
        test_dice = self.dice_metric.aggregate().item()

        # setup the root confusion matrix metrics for aggregated data for the epoch and step
        root_cmm = self.root_confusion_matrix_metrics(
            y_pred=binary_output[:, root_idx, :, :, :].unsqueeze(1),
            y=final_label[:, root_idx, :, :, :].unsqueeze(1),
        )

        root_cmm_agg = self.root_confusion_matrix_metrics_agg(
            y_pred=binary_output[:, root_idx, :, :, :].unsqueeze(1),
            y=final_label[:, root_idx, :, :, :].unsqueeze(1),
        )

        confusion_matrix = get_confusion_matrix(
            y_pred=binary_output[:, root_idx, :, :, :].unsqueeze(1), 
            y=final_label[:, root_idx, :, :, :].unsqueeze(1)
        )
        print("confusion_matrix", confusion_matrix)

        print("logits", logits.shape)
        # Compute and log the precision recall curve
        # if logits.shape[1] == 2:
        #     preds = torch.softmax(logits, dim=1)[:, 1, :, :, :].unsqueeze(1)
        #     # TODO: add sigmoid case
        #     preds = preds.view(-1)
        #     targets = labels.view(-1)
        #     print("shaaape", preds.shape, targets.shape)
        #     pres_rec = BinaryPrecisionRecallCurve(thresholds=10).to(preds.device)
        #     pres_rec(
        #         preds, 
        #         targets.long(),
        #     )
        #     print("preds after content dtype", preds.dtype)
        #     print("targets after content dtype", targets.dtype)
        #     print("computing curve")
        #     precision, recall, thresholds = pres_rec.compute()
        #     print("curve computed")
        #     fig = self.plot_precision_recall_curve(precision, recall)
        #     self.logger.experiment.add_figure(f'Precision-Recall Curve/Batch {batch_idx}', fig, global_step=batch_idx)
        #     pres_rec.reset()

        # setup the background confusion matrix metrics for aggregated data for the epoch and step
        # background_cmm = self.background_confusion_matrix_metrics(
        #     y_pred=background_pred,
        #     y=final_label[:, 0, :, :, :].unsqueeze(1),
        # )

        # background_cmm_agg = self.background_confusion_matrix_metrics_agg(
        #     y_pred=background_pred,
        #     y=final_label[:, 0, :, :, :].unsqueeze(1),
        # )

        dice_metric = self.dice_metric(y_pred=binary_output, y=final_label)
        print("dice_metric", dice_metric)

        root_cmm_step = self.root_confusion_matrix_metrics.aggregate()
        # background_cmm_step = self.background_confusion_matrix_metrics.aggregate()

        model_dir = os.path.dirname(f"./test_model_output/{batch_idx}/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        visualizer = Visualizations()
        for slice in [0.3, 0.5, 0.6, 0.7]:
            visualizer.plot_row(
                batch["image"][0, 0, :, :, :].cpu(),
                logits[0, root_idx, :, :, :].cpu(),
                binary_output[0, root_idx, :, :, :].cpu(),
                slice_frac=slice,
                filename=f"test_model_output/{batch_idx}/prediction_{slice}",
                label=final_label[0, 1, :, :, :].cpu(),
            )

        # For inspection: save the predictions as nifti files
        # Convert to NumPy array

        save_dir = f"./test_model_output/{batch_idx}"

        self._save_mri(save_dir, "label", batch["label"][0][0].cpu().numpy())
        self._save_mri(save_dir, "output", logits[0][root_idx].cpu().numpy())
        self._save_mri(save_dir, "binary_output", binary_output[0][root_idx].cpu().numpy())
        self._save_mri(save_dir, "labels", labels[0][0].cpu().numpy())
        # self._save_mri(save_dir, "mask", mask[0][0].cpu().numpy())

        test_output = {
            # "Test/test_loss": loss,
            "Test/surface_distance": sd,
            "Test/root_precision": root_cmm_step[0] if not thresh else root_cmms[best_percentile][0],
            "Test/root_recall": root_cmm_step[1] if not thresh else root_cmms[best_percentile][1],
            # "Test/root_f1": root_cmm_step[2] if not thresh else root_cmms[best_percentile][2],
            "Test/dice": test_dice,
            # "Test/background_precision": background_cmm_step[0],
            # "Test/background_recall": background_cmm_step[1],
            # "Test/background_f1": background_cmm_step[2],
        }

        for metric_name, metric_value in test_output.items():
            print("logging test_step")
            self.log(
                metric_name,
                metric_value,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                # sync_dist=True  # Uncomment if you are training in a distributed setting and want to synchronize metrics
            )

        self.test_step_outputs.append(test_output)

        self.dice_metric.reset()
        self.surface_distance_metric.reset()
        self.root_confusion_matrix_metrics.reset()

        return test_output

    def plot_precision_recall_curve(self, precision, recall):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(recall.cpu(), precision.cpu(), marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)

        fig = plt.gcf()
        plt.close()
        return fig

    def plot_f1_scores(self, x_values, path):
        # Plotting the F1 scores
        plt.figure(figsize=(15, 6))
        
        i = 1
        for f1s in self.root_f1s_arr:
            plt.plot(x_values, f1s, marker='o', linestyle='-', label=f"MRI scan {i}")
            i = i+1
        plt.title('F1 Scores for Different Runs')
        plt.xlabel('Run', fontsize=8)
        plt.ylabel('F1 Score')
        plt.grid(True)
        plt.legend()
        plt.xticks(x_values, rotation=45)

        plt.savefig(path)

    def on_test_epoch_end(self):
        # 1. Gather Individual Metrics
        # mean_test_loss = torch.stack(
        #     [x["Test/test_loss"] for x in self.test_step_outputs]
        # )
        mean_test_loss = mean_test_loss.mean()
        mean_test_dice = self.dice_metric.aggregate().item()
        mean_test_sd = self.surface_distance_metric.aggregate().item()

        self.metric_values.append(mean_test_dice)

        root_cmm = self.root_confusion_matrix_metrics_agg.aggregate()

        print("root_cmm", root_cmm)
        # background_cmm = self.background_confusion_matrix_metrics_agg.aggregate()

        tensorboard_logs = {
            "Test/avg_test_dice": mean_test_dice,
            # "Test/avg_test_loss": mean_test_loss,
            "Test/avg_surface_distance": mean_test_sd,
            "Test/avg_root_precision": root_cmm[0],
            "Test/avg_root_recall": root_cmm[1],
            "Test/avg_root_f1": root_cmm[2],
            # "Test/avg_background_precision": background_cmm[0],
            # "Test/avg_background_recall": background_cmm[1],
            # "Test/avg_background_f1": background_cmm[2],
        }

        for metric_name, metric_value in tensorboard_logs.items():
            print("log", metric_name)
            self.log(
                metric_name,
                metric_value,
                # on_step=False,
                # on_epoch=True,
                # prog_bar=True,
                # logger=True,
                # sync_dist=True  # Uncomment if you are training in a distributed setting and want to synchronize metrics
            )

        # Clear validation step outputs and reset metrics for all processes
        self.test_step_outputs.clear()
        self.dice_metric.reset()
        self.root_confusion_matrix_metrics_agg.reset()
        # self.background_confusion_matrix_metrics_agg.reset()

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
            print("num_of_classes", num_of_classes)
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
            print("Applying class weight")
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
