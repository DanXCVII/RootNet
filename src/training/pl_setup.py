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
    DiceLoss,
)
from mri_dataloader import MRIDataLoader
from models import MyUpscaleSwinUNETR, UNet

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
    UNET = 1
    UPSCALESWINUNETR = 2


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
        Pytorch Lightning Module used for training and testing of the provided DNN model

        Args:
        - learning_rate: learning rate for the optimizer
        - model: the model to train
        - model_params: the parameters of the DNN
        - patch_size: the size of the patches
        - class_weight: the class weights
        - output_activation: the output activation function (SIGMOID or SOFTMAX)
        """

        super().__init__()
        self.save_hyperparameters()

        if model is None:
            raise ValueError("model cannot be None")
        elif model == ModelType.UNET.name:
            self.model = UNet(**model_params, patch_size=patch_size)
        elif model == ModelType.UPSCALESWINUNETR.name:
            self.model = MyUpscaleSwinUNETR(**model_params, patch_size=patch_size)
        else:
            raise ValueError("Model must be of type ModelType.")

        # Loss function and metrics
        if class_weight == [0, 1]:
            include_background = False
            print("include background:", include_background)
        else:
            include_background = True
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
            include_background=False,
            reduction="mean",
            get_not_nans=False,
            percentile=95,
        )
        self.surface_distance_metric = SurfaceDistanceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False,
            symmetric=True,
        )
        self.root_confusion_matrix_metrics = ConfusionMatrixMetric(
            include_background=False,
            metric_name=("precision", "recall", "f1 score"),
            reduction="mean",
            get_not_nans=False,
        )
        self.background_confusion_matrix_metrics = ConfusionMatrixMetric(
            include_background=False,
            metric_name=("precision", "recall", "f1 score"),
            reduction="mean",
            get_not_nans=False,
        )
        self.root_confusion_matrix_metrics_agg = ConfusionMatrixMetric(
            include_background=False,
            metric_name=("precision", "recall", "f1 score"),
            reduction="mean",
            get_not_nans=False,
        )
        self.background_confusion_matrix_metrics_agg = ConfusionMatrixMetric(
            include_background=False,
            metric_name=("precision", "recall", "f1 score"),
            reduction="mean",
            get_not_nans=False,
        )
        self.dice_metric = DiceMetric(
            include_background=False,
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
        self.lr_scheduler = None
        self.my_current_epoch = 0

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

        print("configure optimizers")
        print("current epoch optim", self.my_current_epoch)
        self.lr_scheduler = ChainedScheduler(
            optimizer,
            T_0=6,
            T_mul=1,
            eta_min=1e-5,
            gamma=0.8,
            max_lr=self.learning_rate,
            warmup_steps=5,
            last_epoch=self.my_current_epoch,
        )

        # return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": self.lr_scheduler, "interval": "epoch"},
        }

    def lr_schedulers(self):
        return self.lr_scheduler

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

            print("learning rate: ", self.trainer.optimizers[0].param_groups[0]["lr"])

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
                )

                self.log(
                    "Train/learning_rate",
                    self.trainer.optimizers[0].param_groups[0]["lr"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )

                # Clear training step outputs
                self.training_step_outputs.clear()

    def _get_soil_root_prob(self, logits, labels):
        """
        Calculates multiple segmentation metrics for the soil and root predictions and labels

        Args:
        - logits: the model output
        - labels: the ground truth labels

        Returns:
        - post_pred: the probability of each voxel belonging to the root
        - binary_output: the binary output of the model
        - final_label: the post processed labels
        - background_pred: the background prediction
        - root_idx: the index of the root in the label (1 for softmax, 0 for sigmoid)
        """
        if self.output_activation == OutputActivation.SIGMOID.name:
            post_pred = torch.sigmoid(logits)
            binary_output = (post_pred >= 0.5).int()

            final_label = torch.stack(
                [self.post_label(i) for i in decollate_batch(labels)]
            )
            root_idx = 0

            # the background probability is the inverse of the root probability
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
            :,
            :,
            : images.shape[2] * 2,
            : images.shape[3] * 2,
            : images.shape[4] * 2,
        ]

        logits = sliding_window_inference(
            images,
            self.patch_size,
            sw_batch_size,
            self.forward,
        )

        loss = self.loss_function(logits, labels)

        _, binary_output, final_label, background_pred, root_idx = (
            self._get_soil_root_prob(logits, labels)
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
            )

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

        mri_full_path = (
            f"{mri_path}/{name}_{mri_shape[0]}x{mri_shape[1]}x{mri_shape[2]}.nii.gz"
        )

        mri_ops.save_mri(mri_full_path, mri_data)

    def test_step(self, batch, batch_idx):
        thresh = True
        visualize = False
        save_segmentations = False

        images, labels = batch["image"], batch["label"]
        sw_batch_size = 2

        labels = labels[
            :,
            :,
            : images.shape[2] * 2,
            : images.shape[3] * 2,
            : images.shape[4] * 2,
        ]

        logits = sliding_window_inference(
            images,
            self.patch_size,
            sw_batch_size,
            self.forward,
        )

        _, binary_output, final_label, _, root_idx = self._get_soil_root_prob(
            logits, labels
        )

        # apply thresholding
        if thresh:
            best_root_cmm, binary_output = self._apply_thresholding(
                batch["image"],
                final_label,
                root_idx,
                batch_idx,
                min_tresh=0.9975,
                max_thresh=0.9999,
                step_size=0.00005,
            )

        ###################### Metrics #######################

        sd = self.surface_distance_metric(
            y_pred=binary_output[:, root_idx, :, :, :].unsqueeze(1),
            y=final_label[:, root_idx, :, :, :].unsqueeze(1),
        )

        self.dice_metric(
            y_pred=binary_output[:, root_idx, :, :, :].unsqueeze(1),
            y=final_label[:, root_idx, :, :, :].unsqueeze(1),
        )
        test_dice = self.dice_metric.aggregate().item()

        self.root_confusion_matrix_metrics(
            y_pred=binary_output[:, root_idx, :, :, :].unsqueeze(1),
            y=final_label[:, root_idx, :, :, :].unsqueeze(1),
        )

        self.root_confusion_matrix_metrics_agg(
            y_pred=binary_output[:, root_idx, :, :, :].unsqueeze(1),
            y=final_label[:, root_idx, :, :, :].unsqueeze(1),
        )

        confusion_matrix = get_confusion_matrix(
            y_pred=binary_output[:, root_idx, :, :, :].unsqueeze(1),
            y=final_label[:, root_idx, :, :, :].unsqueeze(1),
        )

        root_cmm_step = self.root_confusion_matrix_metrics.aggregate()
        # background_cmm_step = self.background_confusion_matrix_metrics.aggregate()

        ########### Further Result Inspections ############

        if visualize:
            # visualize some example slices of the segmentations
            # slices to visualize: e.g. 0.3 stands for the slice at 30% of the image vertically
            visualized_slices = [0.3, 0.5, 0.6, 0.7]
            model_dir = os.path.dirname(f"./test_output/{batch_idx}/")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            visualizer = Visualizations()
            for slice in visualized_slices:
                visualizer.plot_row(
                    batch["image"][0, 0, :, :, :].cpu(),
                    logits[0, root_idx, :, :, :].cpu(),
                    binary_output[0, root_idx, :, :, :].cpu(),
                    slice_frac=slice,
                    filename=f"test_output/{batch_idx}/prediction_{slice}",
                    label=final_label[0, 1, :, :, :].cpu(),
                )
        if save_segmentations:
            # saves the segmented images for further inspection
            post_pred = torch.softmax(logits, dim=1)

            save_dir = f"./test_output/{batch_idx}"

            self._save_mri(save_dir, "label", batch["label"][0][0].cpu().numpy())
            self._save_mri(save_dir, "root_prob", post_pred[0][root_idx].cpu().numpy())
            self._save_mri(
                save_dir, "binary_output", binary_output[0][root_idx].cpu().numpy()
            )

        test_metrics = {
            "Test/surface_distance": sd,
            "Test/root_precision": (
                root_cmm_step[0] if not thresh else best_root_cmm[0]
            ),
            "Test/root_recall": (root_cmm_step[1] if not thresh else best_root_cmm[1]),
            "Test/root_f1": (root_cmm_step[2] if not thresh else best_root_cmm[2]),
            "Test/dice": test_dice,
            # "Test/background_precision": background_cmm_step[0],
            # "Test/background_recall": background_cmm_step[1],
            # "Test/background_f1": background_cmm_step[2],
        }

        for metric_name, metric_value in test_metrics.items():
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

        self.test_step_outputs.append(test_metrics)

        self.dice_metric.reset()
        self.surface_distance_metric.reset()
        self.root_confusion_matrix_metrics.reset()

        return test_metrics

    def _apply_thresholding(
        self,
        image,
        label,
        root_idx,
        batch_idx,
        min_tresh,
        max_thresh,
        step_size,
    ):
        """
        Iterates through a range of percentiles to find the best threshold for the root prediction

        Args:
        - image: the MRI image
        - label: the ground truth label
        - root_idx: the index of the root in the label (1 for softmax, 0 for sigmoid)
        - batch_idx: the index of the batch
        - min_thresh: the minimum threshold value (inclusive)
        - max_thresh: the maximum threshold value (inclusive)
        - step_size: the step size for the thresholding

        Returns:
        - best_root_cmm: the confusion matrix metrics for the best threshold (regarding root f1 score)
        - binary_output: the binary output of the model
        """
        root_cmms = []
        best_percentile = 0

        # start timer
        min_thresh = 0.9975
        steps = round(((max_thresh - min_tresh) / step_size) + 2)

        binary_output_arr = []

        # iterate over the number of different thresholds
        for i in range(steps):
            # threshold the image and upscale the mask for the root prediction
            percentile = torch.quantile(image, min_thresh + i * step_size)
            mask = (image > percentile).float()
            mask_root_upscaled = F.interpolate(
                mask,
                size=(mask.shape[2] * 2, mask.shape[3] * 2, mask.shape[4] * 2),
                mode="trilinear",
                align_corners=False,
            )
            mask_root_upscaled = (mask_root_upscaled > 0).float()

            # calculate the inverse of the root mask for the background prediction and combine both
            inverse_bg_upscaled = 1 - mask_root_upscaled
            mask_combined = torch.cat((inverse_bg_upscaled, mask_root_upscaled), dim=1)

            if self.output_activation == OutputActivation.SIGMOID.name:
                binary_output = mask_root_upscaled
            elif self.output_activation == OutputActivation.SOFTMAX.name:
                binary_output = mask_combined
            else:
                raise ValueError("Output activation must be of type OutputActivation.")

            binary_output_arr.append(binary_output.cpu().numpy())

            # calculate the confusion matrix metrics for the thresholding
            self.root_confusion_matrix_metrics(
                y_pred=binary_output[:, root_idx, :, :, :].unsqueeze(1),
                y=label[:, root_idx, :, :, :].unsqueeze(1),
            )
            root_cmms.append(self.root_confusion_matrix_metrics.aggregate())
            self.root_confusion_matrix_metrics.reset()

        # create a list with only the f1 scores and find the best performing threshold percentile
        root_f1s = np.array([array[2].cpu().numpy() for array in root_cmms])
        self.root_f1s_arr.append(root_f1s)
        best_percentile = np.argmax(root_f1s)

        thresh_percentiles = [min_thresh + i * step_size for i in range(steps)]

        # save the f1 scores and percentile values to a csv file
        combined_data = np.column_stack((thresh_percentiles, root_f1s))
        np.savetxt(
            f"./threshold_output/x_value_dice_{batch_idx}.txt",
            combined_data,
            delimiter=",",
            header="x_value,dice",
            comments="",
        )

        self._plot_f1_scores(
            thresh_percentiles,
            f"./threshold_output/f1_scores_{batch_idx}.png",
        )

        # apply the best threshold to the image
        binary_output = torch.where(
            mask_root_upscaled > 0, torch.ones_like(binary_output), binary_output
        )

        binary_output = binary_output.to("cuda")

        return root_cmms[best_percentile], binary_output

    def _plot_f1_scores(self, x_values, path):
        # Plotting the F1 scores
        plt.figure(figsize=(15, 6))

        i = 1
        for f1s in self.root_f1s_arr:
            plt.plot(x_values, f1s, marker="o", linestyle="-", label=f"MRI scan {i}")
            i = i + 1
        plt.title("F1 Scores for Different Runs")
        plt.xlabel("Run", fontsize=8)
        plt.ylabel("F1 Score")
        plt.grid(True)
        plt.legend()
        plt.xticks(x_values, rotation=45)

        plt.savefig(path)

    def on_test_epoch_end(self):
        surface_distances = [metrics['Test/surface_distance'].cpu() for metrics in self.test_step_outputs]
        
        mean_test_sd = torch.stack(surface_distances).mean().item()

        self.metric_values.append(mean_test_dice)

        root_cmm = self.root_confusion_matrix_metrics_agg.aggregate()

        print("root_cmm", root_cmm)
        # background_cmm = self.background_confusion_matrix_metrics_agg.aggregate()

        tensorboard_logs = {
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
            )

        # Clear validation step outputs and reset metrics for all processes
        self.test_step_outputs.clear()
        self.dice_metric.reset()
        self.root_confusion_matrix_metrics_agg.reset()

        # Return the logs only from the main
        return {"log": tensorboard_logs}

