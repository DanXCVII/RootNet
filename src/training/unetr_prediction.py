from typing import Tuple, Union

import sys
import os
import nibabel as nib
import numpy as np
import re
import json

sys.path.append("../")

import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference
from monai.transforms import (
    EnsureChannelFirstd,
    Resized,
    Compose,
    NormalizeIntensityd,
    LoadImaged,
    ScaleIntensityRanged,
)
from pl_setup import MyUNETRWrapper

import torch

"""
Description:    This script contains the pipeline for the prediction of MySwinUNETR etc.
                It loads the model from a checkpoint and applies it to the given data which
                is stored under prediction_data. It then plots the image, label, output and
                binary output for a given horizontal slice.
Usage:  Simply execute the script
Example: python unetr_prediction.py
"""

class ImagePredictionPipeline:
    def __init__(self, checkpoint_path, train_params):
        self.train_params = train_params
        self.prediction_transform = Compose(
            [
                LoadImaged(keys=["image", "label"], allow_missing_keys=True),
                EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
                # ScaleIntensityRanged(
                #     keys=["image"],
                #     a_min=0,
                #     a_max=30000,
                #     b_min=0.0,
                #     b_max=1.0,
                #     clip=True,
                # ),
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
                # Resized( # TODO: remove
                #     keys=["image"],
                #     spatial_size=(402, 458, 458),
                #     mode="trilinear",
                # ),
            ]
        )
        self.net = MyUNETRWrapper.load_from_checkpoint(
            checkpoint_path
        )
        self.net.eval()
        self.net.to(torch.device("cuda"))

    def _transform_data(self, prediction_data):
        """
        applies the prediction_transform to the given data

        Args:
        - prediction_data: list of dictionaries containing the image and label paths
        """
        return self.prediction_transform(prediction_data)

    def _get_output(self, img, img_shape):
        """
        applies the model to the given image and returns the predictions

        Args:
        - img: image tensor
        """
        with torch.no_grad():
            test_input = img.cuda()
            val_outputs = sliding_window_inference(
                test_input, img_shape, 1, self.net, overlap=0.2
            )
            return val_outputs

    def _plot_prediction(self, img, val_outputs, slice_num, filename, label=None):
        """
        plots the image, label, output and binary output for a given horizontal slice

        Args:
        - img: image tensor
        - label: label tensor
        - val_outputs: output tensor
        - slice_num: slice number to plot
        - filename: filename to save the plot
        """
        total_imges = 0
        if label is None:
            total_imges = 3
        else:
            total_imges = 4
        plt.figure("check", (18, 6))

        # plot data image
        plt.subplot(1, total_imges, 1)
        plt.title("image")
        plt.imshow(img.cpu().numpy()[0, 0, :, :, slice_num], cmap="gray")

        # plot prediction of NN
        plt.subplot(1, total_imges, 2)
        plt.title("output")
        plt.imshow(val_outputs.detach().cpu()[0, 0, :, :, slice_num * 2])

        # for softmax output
        binary_prediction = torch.argmax(val_outputs.detach().cpu(), dim=1)
        print("binary_prediction shape", binary_prediction.shape)
        # for sigmoid output
        # binary_prediction = (val_outputs.detach().cpu() >= 0.5).int()

        # plot binary prediction of NN
        plt.subplot(1, total_imges, 3)
        plt.title("binary output")
        plt.imshow(binary_prediction[0, :, :, slice_num * 2])

        # plot the ground truth if existent
        if label is not None:
            plt.subplot(1, total_imges, 4)
            plt.title("label")
            plt.imshow(label.cpu().numpy()[0, 0, :, :, slice_num * 2])

        plt.savefig(f"./{filename}.png")

    def save_as_nifti(self, img, filename):
        """
        saves the given image as nifti file

        Args:
        - img: image tensor or numpy array
        - filename: filename to save the image
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()[0, 0, :, :, :]
        # img = img.astype("float32")
        # img = img.astype("int16")
        # img = img.transpose(2, 1, 0)
        affine_transformation = np.array(
            [
                [0.1, 0.0, 0.0, 0.0],
                [0.0, 0.027, 0.0, 0.0],
                [0.0, 0.0, 0.027, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        img = nib.Nifti1Image(img, affine_transformation)

        nib.save(img, filename)

    def save_as_raw(self, img, filename):
        """
        saves the given image as raw file

        Args:
        - img: image tensor
        - filename: filename to save the image
        """
        img = img.cpu()[0, 0, :, :, :]
        print("img", img.dtype)
        img = img * 30000
        img = img.to(torch.int16)
        # img = img.transpose(2, 1, 0)
        img = img.numpy()
        img.astype("int16").tofile(
            f"{filename}{img.shape[-1]}x{img.shape[-2]}x{img.shape[-3]}.raw"
        )

    def _convert_images_to_nifti(self, prediction_data):
        """
        Converts the images in the prediction data to nifti files and saves them in the same directory
        as the raw files

        Args:
        - prediction_data: list of dictionaries containing the image and label paths
        """
        for i in range(0, len(prediction_data)):
            if prediction_data[i]["image"].endswith(".raw"):
                filename = prediction_data[i]["image"].split(".raw")[0]
                print("filename", filename)
                numbers = re.findall(r"\d+", filename)
                last_three_numbers = numbers[-3:]
                # read the raw file
                img = np.fromfile(prediction_data[i]["image"], dtype=np.int16)
                # reshape the image
                img = img.reshape(
                    int(last_three_numbers[2]),
                    int(last_three_numbers[1]),
                    int(last_three_numbers[0]),
                )

                self.save_as_nifti(img, f"{filename}.nii.gz")
                prediction_data[i]["image"] = f"{filename}.nii.gz"

        return prediction_data

    def predict_and_plot(self, prediction_data, slice_nums, plot_path, img_shape):
        """
        for given prediction data, does the necessary transformations, applys the model to do the prediction
        and plots the image, label, output and binary output for a given horizontal slice

        Args:
        - prediction_data: list of dictionaries containing the image and label paths
        """
        prediction_data = self._convert_images_to_nifti(prediction_data)

        transformed_data = self._transform_data(prediction_data)
        print("len transform", len(transformed_data))
        print("len prediction", len(prediction_data))

        iterations = len(transformed_data)
        for i in range(0, iterations):
            filename = os.path.basename(prediction_data[i]["image"]).split("_res")[0]
            os.makedirs(f"{plot_path}/{filename}", exist_ok=True)

            test_input = transformed_data[i]["image"].unsqueeze(1)
            if "label" in transformed_data[i]:
                test_labels = transformed_data[i]["label"].unsqueeze(1)
            else:
                test_labels = None

            val_outputs = self._get_output(test_input, img_shape)
            binary_prediction = (val_outputs.detach().cpu() >= 0.5).int()

            # self.save_as_nifti(val_outputs, "example_predictions/prediction.nii.gz")

            self.save_as_raw(test_input, f"{plot_path}/{filename}/input")
            self.save_as_raw(val_outputs, f"{plot_path}/{filename}/prediction")
            if "label" in transformed_data[i]:
                self.save_as_raw(test_labels, f"{plot_path}/{filename}/label")
            self.save_as_raw(
                binary_prediction, f"{plot_path}/{filename}/binary_prediction"
            )

            for slice_num in slice_nums:
                self._plot_prediction(
                    test_input,
                    val_outputs,
                    slice_num=slice_num,
                    label=test_labels,
                    filename=f"{plot_path}/{filename}/prediction_{slice_num}",
                )


# Example usage:
prediction_data = [
    {
        "image": "../../data/generated/validation/Crypsis_aculeata_Clausnitzer_1994/clay/sim_days_5-initial_-900-noise_0.9/Crypsis_aculeata_Clausnitzer_1994_day_5_SNR_3_res_229x229x201.nii.gz",
        "label": "../../data/generated/validation/Crypsis_aculeata_Clausnitzer_1994/clay/sim_days_5-initial_-900-noise_0.9/label_Crypsis_aculeata_Clausnitzer_1994_day_5_SNR_3_res_459x459x403.nii.gz",
    },
    {
        "image": "../../data/generated/validation/Glycine_max/sand/sim_days_5-initial_-25-noise_0.8/Glycine_max_day_5_SNR_3_res_229x229x201.nii.gz",
        "label": "../../data/generated/validation/Glycine_max/sand/sim_days_5-initial_-25-noise_0.8/label_Glycine_max_day_5_SNR_3_res_459x459x403.nii.gz",
    },
    {"image": "../../data_old/real/III_Sand_1W_DAP14_256x256x131.nii.gz"},
    {"image": "../../data_old/real/III_Soil_1W_DAP14_256x256x186.nii.gz"},
    # {
    #     "image": "../../data/generated/test/Glycine_max_Moraes2020_opt2/clay/sim_days_5-initial_-600-noise_0.8/Glycine_max_Moraes2020_opt2_day_5_SNR_3_res_229x229x201.nii.gz",
    #     "label": "../../data/generated/test/Glycine_max_Moraes2020_opt2/clay/sim_days_5-initial_-600-noise_0.8/label_Glycine_max_Moraes2020_opt2_day_5_SNR_3_res_458x458x402.nii.gz",
    # }
]
# TODO: remove *2 for the plot input and the Resized

model_name = "weight_1.0-1.03_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0008-upsample_end_False"
checkpoint_dir = f"../../runs/{model_name}/"
# load the model params from the checkpoint dir (../../{model_name}/train_params.json) and store them in a dictionary
with open(f"{checkpoint_dir}train_params.json") as f:
    train_params = json.load(f)

pipeline = ImagePredictionPipeline(f"{checkpoint_dir}best_metric_model.ckpt", train_params)


img_shape = train_params["img_shape"]
plot_path = f"example_predictions/{model_name}"
pipeline.predict_and_plot(
    prediction_data, [100, 80, 75, 50, 125, 200], plot_path, img_shape
)
