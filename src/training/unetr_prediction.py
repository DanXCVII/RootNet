from typing import Tuple, Union

import sys
import os
import nibabel as nib
import numpy as np

sys.path.append("../")

import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
)
from unetr_monai import MyUNETRWrapper

import torch


class ImagePredictionPipeline:
    def __init__(self, checkpoint_path):
        self.prediction_transform = Compose(
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
        self.net = MyUNETRWrapper.load_from_checkpoint(checkpoint_path)
        self.net.eval()
        self.net.to(torch.device("cuda"))

    def _transform_data(self, prediction_data):
        """
        applies the prediction_transform to the given data

        Args:
        - prediction_data: list of dictionaries containing the image and label paths
        """
        return self.prediction_transform(prediction_data)

    def _get_output(self, img):
        """
        applies the model to the given image and returns the predictions

        Args:
        - img: image tensor
        """
        with torch.no_grad():
            test_input = img.cuda()
            val_outputs = sliding_window_inference(
                test_input, (128, 128, 128), 1, self.net, overlap=0.2
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
        
        binary_prediction = (val_outputs.detach().cpu() >= 0.5).int()
        print("binary_prediction shape", binary_prediction.shape)

        # plot binary prediction of NN
        plt.subplot(1, total_imges, 3)
        plt.title("binary output")
        plt.imshow(binary_prediction[0, 0, :, :, slice_num * 2])

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
        - img: image tensor
        - filename: filename to save the image
        """
        img = img.cpu().numpy()[0, 0, :, :, :]
        # img = img.astype("float32")
        # img = img.astype("int16")
        # img = img.transpose(2, 1, 0)
        affine_transformation = np.array(
            [
                [self.resz, 0.0, 0.0, 0.0],
                [0.0, self.resy, 0.0, 0.0],
                [0.0, 0.0, self.resx, 0.0],
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
        img.astype("int16").tofile(f"{filename}{img.shape[-1]}x{img.shape[-2]}x{img.shape[-3]}.raw")

    def predict_and_plot(self, prediction_data, slice_nums):
        """
        for given prediction data, does the necessary transformations, applys the model to do the prediction
        and plots the image, label, output and binary output for a given horizontal slice

        Args:
        - prediction_data: list of dictionaries containing the image and label paths
        """
        transformed_data = self._transform_data(prediction_data)
        print("len transform", len(transformed_data))
        print("len prediction", len(prediction_data))
        
        iterations = len(transformed_data)
        for i in range(0, iterations):
            print("iteration", i)
            filename = os.path.basename(prediction_data[i]["image"]).split("_res")[0]
            os.makedirs(f"example_predictions/{filename}", exist_ok=True)

            test_input = transformed_data[i]["image"].unsqueeze(1)
            test_labels = transformed_data[i]["label"].unsqueeze(1)

            val_outputs = self._get_output(test_input)
            binary_prediction = (val_outputs.detach().cpu() >= 0.5).int()

            # self.save_as_nifti(val_outputs, "example_predictions/prediction.nii.gz")
            
            self.save_as_raw(test_input, f"example_predictions/{filename}/input")
            self.save_as_raw(val_outputs, f"example_predictions/{filename}/prediction")
            self.save_as_raw(test_labels, f"example_predictions/{filename}/label")
            self.save_as_raw(binary_prediction, f"example_predictions/{filename}/binary_prediction")


            for slice_num in slice_nums:
                self._plot_prediction(
                    test_input,
                    val_outputs,
                    slice_num=slice_num,
                    label=test_labels,
                    filename=f"example_predictions/{filename}/prediction_{slice_num}",
                )


# Example usage:
prediction_data = [
    {
        "image": "../../data/generated/training/Bench_lupin/loam/sim_days_8-initial_-50-noise_0.6/Bench_lupin_day_8_SNR_3_res_229x229x201.nii.gz",
        "label": "../../data/generated/training/Bench_lupin/loam/sim_days_8-initial_-50-noise_0.6/label_Bench_lupin_day_8_SNR_3_res_458x458x402.nii.gz",
    },
    {
        "image": "../../data/generated/training/Bench_lupin/sand/sim_days_10-initial_-25-noise_0.9/Bench_lupin_day_10_SNR_3_res_229x229x201.nii.gz",
        "label": "../../data/generated/training/Bench_lupin/sand/sim_days_10-initial_-25-noise_0.9/label_Bench_lupin_day_10_SNR_3_res_458x458x402.nii.gz",
    }
    # {
    #     "image": "../../data/generated/test/Glycine_max_Moraes2020_opt2/clay/sim_days_5-initial_-600-noise_0.8/Glycine_max_Moraes2020_opt2_day_5_SNR_3_res_229x229x201.nii.gz",
    #     "label": "../../data/generated/test/Glycine_max_Moraes2020_opt2/clay/sim_days_5-initial_-600-noise_0.8/label_Glycine_max_Moraes2020_opt2_day_5_SNR_3_res_458x458x402.nii.gz",
    # }
]

pipeline = ImagePredictionPipeline("../../runs/best_metric_model_split.ckpt")

pipeline.predict_and_plot(prediction_data, [100, 50, 125, 150, 175, 200])
