from typing import Tuple, Union

import sys
import os

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
                        
        plt.savefig(f"./example_predictions/{filename}.png")

    def predict_and_plot(self, prediction_data, slice_nums):
        """
        for given prediction data, does the necessary transformations, applys the model to do the prediction
        and plots the image, label, output and binary output for a given horizontal slice

        Args:
        - prediction_data: list of dictionaries containing the image and label paths
        """
        transformed_data = self._transform_data(prediction_data)
        test_input = transformed_data[0]["image"]
        test_input = torch.unsqueeze(test_input, 1)
        test_labels = transformed_data[0]["label"]
        test_labels = torch.unsqueeze(test_labels, 1)

        val_outputs = self._get_output(test_input)

        for slice_num in slice_nums:
            self._plot_prediction(
                test_input,
                val_outputs,
                slice_num=slice_num,
                label=test_labels,
                filename=f"prediction_{slice_num}",
                
            )


# Example usage:
prediction_data = [
    {
        "image": "../../data/generated/test/Glycine_max_Moraes2020_opt2/clay/sim_days_5-initial_-600-noise_0.8/Glycine_max_Moraes2020_opt2_day_5_SNR_3_res_229x229x201.nii.gz",
        "label": "../../data/generated/test/Glycine_max_Moraes2020_opt2/clay/sim_days_5-initial_-600-noise_0.8/label_Glycine_max_Moraes2020_opt2_day_5_SNR_3_res_458x458x402.nii.gz",
    }
]

pipeline = ImagePredictionPipeline("../../runs/best_metric_model-v6.ckpt")

pipeline.predict_and_plot(prediction_data, [100, 50, 125, 150, 175, 200])
