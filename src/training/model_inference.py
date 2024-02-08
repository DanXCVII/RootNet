import sys
import os
import json

sys.path.append("../")

import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference
from pl_setup import MyUNETRWrapper
from mri_dataloader import MRIDataLoader
from utils import Visualizations
from utils import MRIoperations

import torch

"""
Description:    This script contains the pipeline for the prediction of MyUpscaleSwinUNETR etc.
                It loads the model from a checkpoint and applies it to the given data which
                is stored under prediction_data. It then plots the image, label, output and
                binary output for a given horizontal slice.
Usage:  Simply execute the script
Example: python unetr_prediction.py
"""


class ImagePredictionPipeline:
    def __init__(self, checkpoint_path, train_params):
        self.train_params = train_params

        my_dl = MRIDataLoader(
            "../../data", 1, True, 1, (96, 96, 96), allow_missing_keys=True
        )
        self.prediction_transform = my_dl.cache_transforms

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

    def _get_output(self, img, img_shape):
        """
        applies the model to the given image and returns the predictions

        Args:
        - img: image tensor
        - img_shape: patch size for the sliding window inference
        """
        with torch.no_grad():
            test_input = img.cuda()
            val_outputs = sliding_window_inference(
                test_input, img_shape, 1, self.net, overlap=0.2
            )
            return val_outputs

    def _plot_prediction(self, img, val_outputs, slice_frac, filename, label=None):
        """
        plots the image, label, output and binary output for a given horizontal slice

        Args:
        - img: image tensor
        - label: label tensor
        - val_outputs: output tensor
        - slice_frac: fraction of the image, where the slice should be taken
        - filename: filename to save the plot
        """
        total_imges = 0
        if label is None:
            total_imges = 3
        else:
            total_imges = 4
        plt.figure("check", (18, 6))

        img_shape = img.shape
        slice_num = int(img_shape[-1] * slice_frac)

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
                # read the raw file
                _, img = MRIoperations().load_mri(prediction_data[i]["image"])

                MRIoperations().save_mri(f"{filename}.nii.gz", img)
                prediction_data[i]["image"] = f"{filename}.nii.gz"

        return prediction_data

    def predict_and_plot(self, prediction_data, slices, plot_path, img_shape):
        """
        for given prediction data, does the necessary transformations, applys the model to do the prediction
        and plots the image, label, output and binary output for a given horizontal slice

        Args:
        - prediction_data: list of dictionaries containing the image and label paths
        - slices: list of fractions of the image, for which slices should be plotted
        - plot_path: path to save the plots
        - img_shape: patch size for the sliding window inference
        """
        prediction_data = self._convert_images_to_nifti(prediction_data)

        transformed_data = self._transform_data(prediction_data)

        iterations = len(transformed_data)
        for i in range(0, iterations):
            filename = os.path.basename(prediction_data[i]["image"]).split("_res")[0]
            os.makedirs(f"{plot_path}/{filename}", exist_ok=True)

            # add a batch dimension to the image tensor to make it a valid input for the model
            test_input = transformed_data[i]["image"].unsqueeze(1)
            if "label" in transformed_data[i]:
                test_labels = transformed_data[i]["label"].unsqueeze(1)
            else:
                test_labels = None

            val_outputs = self._get_output(test_input, img_shape)
            binary_prediction = (val_outputs.detach().cpu() >= 0.5).int()

            # save the images as nifti files (multiplying with 30000 to get integer values which can
            # be displayed in ImageJ, also roughly the range of an original MRI)
            ops = MRIoperations()
            numpy_input = test_input[0, 0, :, :, :].cpu().numpy() * 30000
            numpy_outputs = val_outputs[0, 0, :, :, :].cpu().numpy() * 30000
            numpy_binary_prediction = (
                binary_prediction[0, :, :, :].cpu().numpy() * 30000
            )

            if "label" in transformed_data[i]:
                numpy_labels = test_labels[0, 0, :, :, :].cpu().numpy() * 30000
                ops.save_mri(f"{plot_path}/{filename}/label.nii.gz", numpy_labels)

            ops.save_mri(f"{plot_path}/{filename}/input.nii.gz", numpy_input)
            ops.save_mri(f"{plot_path}/{filename}/prediction.nii.gz", numpy_outputs)
            ops.save_mri(
                f"{plot_path}/{filename}/binary_prediction.nii.gz",
                numpy_binary_prediction,
            )

            # Depending on the activation function (softmax or sigmoid) tht probability of the root
            # being present is either the first or the second channel
            if val_outputs.shape[1] == 2:
                binary_output = torch.argmax(val_outputs.detach().cpu(), dim=1)
            else:
                binary_output = (val_outputs.detach().cpu() >= 0.5).int()

            # plot different slices of the image
            for slice_frac in slices:
                Visualizations().plot_row(
                    test_input[0, 0, :, :, :],
                    val_outputs[0, 0, :, :, :],
                    binary_output=binary_output[0, :, :, :],
                    slice_frac=slice_frac,
                    label=(
                        test_labels[0, 0, :, :, :] if test_labels is not None else None
                    ),
                    filename=f"{plot_path}/{filename}/prediction_{slice_frac}",
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
    {"image": "../../data/real/III_Sand_1W_DAP14_res_256x256x131.nii.gz"},
    {"image": "../../data/real/III_Sand_3D_DAP14_res_256x256x191.nii.gz"},
    {"image": "../../data/real/III_Soil_3D_DAP15_res_256x256x199.nii.gz"},
    {"image": "../../data/real/IV_Sand_3D_DAP8_res_256x256x192.nii.gz"},
    {"image": "../../data/real/IV_Soil_1W_DAP9_res_256x256x136.nii.gz"},
    {"image": "../../data/real/IV_Soil_3D_DAP8_res_256x256x193.nii.gz"},
    {"image": "../../data/real/III_Soil_1W_DAP14_res_256x256x186.nii.gz"},
]
# TODO: remove *2 for the plot input and the Resized

model_name = "weight_1_0.6_Data_DICE_softmax_UPSCALESWINUNETR-img_shape_96-feat_36-upscale_True-out_channels_2-lr_0.0008-upsample_end_False"
checkpoint_dir = f"../../runs/{model_name}/"

# load the model params from the checkpoint dir (../../{model_name}/train_params.json) and store them in a dictionary
with open(f"{checkpoint_dir}train_params.json") as f:
    train_params = json.load(f)

pipeline = ImagePredictionPipeline(
    f"{checkpoint_dir}best_metric_model.ckpt", train_params
)


img_shape = train_params["img_shape"]
plot_path = f"example_predictions/{model_name}"
pipeline.predict_and_plot(
    prediction_data,
    [0.1, 0.3, 0.5, 0.6, 0.7],
    plot_path,
    img_shape,
)
