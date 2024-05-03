import sys
import os
import json

sys.path.append("../")

import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference
from pl_setup import MyPlSetup
from mri_dataloader import MRIDataLoader
from utils import Visualizations
from utils import MRIoperations

import torch
import argparse

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

        self.net = MyPlSetup.load_from_checkpoint(checkpoint_path)
        self.net.eval()
        self.net.to(torch.device("cuda"))

    def _transform_data(self, prediction_data):
        """
        applies the prediction_transform to the given data

        Args:
        - prediction_data: list of dictionaries containing the image and label paths
        """
        return self.prediction_transform(prediction_data)

    def _get_output(self, img, patch_size):
        """
        applies the model to the given image and returns the predictions

        Args:
        - img: image tensor
        - patch_size: patch size for the sliding window inference
        """
        with torch.no_grad():
            test_input = img.cuda()
            logits = sliding_window_inference(
                test_input, patch_size, 1, self.net, overlap=0.2
            )
            return logits

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

    def predict_and_plot(self, prediction_data, slices, plot_path, patch_size):
        """
        for given prediction data, does the necessary transformations, applys the model to do the prediction
        and plots the image, label, output and binary output for a given horizontal slice

        Args:
        - prediction_data: list of dictionaries containing the image and label paths
        - slices: list of fractions of the image, for which slices should be plotted
        - plot_path: path to save the plots
        - patch_size: patch size for the sliding window inference
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

            logits = self._get_output(test_input, patch_size)

            # apply sigmoid to the output to get the binary prediction

            print("logits min", logits.min())
            print("logits max", logits.max())
            print("logits", logits.shape)

            if logits.shape[1] == 2:
                post_activation = torch.softmax(logits, dim=1)
                binary_output = torch.argmax(post_activation.detach().cpu(), dim=1).unsqueeze(1)
                print("binary_output after", binary_output.shape)
                root_idx = 1
            else:
                post_activation = torch.sigmoid(logits)
                binary_output = (post_activation.detach().cpu() >= 0.5).int()
                root_idx = 0


            # save the images as nifti files (multiplying with 30000 to get integer values which can
            # be displayed in ImageJ, also roughly the range of an original MRI)

            ops = MRIoperations()
            numpy_input = test_input[0, 0, :, :, :].cpu().numpy() * 30000
            numpy_outputs = (
                post_activation[0, root_idx, :, :, :].cpu().numpy() * 30000
            )
            numpy_binary_prediction = binary_output[0, 0, :, :, :].cpu().numpy()
            print("numpy_binary_prediction", numpy_binary_prediction.dtype)

            # Depending on the activation function (softmax or sigmoid) tht probability of the root
            # being present is either the first or the second channel

            if "label" in transformed_data[i]:
                numpy_labels = test_labels[0, 0, :, :, :].cpu().numpy() * 30000
                ops.save_mri(f"{plot_path}/{filename}/label_{numpy_labels.shape[0]}x{numpy_labels.shape[1]}x{numpy_labels.shape[2]}.nii.gz", numpy_labels)

            numpy_binary_prediction = numpy_binary_prediction.astype("float32")

            ops.save_mri(f"{plot_path}/{filename}/input_{numpy_input.shape[0]}x{numpy_input.shape[1]}x{numpy_input.shape[2]}.nii.gz", numpy_input)
            ops.save_mri(
                f"{plot_path}/{filename}/prediction_scaled_{numpy_outputs.shape[0]}x{numpy_outputs.shape[1]}x{numpy_outputs.shape[2]}.nii.gz",
                numpy_outputs,
            )
            ops.save_mri(
                f"{plot_path}/{filename}/binary_prediction_{numpy_binary_prediction.shape[0]}x{numpy_binary_prediction.shape[1]}x{numpy_binary_prediction.shape[2]}.nii.gz",
                numpy_binary_prediction,
            )

            # plot different slices of the image
            for slice_frac in slices:
                Visualizations().plot_row(
                    test_input[0, 0, :, :, :],
                    post_activation[0, root_idx, :, :, :],
                    binary_output=binary_output[0, 0, :, :, :],
                    slice_frac=slice_frac,
                    label=(
                        test_labels[0, 0, :, :, :] if test_labels is not None else None
                    ),
                    filename=f"{plot_path}/{filename}/prediction_{slice_frac}",
                )

def main(model_name):
    prediction_data = [
        # {
        #     "image": "../../data_new/generated/validation/Crypsis_aculeata_Clausnitzer_1994/clay/sim_days_5-initial_-900-noise_0.9/Crypsis_aculeata_Clausnitzer_1994_day_5_SNR_3_res_229x229x201.nii.gz",
        #     "label": "../../data_new/generated/validation/Crypsis_aculeata_Clausnitzer_1994/clay/sim_days_5-initial_-900-noise_0.9/label_Crypsis_aculeata_Clausnitzer_1994_day_5_SNR_3_res_459x459x403.nii.gz",
        # },
        # {
        #     "image": "../../data_new/generated/validation/Glycine_max/sand/sim_days_5-initial_-25-noise_0.8/Glycine_max_day_5_SNR_3_res_229x229x201.nii.gz",
        #     "label": "../../data_new/generated/validation/Glycine_max/sand/sim_days_5-initial_-25-noise_0.8/label_Glycine_max_day_5_SNR_3_res_459x459x403.nii.gz",
        # },
        {
            "image": "../../data_new/real/III_Sand_1W_DAP14_res_256x256x131.nii.gz",
            "label": "../../data_new/real/label_III_Sand_1W_DAP14_res_512x512x262.nii.gz"
        },
        {
            "image": "../../data_new/real/III_Sand_3D_DAP14_res_256x256x191.nii.gz",
            "label": "../../data_new/real/label_III_Sand_3D_DAP14_res_512x512x382.nii.gz"
        },
        # {
        #     "image": "../../data_new/real/III_Soil_3D_DAP15_res_256x256x199.nii.gz",
        #     "label": "../../data_new/real/label_III_Soil_3D_DAP15_res_512x512x398.nii.gz"
        # },
        # {
        #     "image": "../../data_new/real/IV_Sand_3D_DAP8_res_256x256x192.nii.gz",
        #     "label": "../../data_new/real/label_IV_Sand_3D_DAP8_res_512x512x384.nii.gz"
        # },
        # {
        #     "image": "../../data_new/real/IV_Soil_1W_DAP9_res_256x256x136.nii.gz",
        #     "label": "../../data_new/real/label_IV_Soil_1W_DAP9_res_512x512x272.nii.gz"
        # },
        # {
        #     "image": "../../data_new/real/IV_Soil_3D_DAP8_res_256x256x193.nii.gz",
        #     "label": "../../data_new/real/label_IV_Soil_3D_DAP8_res_512x512x386.nii.gz"
        # },
        # {
        #     "image": "../../data_new/real/III_Soil_1W_DAP14_res_256x256x186.nii.gz",
        #     "label": "../../data_new/real/label_III_Soil_1W_DAP14_res_512x512x372.nii.gz"
        # },
    ]

    checkpoint_dir = f"../../runs/{model_name}/"

    # load the model params from the checkpoint dir (../../{model_name}/train_params.json) and store them in a dictionary
    with open(f"{checkpoint_dir}train_params.json") as f:
        train_params = json.load(f)

    pipeline = ImagePredictionPipeline(
        f"{checkpoint_dir}best_metric_model.ckpt", train_params
    )

    patch_size = train_params["patch_size"]
    plot_path = f"example_predictions/{model_name}"

    pipeline.predict_and_plot(
        prediction_data,
        [0.1, 0.3, 0.5, 0.6, 0.7],
        plot_path,
        patch_size,
    )


# Example usage:

# TODO: remove *2 for the plot input and the Resized

if __name__ == "__main__":
    print("Test")
    # Create the parser
    parser = argparse.ArgumentParser(description="Test run script")

    # Add arguments
    parser.add_argument("--model_name", "-m", type=str, help="The name of the model in the run directory inside the RootNet/runs folder")
    
    # Parse the arguments
    args = parser.parse_args()

    main(
        args.model_name
    )