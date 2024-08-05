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
    def __init__(self, checkpoint_path, train_params, patch_size):
        self.train_params = train_params

        my_dl = MRIDataLoader(
            "../../data", 
            1, 
            True, 
            1, 
            patch_size, 
            allow_missing_keys=True,
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

    def _predict(self, img, patch_size):
        """
        applies the model to the given image and returns the predictions

        Args:
        - img: image tensor
        - patch_size: patch size for the sliding window inference
        """
        with torch.no_grad():
            prediction_img = img.cuda()
            logits = sliding_window_inference(
                prediction_img, patch_size, 1, self.net, overlap=0.2
            )

            # Depending on the activation function (softmax or sigmoid) the probability of the root
            # being present is either the first or the second channel
            if logits.shape[1] == 2:
                post_activation = torch.softmax(logits, dim=1)
                binary_output = torch.argmax(post_activation.detach().cpu(), dim=1).unsqueeze(1)
                root_idx = 1
            else:
                post_activation = torch.sigmoid(logits)
                binary_output = (post_activation.detach().cpu() >= 0.5).int()
                root_idx = 0

            return post_activation, binary_output

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
            # create the directory for storing the plots
            filename = os.path.basename(prediction_data[i]["image"]).split("_res")[0]
            os.makedirs(f"{plot_path}/{filename}", exist_ok=True)

            # add a batch dimension to the image tensor to make it a valid input for the model
            prediction_img = transformed_data[i]["image"].unsqueeze(1)
            if "label" in transformed_data[i]:
                prediction_label = transformed_data[i]["label"].unsqueeze(1)
            else:
                prediction_label = None

            post_activation, binary_output = self._predict(prediction_img, patch_size)

            # save the images as nifti files
            # - multiply prediction_img by 30000 because it was normalized to [0, 1] before -> makes the img
            #   visualizable in imageJ
            # - multiply post_activation by 30000 because it contains the probability of the root being present

            ops = MRIoperations()

            np_prediction_img = prediction_img[0, 0, :, :, :].cpu().numpy() * 30000
            np_post_activation = (post_activation[0, root_idx, :, :, :].cpu().numpy() * 30000)
            np_binary_output = binary_output[0, 0, :, :, :].cpu().numpy()

            if "label" in transformed_data[i]:
                numpy_labels = prediction_label[0, 0, :, :, :].cpu().numpy() * 30000
                ops.save_mri(f"{plot_path}/{filename}/label_{numpy_labels.shape[0]}x{numpy_labels.shape[1]}x{numpy_labels.shape[2]}.nii.gz", numpy_labels)

            np_binary_output = np_binary_output.astype("float32")

            ops.save_mri(f"{plot_path}/{filename}/input_{np_prediction_img.shape[0]}x{np_prediction_img.shape[1]}x{np_prediction_img.shape[2]}.nii.gz", np_prediction_img)
            ops.save_mri(f"{plot_path}/{filename}/prediction_scaled_{np_post_activation.shape[0]}x{np_post_activation.shape[1]}x{np_post_activation.shape[2]}.nii.gz", np_post_activation)
            ops.save_mri(f"{plot_path}/{filename}/binary_prediction_{np_binary_output.shape[0]}x{np_binary_output.shape[1]}x{np_binary_output.shape[2]}.nii.gz", np_binary_output)

            # plot different slices of the image with its prediction
            for slice_frac in slices:
                Visualizations().plot_row(
                    prediction_img[0, 0, :, :, :],
                    post_activation[0, root_idx, :, :, :],
                    binary_output=binary_output[0, 0, :, :, :],
                    slice_frac=slice_frac,
                    label=(
                        prediction_label[0, 0, :, :, :] if prediction_label is not None else None
                    ),
                    filename=f"{plot_path}/{filename}/prediction_{slice_frac}",
                )

def main(model_name):
    prediction_data = [
        { # synthetic data
            "image": "../../data_final/generated/validation/my_Glycine_max/loam/sim_days_6-initial_-35/my_Glycine_max_day_6_res_237x237x201.nii.gz",
            "label": "../../data_final/generated/validation/my_Glycine_max/loam/sim_days_6-initial_-35/label_my_Glycine_max_day_6_res_474x474x402.nii.gz",
        },
        {
            "image": "../../data_final/generated/validation/my_Moraesetal_2020/loam/sim_days_9-initial_-415/my_Moraesetal_2020_day_9_res_237x237x151.nii.gz",
            "label": "../../data_final/generated/validation/my_Moraesetal_2020/loam/sim_days_9-initial_-415/label_my_Moraesetal_2020_day_9_res_474x474x302.nii.gz",
        },
        {
            "image": "../../data_new/real/III_Sand_1W_DAP14_res_256x256x131.nii.gz",
            "label": "../../data_new/real/label_III_Sand_1W_DAP14_res_512x512x262.nii.gz"
        },
        {
            "image": "../../data_new/real/III_Sand_3D_DAP14_res_256x256x191.nii.gz",
            "label": "../../data_new/real/label_III_Sand_3D_DAP14_res_512x512x382.nii.gz"
        },
        {
            "image": "../../data_new/real/III_Soil_3D_DAP15_res_256x256x199.nii.gz",
            "label": "../../data_new/real/label_III_Soil_3D_DAP15_res_512x512x398.nii.gz"
        },
        {
            "image": "../../data_new/real/IV_Sand_3D_DAP8_res_256x256x192.nii.gz",
            "label": "../../data_new/real/label_IV_Sand_3D_DAP8_res_512x512x384.nii.gz"
        },
        {
            "image": "../../data_new/real/IV_Soil_1W_DAP9_res_256x256x136.nii.gz",
            "label": "../../data_new/real/label_IV_Soil_1W_DAP9_res_512x512x272.nii.gz"
        },
        {
            "image": "../../data_new/real/IV_Soil_3D_DAP8_res_256x256x193.nii.gz",
            "label": "../../data_new/real/label_IV_Soil_3D_DAP8_res_512x512x386.nii.gz"
        },
        {
            "image": "../../data_new/real/III_Soil_1W_DAP14_res_256x256x186.nii.gz",
            "label": "../../data_new/real/label_III_Soil_1W_DAP14_res_512x512x372.nii.gz"
        },
    ]

    checkpoint_dir = f"../../runs/{model_name}/"

    # load the model params from the checkpoint dir (../../{model_name}/train_params.json) and store them in a dictionary
    with open(f"{checkpoint_dir}train_params.json") as f:
        train_params = json.load(f)

    patch_size = train_params["patch_size"]
    plot_path = f"example_predictions/{model_name}"

    pipeline = ImagePredictionPipeline(
        f"{checkpoint_dir}best_metric_model.ckpt", 
        train_params, 
        patch_size,
    )

    pipeline.predict_and_plot(
        prediction_data,
        [0.1, 0.3, 0.5, 0.6, 0.7],
        plot_path,
        patch_size,
    )



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