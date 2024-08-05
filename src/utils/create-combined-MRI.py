import sys

with open("../../DUMUX_path.txt", "r") as file:
    DUMUX_path = file.read()

sys.path.append(f"{DUMUX_path}/CPlantBox")
sys.path.append(f"{DUMUX_path}/CPlantBox/experimental/parametrisation/")
sys.path.append(f"{DUMUX_path}/CPlantBox/src")
sys.path.append("..")

offset = (4.1599, -8.2821, -0.4581)

import numpy as np
import argparse
import matplotlib.pyplot as plt
import nibabel as nib
from data.virtual_mri_generation import Virtual_MRI

import plantbox as pb
import rsml.rsml_reader as rsml_reader
import os


"""
Decription:     Shifts the rsml file towards the same boundary as the MRI file and then visualizes the MRI 
                with the label on top of it.
Usage: Specify a path to the rsml file and corresponding MRI scan.
Example: python3 center_root_MRI.py -m "mri_file_path" -r "rsml_file_path"
"""

class CombinedMRI:
    def __init__(self, rsml_path, mri_path) -> None:
        self.rsml_path = rsml_path
        self.mri_path = mri_path

        img = nib.load(mri_path)
        image_data = img.get_fdata()

        self.image_data = image_data.astype("int16")
        self.affine_matrix = img.affine

    def _get_root_data_from_rsml(self, rsml_path):
        """
        Gets the pb.SegmentAnalyser for the given rsml file

        Args:
        - rsml_path: path to the rsml file

        Returns:
        - segana: pb.SegmentAnalyser
        """
        polylines, properties, functions, _ = rsml_reader.read_rsml(rsml_path)

        nodes, segs = rsml_reader.get_segments(polylines, properties)
        seg_radii = rsml_reader.get_parameter(polylines, functions, properties)[0][:-1]

        segs_ = [pb.Vector2i(s[0], s[1]) for s in segs]  # convert to CPlantBox types
        nodes_ = [pb.Vector3d(n[0], n[1], n[2]) for n in nodes]
        segRadii = np.zeros((segs.shape[0], 1))  # convert to paramter per segment
        segCTs = np.zeros((segs.shape[0], 1))

        segana = pb.SegmentAnalyser(nodes_, segs_, segCTs, segRadii)

        return (
            segana.nodes,
            segana.segments,
            seg_radii,
        )

    def get_translation_border(self, source_array, target, width, depth) -> np.array:
        z_values = source_array[:, 2]
        y_values = source_array[:, 1]
        x_values = source_array[:, 0]

        min_root_y = np.min(y_values)
        min_root_x = np.min(x_values)

        max_root_z = np.max(z_values)

        target_min_x = -width  # Replace with target boundary x-axis
        target_min_y = -width  # Replace with target boundary y-axis
        target_min_z = 0  # Replace with target boundary z-axis

        translation = np.array(
            [
                -(target_min_x - min_root_x),
                -(target_min_y - min_root_y),
                -(target_min_z - max_root_z),
            ]
        )

        return translation

    def combine_label_mri(self):
        print(self.affine_matrix[1, 1])
        print(self.affine_matrix[0, 0])
        width = (self.affine_matrix[1, 1] * self.image_data.shape[1]) / 2
        depth = self.affine_matrix[0, 0] * self.image_data.shape[0]

        print("width:", width)
        print("depth:", depth)
        print(self.affine_matrix[0, 0])

        nodes, _, _ = self._get_root_data_from_rsml(self.rsml_path)
        points_array = np.array(nodes)

        translation = self.get_translation_border(
            points_array,
            self.image_data,
            width,
            depth,
        )

        my_root = Virtual_MRI(
            rsml_path=self.rsml_path,
            res_mri=(
                self.affine_matrix[2, 2],
                self.affine_matrix[1, 1],
                self.affine_matrix[0, 0],
            ),
            radius=width,
            depth=depth,
            offset=translation,
        )

        label, _ = my_root.create_virtual_root_mri(".", label=True)

        label = label * 30000

        combined = self.image_data * 0.5 + label * 0.5

        # get the file name of the rsml file
        mri_filename = os.path.basename(self.mri_path).split(".")[0]
        mri_dir = os.path.dirname(self.mri_path)

        resolution_tag = self.mri_path.split("_")[-1].split(".")[0]

        label.astype("int16").tofile(f"{mri_dir}/label_{mri_filename}.raw")

        image_name = f"combined_{resolution_tag}"
        combined.astype("int16").tofile(f"{mri_dir}/{image_name}.raw")
        print(f"saved combined image as {image_name}.raw")

        self.get_translation_border


# Convert the list to a numpy array -
# Define your main function
# Create the parser
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Script for visualizing an MRI with label data"
    )

    # Add arguments
    parser.add_argument(
        "--mri_path", "-m", type=str, help="path to the mri file (.nii.gz)"
    )
    parser.add_argument(
        "--rsml_path", "-r", type=str, help="path to the rsml file (.rmsl)"
    )

    # Parse the arguments
    args = parser.parse_args()

    rsml_path = args.rsml_path
    mri_path = args.mri_path

    # Combine the MRI and label
    combined = CombinedMRI(rsml_path, mri_path)  # args.rsml_path, args.mri_path)
    combined.combine_label_mri()

# Example usage:
# python3 create-combined-MRI.py -m /Users/daniel/Desktop/FZJ/CPlantBox/DUMUX/CPlantBox/tutorial/examples_segmentation/RootNet/src/data/virtual_mri_generation/test_data/convert/IV_Soil_3D_DAP8_256x256x193.nii.gz -r "/Users/daniel/Desktop/FZJ/Echte Daten/tobias_mri/IV_Soil_3D_DAP8_256x256x193/roots_vr_21.rsml"
