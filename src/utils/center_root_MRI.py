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
Description:    Shifts the rsml file towards the same boundary as the MRI file. This is important
                because the visualize function of the MRI (and label) only visualizes a bounding box
                centered at 0, 0, 0 which also should be the norm because, the seed should be placed
                there in global coordinates.
Usage: Specify a path to the rsml file and corresponding MRI scan.
Example: python3 center_root_MRI.py -m "mri_file_path" -r "rsml_file_path"
"""


class MoveMRI:
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

    def get_translation_border(self, source_array, width, depth) -> np.array:
        """
        Calculate and return the translation needed to move the source_array coordinates to the target boundary,
        which is defined by the width and depth.

        Args:
        - source_array (np.array): array of points (coordinates)
        - width (float): width of the MRI (target where the root should be placed)
        - depth (float): depth of the MRI (target where the root should be placed)

        Returns:
        - translation (np.array): translation needed to move the source_array to the target boundary
        """
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

    def center_label_mri(self):
        """
        Calclates the translation needed to move the root to the target boundary. Then creates a virtual MRI
        of the root signal and return it (the label grid)

        Returns:
        - label_grid (np.array): virtual MRI numpy array of the root signal
        """
        width = (self.affine_matrix[1, 1] * self.image_data.shape[1]) / 2
        depth = self.affine_matrix[0, 0] * self.image_data.shape[0]

        print("width:", width)
        print("depth:", depth)
        print(self.affine_matrix[0, 0])

        nodes, _, _ = self._get_root_data_from_rsml(self.rsml_path)
        points_array = np.array(nodes)

        translation = self.get_translation_border(
            points_array,
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
            width=width,
            depth=depth,
            offset=translation,
            scale_factor=2,
        )

        label_grid, _ = my_root.create_virtual_root_mri(".", label=True)

        return label_grid


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
    combined = MoveMRI(rsml_path, mri_path)  # args.rsml_path, args.mri_path)
    combined.center_label_mri()



# Example usage:
# python3 center_root_MRI.py -m /Users/daniel/Desktop/FZJ/CPlantBox/DUMUX/CPlantBox/tutorial/examples_segmentation/RootNet/src/data/virtual_mri_generation/test_data/convert/III_Sand_3D_DAP14_256x256x191.nii.gz -r "/Users/daniel/Desktop/FZJ/Echte Daten/tobias_mri/III_Sand_3D_DAP14_256x256x191/roots_vr_7.rsml"
