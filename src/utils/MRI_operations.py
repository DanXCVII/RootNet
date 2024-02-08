import numpy as np
import nibabel as nib
import re

"""
Description:    This class contains functions to load and save MRI data for common file formats.
"""

class MRIoperations:
    def __init__(self):
        pass

    def load_mri(self, mri_path):
        """
        Loads the MRI data from a given path which can be a .nii.gz or .raw file.

        Args:
        - mri_path (str): Path to the MRI file.

        Returns:
        - affine_matrix (np.ndarray): Affine matrix of the MRI if it is in format .nii.gz or .nii. Otherwise None.
        - image_data (np.ndarray): MRI data.
        """

        if mri_path.endswith(".nii.gz") or mri_path.endswith(".nii"):
            img = nib.load(mri_path)
            image_data = img.get_fdata()
            affine_matrix = img.affine
        elif mri_path.endswith(".raw"):
            image_data = np.fromfile(mri_path, dtype="int16")
            
            pattern = r"_(\d+)x(\d+)x(\d+)"
            match = re.search(pattern, mri_path)

            if match:
                width, height, depth = match.groups()

                resolution = (int(depth), int(height), int(width))
            
            image_data = image_data.reshape(resolution)
            affine_matrix = None

        return affine_matrix, image_data

    def save_mri(self, mri_path, mri_data):
        """
        Saves the MRI data to the given path as the given file format depending on the file extension.
        Supported file formats are .nii.gz and .raw.

        Args:
        - mri_path (str): Path to the MRI file.
        - mri_data (np.ndarray): MRI data.
        """
        if mri_path.endswith(".nii.gz") or mri_path.endswith(".nii"):
            affine_transformation = np.array(
                [
                    [1, 0.0, 0.0, 0.0],
                    [0.0, 0.27, 0.0, 0.0],
                    [0.0, 0.0, 0.27, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            img = nib.Nifti1Image(mri_data, affine_transformation)
            print(f"Saving MRI to {mri_path}")
            nib.save(img, mri_path)
        elif mri_path.endswith(".raw"):
            mri_data.astype("int16").tofile(mri_path)

    