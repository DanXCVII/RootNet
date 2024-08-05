import numpy as np
import nibabel as nib
import re


class MRIoperations:
    """
    Class for loading and saving MRI files in the .nii.gz, .nii, and .raw formats. 
    """
    def __init__(self):
        pass

    def load_mri(self, mri_path):
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
            # affine_matrix = None

            affine_matrix = np.array(
                [
                    [0.027, 0.0, 0.0, 0.0],
                    [0.0, 0.027, 0.0, 0.0],
                    [0.0, 0.0, 0.1, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        else:
            raise ValueError("MRI file must be .nii.gz, .nii, or .raw")

        return affine_matrix, image_data

    def save_mri(self, mri_path, mri_data):
        if mri_path.endswith(".nii.gz") or mri_path.endswith(".nii"):
            affine_transformation = np.array(
                [
                    [0.027, 0.0, 0.0, 0.0],
                    [0.0, 0.027, 0.0, 0.0],
                    [0.0, 0.0, 0.1, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            img = nib.Nifti1Image(mri_data, affine_transformation)
            print(f"Saving MRI to {mri_path}")
            nib.save(img, mri_path)
        elif mri_path.endswith(".raw"):

            mri_data.astype("int16").tofile(mri_path)

    def load_section_mri(self, mri_path, point_one, point_two):
        _, mri = self.load_mri(mri_path)

        min_x, min_y, min_z = (
            min(point_one[0], point_two[0]),
            min(point_one[1], point_two[1]),
            min(point_one[2], point_two[2]),
        )

        max_x, max_y, max_z = (
            max(point_one[0], point_two[0]),
            max(point_one[1], point_two[1]),
            max(point_one[2], point_two[2]),
        )

        print(min_x, min_y, min_z)
        print(max_x, max_y, max_z)

        section = mri[min_z : max_z + 1, min_y : max_y + 1, min_x : max_x + 1]

        return section
