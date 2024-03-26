import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform
import SimpleITK as sitk
import nibabel as nib
import scipy.ndimage
import cv2
import copy
import argparse
import json
import ast
import re
import os


class MatchMRI:
    def __init__(self, mri_path, label_path) -> None:
        self.mri_path = mri_path
        pattern = r"(\d+)x(\d+)x(\d+)"
        # Search for the pattern in the filename
        match = re.search(pattern, label_path)

        dimensions = tuple(map(int, match.groups()))
        self.voxel_width = dimensions[1]
        self.voxel_depth = dimensions[2]

        self.label = np.fromfile(label_path, dtype="int16").reshape(
            (self.voxel_depth, self.voxel_width, self.voxel_width)
        )

        self.img = np.fromfile(mri_path, dtype="int16").reshape(
            (self.voxel_depth, self.voxel_width, self.voxel_width)
        )

        # img = nib.load(mri_path)
        # self.img = img.get_fdata()

    def _calc_transform_for_given_points(self, source_points, target_points):
        # # Convert lists to numpy arrays for convenience
        # src_pts = np.array(source_points)
        # tgt_pts = np.array(target_points)

        # # Ensure that the points are arrays of shape (N, 3)
        # assert src_pts.shape[1] == 3 and tgt_pts.shape[1] == 3

        # # Calculate centroids of the source and target points
        # centroid_src = np.mean(src_pts, axis=0)
        # centroid_tgt = np.mean(tgt_pts, axis=0)

        # # Center the points around their centroids
        # src_centered = src_pts - centroid_src
        # tgt_centered = tgt_pts - centroid_tgt

        # # Singular Value Decomposition (SVD) for rotation
        # H = src_centered.T @ tgt_centered
        # U, S, Vt = np.linalg.svd(H)
        # R = Vt.T @ U.T

        # # Correct rotation matrix to ensure a right-handed coordinate system
        # if np.linalg.det(R) < 0:
        #     Vt[-1, :] *= -1
        #     R = Vt.T @ U.T

        # # Calculate scale factor
        # norm_src = np.linalg.norm(src_centered, axis=1)
        # norm_tgt = np.linalg.norm(tgt_centered @ R.T, axis=1)
        # scale = np.mean(norm_tgt / norm_src)
        # scale = 1  # remove this line to use the calculated scale

        # # Calculate translation
        # T = centroid_tgt - scale * (R @ centroid_src)

        # # Construct the transformation matrix
        # transformation_matrix = np.eye(4)
        # transformation_matrix[:3, :3] = scale * R
        # transformation_matrix[:3, 3] = T

        ##############################

        # Convert lists to numpy arrays for convenience
        src_pts = np.array(source_points)
        tgt_pts = np.array(target_points)

        # Ensure that the points are arrays of shape (N, 3)
        assert src_pts.shape[1] == 3 and tgt_pts.shape[1] == 3

        # Calculate centroids of the source and target points
        centroid_src = np.mean(src_pts, axis=0)
        centroid_tgt = np.mean(tgt_pts, axis=0)

        centroid_src_marker = o3d.geometry.PointCloud()
        centroid_src_marker.points = o3d.utility.Vector3dVector([centroid_src])

        centroid_tgt_marker = o3d.geometry.PointCloud()
        centroid_tgt_marker.points = o3d.utility.Vector3dVector([centroid_tgt])

        angle_x = np.radians(180)
        R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_x), -np.sin(angle_x)],
                [0, np.sin(angle_x), np.cos(angle_x)],
            ]
        )

        # Construct the full transformation matrix
        shift_globel_center_transform = np.eye(4)
        shift_globel_center_transform[
            :3, 3
        ] = -centroid_src  # -translation_center_shift

        rotate_transform = np.eye(4)
        rotate_transform[:3, :3] = R_x

        shift_target_center_transform = np.eye(4)
        shift_target_center_transform[:3, 3] = centroid_tgt

        final_transform = np.dot(
            np.dot(shift_target_center_transform, rotate_transform),
            shift_globel_center_transform,
        )

        return final_transform, centroid_src_marker, centroid_tgt_marker

    def _apply_transformation(self, source_array, transformation_matrix):
        # Invert the matrix because scipy.ndimage.affine_transform applies the inverse
        inverse_matrix = np.linalg.inv(transformation_matrix)

        # Extract the rotation part and the translation part from the inverted matrix
        rotation = inverse_matrix[:3, :3]
        translation = inverse_matrix[:3, 3]

        # Apply the affine transformation
        transformed_array = scipy.ndimage.affine_transform(
            source_array,
            rotation,
            offset=translation,
            output_shape=source_array.shape,
            order=1,
        )

        return transformed_array

    def _calc_tranform_icp(self, source_pcd, target_pcd, threshold):
        # ICP registration
        icp = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000),
        )

        # Get the transformation matrix
        transformation_matrix = icp.transformation

        return icp.inlier_rmse, transformation_matrix

    # Convert numpy arrays to point clouds
    def _numpy_array_to_point_cloud(self, array_3d):
        # Get the coordinates of non-zero values (you can adjust this criteria)
        points = np.argwhere(array_3d > 0.5)  # Example threshold, adjust as needed

        return points

    def transform_points(self, points, transformation_matrix):
        """
        Apply the transformation matrix to the points.
        Points are given as an array of shape (N, 3).
        The transformation matrix is a 4x4 matrix.
        """
        # Convert points to homogeneous coordinates (add a column of ones)
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
        # Apply the transformation matrix to the points
        transformed_points = points_homogeneous @ transformation_matrix.T
        # Convert back from homogeneous to Cartesian coordinates
        transformed_points_cartesian = (
            transformed_points[:, :3] / transformed_points[:, [3]]
        )
        return transformed_points_cartesian

    def calculate_distances(self, source_points, target_points):
        """
        Calculate the Euclidean distances between corresponding source and target points.
        """
        distances = np.sqrt(np.sum((source_points - target_points) ** 2, axis=1))

        return distances

    def _calculate_and_apply_transform(
        self,
        source_points,
        target_points,
        grid,
        icp_target_cloud,
    ):
        label_points = np.argwhere(grid > 0.5)

        label_pcd = o3d.geometry.PointCloud()
        label_pcd.points = o3d.utility.Vector3dVector(label_points)

        if len(source_points):
            source_marker_pcd = o3d.geometry.PointCloud()
            source_marker_pcd.points = o3d.utility.Vector3dVector(source_points)

            target_marker_pcd = o3d.geometry.PointCloud()
            target_marker_pcd.points = o3d.utility.Vector3dVector(target_points)

            (
                points_transform,
                cent_source_marker,
                cent_target_marker,
            ) = self._calc_transform_for_given_points(
                source_points,
                target_points,
            )

            global_center_coord = np.array([0, 0, 0])
            global_center_marker = o3d.geometry.PointCloud()
            global_center_marker.points = o3d.utility.Vector3dVector(
                [global_center_coord]
            )

            # self._visualize_point_clouds(
            #     [
            #         label_pcd,
            #         icp_target_cloud,
            #         cent_source_marker,
            #         cent_target_marker,
            #         global_center_marker,
            #     ]
            # )

            label_pcd = label_pcd.transform(points_transform)

            # self._visualize_point_clouds(
            #     [
            #         label_pcd,
            #         icp_target_cloud,
            #         cent_source_marker,
            #         cent_target_marker,
            #         global_center_marker,
            #     ]
            # )

        _, pre_transform = self._calc_tranform_icp(
            label_pcd,
            icp_target_cloud,
            500,
        )

        label_pcd.transform(pre_transform)

        rsme, icp_transform = self._calc_tranform_icp(
            label_pcd,
            icp_target_cloud,
            threshold=10,
        )

        combined_transform = np.dot(
            icp_transform,
            pre_transform,
        )  # selected_point_transform)

        return (rsme, combined_transform, label_pcd.transform(icp_transform))

    def _visualize_point_clouds(self, pcd_list):
        # Create a visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add the point clouds to the visualizer
        for pcd in pcd_list:
            vis.add_geometry(pcd)

        # Set colors for the point clouds (e.g., transformed as red and original as blue)
        if len(pcd_list) == 2:
            pcd_list[0].paint_uniform_color([1, 0, 0])
            pcd_list[1].paint_uniform_color([0, 0, 1])
        if len(pcd_list) == 5:
            pcd_list[0].paint_uniform_color([1, 0, 0])  # red
            pcd_list[1].paint_uniform_color([0, 0, 1])  # blue
            pcd_list[2].paint_uniform_color([0, 1, 0])  # green
            pcd_list[3].paint_uniform_color([1, 1, 0])  # yellow
            pcd_list[4].paint_uniform_color([0, 1, 1])  # cyan

        # Run the visualizer
        vis.run()
        vis.destroy_window()

    def match_images(self, point_pairs):  # TODO: label)
        img_marker = np.array([pair[0] for pair in point_pairs])
        label_marker = np.array([pair[1] for pair in point_pairs])

        flipped_label_markers = np.array(
            [
                (point[0], point[1], self.voxel_width - point[2])
                for point in label_marker
            ]
        )
        print("flipped_label_markers:", flipped_label_markers)

        # filter the image such that it mostly contains the root signal
        image_filtered = np.where(
            self.img > np.percentile(self.img, 99.97),
            self.img,
            0,
        )
        print("image_filtered len:", len(image_filtered))

        # Convert it to a point cloud
        img_points = np.argwhere(image_filtered)

        img_pcd = o3d.geometry.PointCloud()
        img_pcd.points = o3d.utility.Vector3dVector(img_points)

        rsme, transform, transformed_pcd = self._calculate_and_apply_transform(
            label_marker,
            img_marker,
            np.copy(self.label),
            img_pcd,
        )

        flipped_label = np.flip(self.label, axis=2)

        (
            rsme_flipped,
            transform_flipped,
            transformed_pcd_flipped,
        ) = self._calculate_and_apply_transform(
            flipped_label_markers,
            img_marker,
            flipped_label,
            img_pcd,
        )

        print("rsme:", rsme)
        print("rsme_flipped:", rsme_flipped)

        image_pcd = o3d.geometry.PointCloud()
        image_pcd.points = o3d.utility.Vector3dVector(img_points)

        if rsme < rsme_flipped and not rsme == 0:
            self._visualize_point_clouds([transformed_pcd, image_pcd])  # , image_pcd])

            transformed_label = self._apply_transformation(self.label, transform)
        else:
            self._visualize_point_clouds(
                [transformed_pcd_flipped, image_pcd]
            )  # , image_pcd])

            transformed_label = self._apply_transformation(flipped_label, transform)

        # mri filename
        mri_filename = os.path.basename(self.mri_path).split(".")[0]
        # Save the transformed label
        transformed_label.astype("int16").tofile(f"corrected_label_{mri_filename}.raw")


my_point_pairs = [
    (np.array([95, 197, 53]), np.array([95, 39, 31])),
    (np.array([165, 116, 100]), np.array([165, 143, 69])),
    (np.array([70, 200, 52]), np.array([70, 38, 27])),
    # (np.array([26, 236, 62]), np.array([26, 29, 10])),
]

if __name__ == "__main__":
    # Create the parser

    parser = argparse.ArgumentParser(
        description="Script for visualizing an MRI with label data"
    )

    # example execution
    # python3 test-match-label-mri.py -m ../data/virtual_mri_generation/test_data/convert/III_Sand_1W_DAP14_256x256x131.nii.gz -r ../data/label_roots_vr_18_SNR_3_res_256x256x131.raw -p "[[[4, 60, 76], [4, 142, 171], [29, 139, 41]], [[29, 64, 205], [51, 118, 46], [51, 84, 199]]]"
    # Add arguments

    parser.add_argument(
        "--mri_path", "-m", type=str, help="path to the mri file of type .nii"
    )

    parser.add_argument(
        "--label_path", "-l", type=str, help="path to the .raw file of the label"
    )

    parser.add_argument(
        "--point_pairs",
        "-p",
        type=str,
        help="nested array as a JSON string for the point pairs (points specified as (z, y, x) - first in list is the MRI, second is the label)",
    )

    # Parse the arguments
    args = parser.parse_args()
    mri_path = args.mri_path
    label_path = args.label_path
    point_pairs = ast.literal_eval(args.point_pairs)

    print("point_pairs:", point_pairs)

    # Combine the MRI and label
    match_mri = MatchMRI(mri_path, label_path)  # args.rsml_path, args.mri_path)
    match_mri.match_images(my_point_pairs)



# python3 match-label-mri.py -m "/Users/daniel/Desktop/FZJ/CPlantBox/DUMUX/CPlantBox/tutorial/examples_segmentation/RootNet/src/data/virtual_mri_generation/test_data/convert/IV_Soil_3D_DAP8_256x256x193.raw" -l "/Users/daniel/Desktop/FZJ/CPlantBox/DUMUX/CPlantBox/tutorial/examples_segmentation/RootNet/src/data/virtual_mri_generation/test_data/convert/label_IV_Soil_3D_DAP8_256x256x193.raw"

# TODO: Something is wrong with the calculation of the translation. It doesn't shift the points, such that the centroids align.
#       Regarding the rotation, check out whether it improves the result, when first shifting the points to the center to apply the rotation
#       and in the end shifting them back to the original position.
