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
from scipy.ndimage import zoom
import center_root_MRI


from MRI_operations import MRIoperations

"""
Description:    Uses the rsml file to create a label MRI. The label MRI which is potentially not aligned with the
                real MRI is then matched with it. This is done by first centering the rsml, of which the label MRI
                is created. Then a point cload is created of the label MRI and the real MRI (which is filtered to
                mostly contain the root signal (99.96th percentile)). The label MRI is then transformed to match the 
                real MRI by first applying different rotations and then ICP for final optimization. The rotations
                are needed because ICP primarily does the final optimization and not the initial alignment, which 
                can be off by a lot. The final transformation is then applied to the label MRI, which is then saved.
Example: python3 match-label-mri.py -m "./example_data/III_Sand_1W_DAP14_256x256x131.rsml" -l "./example_data/label_roots_vr_18_res_512x512x262.raw"
"""


class MatchMRI:
    def __init__(self, mri_path, rsml_path) -> None:
        """
        Initializes the params which are the real MRI image, the label MRI image and the upscaled MRI image.
        The upscaled version is needed because the label MRI has twice the resolution of the real MRI.

        Args:
        - mri_path: path to the real MRI image
        - rsml_path: path to the rsml file
        """
        self.mri_path = mri_path

        _, self.rsml_img = center_root_MRI.MoveMRI(rsml_path, mri_path)
        self.label_img = mri_mover.center_label_mri()

        _, img = MRIoperations().load_mri(mri_path)

        zoom_factors = (2, 2, 2)

        # Perform the upscaling with interpolation
        self.upscaled_img = zoom(img, zoom_factors, order=1)

    def _calc_transform_for_given_points(self, source_point, target_point, degrees):
        """
        First calculates the transformation matrix to center the source points. Then the transform needed
        to rotate them for a given degree and finally the transform to shift them to the center of the
        target points.

        Args:
        - source_point: points array
        - target_point: points array
        - degrees: int

        Returns:
        - final_transform: The final transformation matrix containing the combined transformations for the
                            source array
        """
        centroid_src_marker = o3d.geometry.PointCloud()
        centroid_src_marker.points = o3d.utility.Vector3dVector([source_point])

        centroid_tgt_marker = o3d.geometry.PointCloud()
        centroid_tgt_marker.points = o3d.utility.Vector3dVector([target_point])

        angle_x = np.radians(degrees)
        R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_x), -np.sin(angle_x)],
                [0, np.sin(angle_x), np.cos(angle_x)],
            ]
        )

        # Construct the full transformation matrix
        shift_globel_center_transform = np.eye(4)
        shift_globel_center_transform[:3, 3] = (
            -source_point
        )  # -translation_center_shift

        rotate_transform = np.eye(4)
        rotate_transform[:3, :3] = R_x

        shift_target_center_transform = np.eye(4)
        shift_target_center_transform[:3, 3] = target_point

        final_transform = np.dot(
            np.dot(shift_target_center_transform, rotate_transform),
            shift_globel_center_transform,
        )

        return final_transform, centroid_src_marker, centroid_tgt_marker

    def _apply_transformation(self, source_array, transformation_matrix):
        """
        Ally a given transformation matrix to the source array.

        Args:
        - source_array: np.array
        - transformation_matrix: np.array

        Returns:
        - transformed_array: the transformed array
        """
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
        """
        Applys the ICP algorithm to the source and target point cloud, such that the source point cloud
        is aligned with the target point cloud.

        Args:
        - source_pcd: o3d.geometry.PointCloud
        - target_pcd: o3d.geometry.PointCloud

        Returns:
        - icp.inlier_rmse: float
        """
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

    def calculate_rmse(self, source_cloud, target_cloud):
        # Ensure source_cloud and target_cloud are o3d.geometry.PointCloud objects
        if not isinstance(source_cloud, o3d.geometry.PointCloud) or not isinstance(
            target_cloud, o3d.geometry.PointCloud
        ):
            raise TypeError(
                "source_cloud and target_cloud must be open3d.geometry.PointCloud objects"
            )

        # Convert target cloud to KDTree for nearest neighbor search
        target_kdtree = o3d.geometry.KDTreeFlann(target_cloud)

        squared_distances = []
        for point in source_cloud.points:
            # For each point in the source cloud, find its nearest neighbor in the target cloud
            _, idx, dist_squared = target_kdtree.search_knn_vector_3d(point, 1)
            squared_distances.append(dist_squared[0])

        # Calculate mean of squared distances
        mean_squared_error = np.mean(squared_distances)

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error)

        return rmse

    def _calculate_and_apply_transform(
        self,
        source_array,
        target_pcd,
    ):
        """
        Applies different degrees of rotations and then ICP for final adjustment. It returns the best
        transformation matrix and the transformed point cloud with its corresponding RSME.

        Args:
        - source_array: np.array array for which the transformation matrix should be calculated to match
                        the target point cloud
        - target_pcd: o3d.geometry.PointCloud target point cloud to which the source point cloud should be matched

        Returns:
        - min_rsme: float the minimum RSME value calculated for the different rotations
        - best_transform: np.array the best transformation matrix
        - source_pcd.transform(best_transform): o3d.geometry.PointCloud the transformed source point cloud which
                                                should be matched with the target point cloud
        """
        rsme_list = []
        transfroms_list = []

        label_points = np.argwhere(source_array > 0.5)

        # Iterate over multiple rotations
        for degree in range(0, 360, 90):
            print("Calculating transformation for degree:", degree)
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(label_points)

            # Create a scaling matrix for reducing the size of the point cloud to match the
            # mri size (because the label has twice the resolution)

            # Apply the transformation
            # source_pcd = source_pcd.transform(scaling_matrix)

            source_center = source_pcd.get_center()
            target_center = target_pcd.get_center()

            source_marker_pcd = o3d.geometry.PointCloud()
            source_marker_pcd.points = o3d.utility.Vector3dVector([source_center])

            target_marker_pcd = o3d.geometry.PointCloud()
            target_marker_pcd.points = o3d.utility.Vector3dVector([target_center])

            (
                points_transform,
                cent_source_marker,
                cent_target_marker,
            ) = self._calc_transform_for_given_points(
                source_center,
                target_center,
                degree,
            )

            source_pcd = source_pcd.transform(points_transform)

            _, pre_transform = self._calc_tranform_icp(
                source_pcd,
                target_pcd,
                200,
            )

            source_pcd.transform(pre_transform)

            rsme, icp_transform = self._calc_tranform_icp(
                source_pcd,
                target_pcd,
                threshold=10,
            )

            combined_transform = np.dot(
                icp_transform,
                np.dot(pre_transform, points_transform),
            )  # selected_point_transform)

            transformed_pcd = source_pcd.transform(icp_transform)

            rsme = self.calculate_rmse(transformed_pcd, target_pcd)
            print("final rsme:", rsme)

            rsme_list.append(rsme)
            transfroms_list.append(combined_transform)

        min_rsme = min(rsme_list)
        min_rsme_index = rsme_list.index(min_rsme)
        best_transform = transfroms_list[min_rsme_index]

        # Exclude zeros from array1 and find the minimum value
        non_zero_array1 = [value for value in rsme_list if value > 0]
        min_rsme = np.min(non_zero_array1)

        # Find the index of the minimum value in the original array1
        min_index = rsme_list.index(min_rsme)

        # Get the corresponding value from array2
        best_transform = transfroms_list[min_index]

        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(label_points)

        return (
            min_rsme,
            best_transform,
            source_pcd.transform(best_transform),
        )

    def _visualize_point_clouds(self, pcd_list):
        """
        Visualizes the given point clouds. For 2 different point clouds, the first one is colored red and the second
        one is colored blue. For 5 different point clouds, the first one is colored red, the second one is colored blue,
        the third one is colored green, the fourth one is colored yellow and the fifth one is colored cyan.

        Args:
        - pcd_list: list of o3d.geometry.PointCloud
        """
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

    def match_images(self):
        """
        Matches the label MRI with the real MRI. It does it by first filtering the real MRI such that it mostly
        contains the root signal (by showing only datapoints with a high intensity). Then it converts the filtered
        MRI to a point cloud and calculates the transformation needed to match the label MRI with the real MRI.

        Returns:
        - transformed_label: np.array the transformed label MRI
        """
        # filter the image such that it mostly contains the root signal
        image_filtered = np.where(
            self.upscaled_img > np.percentile(self.upscaled_img, 99.96),
            self.upscaled_img,
            0,
        )

        label_voxel_width = self.label_img.shape[1]
        label_voxel_depth = self.label_img.shape[0]

        # Convert it to a point cloud
        img_points = np.argwhere(image_filtered)

        img_pcd = o3d.geometry.PointCloud()
        img_pcd.points = o3d.utility.Vector3dVector(img_points)

        (
            rsme,
            transform,
            transformed_pcd,
        ) = self._calculate_and_apply_transform(
            np.copy(self.label_img),
            img_pcd,
        )

        flipped_label = np.flip(self.label_img, axis=2)

        (
            rsme_flipped,
            transform_flipped,
            transformed_pcd_flipped,
        ) = self._calculate_and_apply_transform(
            flipped_label,
            img_pcd,
        )

        print("rsme:", rsme)
        print("rsme_flipped:", rsme_flipped)

        image_pcd = o3d.geometry.PointCloud()
        image_pcd.points = o3d.utility.Vector3dVector(img_points)

        if rsme < rsme_flipped and not rsme == 0:
            self._visualize_point_clouds([transformed_pcd, image_pcd])  # , image_pcd])

            transformed_label = self._apply_transformation(self.label_img, transform)
        else:
            self._visualize_point_clouds(
                [transformed_pcd_flipped, image_pcd]
            )  # , image_pcd])

            transformed_label = self._apply_transformation(
                flipped_label, transform_flipped
            )

        # mri filename
        mri_filename = os.path.basename(self.mri_path)

        label_filename = f"label_{mri_filename[:mri_filename.rfind('_')]}_{label_voxel_width}x{label_voxel_width}x{label_voxel_depth}"

        # transformed_label = transformed_label.swapaxes(1, 2)
        MRIoperations().save_mri(label_filename + ".nii.gz", transformed_label)


if __name__ == "__main__":
    # Create the parser

    parser = argparse.ArgumentParser(
        description="Script for visualizing an MRI with label data"
    )

    parser.add_argument(
        "--mri_path", "-m", type=str, help="path to the mri file of type .nii.gz or .raw"
    )

    parser.add_argument(
        "--rsml_path", "-l", type=str, help="path to the corresponding rsml file"
    )

    # Parse the arguments
    args = parser.parse_args()
    mri_path = args.mri_path
    label_path = args.label_path

    # Combine the MRI and label
    match_mri = MatchMRI(mri_path, label_path)  # args.rsml_path, args.mri_path)
    match_mri.match_images()

