import sys

with open("../../DUMUX_path.txt", "r") as file:
    DUMUX_path = file.read()

sys.path.append(f"{DUMUX_path}/CPlantBox")
sys.path.append(f"{DUMUX_path}/CPlantBox/experimental/parametrisation/")
sys.path.append(f"{DUMUX_path}/CPlantBox/src")
sys.path.append("..")

import plantbox as pb
import rsml.rsml_reader as rsml_reader
import numpy as np
import functional.bresenham3D as bres3D
import matplotlib.pyplot as plt
import rsml_reader as rsml
from scipy import ndimage
from .interpolate_water_sim import VTUInterpolatedGrid
from noise import pnoise3
from skimage import exposure
from scipy.ndimage import gaussian_filter
import re

# import pygorpho as pg
from skimage.morphology import ball
import math
from scipy.ndimage import distance_transform_edt
import random
import os
from typing import Tuple

from utils import FourierSynthesis
from utils import expand_volume_with_blending
from utils import MRIoperations


def get_min_max_numpy(array):
    """
    returns the min and max value of a numpy array
    """
    non_nan_indices = np.argwhere(~np.isnan(array.flat))
    first_non_nan_value = array.flat[non_nan_indices[0][0]]

    new = np.nan_to_num(array, nan=first_non_nan_value)

    min = new.min()
    max = new.max()

    return min, max


class Virtual_MRI:
    def __init__(
        self,
        rsml_path,
        soil_type=None,
        vtu_path=None,
        seganalyzer=None,
        radius=3,
        depth=20,
        res_mri=[0.027, 0.027, 0.1],
        scale_factor=1,
        offset=(0, 0, 0),
    ):
        """
        creates a virtual MRI for the given root system with simulating noise based on the water content of the soil.
        - rootsystem: root system object (pb.RootSystem)
        - rsml_path: path to the rsml file of the root system # currently not used but if it is possible to read the
                     rsml file and get the seganalyser from it, it would be better
        - vtu_path: path to the vtu file of the water simulation
        - perlin_noise_intensity: intensity of how much perlin noise should be added (value between 0 and 1)
        - width: width of the soil container
        - depth: depth of the soil container
        - resolution: resolution of the MRI
        """
        self.resx = res_mri[0] / scale_factor
        self.resy = res_mri[1] / scale_factor
        self.resz = res_mri[2] / scale_factor
        self.res_mri = res_mri
        self.scale_factor = scale_factor
        self.width = radius
        self.depth = depth
        self.rsml_path = rsml_path
        self.soil_type = soil_type
        self.vtu_path = vtu_path
        if seganalyzer is None:
            self.nodes, self.segs, self.seg_radii = self._get_root_data_from_rsml(
                rsml_path
            )
        else:
            self.segana = seganalyzer
            self.nodes = self.segana.nodes
            self.segs = self.segana.segments
            self.seg_radii = self.segana.getParameter("radius")
        self.water_intensity_grid = None
        self.offset = offset

        self.nx = int(self.width * 2 / res_mri[0]) * self.scale_factor
        self.ny = int(self.width * 2 / res_mri[1]) * self.scale_factor
        self.nz = int(self.depth / res_mri[2]) * self.scale_factor

        if self.soil_type == "loam":
            self.max_root_signal_intensity = 16000
        elif self.soil_type == "sand":
            self.max_root_signal_intensity = 18000
        else:
            # case that usually only applies if no noise is added for visualization purposes
            self.max_root_signal_intensity = 5000

    def _generate_perlin_noise_3d(self, shape, scale_x, scale_y, scale_z) -> np.array:
        """
        Generate a 3D numpy array of Perlin noise.

        Args:
        - shape: The shape of the generated array (tuple of 3 ints).
        - scale: The scale of noise.

        Returns:
        - np.array: 3D array of perlin noise values in the range [-1, 1].
        """
        noise = np.zeros(shape)

        base = np.random.randint(0, 2000)

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    noise_value = pnoise3(
                        i / scale_x,
                        j / scale_y,
                        k / scale_z,
                        octaves=4,
                        persistence=2,
                        lacunarity=2,
                        repeatx=256,
                        repeaty=256,
                        repeatz=200,
                        base=base,
                    )
                    # Normalize to [0, 1]
                    # noise_value = (noise_value + 1) / 2

                    noise[i][j][k] = noise_value

        return noise

    def _get_root_data_from_rsml(self, rsml_path) -> pb.SegmentAnalyser:
        """
        Gets the pb.SegmentAnalyser for the given rsml file

        Args:
        - rsml_path: path to the rsml file

        Returns:
        - segana: pb.SegmentAnalyser
        """
        polylines, properties, functions, _ = rsml_reader.read_rsml(rsml_path)
        print()

        nodes, segs = rsml_reader.get_segments(polylines, properties)
        seg_radii = rsml_reader.get_parameter(polylines, functions, properties)[0][:-1]

        segs_ = [pb.Vector2i(s[0], s[1]) for s in segs]  # convert to CPlantBox types
        nodes_ = [pb.Vector3d(n[0], n[1], n[2]) for n in nodes]
        segRadii = np.zeros((segs.shape[0], 1))  # convert to paramter per segment
        segCTs = np.zeros((segs.shape[0], 1))

        segana = pb.SegmentAnalyser(nodes_, segs_, segCTs, segRadii)

        print("nodes shape", len(segana.nodes))
        print("segments shape", len(segana.segments))
        print("radius shape", len(seg_radii))

        return (
            segana.nodes,
            segana.segments,
            seg_radii,
        )

    def _get_dimensions_container_array(self, nx, ny, nz) -> Tuple[int, int, int]:
        """
        creates the dimensions of the container array, which must consist of consecutive values for each pixel
        representing a gray intensity value for the MRI

        Args:
        - nx, ny, nz: number of pixels in x, y and z direction

        Returns:
        - my_x, my_y, my_z: dimensions of the container array
        """

        min_x = -int(nx / 2)
        min_y = -int(ny / 2)

        my_x = np.arange(min_x, min_x + nx)
        my_y = np.arange(min_y, min_y + ny)
        my_z = np.arange(0, -nz, -1)

        return my_x, my_y, my_z

    def _get_root_segment_idx(
        self,
        segment,
        nodes,
        x_int,
        y_int,
        z_int,
        res,
    ) -> np.array:
        """
        calculates the indexes in the mri grid, where the root segment is located

        Args:
        - segment: root segment
        - nodes: nodes of the root system
        - x_int, y_int, z_int: coordinates of the mri grid
        """
        n1, n2 = nodes[segment.x], nodes[segment.y]

        voxel_offset_x = np.around(self.offset[0] / res["x"])
        voxel_offset_y = np.around(self.offset[1] / res["y"])
        voxel_offset_z = np.around(self.offset[2] / res["z"])

        (x1, y1, z1) = [
            np.around(n1.x / res["x"]) - voxel_offset_x,
            np.around(n1.y / res["y"]) - voxel_offset_y,
            np.around(n1.z / res["z"]) - voxel_offset_z,
        ]
        (x2, y2, z2) = [
            np.around(n2.x / res["x"]) - voxel_offset_x,
            np.around(n2.y / res["y"]) - voxel_offset_y,
            np.around(n2.z / res["z"]) - voxel_offset_z,
        ]
        # contains all points on the segment
        ListOfPoints = np.array(bres3D.Bresenham3D(x1, y1, z1, x2, y2, z2))

        allidx_ = []
        # searches the points in the 3d structure, which correspond to root segments
        for j in range(0, len(ListOfPoints)):
            # ListOfPoints[j,0] is the same as ListOfPoints[j][0]
            xidx = np.where(x_int == ListOfPoints[j, 0])
            yidx = np.where(y_int == ListOfPoints[j, 1])
            zidx = np.where(z_int == ListOfPoints[j, 2])
            # if the point of the segment is in the 3d structure
            if xidx[0].size > 0 and yidx[0].size > 0 and zidx[0].size > 0:
                a = [int(xidx[0][0]), int(yidx[0][0]), int(zidx[0][0])]
                allidx_.append(a)

        if len(allidx_) < 1:
            print("no idx found")
            print("ListOfPoints", ListOfPoints)
            print("x1, y1, z1", x1, y1, z1)
            print("x2, y2, z2", x2, y2, z2)

        return np.array(allidx_)

    def ellipsoid(self, radius_x, radius_y, radius_z, dtype=np.int16) -> np.array:
        """
        Create a 3D ellipsoidal structuring element.

        Parameters:
        - radius_x, radius_y, radius_z: Radii along the x, y, and z axes.
        - dtype: Desired data type of the output (default is np.int16).

        Returns:
        - A 3D numpy array representing the ellipsoidal structuring element.
        """

        # Create a 3D grid of coordinates
        x, y, z = np.ogrid[
            -radius_x : radius_x + 1, -radius_y : radius_y + 1, -radius_z : radius_z + 1
        ]

        # Evaluate the ellipsoidal equation for each coordinate
        ellipsoid = (x / radius_x) ** 2 + (y / radius_y) ** 2 + (z / radius_z) ** 2 <= 1

        return ellipsoid.astype(dtype)

    def ellipsoid_gaussian(self, radius_x, radius_z, sigma) -> np.array:
        """
        Create a 3D ellipsoidal Gaussian structuring element.

        - radius_x: radius along the x and y axes of the ellipsoid.
        - radius_z: radius along the z axis of the ellipsoid.
        - sigma: Standard deviation for the Gaussian distribution.
        """
        width = radius_x * 2
        height = radius_z * 2
        # Create a grid of coordinates
        x, y, z = np.indices((width, width, height))
        center_x, center_y, center_z = width // 2, width // 2, height // 2

        # Calculate the Gaussian distribution
        gaussian = np.exp(
            -((x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2)
            / (2 * sigma**2)
        )

        # Apply ellipsoidal condition
        ellipsoidal_mask = ((x - center_x) / radius_x) ** 2 + (
            (y - center_y) / radius_x
        ) ** 2 + ((z - center_z) / radius_z) ** 2 <= 1

        gaussian = gaussian * ellipsoidal_mask

        return gaussian

    def _get_binary_dilation_root_segment_idx(
        self,
        grid,
        radius,
        res,
        binary,
    ) -> np.array:
        """
        Expands the coordinates where the root segment is located by its radius and returns the indexes of the
        expanded coordinates

        Args:
        - grid: 3d array of the root container with dimensions (nx, ny, nz)
        - radius: radius of the root segment
        - res: resolution of the MRI
        - binary: boolean, if True, the root segment is expanded by a binary dilation (for the label). If False, the root segment
                    is expanded by a gaussian dilation (for more realism)
        """
        # if radius < 0.1: # TODO: Check if required
        #     radius = 0.1
        radius_x = int(np.around(radius / res["x"]))
        radius_z = int(np.around(radius / res["z"]))
        radius_z = radius_z if radius_z > 0 else 1

        if binary:
            selem = self.ellipsoid(radius_x, radius_x, radius_z)

            grid = ndimage.binary_dilation(grid, structure=selem).astype(grid.dtype)
        else:
            selem = self.ellipsoid_gaussian(radius_x, radius_z, radius_x)

            grid = ndimage.grey_dilation(grid, structure=selem)

            grid = grid - 1

        idx = np.nonzero(grid)
        non_zero_values = grid[idx]

        return idx, non_zero_values

    def get_noise_volume(self, noise_dir) -> np.array:
        """
        reads a random noise volume based on the soil type from the data assets

        Args:
        - noise_dir: directory with the noise volumes

        Returns:
        - water_content: water content of the read noise volume
        - noise_volume: 3d (numpy) array of the noise volume
        """

        raw_files = [
            file
            for file in os.listdir(noise_dir)
            if file.endswith(".raw") and self.soil_type in file
        ]
        random_file = random.choice(raw_files)

        random_noise_filename = os.path.join(noise_dir, random_file)

        _, noise_volume = MRIoperations().load_mri(random_noise_filename)

        # extract the water content from the filename
        water_content_match = re.search(
            r"_(\d+)_\d+x\d+x\d+\.raw$", random_noise_filename
        )

        if water_content_match:
            water_content = (
                float(water_content_match.group(1)) / 100
            )  # Convert to decimal
        else:
            # throw error that the filename should contain the percentage
            raise ValueError(
                'The noise filename in data_assets/noise should contain the water content as "..._XX_resolution.raw"'
            )

        noise_volume = np.swapaxes(noise_volume, 0, 2)

        return water_content, noise_volume

    def _get_fourier_noise(self, shape, noise_dir, overlap) -> np.array:
        """
        generates a fourier noise volume for the given shape given a directory with noise volumes. It does so
        by generating multiple noise volumes by a fourier synthesis, applying histogram matching to them and
        expanding them by blending them together with the specified overlap

        Args:
        - shape: target shape of the noise volume
        - noise_dir: directory with noise volumes to generate the noise from
        - overlap: overlap of the noise volumes in voxels

        Returns:
        - water_content: water content of the noise volume
        - scaled_expanded_noise: 3d (numpy) array of the noise volume
        """
        water_content, original_noise = self.get_noise_volume(noise_dir)
        noise_gen = FourierSynthesis(original_noise)

        # Calculate how often the noise volume can fit into the larger volume (target shape) with the given overlap
        # information needed for how many new noise volumes need to be generated
        fit_count = tuple(
            math.ceil(target_volume / (sub_volume - overlap))
            for target_volume, sub_volume in zip(shape, original_noise.shape)
        )

        total_fit_count = fit_count[0] * fit_count[1] * fit_count[2]

        # Generate new noise volumes (amount equal to how often the noise volume fits into the target shape)
        noise_volumes = noise_gen.generate_new_texture(original_noise, total_fit_count)

        # Apply histogram matching to the generated noise volumes
        for i in range(len(noise_volumes)):
            # Flatten the 3D images to 1D
            original_noise_flat = original_noise.flatten()
            fourier_noise_flat = noise_volumes[i].flatten()

            # Match histograms
            matched_img_flat = exposure.match_histograms(
                fourier_noise_flat, original_noise_flat
            )

            # Reshape back to the original 3D shape
            matched_img = matched_img_flat.reshape(original_noise.shape)

            noise_volumes[i] = matched_img

        # expand the noise volumes to the target shape
        expanded_noise = expand_volume_with_blending(noise_volumes, shape, overlap)

        return water_content, expanded_noise

    def _add_noise_to_grid(self, root_grid, water_content_grid) -> np.array:
        """
        Adds gaussian and perlin noise to the MRI and scales it according to the water saturation of the soil

        Args:
        - grid: 3d (numpy) array of the root container with dimensions (nx, ny, nz)
        - water_content_grid: 3d (numpy) array of the water intensity (nx, ny, nz)

        Returns:
        - grid_noise: 3d (numpy) array of the root container with added noise
        """
        # generate the two noise for the soil and the background (background - almost black part of the MRI)
        water_content_soil, fourier_noise_soil = self._get_fourier_noise(
            root_grid.shape, "../../data_assets/noise/soil_noise/", 10
        )
        _, fourier_noise_background = self._get_fourier_noise(
            root_grid.shape, "../../data_assets/noise/background_noise/", 10
        )

        # scale the fourier noise to water content of 100% to later adjust it to the water content of the soil
        fourier_noise_rescaled = fourier_noise_soil / water_content_soil

        # combine the two noises
        water_scaled_fourier_noise = water_content_grid * fourier_noise_rescaled

        # use the fourier noise for the root signal intensity
        # root_noise = self._rescale_image(
        #     fourier_noise_soil,
        #     0.9,
        #     1,
        # )

        # create the noisy root signal, which only contains the root with some noise
        # root_noise[root_grid == 0] = 0

        # scale the water content between min_water_content_scale and 1 where 1 is 100% meaning maximum water content (0.42 because
        # the maximum water content is 42% for both sand and loam) -- / 0.8 + 0.5 is to scale it between 50 and 100%
        min_water_content_scale = 0.5
        root_water_scale = (
            water_content_grid / 0.42 * (1 - min_water_content_scale)
            + min_water_content_scale
        )

        # apply the fourier noise on the root signal (seperated from the soil noise just in case
        # we want to apply different noise to the root and the soil)
        noisy_root = (
            # root_noise_inv * root_grid
            root_grid
            # * root_noise
            * root_water_scale
        )

        # For SNR ratio: Calculating the variance of the root signal and outputting it
        # positive_values = noisy_root[noisy_root > 0]
        # mean = np.mean(positive_values)
        # print("noisy root variance", np.var(positive_values - mean))

        # Calculating the variance of the soil signal and outputting it
        # soil_values = water_scaled_fourier_noise[water_scaled_fourier_noise > 0]
        # mean = np.mean(soil_values)
        # print("soil variance", np.var(soil_values - mean))

        # mask the background noise, such that it only applies outside of the root-soil cylinder
        fourier_noise_background[water_content_grid != 0] = 0

        # set the values for the root which are lower than the surrounding soil to a fraction of the surrounding
        # soil and the root signal intensity
        root_background_fraction = 0.9
        # create a mask for the water content grid, such that only the part remains, which is part of the root
        water_root_filtered = np.where(noisy_root > 0, water_scaled_fourier_noise, 0)
        noisy_root[water_root_filtered > noisy_root] = water_root_filtered[
            water_root_filtered > noisy_root
        ] * root_background_fraction + noisy_root[water_root_filtered > noisy_root] * (
            1 - root_background_fraction
        )

        water_scaled_fourier_noise[root_grid > 0] = 0
        print("water_scaled_fourier_noise", water_scaled_fourier_noise.shape)
        print("noisy_root", noisy_root.shape)
        print("background_noise", fourier_noise_background.shape)

        fourier_noise_background = fourier_noise_background * water_content_grid.mean()

        water_scaled_fourier_noise[water_scaled_fourier_noise < 0] = 0

        final_mri = water_scaled_fourier_noise + noisy_root + fourier_noise_background

        final_mri[final_mri < 0] = 0

        return final_mri

    def _get_grid_water_content(self, X, Y, Z) -> np.array:
        """
        calculates the water content of the soil for the given coordinates by interpolation of the information
        from the water simulation from the .vtu file.

        Args:
        - X, Y, Z: respective coordinate in the root container for which we want the water simulation value
        - grid_values: 3d (numpy) array of the root container with dimensions (nx, ny, nz)

        Returns:
        - water_intensity_grid: 3d (numpy) array of the water intensity
        """

        interpolator = VTUInterpolatedGrid(
            self.vtu_path, resolution=[self.resx, self.resy, self.resz]
        )
        print("vtu_path", self.vtu_path)

        print("interpolator initialized")

        _, grid_data = interpolator.process(
            interpolating_coords=[X[:-1], Y[:-1], Z[:-1]]
        )

        water_intensity_grid = np.nan_to_num(grid_data, nan=0)
        water_intensity_grid = water_intensity_grid.reshape(self.nx, self.ny, self.nz)

        # # previously used to check, if the water intensity grid scaling is working because the difference
        # # in the saturation of the soil with water is very small depending on the soil type
        # scaled_arr = (grid_data - min_gd) / (max_gd - min_gd)

        return water_intensity_grid

    def _print_progress_bar(self, iteration, total, info="", bar_length=50):
        """
        prints a progress bar to the console, which is updated with each iteration

        Args:
        - iteration: current iteration
        - total: total number of iterations
        - info: additional information to be printed
        - bar_length: length of the progress bar
        """
        progress = iteration / total
        arrow = "=" * int(round(progress * bar_length) - 1) + ">"
        spaces = " " * (bar_length - len(arrow))

        sys.stdout.write(f"\rProgress: [{arrow + spaces}] {int(progress*100)}% {info}")
        sys.stdout.flush()  # This is important to ensure the progress is updated

    def _add_root_to_v_mri(self, mri_grid, xx, yy, zz, res, binary):
        """
        adds the root to the MRI grid by adding a white color, where the root is located and a light
        grey depending on how much of a root segment is present in a cell.

        Args:
        - mri_grid: 3d (numpy) array of the root container with dimensions (nx, ny, nz)
        - xx, yy, zz: respective coordinate in the root container
        - res: resolution of the MRI
        """

        idxrad = np.argsort(self.seg_radii)

        cellvol = res["x"] * res["y"] * res["z"]

        iteration = 1
        total_segs = len(self.segs)
        for k, _ in enumerate(self.segs):
            # The list allidx will eventually contain the discretized 3D indices in the grid for all the points
            # along the segment. This part of simulation/visualization is discretizing the root system into a 3D grid,
            # and allidx_ is helping you keep track of which grid cells are occupied by the segment.
            root_signal_intensity = 1 if binary else random.uniform(0.9, 1)

            allidx = self._get_root_segment_idx(
                self.segs[idxrad[k]], self.nodes, xx, yy, zz, res
            )

            if len(allidx) < 1:
                print("warning: root segment out of scope")

            self._print_progress_bar(
                iteration,
                len(self.segs),
                info="Adding root segment {} of {}".format(iteration, total_segs),
            )

            mri_grid_zero = np.zeros(mri_grid.shape)
            # checks if the diameter is greater than the resolution
            if np.round(self.seg_radii[idxrad[k]] * 2 / res["x"]) > 1:
                if len(allidx) > 0:
                    # set the element of the root to 1, indicating that it is present
                    mri_grid_zero[allidx[:, 0], allidx[:, 1], allidx[:, 2]] = 1
                    mri_grid_zero
                    # idx contains the indices of the binary dilation across the root segment

                    # Extracting min and max along each dimension
                    min_x, min_y, min_z = np.min(allidx, axis=0)
                    max_x, max_y, max_z = np.max(allidx, axis=0)

                    # Define the extension of the sub-volume which is by the radius lager
                    n = int(self.seg_radii[idxrad[k]] / res["x"] + 1)

                    # Adjusting the min and max values, while ensuring they remain within the valid range of the array
                    min_x = max(0, min_x - n)
                    min_y = max(0, min_y - n)
                    min_z = max(0, min_z - n)

                    max_x = min(mri_grid_zero.shape[0] - 1, max_x + n)
                    max_y = min(mri_grid_zero.shape[1] - 1, max_y + n)
                    max_z = min(mri_grid_zero.shape[2] - 1, max_z + n)

                    # Slicing the 3D grid to extract the expanded sub-volume
                    expanded_sub_volume = mri_grid_zero[
                        min_x : max_x + 1,
                        min_y : max_y + 1,
                        min_z : max_z + 1,
                    ]

                    indices, values = self._get_binary_dilation_root_segment_idx(
                        expanded_sub_volume,
                        self.seg_radii[idxrad[k]],
                        res,
                        binary,
                    )
                    for idx, value in zip(zip(*indices), values):
                        expanded_sub_volume[idx[0], idx[1], idx[2]] = (
                            max(expanded_sub_volume[idx[0], idx[1], idx[2]], value)
                            * root_signal_intensity
                        )

                    mri_grid_zero[
                        min_x : max_x + 1,
                        min_y : max_y + 1,
                        min_z : max_z + 1,
                    ] = expanded_sub_volume

                    indices = np.nonzero(mri_grid_zero)
                    values = mri_grid_zero[indices]

                    for idx, value in zip(zip(*indices), values):
                        mri_grid[idx[0], idx[1], idx[2]] = max(
                            mri_grid[idx[0], idx[1], idx[2]], value
                        )

            else:
                estlen = res[
                    "x"
                ]  # estimated segment length within voxel: very rough estimation
                rootvol = self.seg_radii[idxrad[k]] ** 2 * math.pi * estlen

                # set how the intensity of the root signal should be
                frac = rootvol / cellvol
                if frac > 1:
                    frac = 1
                # sets
                if len(allidx) > 0:
                    mri_grid_zero[allidx[:, 0], allidx[:, 1], allidx[:, 2]] = (
                        1 * root_signal_intensity
                    )

                # set root voxels to the appropriate value
                idx = np.argwhere(mri_grid_zero == 1)
                for x, y, z in idx:
                    mri_grid[x, y, z] = frac * root_signal_intensity

            iteration += 1

        mri_grid[mri_grid > 1] = 1

        if not binary:
            # scale the root signal intensity to the maximum signal intensity
            mri_grid = mri_grid * self.max_root_signal_intensity

        mri_grid = np.floor(mri_grid).astype(int)

        print(
            "\n"
            + "\033[33m"
            + "===================================================="  # Green text
            + "\n"
            + "||        Adding root segments: COMPLETE!         ||"
            + "\n"
            + "===================================================="
            + "\033[0m"
        )  # Reset text color

        return mri_grid

    def _rescale_image(self, image, new_min, new_max):
        """
        Rescale the image to have a new minimum and maximum.

        - image: numpy array representing the image.
        - new_min: New minimum value for the rescaled image.
        - new_max: New maximum value for the rescaled image.
        """
        # Find the original minimum and maximum values of the image
        orig_min = np.min(image)
        orig_max = np.max(image)

        # Rescale the image
        rescaled_image = (image - orig_min) / (orig_max - orig_min)  # Scale to 0-1
        rescaled_image = (
            rescaled_image * (new_max - new_min) + new_min
        )  # Scale to new_min-new_max

        return rescaled_image

    def _create_3d_array(self, x_dim, y_dim, z_dim):
        """
        Create a 3D numpy array with specified dimensions where values range from 0.2 to 0.4 along the z-axis linearly.
        Each slice in the z-axis will contain only one unique value.

        Args:
        - z_dim (int): Number of slices along the z-axis.
        - y_dim (int): Dimension along the y-axis.
        - x_dim (int): Dimension along the x_dim axis.

        Returns:
        - np.ndarray: A 3D array with linearly increasing values along the z-axis.
        """
        # Generate linearly spaced values between 0.2 and 0.4 for the z-axis
        z_values = np.linspace(0.15, 0.4, z_dim)

        # Create an empty 3D array
        array = np.zeros((x_dim, y_dim, z_dim))

        # Assign the z_values to each slice along the z-axis
        for i, value in enumerate(z_values):
            array[:, :, i] = value

        return array

    # TODO: remove
    def generate_random_array(self, shape, value_range):
        """
        Generate a numpy array with a given shape and random values in a specified range.

        - Tuple representing the shape of the array.
        - Tuple (min, max) representing the range of values.
        """
        min_val, max_val = value_range
        return np.random.uniform(min_val, max_val, shape)

    def create_virtual_root_mri(
        self,
        mri_output_path,
        add_noise=True,
        label=False,
    ) -> Tuple[np.array, int]:
        """
        creates a virtual MRI for the given root system with simulating noise based on the water content of the soil.

        Args:
        - mri_output_path: path to the folder, where the generated virtual MRI should be saved
        - add_noise: boolean, if True, noise, scaled by the water intensity is added to the MRI
        - label: boolean, if True, the MRI is saved as a numpy file with values 0 and 1, where 1 indicates the presence of a root
                        if Flase, the MRI is saved as a raw file similar to a real MRI

        Returns:
        - mri_final_grid: 3d (numpy) array of the root container
        """

        # throw an error if self.soil_type is None
        if self.soil_type is None and add_noise is True:
            raise ValueError("The soil type must be specified to add noise to the MRI")

        # xx etc. are lists of consecutive values, which represent the coordinates of the MRI when scaled with
        # the resolution
        xx, yy, zz = self._get_dimensions_container_array(self.nx, self.ny, self.nz)

        # create a 3d array with the dimensions of the MRI
        mri_grid = np.zeros((self.nx, self.ny, self.nz))

        # set dictionaries with the resolution and pass them to the function, which adds the root to the MRI
        dict_res = {"x": self.resx, "y": self.resy, "z": self.resz}

        mri_grid = self._add_root_to_v_mri(
            mri_grid,
            xx,
            yy,
            zz,
            dict_res,
            binary=label,
        )

        # TODO: maybe remove if not good
        if not label:
            mri_grid = gaussian_filter(mri_grid, sigma=(0.6, 0.6, 0.15))

        # create a grid of the actual coordinates, represented by each voxel of the MRI
        X = np.linspace(
            -1 * self.width,
            -1 * self.width + self.nx * self.resx,
            self.nx + 1,
        )
        Y = np.linspace(
            -1 * self.width,
            -1 * self.width + self.ny * self.resy,
            self.ny + 1,
        )
        Z = np.linspace(
            0,
            -self.nz * self.resz,
            self.nz + 1,
        )

        # add noise scaled by the water content in the soil to make it more realistic (more noise where more
        # water is present)
        if add_noise and not label:
            # calculate the water content of the grid
            # for debugging something not related to the water content uncomment the following line and comment the next one
            # for faster execution
            # water_grid = self._create_3d_array(mri_grid.shape[0], mri_grid.shape[1], mri_grid.shape[2])
            # water_grid = self.generate_random_array(mri_grid.shape, (0.39, 0.42))
            water_grid = self._get_grid_water_content(X, Y, Z)
            print("water_grid", water_grid.shape)
            # add noise to the MRI scaled by the water content
            mri_grid = self._add_noise_to_grid(mri_grid, water_grid)

        mri_final_grid = mri_grid

        root_system_name = self.rsml_path.split("/")[-1].split(".")[0]
        filename = f"{mri_output_path}/{'label_' if label else ''}{root_system_name}_res_{self.nx}x{self.ny}x{self.nz}"

        # for the labeling save it compressed as a h5 file
        mri_final_grid = mri_final_grid.astype("int16")

        if label:
            # replace the values in mri_final_grid with 0 and 1, where 1 indicates the presence of a root
            mri_final_grid[mri_final_grid > 0] = 1

        print("mri_final_grid min", mri_final_grid.min())
        print("mri_final_grid max", mri_final_grid.max())

        # save as a nifti file
        mri_ops = MRIoperations()
        mri_ops.save_mri(filename + ".nii.gz", mri_final_grid)

        print(filename)

        # create a file containing the root idx (label where the root is present in the original MRI)

        return mri_final_grid, filename


# Example usage:
# offset = (4.1599, -8.2821, -0.4581)
#
# my_root = Virtual_MRI(
#     rsml_path="./roots_vr_18.rsml",
#     res_mri=(0.027, 0.027, 0.1),
#     width=3.46,  # TODO: adjust
#     depth=13.1,  # TODO: adjust
#     offset=offset,
# )
# my_root.create_virtual_root_mri(".", label=True)
