"""simulates a root system, which is rasterized to a given resolution. To mimic an MRI image, Gaussian noise is additionally added"""
import sys

sys.path.append("/Users/daniel/Desktop/FZJ/CPlantBox/DUMUX/CPlantBox")
sys.path.append(
    "/Users/daniel/Desktop/FZJ/CPlantBox/DUMUX/CPlantBox/experimental/parametrisation/"
)
sys.path.append("/Users/daniel/Desktop/FZJ/CPlantBox/DUMUX/CPlantBox/src")

import plantbox as pb
import rsml.rsml_reader as rsml_reader
import visualisation.vtk_plot as vp
import numpy as np
import functional.bresenham3D as bres3D
import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK
import rsml_reader as rsml
import scipy
from .root_growth_simulation import RootSystemSimulation as RSSim
from scipy import ndimage
from .interpolate_water_sim import VTUInterpolatedGrid
from noise import pnoise3

# import pygorpho as pg
import skimage
from skimage.morphology import ball
import math
from scipy.ndimage import distance_transform_edt
import nibabel as nib
import random
import os
from typing import Tuple
import time


def get_min_max_numpy(array):
    """
    since the min max function of numpy does
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
        vtu_path,
        perlin_noise_intensity,
        seganalyzer=None,
        width=3,
        depth=20,
        res_mri=[0.027, 0.027, 0.1],
        snr=3,
        scale_factor=1,
    ):
        """
        creates a virtual MRI for the given root system with simulating noise based on the water content of the soil.

        Parameters:
        - rootsystem: root system object (pb.RootSystem)
        - rsml_path: path to the rsml file of the root system # currently not used but if it is possible to read the
                     rsml file and get the seganalyser from it, it would be better
        - vtu_path: path to the vtu file of the water simulation
        - perlin_noise_intensity: intensity of how much perlin noise should be added (value between 0 and 1)
        - width: width of the soil container
        - depth: depth of the soil container
        - resolution: resolution of the MRI
        - snr: signal to noise ratio
        """
        self.resx = res_mri[0] / scale_factor
        self.resy = res_mri[1] / scale_factor
        self.resz = res_mri[2] / scale_factor
        self.res_mri = res_mri
        self.scale_factor = scale_factor
        self.width = width
        self.depth = depth
        self.rsml_path = rsml_path
        self.vtu_path = vtu_path
        self.perlin_noise_intensity = perlin_noise_intensity
        if seganalyzer is None:
            self.segana = self._get_root_data_from_rsml(rsml_path)
        else:
            self.segana = seganalyzer
        self.snr = snr
        self.water_intensity_grid = None

        self.nx = int(self.width * 2 / self.resx)
        self.ny = int(self.width * 2 / self.resy)
        self.nz = int(self.depth / self.resz)

        self.max_signal_intensity = 30000
        self.max_root_signal_intensity = 30000

    def _add_gaussian_noise(self, image, sigma, water_intensity_grid) -> np.array:
        """
        adds gaussian noise to the image with the given mean and variance

        Args:
        image (numpy array): image to which the noise is added
        mean (float): mean of the gaussian distribution
        sigma (float): variance of the gaussian distribution

        Returns:
        noisy_image (numpy array): 3d image with added gaussian noise
        """
        row, col, ch = image.shape
        gauss = np.random.normal(0, sigma, (row, col, ch))

        gauss_water_scaled = np.multiply(gauss, water_intensity_grid)

        noisy_plus_image = image + gauss_water_scaled

        return noisy_plus_image

    def _add_perlin_noise(self, image, water_intensity_grid) -> np.array:
        """
        Add Perlin noise to a 3D image.

        Parameters:
        - image: Input 3D image (numpy array).
        - intensity: Intensity of the noise (multiplier for noise values).

        Returns:
        - noisy_image: 3D image with added Perlin noise.
        """
        noise_array = self._generate_perlin_noise_3d(image.shape, 5, 5, 15)

        # Scale noise to fit the desired intensity
        noise_array_scaled = (
            self.perlin_noise_intensity
            * water_intensity_grid
            * ((self.max_signal_intensity * 2 * noise_array))
        )

        # Add noise to the image
        image_plus_noise = image + noise_array_scaled

        # Clip values to ensure they remain in the valid range [0, self.max_signal_strength]
        noisy_image = np.clip(image_plus_noise, 0, self.max_signal_intensity).astype(
            np.int16
        )

        return noisy_image

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
                        base=0,
                    )
                    # Normalize to [0, 1]
                    # noise_value = (noise_value + 1) / 2

                    noise[i][j][k] = noise_value

        return noise

    def _isConsecutive(self, A) -> bool:
        """
        checks if the array only contains consecutive values like [1,2,3,4,5], nothing else is accepted

        Args:
        - A: array to be checked

        Returns:
        - bool: True if the array only contains consecutive values, False otherwise
        """
        if len(A) <= 1:
            return True

        minimum = min(A)
        maximum = max(A)

        if maximum - minimum != len(A) - 1:
            return False

        visited = set()
        for i in A:
            if i in visited:
                return False
            visited.add(i)
        return True

    def _get_root_data_from_rsml(self, rsml_path) -> pb.SegmentAnalyser:
        """
        Gets the pb.SegmentAnalyser for the given rsml file

        Args:
        - rsml_path: path to the rsml file

        Returns:
        - segana: pb.SegmentAnalyser
        """
        polylines, properties, functions, _ = rsml_reader.read_rsml(rsml_path)

        nodes, segs = rsml_reader.get_segments(polylines, properties)
        segs_ = [pb.Vector2i(s[0], s[1]) for s in segs]  # convert to CPlantBox types
        nodes_ = [pb.Vector3d(n[0], n[1], n[2]) for n in nodes]
        segRadii = np.zeros((segs.shape[0], 1))  # convert to paramter per segment
        segCTs = np.zeros((segs.shape[0], 1))

        segana = pb.SegmentAnalyser(nodes_, segs_, segCTs, segRadii)

        return segana

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
        (x1, y1, z1) = [
            np.around(n1.x / res["x"]),
            np.around(n1.y / res["y"]),
            np.around(n1.z / res["z"]),
        ]
        (x2, y2, z2) = [
            np.around(n2.x / res["x"]),
            np.around(n2.y / res["y"]),
            np.around(n2.z / res["z"]),
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

    def _get_binary_delation_root_segment_idx(self, grid, radius, res) -> np.array:
        """
        Expands the coordinates where the root segment is located by its radius and returns the indexes of the
        expanded coordinates

        Args:
        - array: 3d array of the root container with dimensions (nx, ny, nz)
        - radius: radius of the root segment
        """
        # if radius < 0.1: # TODO: Check if required
        #     radius = 0.1
        width = int(np.around(radius / res["x"]))
        height = int(np.around(radius / res["z"]))
        height = height if height > 0 else 1

        selem = self.ellipsoid(width, width, height)

        grid = ndimage.binary_dilation(grid, structure=selem).astype(grid.dtype)
        # grid = np.reshape(grid, (dims['x']*dims['y']*dims['z']))
        idx = np.argwhere(grid == 1)

        return idx

    def _add_noise_to_grid(self, grid, water_intensity_grid) -> np.array:
        """
        Adds gaussian and perlin noise to the MRI and scales it according to the water saturation of the soil

        Args:
        - grid: 3d (numpy) array of the root container with dimensions (nx, ny, nz)
        - water_intensity_grid: 3d (numpy) array of the water intensity (nx, ny, nz)

        Returns:
        - grid_noise: 3d (numpy) array of the root container with added noise
        """
        ##### Gaussian Noise #####
        # calculation of the power signal but since all MRIs are very similar, calculating it, doesn't
        # really add value and just makes it more complicated, so the Power noise (pn) is just set to
        # a fixed value
        # ps = np.sum((grid - np.mean(grid)) ** 2) / (grid).size
        pn = 8 * 100000000
        sigma = np.sqrt(pn)

        # gaussian noise is added to the grid and then the grid cut off at min and max grey value intensity
        grid_noise = self._add_gaussian_noise(grid, sigma, water_intensity_grid)
        grid_noise = np.clip(grid_noise, 0, self.max_signal_intensity).astype(np.int16)

        ##### Perlin Noise #####
        grid_noise = self._add_perlin_noise(grid_noise, water_intensity_grid)
        grid_noise = np.clip(grid_noise, 0, self.max_signal_intensity).astype(np.int16)

        return grid_noise

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

        _, grid_data = interpolator.process(interpolating_coords=[X[:-1], Y[:-1], Z[:-1]])

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

    def _add_root_to_v_mri(self, mri_grid, xx, yy, zz, root_signal_intensity, res):
        """
        adds the root to the MRI grid by adding a white color, where the root is located and a light
        grey depending on how much of a root segment is present in a cell.

        Args:
        - mri_grid: 3d (numpy) array of the root container with dimensions (nx, ny, nz)
        - xx, yy, zz: respective coordinate in the root container
        - root_signal_intensity: intensity of the root signal
        - res: resolution of the MRI
        """

        nodes = self.segana.nodes
        segs = self.segana.segments
        radius = self.segana.getParameter("radius")

        idxrad = np.argsort(radius)

        cellvol = res["x"] * res["y"] * res["z"]

        iteration = 1
        total_segs = len(segs)
        for k, _ in enumerate(segs):
            # The list allidx will eventually contain the discretized 3D indices in the grid for all the points
            # along the segment. This part of simulation/visualization is discretizing the root system into a 3D grid,
            # and allidx_ is helping you keep track of which grid cells are occupied by the segment.
            allidx = self._get_root_segment_idx(segs[idxrad[k]], nodes, xx, yy, zz, res)

            if len(allidx) < 1:
                print("warning: root segment out of scope")

            self._print_progress_bar(
                iteration,
                len(segs),
                info="Adding root segment {} of {}".format(iteration, total_segs),
            )

            mri_grid_zero = np.zeros(mri_grid.shape)
            # checks if the diameter is greater than the resolution
            if np.round(radius[idxrad[k]] * 2 / res["x"]) > 1:
                if len(allidx) > 0:
                    # set the element of the root to 1, indicating that it is present
                    mri_grid_zero[allidx[:, 0], allidx[:, 1], allidx[:, 2]] = 1
                    mri_grid_zero
                    # idx contains the indices of the binary dilation across the root segment

                    # Extracting min and max along each dimension
                    min_x, min_y, min_z = np.min(allidx, axis=0)
                    max_x, max_y, max_z = np.max(allidx, axis=0)

                    # Define the extension of the sub-volume which is by the radius lager
                    n = int(radius[idxrad[k]] / res["x"] + 1)

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

                    idx = self._get_binary_delation_root_segment_idx(
                        expanded_sub_volume,
                        radius[idxrad[k]],
                        res,
                    )
                    for x, y, z in idx:
                        expanded_sub_volume[x, y, z] = root_signal_intensity

                    mri_grid_zero[
                        min_x : max_x + 1,
                        min_y : max_y + 1,
                        min_z : max_z + 1,
                    ] = expanded_sub_volume

                    idx = np.argwhere(mri_grid_zero == root_signal_intensity)

                    for x, y, z in idx:
                        mri_grid[x, y, z] = root_signal_intensity

            else:
                estlen = res[
                    "x"
                ]  # estimated segment length within voxel: very rough estimation
                rootvol = radius[idxrad[k]] ** 2 * math.pi * estlen

                # set how the intensity of the root signal should be
                frac = rootvol / cellvol
                if frac > 1:
                    frac = 1
                # sets
                if len(allidx) > 0:
                    mri_grid_zero[allidx[:, 0], allidx[:, 1], allidx[:, 2]] = 1

                # set root voxels to the appropriate value
                idx = np.argwhere(mri_grid_zero == 1)
                for x, y, z in idx:
                    mri_grid[x, y, z] = int(np.floor(frac * root_signal_intensity))

            iteration += 1

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

    def _get_avg_numpy_array_non_zero(self, array) -> float:
        """
        calculates the average of the non zero values of a numpy array

        Args:
        - array: numpy array
        """
        non_zero = array[array != 0]
        if non_zero.size == 0:
            return 0
        else:
            return np.mean(non_zero)

    def _save_as_nifti(self, mri_grid, filename):
        """
        saves the mri grid as a nifti file

        Args:
        - mri_grid: 3d (numpy) array of the root container with dimensions (nx, ny, nz)
        - filename: path to the nifti file
        """
        # since all the mris have swapped z and x coordinate, the dimensions have to be swapped as well
        affine_transformation = np.array(
            [
                [self.resz, 0.0, 0.0, 0.0],
                [0.0, self.resy, 0.0, 0.0],
                [0.0, 0.0, self.resx, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # create a nifti file
        img = nib.Nifti1Image(mri_grid, affine_transformation)
        # save the nifti file
        nib.save(img, filename)

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

        # xx etc. are lists of consecutive values, which represent the coordinates of the MRI when scaled with
        # the resolution
        xx, yy, zz = self._get_dimensions_container_array(self.nx, self.ny, self.nz)

        # create a 3d array with the dimensions of the MRI
        mri_grid = np.zeros((self.nx, self.ny, self.nz))

        # set dictionaries with the resolution and pass them to the function, which adds the root to the MRI
        dict_res = {"x": self.resx, "y": self.resy, "z": self.resz}
        self._add_root_to_v_mri(
            mri_grid,
            xx,
            yy,
            zz,
            self.max_signal_intensity,
            dict_res,
        )

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

        # calculate the water content of the grid
        if not label:
            water_grid = self._get_grid_water_content(X, Y, Z)
            # scale the root signal intensity slightly by the water content (max intensity influence is a
            # reduction by 30%)
            min_scale_root = 0.7
            max_scale_root = 1
            scaled_array_root = (
                water_grid * (max_scale_root - min_scale_root) + min_scale_root
            )
            mri_grid = mri_grid * scaled_array_root

        # add noise scaled by the water content in the soil to make it more realistic (more noise where more
        # water is present)
        if add_noise and not label:
            # add noise to the MRI scaled by the water content
            mri_grid = self._add_noise_to_grid(mri_grid, water_grid)

        # do the necessary tranformations to the grid, such that it has the same format as an original MRI
        nx = int(self.width * 2 / self.resx) * self.scale_factor
        ny = int(self.width * 2 / self.resy) * self.scale_factor
        nz = int(self.depth / self.resz) * self.scale_factor

        mri_final_grid = mri_grid[:nx, :ny, :nz]
        mri_final_grid = np.swapaxes(mri_grid, 0, 2)
        mri_final_grid = mri_final_grid[::-1]
        root_system_name = self.rsml_path.split("/")[-1].split(".")[0]
        filename = f"{mri_output_path}/{'label_' if label else ''}{root_system_name}_SNR_{self.snr}_res_{self.nx}x{self.ny}x{self.nz}"

        # for the labeling save it compressed as a h5 file
        if not label:
            mri_final_grid.astype("int16").tofile(filename + ".raw")
        else:
            # replace the values in mri_final_grid with 0 and 1, where 1 indicates the presence of a root
            mri_final_grid[mri_final_grid > 0] = 1

        # save as a nifti file
        self._save_as_nifti(mri_final_grid, filename + ".nii.gz")

        print(filename)

        # create a file containing the root idx (label where the root is present in the original MRI)

        return mri_final_grid, filename


# Example usage:
# rssim = RSSim("Anagallis_femina_Leitner_2010", "../../../../data/generated/root_systems", 3, 20)
# anas, filenames = rssim.run_simulation([10, 11])

# for i in range(len(anas)):
#     my_vi = Virtual_MRI(anas[i], "../../../data/generated/{}".format(filenames[i]), "../../../../soil_simulation_data/generated_20_-90.8_7.1.vtu", 0.5)
#     my_vi.create_virtual_root_mri()
