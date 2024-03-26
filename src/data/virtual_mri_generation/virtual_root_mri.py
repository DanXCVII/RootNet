import sys

with open("../../DUMUX_path.txt", "r") as file:
    DUMUX_path = file.read()

sys.path.append(f"{DUMUX_path}/CPlantBox")
sys.path.append(f"{DUMUX_path}/CPlantBox/experimental/parametrisation/")
sys.path.append(f"{DUMUX_path}/CPlantBox/src")
sys.path.append("..")

import numpy as np
from typing import Tuple
import random
import math
from scipy import ndimage
import plantbox as pb
import rsml.rsml_reader as rsml_reader
import functional.bresenham3D as bres3D


class VirtualRootMRI:
    def __init__(
        self,
        rsml_file,
        root_signal_intensity,
        res_mri,
        width,
        depth,
        binary,
    ):
        self.binary = binary

        self.resx = res_mri[0]
        self.resy = res_mri[1]
        self.resz = res_mri[2]

        self.nx = int(width * 2 / self.resx)
        self.ny = int(width * 2 / self.resy)
        self.nz = int(depth / self.resz)

        self.root_signal_intensity = root_signal_intensity

        self.segana = self._get_root_data_from_rsml(rsml_file)

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
            selem = self.ellipsoid_gaussian(radius_x, radius_z, radius_x * 0.75)

            grid = ndimage.grey_dilation(grid, structure=selem)

            grid = grid - 1

        idx = np.nonzero(grid)
        non_zero_values = grid[idx]

        return idx, non_zero_values

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

    def _get_root_segment_idx(
        self,
        segment,
        nodes,
        x_int,
        y_int,
        z_int,
        res,
        offset,
    ) -> np.array:
        """
        calculates the indexes in the mri grid, where the root segment is located

        Args:
        - segment: root segment
        - nodes: nodes of the root system
        - x_int, y_int, z_int: coordinates of the mri grid
        """
        voxel_offset_x = np.around(offset[0] / res["x"])
        voxel_offset_y = np.around(offset[1] / res["y"])
        voxel_offset_z = np.around(offset[2] / res["z"])

        n1, n2 = nodes[segment.x], nodes[segment.y]
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
            print("\nno idx found")
            print("ListOfPoints\n", ListOfPoints)
            print("x1, y1, z1", x1, y1, z1)
            print("x2, y2, z2", x2, y2, z2)

        return np.array(allidx_)

    def _add_root_to_v_mri(
        self,
        mri_grid,
        xx,
        yy,
        zz,
        res,
        binary,
        offset,
    ):
        """
        adds the root to the MRI grid by adding a white color, where the root is located and a light
        grey depending on how much of a root segment is present in a cell.

        Args:
        - mri_grid: 3d (numpy) array of the root container with dimensions (nx, ny, nz)
        - xx, yy, zz: respective coordinate in the root container
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
            root_signal_intensity = 1 if binary else random.uniform(0.9, 1)

            allidx = self._get_root_segment_idx(
                segs[idxrad[k]],
                nodes,
                xx,
                yy,
                zz,
                res,
                offset,
            )

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

                    indices, values = self._get_binary_dilation_root_segment_idx(
                        expanded_sub_volume,
                        radius[idxrad[k]],
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
                rootvol = radius[idxrad[k]] ** 2 * math.pi * estlen

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
                    mri_grid[x, y, z] = frac * 1 * root_signal_intensity

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

    def create_mri(self, offset):
        # create a 3d array with the dimensions of the MRI
        mri_grid = np.zeros((self.nx, self.ny, self.nz))

        # set dictionaries with the resolution and pass them to the function, which adds the root to the MRI
        dict_res = {"x": self.resx, "y": self.resy, "z": self.resz}

        # xx etc. are lists of consecutive values, which represent the coordinates of the MRI when scaled with
        # the resolution
        xx, yy, zz = self._get_dimensions_container_array(self.nx, self.ny, self.nz)

        mri_grid = self._add_root_to_v_mri(
            mri_grid,
            xx,
            yy,
            zz,
            dict_res,
            self.binary,
            offset,
        )
        print("mri_grid min:", mri_grid.min())
        print("mri_grid max:", mri_grid.max())

        # save the MRI as a raw file
        mri_grid.astype("int16").tofile(f"test_{self.nx}x{self.ny}x{self.nz}.raw")

        return mri_grid


my_root = VirtualRootMRI(
    rsml_file="./roots_vr_18.rsml",
    root_signal_intensity=100,
    res_mri=(0.027, 0.027, 0.1),
    width=4,  # TODO: adjust
    depth=15.1,  # TODO: adjust
    binary=True,
)
offset = (4.1599, -8.2821, -0.7981)
my_root.create_mri(offset=offset)

# the resolution together with the real dimensions of the soil container specify the resolution of the MRI
# ? could also not work but then simply add a padding with zeros around the container to make it fit - actually should work?
