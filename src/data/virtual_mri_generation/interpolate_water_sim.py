import numpy as np
import vtk
from scipy.interpolate import LinearNDInterpolator
import SimpleITK as sitk
import time
from typing import Tuple

from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv


class VTUInterpolatedGrid:
    """
    For given coordinates and a given soil water simulation, this class computes the values of the water
    for the given coordinates by interpolating the values of the soil water simulation.

    Parameters:
    - vtu_path (str): Path to the vtu file containing the soil water simulation data.
    - resolution (list): Resolution of the grid, which will be interpolated.
    - scale_factor (int): Factor by which the grid will be scaled. Default is 100 because the output of CPlantBox
                            is factor 100 smaller than we need it to be, maybe due to different units.
    - interpolating_coords (list): List of 3 arrays, containing the x, y and z coordinates of the points, for which
                                    the water sim values will be set by the interpolator.
    """

    def __init__(self, vtu_path, resolution=[0.027, 0.027, 0.1], scale_factor=100):
        self.vtu_path = vtu_path
        self.resolution = resolution
        self.scale_factor = scale_factor

    def load_vtu(self):
        """
        loads the vtu file, containing the simulation data for the water soil root simulation
        """
        # wait for 3 seconds to make sure the file is written
        time.sleep(3)
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(self.vtu_path)
        reader.Update()

        return reader.GetOutput()

    def extract_vtu_data(self, grid):
        """
        extracts the cells, points and cell data from the soil model

        Args:
        - grid (vtk.vtkUnstructuredGrid): The grid to extract the data from.

        Returns:
        - points (np.ndarray): Array containing the coordinates of the points of the mesh.
        """
        # setting the center coordinates of the cells
        points = vtk_to_numpy(grid.GetPoints().GetData())
        cells = vtk_to_numpy(grid.GetCells().GetConnectivityArray()).reshape(-1, 8)
        cell_data = vtk_to_numpy(grid.GetCellData().GetArray(3))

        return points, cells, cell_data

    def _get_cylindar_surface_data(self, grid) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the datapoints of the mesh spanning the surface of the cylinder and the associated data values
        which are averaged over the cells that use the point.

        Returns:
        - surface_coords (np.ndarray): Array containing the coordinates of the surface points.
        - data_values_arr (np.ndarray): Array containing the data values of the surface points.
        """
        # setting the surface of the cylinder
        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputData(grid)
        surface_filter.Update()

        surface_points = surface_filter.GetOutput().GetPoints()
        surface_coords = np.array(
            [
                surface_points.GetPoint(i)
                for i in range(surface_points.GetNumberOfPoints())
            ]
        )

        # setting the surface of the cylinder
        cell_data = vtk_to_numpy(surface_filter.GetOutput().GetCellData().GetArray(3))

        coords_arr = []
        data_values_arr = []
        for i in range(surface_points.GetNumberOfPoints()):
            coord = surface_coords[i]
            associated_data = []

            # Loop through cells to find those that use the current point
            for j in range(surface_filter.GetOutput().GetNumberOfCells()):
                cell = surface_filter.GetOutput().GetCell(j)
                if i in [cell.GetPointId(k) for k in range(cell.GetNumberOfPoints())]:
                    associated_data.append(cell_data[j])

            # Store the average data value for the point (you can adjust this to median, mode, etc. if desired)
            if associated_data:
                data_values_arr.append(sum(associated_data) / len(associated_data))

        return surface_coords, np.array(data_values_arr)

    def scale_grid(self, grid):
        """
        scales the grid by the scale factor given in the constructor (this is required since the output
        of CPlnatBox is factor 100 smaller than we need it to be, maybe due to different units)

        Args:
        - grid (vtk.vtkUnstructuredGrid): The grid to scale.

        Returns:
        - vtk.vtkUnstructuredGrid: The scaled grid.
        """
        transform = vtk.vtkTransform()
        transform.Scale(self.scale_factor, self.scale_factor, self.scale_factor)

        transform_filter = vtk.vtkTransformFilter()
        transform_filter.SetInputData(grid)
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        return transform_filter.GetOutput()

    def visualize_3d_coords(self, coords, data, path):
        """
        visualizes the given coordinates and the associated data values in a 3d plot. The data values are
        colored based on how high they are. The figure is saved to the given path.

        Args:
        - coords (np.ndarray): Array containing the coordinates of the points to visualize.
        - data (np.ndarray): Array containing the data values of the points to visualize.
        - path (str): Path to save the figure to.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=data, cmap="jet")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # add a color bar which maps values to colors
        fig.colorbar(
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=data, cmap="jet"),
            label="Cell Data",
        )

        plt.savefig(path)

    def create_3d_coords_array(self, points):
        """
        creates a 3d coordinate array spanning evenly spaced across the min and max coords of the soil model

        Args:
        - points (np.ndarray): Array containing the coordinates of the points, for which the water sim values will be set
                               by the interpolator.

        Returns:
        - np.ndarray: Array containing the coordinates of the points, for which the water sim values will be set
                      by the interpolator.
        """
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)

        x_width = max_coords[0] - min_coords[0]
        y_width = max_coords[1] - min_coords[1]
        z_width = max_coords[2] - min_coords[2]

        x = np.linspace(min_coords[0], max_coords[0], int(x_width / self.resolution[0]))
        y = np.linspace(min_coords[1], max_coords[1], int(y_width / self.resolution[1]))
        z = np.linspace(min_coords[2], max_coords[2], int(z_width / self.resolution[2]))

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        return np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)

    def _set_3d_coords_array(self, coords):
        """
        Covert the given 3 arrays of coordinates into a 3d coordinate array which can be used for interpolation.

        Args:
        - coords (list): List of 3 arrays, containing the x, y and z coordinates of the points, for which
                         the water sim values will be set by the interpolator.

        Returns:
        - np.ndarray: Array containing the coordinates of the points, for which the water sim values will be set
                      by the interpolator.
        """
        X, Y, Z = np.meshgrid(coords[0], coords[1], coords[2], indexing="ij")

        return np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)

    def _calculate_cell_centers(self, points, cells) -> np.ndarray:
        """
        Calculate the centers of the given cells of a mesh.

        Args:
        - points (np.ndarray): Array containing the coordinates of the points of the mesh.
        - cells (np.ndarray): Array containing the cells of the mesh.

        Returns:
        - cell_centers (np.ndarray): Array containing the center coordinates of the cells of the mesh.
        """
        cell_centers = []
        for cell in cells:
            cell_coords = points[cell]
            center = cell_coords.mean(axis=0)
            cell_centers.append(center)

        return np.array(cell_centers)

    def process(self, interpolating_coords=None):
        """
        main function which needs to be run to interpolate the data of the soil water simulation and
        set the values for the given coordinates accordingly. The interpolated data is saved to an external
        file if multiple runs are made for debugging purposes.

        Args:
        - interpolating_coords (list): List of 3 arrays, containing the x, y and z coordinates of the points, for which
                                       the water sim values will be set by the interpolator.

        Returns:
        - new_grid_coords (np.ndarray): Array containing the coordinates of the points, for which the water sim values
                                        will be set by the interpolator.
        - new_grid_interpolated_data (np.ndarray): Array containing the interpolated data values of the points (corresponding
                                                  to the coordinates in new_grid_coords)
        """
        grid = self.load_vtu()
        grid = self.scale_grid(grid)
        points, cells, cell_data = self.extract_vtu_data(grid)

        if interpolating_coords is None:
            interpolating_coords = self.create_3d_coords_array(points)
        else:
            interpolating_coords = self._set_3d_coords_array(interpolating_coords)

        cell_centers = self._calculate_cell_centers(points, cells)
        mesh_surface_coords, mesh_surface_data = self._get_cylindar_surface_data(grid)

        grid_coords = np.concatenate((cell_centers, mesh_surface_coords), axis=0)
        grid_data = np.concatenate((cell_data, mesh_surface_data), axis=0)

        start_time = time.time()
        interpolator = LinearNDInterpolator(grid_coords, grid_data)

        # self.visualize_3d_coords(grid_coords, grid_data, "./vis/original_grid.png")

        print("Start interpolation")

        interpolated_data = interpolator(interpolating_coords)

        end_time = time.time()
        print("--- %s seconds ---" % (end_time - start_time))

        print("interpolated_data", interpolated_data.shape)

        return interpolating_coords, interpolated_data


# # Example Usage
# width = 3
# depth = 20

# resx = 0.027
# resy = 0.027
# resz = 0.1

# nx = int(width * 2 / resx)
# ny = int(width * 2 / resy)
# nz = int(depth / resz)

# X = np.linspace(
#     -1 * width,
#     -1 * width + nx * resx,
#     nx + 1,
# )
# Y = np.linspace(
#     -1 * width,
#     -1 * width + ny * resy,
#     ny + 1,
# )
# Z = np.linspace(
#     0,
#     -nz * resz,
#     nz + 1,
# )

# interpolator = VTUInterpolatedGrid(
#     "/Users/daniel/Desktop/FZJ/CPlantBox/DUMUX/CPlantBox/tutorial/examples_segmentation/RootNet/src/data/virtual_mri_generation/vtu_loam_2.vtu",
#     resolution=[resx, resy, resz],
# )
# _, grid_data = interpolator.process(
#     interpolating_coords=[X[:-1], Y[:-1], Z[:-1]],
# )
