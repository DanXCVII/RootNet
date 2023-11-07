import numpy as np
import vtk
from scipy.interpolate import LinearNDInterpolator
import SimpleITK as sitk
import time

from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

    def __init__(self, vtu_path, resolution=[0.1, 0.1, 0.1], scale_factor=100):
        self.vtu_path = vtu_path
        self.resolution = resolution
        self.scale_factor = scale_factor
        self.unstructured_grid = None
        self.points = None
        self.cells = None
        self.cell_data = None
        self.new_grid_coords = None
        self.new_grid_interpolated_data = None

    def load_vtu(self):
        """
        loads the vtu file, containing the simulation data for the water soil root simulation
        """
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(self.vtu_path)
        reader.Update()

        self.unstructured_grid = reader.GetOutput()

    def extract_vtu_data(self):
        """
        extracts the cells, points and cell data from the soil model
        """
        # setting the center coordinates of the cells
        self.points = vtk_to_numpy(self.unstructured_grid.GetPoints().GetData())
        self.cells = vtk_to_numpy(
            self.unstructured_grid.GetCells().GetConnectivityArray()
        ).reshape(-1, 8)
        self.cell_data = vtk_to_numpy(self.unstructured_grid.GetCellData().GetArray(1))

    def _get_cylindar_surface_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the datapoints of the mesh spanning the surface of the cylinder and the associated data values
        which are averaged over the cells that use the point.

        Returns:
        - surface_coords (np.ndarray): Array containing the coordinates of the surface points.
        - data_values_arr (np.ndarray): Array containing the data values of the surface points.
        """
        # setting the surface of the cylinder
        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputData(self.unstructured_grid)
        surface_filter.Update()

        surface_points = surface_filter.GetOutput().GetPoints()
        surface_coords = np.array(
            [
                surface_points.GetPoint(i)
                for i in range(surface_points.GetNumberOfPoints())
            ]
        )

        # setting the surface of the cylinder
        cell_data = vtk_to_numpy(surface_filter.GetOutput().GetCellData().GetArray(1))

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

    def scale_grid(self):
        """
        scales the grid by the scale factor given in the constructor (this is required since the output
        of CPlnatBox is factor 100 smaller than we need it to be, maybe due to different units)
        """
        transform = vtk.vtkTransform()
        transform.Scale(self.scale_factor, self.scale_factor, self.scale_factor)

        transform_filter = vtk.vtkTransformFilter()
        transform_filter.SetInputData(self.unstructured_grid)
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        self.unstructured_grid = transform_filter.GetOutput()

    def visualize_3d_coords(self, coords, data, path):
        """
        visualizes the given coordinates and the associated data values in a 3d plot. The data values are
        colored based on how high they are. The figure is saved to the given path.
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

    def create_3d_coords_array(self):
        """
        creates a 3d coordinate array spanning evenly spaced across the min and max coords of the soil model
        """
        min_coords = self.points.min(axis=0)
        max_coords = self.points.max(axis=0)

        x_width = max_coords[0] - min_coords[0]
        y_width = max_coords[1] - min_coords[1]
        z_width = max_coords[2] - min_coords[2]

        x = np.linspace(min_coords[0], max_coords[0], int(x_width / self.resolution[0]))
        y = np.linspace(min_coords[1], max_coords[1], int(y_width / self.resolution[1]))
        z = np.linspace(min_coords[2], max_coords[2], int(z_width / self.resolution[2]))

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        self.new_grid_coords = np.stack(
            [X.flatten(), Y.flatten(), Z.flatten()], axis=-1
        )

    def _set_3d_coords_array(self, coords):
        """
        set the array, for which the water sim values will be set by the interpolator
        """
        X, Y, Z = np.meshgrid(coords[0], coords[1], coords[2], indexing="ij")

        self.new_grid_coords = np.stack(
            [X.flatten(), Y.flatten(), Z.flatten()], axis=-1
        )

    def _calculate_cell_centers(self) -> np.ndarray:
        """
        Calculate the centers of the given cells of a mesh

        Returns:
        - cell_centers (np.ndarray): Array containing the center coordinates of the cells of the mesh.
        """
        cell_centers = []
        for cell in self.cells:
            cell_coords = self.points[cell]
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
        """
        self.load_vtu()
        self.scale_grid()
        self.extract_vtu_data()

        if interpolating_coords is None:
            self.create_3d_coords_array()
        else:
            self._set_3d_coords_array(interpolating_coords)

        cell_centers = self._calculate_cell_centers()
        mesh_surface_coords, mesh_surface_data = self._get_cylindar_surface_data()

        grid_coords = np.concatenate((cell_centers, mesh_surface_coords), axis=0)
        grid_data = np.concatenate((self.cell_data, mesh_surface_data), axis=0)

        start_time = time.time()
        interpolator = LinearNDInterpolator(grid_coords, grid_data)

        # self.visualize_3d_coords(grid_coords, grid_data, "vis/grid1.png")

        print("Start interpolation")

        self.new_grid_interpolated_data = interpolator(self.new_grid_coords)
        # np.save("./numpy_saved/interpolated_data_12.npy", self.new_grid_interpolated_data)

        end_time = time.time()
        print("--- %s seconds ---" % (end_time - start_time))

        print("interpolated_data", self.new_grid_interpolated_data.shape)

        return self.new_grid_coords, self.new_grid_interpolated_data

    def visualize(self, plot_path):
        self.visualize_3d_coords(
            self.new_grid_coords,
            self.new_grid_interpolated_data,
            "{}/interpolated_grid.png".format(plot_path),
        )


# Example Usage
# vtu_path = "./soil_simulation_data/generated_20_1704.vtu"
# interpolator = VTUInterpolatedGrid(vtu_path)
# Option 1:
# interpolator.create_3d_coords_array()
# interpolator.process()
# Option 2 (given coordinates for interpolation):
# interpolator.process(interpolating_coords=$coords$)
