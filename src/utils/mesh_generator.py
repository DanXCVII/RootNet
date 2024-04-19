import pygmsh
import gmsh
import meshio
from pygmsh.common.curve_loop import CurveLoop
import os


class MeshGenerator:
    def __init__(
        self,
        path,
        center=(0.0, 0.0, 0.0),
        radius=0.032,
        mesh_size=0.005,
        depth=0.22,
    ):
        """
        Generates a mesh for a cylinder with a specified radius, height and resolution (mesh size).

        Args:
        - center (tuple): Top-Center of the cylinder.
        - radius (float): Radius of the cylinder.
        - mesh_size (float): resolution of the mesh. (0.006 for sand, 0.0032 for clay)
        - depth (float): Depth of the cylinder (positive float)
        - num_layers (int): Number of layers of the mesh from top to bottom.
        """
        self.center = center
        self.path = path
        self.radius = radius
        self.mesh_size = mesh_size
        self.translation_axis = [0.0, 0.0, -depth]
        self.num_layers = round(depth / 0.011)
        print("num_layers:", self.num_layers)

    def create_mesh(self):
        full_path = "{}/cylinder_r_{}_d_{}_res_{}".format(
            self.path, self.radius, self.translation_axis[2], self.mesh_size
        )
        print("mesh Step 1")
        # Check if the mesh already exists
        if os.path.isfile(full_path + ".msh"):
            print("Mesh already exists.")
            return full_path + ".msh"

        print("mesh Step 2")

        gmsh.initialize()
        geom = pygmsh.occ.Geometry()

        # 1. Create a disk with specified parameters
        disk = geom.add_disk(self.center, self.radius, mesh_size=self.mesh_size)

        print("mesh Step 3")

        # 2. Set the RecombineAll option to 1 to recombine the triangles into quadrangles
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 8)

        # 3. Extrude the disk along the specified translation_axis
        geom.extrude(
            disk,
            translation_axis=self.translation_axis,
            num_layers=self.num_layers,
            recombine=True,
        )

        print("mesh Step 4")

        mesh = geom.generate_mesh()

        # 4. Write the mesh to a vtk and msh file
        full_path = "{}/cylinder_r_{}_d_{}_res_{}".format(
            self.path, self.radius, self.translation_axis[2], self.mesh_size
        )
        # mesh.write(full_path+".vtk")
        meshio.write_points_cells(
            full_path + ".msh",
            mesh.points,
            mesh.cells,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            field_data=mesh.field_data,
            file_format="gmsh22",
            binary=False,
        )

        print("mesh Step 5")

        gmsh.finalize()

        return full_path + ".msh"


# Generates a suitable mesh for clay and sand respectively
# mesh_sizes = [0.0032, 0.005, 0.006, 0.01]

# for mesh_size in mesh_sizes:
#     generator = MeshGenerator("../../data_assets/meshes/", mesh_size=mesh_size)
#     generator.create_mesh()
