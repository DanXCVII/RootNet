import sys
sys.path.append("../../../../../../../")
sys.path.append("../../../../../../../src/")

import plantbox as pb
import visualisation.vtk_plot as vp
import numpy as np
import random
import os
import copy

class RootSystemSimulation:
    def __init__(self, model_name, root_save_dir, soil_width, soil_depth, model_path="../../../../../../../modelparameter/structural/rootsystem", container_type="soilcore"):
        """
        Simulates the growth of a root system for a given number of days and the root container parameters.

        Parameters:
        - model_path (str): Path to the model parameters.
        - model_name (str): Name of the model.
        - root_save_dir (str): Path to the folder where the root system simulation results (.rsml) will be saved.
        - container_type (str): Type of container ("soilcore" or "rhizotron"). Default is "soilcore".
        - soil_width (float): Width of the soil container.
        - soil_depth (float): Depth of the soil container.
        """
        self.rs = pb.RootSystem()
        self.model_path = model_path
        self.root_save_dir = root_save_dir
        self.model_name = model_name
        self.container_type = container_type
        self.soil_width = soil_width
        self.soil_depth = soil_depth
        self._set_geometry()

    def _set_geometry(self):
        """
        Sets the geometry for the root system based on the container type.
        """
        if self.container_type == "soilcore":
            self.geometry = pb.SDF_PlantContainer(self.soil_width, self.soil_width, self.soil_depth, False)
        elif self.container_type == "rhizotron":
            self.geometry = pb.SDF_PlantBox(self.soil_width, self.soil_width, self.soil_depth)
        else:
            raise ValueError("Invalid container type.")
        self.rs.setGeometry(self.geometry)
    
    def run_simulation(self, days=[10, 20], seed=np.nan):
        """
        Runs the root system simulation for a given number of days. If multiple days are given,
        the simulation of the same plant will continue.

        Parameters:
        - days (int): Number of days to run the simulation.
        """

        self.rs.readParameters(self.model_path + "/" + self.model_name + ".xml")
        print(self.model_path + "/" + self.model_name + ".xml")
        
        if not np.isnan(seed):
            self.rs.setSeed(seed)
        else:
            self.rs.setSeed(random.randint(0, 100000))

        self.rs.initialize()

        analist = []
        filenames = []

        for day in days:
            self.rs.simulate(day)
            
            ana = pb.SegmentAnalyser(self.rs)
            analist.append(ana)

            # Export results
            filename = f"{self.model_name}_day_{day}"
            filenames.append(f"{filename}.rsml")

            print(f"{self.root_save_dir}/{filename}.rsml")

            self.rs.write(f"{self.root_save_dir}/{filename}.rsml")
            self.rs.write(f"{self.root_save_dir}/{filename}.vtp")

            # os.remove(f"./{filename}.pvd")

        return analist, filenames
        
        # Plot results
        # vp.plot_roots(self.rs, "type")

# Example Usage
# sim = RootSystemSimulation("Anagallis_femina_Leitner_2010", "../../../data/generated/root_systems", 3, 20)
# analist, filenames = sim.run_simulation([11, 12], seed=0)