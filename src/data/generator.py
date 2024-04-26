with open("../../DUMUX_path.txt", "r") as file:
    DUMUX_path = file.read()

import sys

sys.path.append("..")

from virtual_mri_generation import (
    Virtual_MRI,
    RootSystemSimulation,
    SoilWaterSimulation,
)
from utils import MeshGenerator

from multiprocessing.dummy import Pool as ThreadPool
import xml.etree.ElementTree as ET
import os
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import json
import itertools
import shutil
import re
import nibabel as nib
import pprint
import h5py


class DataGenerator:
    def __init__(self, data_assets_path):
        """
        Generates samples (synthetic MRIs) for each plant and soil combination and random params from the
        specidied ranges. Add parameters should me modified, for whatever configuration is needed.
        """
        self.data_assets_path = data_assets_path
        self.param_grid = {
            "root_model_name": [
                "my_Glycine_max",
                "my_Bench_lupin",
                "my_Crypsis_aculeata_Clausnitzer_1994",
                "my_Moraesetal_2020",
            ],
            "soil_type": ["sand", "loam"],
        }
        # fixed parameters and ranges from which a random value will be chosen
        self.params_random = {
            "root_growth_days": [
                int(x) for x in np.around(np.arange(5, 10, 1), 1).tolist()
            ],
            "initial_sand": list(range(-30, -5, 1)),
            "initial_loam": list(range(-500, -20, 5)),
            # "initial_clay": list(range(-1000, -200, 100)),
            "no_noise_probability": 0,
            "min_xy_seed_pos": -0.1,
            "max_xy_seed_pos": 0.1,
            "depth_range": list(
                range(15, 21, 1)
            ),  # min depth, max depth + 1 (because exclusive) in cm
            "radius": [3, 3],  # min radius, max radius in cm
        }
        self.root_model_path = (
            f"{DUMUX_path}/CPlantBox/modelparameter/structural/rootsystem"
        )
        self.soil_water_sim = None

    def _generate_values_float(min_value, max_value, step_size):
        if min_value == max_value:
            return [min_value]

        values = []
        current_value = min_value

        while current_value <= max_value:
            values.append(current_value)
            current_value += step_size
            # To handle floating point precision issues, round to a reasonable number of decimal places
            current_value = round(current_value, 10)
        return values

    def generate_samples_grid(self, data_path, num_samples_per_config):
        """
        Generates samples using the parameter grid config initialized in the constructor.

        Args:
        - data_path: path to the folder where the generated data should be stored
        - num_samples_per_config: number of samples to generate for each combination of parameters (plant and soil type)
        """
        # Get all possible combinations of the parameters
        all_params = [
            dict(zip(self.param_grid, v))
            for v in itertools.product(*self.param_grid.values())
        ]

        # Generate a sample for each combination of parameters
        for param_combination in all_params:
            # Get the number of generated samples for the current combination of parameters
            plant_path = f"{data_path}/{param_combination['root_model_name']}/{param_combination['soil_type']}"
            num_files_plant_soil = 0
            if os.path.exists(plant_path):
                num_files_plant_soil = len(
                    [
                        d
                        for d in os.listdir(plant_path)
                        if os.path.isdir(os.path.join(plant_path, d))
                    ]
                )

            # If there are less than $num_samples_per_config samples for the current combination of parameters, generate the
            # remaining samples
            if num_files_plant_soil < num_samples_per_config:
                i = 0
                while num_samples_per_config - num_files_plant_soil > 0:
                    # choose the parameters for the generated sample at random from the range of possible values
                    root_growth_days = random.choice(
                        self.params_random["root_growth_days"]
                    )
                    initial = random.choice(
                        self.params_random[f"initial_{param_combination['soil_type']}"]
                    )

                    radius = np.round(
                        random.uniform(
                            self.params_random["radius"][0],
                            self.params_random["radius"][1],
                        )
                    )

                    depth = random.choice(self.params_random["depth_range"])

                    meshGen = MeshGenerator(
                        "../../data_assets/meshes/",
                        mesh_size=0.005,
                        depth=depth / 100,
                        radius=radius / 100,
                    )
                    mesh_path = meshGen.create_mesh()

                    config = {
                        "root_model_name": param_combination["root_model_name"],
                        "soil_type": param_combination["soil_type"],
                        "root_growth_days": root_growth_days,
                        "initial": initial,
                        "sim_time": (
                            root_growth_days
                            if root_growth_days < 4
                            else random.choice(
                                [x for x in np.around(np.arange(2, 4, 0.2), 2)]
                            )
                        ),
                        "seed_pos": (
                            random.uniform(
                                self.params_random["min_xy_seed_pos"],
                                self.params_random["max_xy_seed_pos"],
                            ),
                            random.uniform(
                                self.params_random["min_xy_seed_pos"],
                                self.params_random["max_xy_seed_pos"],
                            ),
                        ),
                        "depth": depth,
                        "radius": radius,
                        "mesh_path": mesh_path,
                    }

                    # if a sample with the same config was not already generated, generate it
                    if not os.path.exists(
                        f"{data_path}/{config['root_model_name']}/{config['soil_type']}/sim_days_{config['root_growth_days']}-initial_{config['initial']}"
                    ):
                        self.generate_sample(data_path, config=config)
                        i += 1
                        # update the number of files in the directory, because parallel computation could lead to changes
                        num_files_plant_soil = len(
                            [
                                d
                                for d in os.listdir(plant_path)
                                if os.path.isdir(os.path.join(plant_path, d))
                            ]
                        )

    def get_random_config(self) -> dict:
        """
        Returns a random configuration for the data generator.
        """
        all_root_models = [
            f
            for f in os.listdir(self.root_model_path)
            if os.path.isfile(os.path.join(self.root_model_path, f))
        ]

        random_root_model_name = random.choice(self.param_grid["root_model_name"])
        random_soil_type = random.choice(self.param_grid["soil_type"])
        random_root_growth_days = random.choice(self.params_random["root_growth_days"])
        random_water_sim_days = random.choice(
            random_root_growth_days
            if random_root_growth_days < 4
            else [x for x in np.around(np.arange(2, 4, 0.2), 2)]
        )
        initial = random.choice(self.params_random[f"initial_{random_soil_type}"])

        meshGen = MeshGenerator(
            "../../data_assets/meshes/",
            mesh_size=0.005,
            depth=random.choice(self.params_random["depth_range"]),
            radius=random.uniform(
                self.params_random["radius"][0], self.params_random["radius"][1]
            ),
        )

        mesh_path = meshGen.create_mesh()

        config = {
            "root_model_name": random_root_model_name,
            "soil_type": random_soil_type,
            "root_growth_days": random_root_growth_days,
            "sim_time": random_water_sim_days,
            "mesh_path": mesh_path,
            "initial": initial,
            "seed_pos": random.uniform(
                self.params_random["min_xy_seed_pos"],
                self.params_random["max_xy_seed_pos"],
            ),
        }

        return config

    def get_dirs_without_subdir(self, data_path) -> list:
        """
        Recursively find leaf directories (directories without subdirectories) starting from a given directory.

        Args:
        - directory_path (str): The path to the directory to start the search from.

        Returns:
        - list: A list of directories that do not contain any subdirectories.
        """

        leaf_directories = []

        for root, dirs, files in os.walk(data_path):
            # Check if the current directory has no subdirectories
            if not dirs:
                leaf_directories.append(root)

        return leaf_directories

    def remove_incomplete_samples(self, data_path):
        """
        removes the incomplete samples, where not all files were generated.

        Args:
        - data_path: path to the folder where the generated data is stored
        """
        data_dir = self.get_dirs_without_subdir(data_path)

        for dir in data_dir:
            num_files = sum(
                1
                for item in os.scandir(dir)
                if item.is_file() and item.name != ".DS_Store"
            )
            if num_files < 7:
                shutil.rmtree(dir)

    def generate_sample(self, data_path, config=None):
        """
        Generates a random sample of a virtual MRI scan if no config is provided.

        Args:
        - data_path: path to the folder where the generated data should be stored
        - config: a dict with the required params for the data generation
        """
        # Get random configuration if no config is given
        my_config = config if config else self.get_random_config()

        # For debugging purposes, a specific configuration can be used
        my_config["root_growth_days"] = 1
        my_config["sim_time"] = 0.01
        # my_config["depth"] = 20
        # my_config["soil_type"] = "sand"
        # my_config["root_model_name"] = "my_Crypsis_aculeata_Clausnitzer_1994"

        pprint.pprint(my_config, width=40, indent=4)

        # Check if the config contains all necessary entries for data generation
        # List of entries to check
        data_gen_params = [
            "root_model_name",
            "soil_type",
            "root_growth_days",
            "sim_time",
            "initial",
            "seed_pos",
        ]

        if all(key in my_config for key in data_gen_params):
            print("All entries are in the dictionary.")
        else:
            missing_keys = [key for key in data_gen_params if key not in my_config]
            raise ValueError(f"Params {missing_keys} are missing in the config.")

        # create the folder for the generated data
        data_path = f"{data_path}/{my_config['root_model_name']}/{my_config['soil_type']}/sim_days_{my_config['root_growth_days']}-initial_{my_config['initial']}"
        os.makedirs(data_path, exist_ok=True)

        # add the config for the simulated plant to the data folder
        with open(f"{data_path}/config.json", "w") as json_file:
            json.dump(my_config, json_file, indent=4)

        # For this root model, the
        if my_config["root_model_name"] == "my_Crypsis_aculeata_Clausnitzer_1994":
            my_config["seed_pos"] = (0, 0)

        # Generate a root system
        root_sim = RootSystemSimulation(
            model_name=my_config["root_model_name"],
            root_save_dir=data_path,
            soil_radius=my_config["radius"] - 0.1,
            soil_depth=my_config["depth"] - 0.1,
            seed_pos=(my_config["seed_pos"][0], my_config["seed_pos"][1], 0),
            model_path=f"{DUMUX_path}/CPlantBox/modelparameter/structural/rootsystem",
        )
        analist, filenames = root_sim.run_simulation(
            [my_config["root_growth_days"]]
        )  # , seed=0)

        soil_water_sim_params = {
            "mesh_path": my_config["mesh_path"],
            "rsml_path": data_path + "/{}".format(filenames[0]),
            "output_path": data_path,
            "soil_type": my_config["soil_type"],
            "initial": my_config["initial"],
            "sim_time": my_config["sim_time"],
        }

        # run the soil water simulation (a single instance must be used for all simulations because it otherwise
        # throws an error in DUMUX)
        if self.soil_water_sim is None:
            self.soil_water_sim = SoilWaterSimulation(**soil_water_sim_params)
        else:
            self.soil_water_sim.__init__(**soil_water_sim_params)
        water_sim_file = self.soil_water_sim.run()

        # Generate a virtual MRI scan
        seganalyzer = analist[0]
        rsml_path = data_path + "/{}".format(filenames[0])
        vtu_path = data_path + "/" + water_sim_file

        depth_overflow = 0.2
        width_overflow = 0.2
        my_vi = Virtual_MRI(
            rsml_path,
            soil_type=my_config["soil_type"],
            vtu_path=vtu_path,
            seganalyzer=seganalyzer,
            res_mri=[0.027, 0.027, 0.1],
            depth=my_config["depth"] + depth_overflow,
            radius=my_config["radius"] + width_overflow,
        )
        _, _ = my_vi.create_virtual_root_mri(
            data_path,
            add_noise=True,
        )
        my_vi.__init__(
            rsml_path,
            seganalyzer=seganalyzer,
            res_mri=[0.027, 0.027, 0.1],
            depth=my_config["depth"] + depth_overflow,
            radius=my_config["radius"] + width_overflow,
            scale_factor=2,
        )

        _, _ = my_vi.create_virtual_root_mri(
            data_path,
            add_noise=False,
            label=True,
        )


generator = DataGenerator("../../data_assets")
# generator.remove_incomplete_samples("../../data/generated")
# Generate training data
generator.generate_samples_grid(
    data_path="../../data/generated/training",
    num_samples_per_config=11,
)
# Generate validation data
generator.generate_samples_grid(
    data_path="../../data/generated/validation",
    num_samples_per_config=2,
)
# Generate test data
generator.generate_samples_grid(
    data_path="../../data/generated/test",
    num_samples_per_config=2,
)

# generator.modify_nifti_values("../../data/generated")
