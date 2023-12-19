from virtual_mri_generation import (
    Virtual_MRI,
    RootSystemSimulation,
    SoilWaterSimulation,
)

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
        specidied ranges.
        """
        # parameters for the grid search
        self.param_grid = {
            "root_model_name": [
                "Bench_lupin",
                "Crypsis_aculeata_Clausnitzer_1994",
                "Glycine_max_Moraes2020_opt2",
                "Glycine_max",
                "Moraesetal_2020",
            ],
            "soil_type": ["sand", "loam", "clay"],
        }
        # fixed parameters and ranges from which a random value will be chosen
        self.params_range = {
            "root_growth_days": [
                int(x) for x in np.around(np.arange(5, 11, 1), 1).tolist()
            ],
            "initial_sand": list(range(-30, -5, 5)),
            "initial_loam": list(range(-500, -20, 10)),
            "initial_clay": list(range(-1000, -200, 100)),
            "perlin_noise_intensity": np.around(np.arange(0.5, 1, 0.1), 1),
            "no_noise_probability": 0,
            "min_xy_seed_pos": -0.1,
            "max_xy_seed_pos": 0.1,
        }
        self.soil_type_mesh_path = {
            "sand": f"{data_assets_path}/meshes/cylinder_r_0.032_d_-0.22_res_0.005.msh",
            "loam": f"{data_assets_path}/meshes/cylinder_r_0.032_d_-0.22_res_0.005.msh",
            "clay": f"{data_assets_path}/meshes/cylinder_r_0.032_d_-0.22_res_0.005.msh",
        }
        self.root_model_path = "../../../../../../modelparameter/structural/rootsystem"
        self.soil_water_sim = None

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
                        self.params_range["root_growth_days"]
                    )
                    initial = random.choice(
                        self.params_range[f"initial_{param_combination['soil_type']}"]
                    )

                    config = {
                        "root_model_name": param_combination["root_model_name"],
                        "soil_type": param_combination["soil_type"],
                        "mesh_path": self.soil_type_mesh_path[
                            param_combination["soil_type"]
                        ],
                        "root_growth_days": root_growth_days,
                        "initial": initial,
                        "sim_time": root_growth_days
                        if root_growth_days < 4
                        else random.choice(
                            [x for x in np.around(np.arange(2, 4, 0.2), 2)]
                        ),
                        "perlin_noise_intensity": random.choice(
                            self.params_range["perlin_noise_intensity"]
                        ),
                        "seed_pos": (
                            random.uniform(
                                self.params_range["min_xy_seed_pos"],
                                self.params_range["max_xy_seed_pos"],
                            ),
                            random.uniform(
                                self.params_range["min_xy_seed_pos"],
                                self.params_range["max_xy_seed_pos"],
                            ),
                        ),
                    }

                    # if a sample with the same config was not already generated, generate it
                    if not os.path.exists(
                        f"{data_path}/{config['root_model_name']}/{config['soil_type']}/sim_days_{config['root_growth_days']}-initial_{config['initial']}-noise_{config['perlin_noise_intensity']}"
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
        random_root_growth_days = random.choice(self.params_range["root_growth_days"])
        random_water_sim_days = random.choice(
            random_root_growth_days
            if random_root_growth_days < 4
            else [x for x in np.around(np.arange(2, 4, 0.2), 2)]
        )
        perlin_noise_intensity = random.choice(
            self.params_range["perlin_noise_intensity"]
        )
        mesh_path = self.soil_type_mesh_path[random_soil_type]
        initial = random.choice(self.params_range[f"initial_{random_soil_type}"])

        config = {
            "root_model_name": random_root_model_name,
            "soil_type": random_soil_type,
            "root_growth_days": random_root_growth_days,
            "sim_time": random_water_sim_days,
            "perlin_noise_intensity": perlin_noise_intensity,
            "mesh_path": mesh_path,
            "initial": initial,
            "seed_pos": random.uniform(
                self.params_range["min_xy_seed_pos"],
                self.params_range["max_xy_seed_pos"],
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
        removes the samples, where not all files were generated.
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

        pprint.pprint(my_config, width=40, indent=4)

        # Check if the config contains all necessary entries for data generation
        # List of entries to check
        data_gen_params = [
            "root_model_name",
            "soil_type",
            "root_growth_days",
            "sim_time",
            "perlin_noise_intensity",
            "initial",
            "seed_pos",
        ]

        if all(key in my_config for key in data_gen_params):
            print("All entries are in the dictionary.")
        else:
            missing_keys = [key for key in data_gen_params if key not in my_config]
            raise ValueError(f"Params {missing_keys} are missing in the config.")

        # create the folder for the generated data
        data_path = f"{data_path}/{my_config['root_model_name']}/{my_config['soil_type']}/sim_days_{my_config['root_growth_days']}-initial_{my_config['initial']}-noise_{my_config['perlin_noise_intensity']}"
        os.makedirs(data_path, exist_ok=True)

        # add the config for the simulated plant to the data folder
        with open(f"{data_path}/config.json", "w") as json_file:
            json.dump(my_config, json_file, indent=4)

        # Generate a root system
        width = 3
        depth = 20
        root_sim = RootSystemSimulation(
            my_config["root_model_name"],
            data_path,
            width,
            depth,
            seed_pos=(my_config["seed_pos"][0], my_config["seed_pos"][1], 0),
            model_path="../../../../../../modelparameter/structural/rootsystem",
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
        water_sim_file = "water_sim.vtu"

        # Generate a virtual MRI scan
        seganalyzer = analist[0]
        rsml_path = data_path + "/{}".format(filenames[0])
        vtu_path = data_path + "/" + water_sim_file
        perlin_noise_intensity = my_config["perlin_noise_intensity"]

        my_vi = Virtual_MRI(
            rsml_path,
            vtu_path,
            perlin_noise_intensity,
            seganalyzer=seganalyzer,
            res_mri=[0.027, 0.027, 0.1],
            depth=depth + 0.2,
            width=width + 0.1,
        )
        grid_values, filename = my_vi.create_virtual_root_mri(
            data_path,
            add_noise=True,
        )
        my_vi.__init__(
            rsml_path,
            vtu_path,
            perlin_noise_intensity,
            seganalyzer=seganalyzer,
            res_mri=[0.027, 0.027, 0.1],
            depth=depth + 0.2,
            width=width + 0.1,
            scale_factor=2,
        )

        label_grid_values, filename = my_vi.create_virtual_root_mri(
            data_path,
            add_noise=False,
            label=True,
        )


generator = DataGenerator("../../data_assets")
generator.remove_incomplete_samples("../../data/generated")
# Generate training data
generator.generate_samples_grid(
    data_path="../../data/generated/training",
    num_samples_per_config=18,
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
