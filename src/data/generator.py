from virtual_mri_generation import Virtual_MRI, RootSystemSimulation, SoilWaterSimulation

from multiprocessing.dummy import Pool as ThreadPool
import xml.etree.ElementTree as ET
import os
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import json
import itertools
import shutil
import pprint


class DataGenerator:
    def __init__(self, config=None):
        # parameters for the grid search
        self.param_grid = {
            "root_model_name": 
                [
                    "Bench_lupin",
                    "Crypsis_aculeata_Clausnitzer_1994",
                    "Glycine_max_Moraes2020_opt2",
                    "Moraesetal_2020"
                ],
            "soil_type": ["sand", "loam", "clay"],
        }
        # fixed parameters and ranges from which a random value will be chosen
        self.params_range = {
            "root_growth_days": [int(x) for x in np.around(np.arange(3, 11, 1), 1).tolist()],
            "initial_sand": list(range(-30, -5, 5)),
            "initial_loam": list(range(-500, -20, 10)),
            "initial_clay": list(range(-1000, -200, 100)),
            "perlin_noise_intensity": np.around(np.arange(0.5, 1, 0.1), 1),
            "no_noise_probability": 0,
            "min_xy_seed_pos": -0.1,
            "max_xy_seed_pos": 0.1,
        }
        self.soil_type_mesh_path = {
            "sand": "../../data_assets/meshes/cylinder_r_0.032_d_-0.22_res_0.005.msh",
            "loam": "../../data_assets/meshes/cylinder_r_0.032_d_-0.22_res_0.005.msh",
            "clay": "../../data_assets/meshes/cylinder_r_0.032_d_-0.22_res_0.005.msh",
        }
        self.root_model_path = "../../../../../../modelparameter/structural/rootsystem"
        self.soil_water_sim = None

    def generate_samples_grid(self):
        """
        Generates samples using the parameter grid config initialized in the constructor.
        """
        # Get all possible combinations of the parameters
        all_params = [dict(zip(self.param_grid, v)) for v in itertools.product(*self.param_grid.values())]

        # Generate a sample for each combination of parameters
        for param_combination in all_params:
            "../../data/generated/root_systems"

            # Get the number of generated samples for the current combination of parameters
            plant_path = f"../../data/generated/{param_combination['root_model_name']}/{param_combination['soil_type']}"
            num_files_plant_soil = 0
            if os.path.exists(plant_path):
                num_files_plant_soil = len([d for d in os.listdir(plant_path) if os.path.isdir(os.path.join(plant_path, d))])
            
            # If there are less than 8 samples for the current combination of parameters, generate the
            # remaining samples
            if num_files_plant_soil < 8:
                i = 0
                while i < 3 - num_files_plant_soil:
                    # choose the parameters for the generated sample at random from the range of possible values
                    root_growth_days = random.choice(self.params_range["root_growth_days"])
                    initial = random.choice(self.params_range[f"initial_{param_combination['soil_type']}"])
                   
                    config = {
                        "root_model_name": param_combination["root_model_name"],
                        "soil_type": param_combination["soil_type"],
                        "mesh_path": self.soil_type_mesh_path[param_combination["soil_type"]],
                        "root_growth_days": root_growth_days,
                        "initial": initial,
                        "sim_time": int(root_growth_days) if root_growth_days < 4 else int(random.choice(np.around(np.arange(2, 3, 1), 1))),
                        "perlin_noise_intensity": random.choice(self.params_range["perlin_noise_intensity"]),
                        "seed_pos": (random.uniform(self.params_range["min_xy_seed_pos"], self.params_range["max_xy_seed_pos"]), random.uniform(self.params_range["min_xy_seed_pos"], self.params_range["max_xy_seed_pos"])),
                    }

                    # if a sample with the same config was not already generated, generate it
                    if not os.path.exists(f"../../data/generated/{config['root_model_name']}/{config['soil_type']}/sim_days_{config['root_growth_days']}-initial_{config['initial']}-noise_{config['perlin_noise_intensity']}"):
                        self.generate_sample(config=config)
                        i += 1

    def _set_seed_pos(self, root_system_name, posx, posy):
        """
        Sets the seed position to a random value in the x and y coordinate between min_xy and max_xy

        Args:
        root_system_name: name of the root system where the seed pos should be modified
        min_xy: minimum value which the x and y coordnate should have
        max_xy: maximum value which the x and y coordnate should have
        """
        # Parse the XML file
        filename = f"{self.root_model_path}/{root_system_name}.xml"
        tree = ET.parse(filename)
        root = tree.getroot()
        
        # Find the seed element
        seed = root.find(f".//seed")
        
        if seed:
            # Iterate over parameters to set values for "seedPos.y" and "seedPos.x"
            for param in seed.findall("parameter"):
                if param.get("name") == "seedPos.x":
                    param.set("value", str(posx))
                elif param.get("name") == "seedPos.y":
                    param.set("value", str(posy))

        # Write the modified XML back to the file
        tree.write(filename)

    def get_random_config(self):
        """
        Returns a random configuration for the data generator.
        """
        all_root_models = [f for f in os.listdir(self.root_model_path) if os.path.isfile(os.path.join(self.root_model_path, f))]

        random_root_model_name = random.choice(self.param_grid["root_model_name"])
        random_soil_type = random.choice(self.param_grid["soil_type"])
        random_root_growth_days = random.choice(self.params_range["root_growth_days"])
        random_water_sim_days = random.choice(self.params_range["root_growth_days"] if random_root_growth_days < 4 else [int(x) for x in np.around(np.arange(2, 3, 1), 1)])
        perlin_noise_intensity = random.choice(self.params_range["perlin_noise_intensity"])
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
            "seed_pos": random.uniform(self.params_range["min_xy_seed_pos"], self.params_range["max_xy_seed_pos"]),
        }

        return config
    
    def remove_incomplete_samples(self):
        """
        removes the samples, where not all files were generated.
        """
        data_path = "../../data/generated"
        # iterate over the root models in the generated folder
        for plant_model in os.scandir(data_path):
            # for each directory there (soil type) iterate over the files inside
            if plant_model.is_dir():
                for soil_type in os.scandir(plant_model.path):
                    # for each simulation folder in the soil types iterate ..
                    if soil_type.is_dir():
                        for root_sim in os.scandir(soil_type.path):
                            if root_sim.is_dir():
                                # .. and check how many files are in the folder (should be 5)
                                # .vtu (water sim), .raw (virtual MRI), .raw (root idx), .rsml (root system), .json (config)
                                num_files = sum(1 for item in os.scandir(root_sim.path) if item.is_file())
                                # if there are less than 5 files, remove the folder
                                if num_files < 6:
                                    shutil.rmtree(root_sim.path)
    
    def generate_sample(self, config=None):
        """
        Generates a random sample of a virtual MRI scan if no config is provided.

        Args:
        config: a dict with the required params for the data generation
        """
        # Get random configuration if no config is given
        my_config = config if config else self.get_random_config()

        # Check if the config contains all necessary entries for data generation
        # List of entries to check
        data_gen_params = ["root_model_name", "soil_type", "root_growth_days", "sim_time", "perlin_noise_intensity", "initial", "seed_pos"]

        if all(key in my_config for key in data_gen_params):
            print("All entries are in the dictionary.")
        else:
            missing_keys = [key for key in data_gen_params if key not in my_config]
            raise ValueError(f"Params {missing_keys} are missing in the config.")

        # Set the mesh path based on the soil type
        self._set_seed_pos(my_config["root_model_name"], my_config["seed_pos"][0], my_config["seed_pos"][0])

        # create the folder for the generated data
        data_path = f"../../data/generated/{my_config['root_model_name']}/{my_config['soil_type']}/sim_days_{my_config['root_growth_days']}-initial_{my_config['initial']}-noise_{my_config['perlin_noise_intensity']}"
        os.makedirs(data_path, exist_ok=True)

        # add the config for the simulated plant to the data folder
        with open(f"{data_path}/config.json", "w") as json_file:
            pprint.pprint(my_config, width=40, indent=4)
            json.dump(my_config, json_file, indent=4)

        # Generate a root system
        width = 3
        depth = 20
        root_sim = RootSystemSimulation(my_config["root_model_name"], data_path, width, depth, model_path="../../../../../../modelparameter/structural/rootsystem")
        analist, filenames = root_sim.run_simulation([my_config["root_growth_days"]]) # , seed=0)
        print("filenames", filenames[0])

        soil_water_sim_params = {
            "mesh_path": my_config["mesh_path"],
            "rsml_path": data_path+"/{}".format(filenames[0]),
            "output_path": data_path,
            "soil_type": my_config['soil_type'],
            "initial": my_config["initial"],
            "sim_time": my_config["sim_time"],
        }

        pprint.pprint(soil_water_sim_params, width=40, indent=4)

        # run the soil water simulation (a single instance must be used for all simulations because it otherwise
        # throws an error in DUMUX)
        if self.soil_water_sim is None:
            self.soil_water_sim = SoilWaterSimulation(**soil_water_sim_params)
        else:
            self.soil_water_sim.__init__(**soil_water_sim_params)
        water_sim_file = self.soil_water_sim.run()

        # Generate a virtual MRI scan
        seganalyzer = analist[0]
        rsml_path = data_path+"/{}".format(filenames[0])
        soil_type = my_config["soil_type"]
        vtu_path = data_path+"/"+water_sim_file
        perlin_noise_intensity = my_config["perlin_noise_intensity"]

        my_vi = Virtual_MRI(seganalyzer, rsml_path, soil_type, vtu_path, perlin_noise_intensity, resolution=[0.027, 0.027, 0.1], depth=depth+0.2, width=width+0.1)
        grid_values, filename = my_vi.create_virtual_root_mri(data_path, add_noise=True)
        my_vi.__init__(seganalyzer, rsml_path, soil_type, vtu_path, perlin_noise_intensity, resolution=[0.027/2, 0.027/2, 0.1/2], depth=depth+0.2, width=width+0.1)
        
        grid_values, filename = my_vi.create_virtual_root_mri(data_path, add_noise=False, label=True)#True)
        

        # Combine the data
        # data = self.combine_data(mri, root_system, soil_water_sim)

        return grid_values

generator = DataGenerator()
# generator.remove_incomplete_samples()
generator.generate_samples_grid()