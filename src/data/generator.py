from virtual_mri_generation import Virtual_MRI, RootSystemSimulation, SoilWaterSimulation
from multiprocessing.dummy import Pool as ThreadPool
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
                    "maize_p2",
                    "Moraesetal_2020",
                    "Anagallis_femina_Leitner_2010",
                    "wheat",
                    "Zea_mays_Streuber_2020",
                    "Lupinus_albus_Leitner_2014"
                ],
            "soil_type": ["sand", "loam", "clay"],
        }
        # fixed parameters and ranges from which a random value will be chosen
        self.params_range = {
            "root_growth": np.around(np.arange(3, 11, 1), 1).tolist(),
            "inital_sand": list(range(-900, -100, 100)),
            "initial_loam": list(range(-4000, -1000, 100)),
            "initial_clay": list(range(-12000, -300, 100)),
            "perlin_noise_intensity": np.around(np.arange(0.5, 1, 0.1), 1),
            "no_noise_probability": 0,
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
            
            # If tehre are less than 3 samples for the current combination of parameters, generate the
            # raming samples
            if num_files_plant_soil < 3:
                i = 0
                while i < 3 - num_files_plant_soil:
                    # choose the parameters for the generated sample at random from the range of possible values
                    sim_time = random.choice(self.params_range["root_growth"])

                    if param_combination["soil_type"] == "sand":
                        initial = random.choice(self.params_range["inital_sand"])
                    elif param_combination["soil_type"] == "loam":
                        initial = random.choice(self.params_range["initial_loam"])
                    elif param_combination["soil_type"] == "clay":
                        initial = random.choice(self.params_range["initial_clay"])

                    config = {
                        "root_model_name": param_combination["root_model_name"],
                        "soil_type": param_combination["soil_type"],
                        "root_growth_days": sim_time,
                        "initial": initial,
                        "sim_time": sim_time,
                        "perlin_noise_intensity": random.choice(self.params_range["perlin_noise_intensity"]),
                    }
                    pprint.pprint(config, width=40, indent=4)

                    # if a sample with the same config was not already generated, generate it
                    if not os.path.exists(f"../../data/generated/{config['root_model_name']}/{config['soil_type']}/sim_days_{config['root_growth_days']}-initial_{config['initial']}-noise_{config['perlin_noise_intensity']}"):
                        self.generate_sample(config=config)
                    else:
                        i -= 1
                    
                    i += 1


    def get_random_config(self):
        """
        Returns a random configuration for the data generator.
        """
        root_models = [f for f in os.listdir(self.root_model_path) if os.path.isfile(os.path.join(self.root_model_path, f))]
        soil_types = ["sand", "loam", "clay"]
        root_growth_days = list(range(3, 11))

        random_root_model_name = random.choice(root_models).split(".")[0]
        random_soil_type = random.choice(soil_types)
        random_root_growth_days = [random.choice(root_growth_days)]
        perlin_noise_intensity = np.random.uniform(0.5, 1)
        mesh_path = ""
        if random_soil_type == "sand":
            mesh_path = "../../data_assets/meshes/cylinder_r_0.03_d_-0.2_res_0.006_sand.msh"
        elif random_soil_type == "loam":
            mesh_path = "../../data_assets/meshes/cylinder_r_0.03_d_-0.2_res_0.005_loam.msh"
        elif random_soil_type == "clay":
            mesh_path = "../../data_assets/meshes/cylinder_r_0.03_d_-0.2_res_0.0032_clay.msh"

        config = {
            "root_model_name": random_root_model_name,
            "soil_type": random_soil_type,
            "root_growth_days": random_root_growth_days,
            "perlin_noise_intensity": perlin_noise_intensity,
            "mesh_path": mesh_path,
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
                                if num_files < 5:
                                    shutil.rmtree(root_sim.path)
    
    def generate_sample(self, config=None):
        """
        Generates a random sample of a virtual MRI scan if no config is provided.
        """
        # Get random configuration if no config is given
        my_config = config if config else self.get_random_config()

        if my_config["soil_type"] == "sand":
            mesh_path = "../../data_assets/meshes/cylinder_r_0.032_d_-0.21_res_0.006_sand.msh"
        elif my_config["soil_type"] == "loam":
            mesh_path = "../../data_assets/meshes/cylinder_r_0.032_d_-0.21_res_0.005_loam.msh"
        elif my_config["soil_type"] == "clay":
            mesh_path = "../../data_assets/meshes/cylinder_r_0.032_d_-0.21_res_0.0032_clay.msh"
        

        # root = "Anagallis_femina_Leitner_2010"
        # soil_type = "sand"
        # root_growth_days = [5,10]
        width = 3
        depth = 20

        # soil_water_sim = "../../data/generated/soil_water_simulation"
        # virtual_mri_path = "../../data/generated/virtual_mri"

        # create the folder for the generated data
        data_path = f"../../data/generated/{my_config['root_model_name']}/{my_config['soil_type']}/sim_days_{my_config['root_growth_days']}-initial_{my_config['initial']}-noise_{my_config['perlin_noise_intensity']}"
        os.makedirs(data_path, exist_ok=True)

        # add the config for the simulated plant to the data folder
        with open(f"{data_path}/config.json", "w") as json_file:
            json.dump(my_config, json_file, indent=4)

        # Generate a root system
        root_sim = RootSystemSimulation(my_config["root_model_name"], data_path, width, depth, model_path="../../../../../../modelparameter/structural/rootsystem")
        analist, filenames = root_sim.run_simulation([my_config["root_growth_days"]]) # , seed=0)
        print("filenames", filenames[0])

        # Generate a soil water simulation
        print("mesh_path", mesh_path)
        soil_water_sim_params = {
            "mesh_path": mesh_path,
            "rsml_path": data_path+"/{}".format(filenames[0]),
            "output_path": data_path,
            "soil_type": my_config["soil_type"],
            "initial": my_config["initial"],
            "sim_time": my_config["sim_time"],
        }
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

        my_vi = Virtual_MRI(seganalyzer, rsml_path, soil_type, vtu_path, perlin_noise_intensity)

        grid_values, filename, root_idx = my_vi.create_virtual_root_mri(data_path, add_noise=True)

        # Combine the data
        # data = self.combine_data(mri, root_system, soil_water_sim)

        return grid_values

generator = DataGenerator()
generator.generate_samples_grid()