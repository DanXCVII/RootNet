# Overview

This repository houses the implementation of a pipeline designed to generate virtual Magnetic Resonance Imaging (MRI) scans of root systems for the Master thesis "3D Segmentation of Plant Roots from MRI Images for Enhanced Automated Root Tracing using Deep Neural Networks". The primary objective of this project is to use these virtual MRIs to train a Neural Network for the task of segmenting real MRI root scans.

## General Structure

### [`src`](src) Directory

The [`src`](src) directory contains all source code of this repo and is divided into several subdirectories, each dedicated to a specific aspect of the project's codebase:
- **[`data`](src/data)**: Contains code necessary for synthetic data generation. This subdirectory is used for creating the datasets needed to train the NNs.
- **[`data/virtual_mri_generation`](src/data/virtual_mri_generation)**: Code related to the generation of a single synthetic MRI, including water and root growth simulation.
- **[`models`](src/models)**: Houses the neural network models used for segmentation of the root MRIs.
- **[`training`](src/training)**: All code necessary for setting up the training environment, data loading, and the training process itself.
- **[`utils`](src/utils)**: Contains a collection of utility functions that help organize files and streamline various tasks.
  
### [`data_assets`](data_assets) Directory

The [`data_assets`](data_assets) directory is intended for storing fixed data assets. These assets include in this case the meshes for the simulation of water flow, the used root models for simulation and MRI data, which is used as a ground truth for synthetic MRI generation

### [`runs`](runs) Directory

The runs directory contains the checkpoints of the trained DNNs (weights and other parameters).


### [`data`](data) Directory

The [`data`](data) directory should contain all MRI data, which is used for training and testing of the DNN. This includes the nifti file of the MRIs together with the corresponding label files with twice the resolution. The naming scheme is the following "image_resx_resy_resz.nii.gz" and "label_image_resx\*2_resy\*2_resz\*2.nii.gz" for the MRI image and label image respectively. So the label image has the prefix "label_" and has the adjusted resolution. E.g. "my_Bench_lupin_day_5_res_237x237x151.nii.gz" and "label_my_Bench_lupin_day_5_res_474x474x302.nii.gz". The data used for the corresponding Master thesis can be downloaded [here](https://zenodo.org/records/12806033):
The structure is the following:
- **[`data/test`](data/test)**: Should contain all test data being used for the evaluation of the DNN
- **[`data/generated/training`](data/generated/training)**: Contains all training data for the DNN
- **[`data/generated/validation`](data/generated/validation)**: Contains all validation data for the DNN

## Setup

Since the repo is divided into two parts, the "Data Generation" and the "NN training", the setup for both of these steps is very different. 

### Synthetic Data Generation

Before being able to run the data generation, CPlantBox must be installed with the addition of DUMUX. An instruction on how to do that can be found on the official CPlantBox repo (https://github.com/Plant-Root-Soil-Interactions-Modelling/CPlantBox).
Additionally, modify the file [`DUMUX_path.txt`](DUMUX_path.txt) to the `CPlantBox/DUMUX` directory for the machine you are running it on.

To run the synthetic MRI generation, execute the [`src/data/generator.py`](src/data/generator.py) file inside the [`src/data`](src/data) directory. The required packages can be found in [`src/data/requirements.txt`](src/data/requirements.txt). The generated MRI data is stored in the [`data`](data) folder on the top level.

Another option for running the data generation is to execute the [`src/data/run_parallel.sh`](src/data/run_parallel.sh) file inside [`src/data`](src/data) where the number of parallel runs can be adjusted in the for loop by replacing X in the following: `for i in X` with the number of parallel runs. This script automatically restarts the generation process, if one run fails, which can occasionally happen for the water simulation.

The main code for the generation of a single synthetic MRI can be found in the [`src/data/virtual_mri_generation/create_virtual_MRI.py`](src/data/virtual_mri_generation/create_virtual_MRI.py). The "create_virtual_root_mri(..)" function is executed to generate a single virtual root MRI together with the label file. 

### Neural Network

The training of the DNN was performed on [JURECA](https://doi.org/10.17815/jlsrf-4-121-1) at the FZJ on nodes with 4xA100 GPUs. The configuration can be found in the [`setup_FZJ_booster`](setup_FZJ_booster) directory. An important fix for the checkpoint loading of the neural network for the used pytorch-lightning version for the master thesis is provided [here](https://github.com/Lightning-AI/pytorch-lightning/issues/13695). All code related to the evaluation of the DNN including the training, testing etc. can be found in [`src/training`](src/training). 

The [`src/training/pl_setup.py`](src/training/pl_setup.py) contains the pytorch-lightning module which can be used for the training, testing, and inference of the DNN model stored in ['src/models'](src/models). The dataloader, responsible for the data preprocessing and augmentations applied for the training, can be found in [`src/training/mri_dataloader.py`](src/training/mri_dataloader.py). 

#### Training, Testing

For the execution of the training, the [`setup_FZJ_booster/train_RootNet_run_booster.sbatch`](setup_FZJ_booster/train_RootNet_run_booster.sbatch) can be used which runs the [`src/training/single_train_run.py`](src/training/single_train_run.py). 
The corresponding slurm file for the testing is the [`setup_FZJ_booster/test_RootNet_run_booster.sbatch`](setup_FZJ_booster/test_RootNet_run_jureca_test.sbatch) executing the [`src/training/model_test.py`](src/training/model_test.py)

#### Inference

An example file for how the model can be used for inference is in the [`src/training/model_inference.py`](src/training/model_inference.py)

## Other info

For additional information regarding specific files and what they do, see the documentation of the individual scripts. 