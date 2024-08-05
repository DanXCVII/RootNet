# Virtual MRI Generation Pipeline

## Overview

This repository houses the implementation of a pipeline designed to generate virtual Magnetic Resonance Imaging (MRI) scans of root systems for the Master thesis "3D Segmentation of Plant Roots from MRI Images for Enhanced Automated Root Tracing using Deep Neural Networks". The primary objective of this project is to use these virtual MRIs to train a Neural Network for the task of segmenting real MRI root scans.

## General Structure

### `src` Directory

The `src` directory contains all source code of this repo and is divided into several subdirectories, each dedicated to a specific aspect of the project's codebase:
- **`data`**: Contains code necessary for synthetic data generation. This subdirectory is used for creating the datasets needed to train the NNs.
- **`virtual_mri_generation`**: Code related to the generation of a single synthetic MRI, including water and root growth simulation.
- **`models`**: Houses the neural network models used for segmentation of the root MRIs.
- **`training`**: All code necessary for setting up the training environment, data loading, and the training process itself.
- **`utils`**: Contains a collection of utility functions that help organize files and streamline various tasks.
  
### `data_assets` Directory

The `data_assets` directory is intended for storing fixed data assets. These assets include meshes for the simulation of water flow, the used root models for simulation and MRI data, which is used as a ground truth for synthetic MRI generation


### `data` Directory

The `data`directory should contain all MRI data, which is used for training and testing of the DNN. This includes the nifti file of the MRI together with the corresponding label file with twice the resolution. The naming scheme is the following "image_resx_res_y_resz.nii.gz" for the MRI image and "label_image_resx\*2_resy\*2_resz\*2.nii.gz". E.g. "my_Bench_lupin_day_5_res_237x237x151.nii.gz" and "label_my_Bench_lupin_day_5_res_474x474x302.nii.gz". The data used for the corresponding Master thesis can be downloaded [here](https://zenodo.org/records/12806033): 
The structure is the following:
- **`data/test`**: Contains all test data being used for the evaluation of the DNN
- **`data/generated/training`**: Contains all training data for the DNN
- **`data/generated/validation`**: Contains all validation data for the DNN

## Setup

Before being able to run the data generation, CPlantBox must be installed with the addition of DUMUX. An instruction on how to do that can be found on the official CPlantBox repo (https://github.com/Plant-Root-Soil-Interactions-Modelling/CPlantBox).
Additionally, modify the file `DUMUX_path` to the `CPlantBox/DUMUX` directory for the machine you are running it on.
An important fix for the checkpoint loading of the neural network for some pytorch-lightning versions is provided here:
https://github.com/Lightning-AI/pytorch-lightning/issues/13695

## Synthetic Data Generation

To run the synthetic MRI generation, execute the `generator.py` file inside the `src/data` directory. The required packages can be found in `src/data/requirements.txt`. The generated MRI data is stored in the `data` folder on the top level.

Another option for running the data generation is to execute the `run_parallel.sh` file inside `src/data` where the number of parallel runs can be adjusted in the for loop by replacing X in the following: `for i in X` with the number of parallel runs. This script automatically restarts the generation process, if one run fails, which can occasionally happen for the water simulation.


## 🚧 Project Status: WIP - Work in Progress

This readme will be extended later on


- added data_assets such that one without access to real MRIs can run the repo
- modified local imports, such that a setup on another machine is easier
- mofidied comments to improve readability
- modified the noise generation to only add noise based on the fourier transformation