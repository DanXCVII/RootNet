#!/bin/bash -x
#SBATCH --account=visforai
#SBATCH --mail-user=d.weissen@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:4                  # Use the 4 GPUs available
#SBATCH --ntasks-per-node=4           # When using pl it should always be set to 4
#SBATCH --cpus-per-task=24            # Divide the number of cpus (96) by the number of GPUs (4)
#SBATCH --job-name=Root-Net-Training
#SBATCH --output=output.%j
#SBATCH --error=error.%j
#SBATCH --time=01:30:00
#SBATCH --partition=dc-gpu # -devel

export CUDA_VISIBLE_DEVICES=0,1,2,3
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# cd /p/project/training2324/$USER/RootProj
# source root_net_venv/activate.sh # Now we finally use the fastai module

# cd /p/project/training2324/$USER/RootProj/RootNet/src/models

# srun python unetr_monai.py

cd /p/project/visforai/$USER
source root_net_venv/activate.sh # Now we finally use the fastai module

cd /p/project/visforai/$USER/RootNet/src/training

# train the UpscaleSwinUNETR model: feat: 36
# srun python single_train_run.py -d '../../data_perlin' -b 11 -spv 1 -u True -p 96 96 96 -cw 1 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 1, "feature_size": 36, "upsample_end": "True"}' -me 60
# train the UpscaleSwinUNETR model: feat: 24
# srun python single_train_run.py -d '../../data_final' -b 2 -spv 4 -u True -p 96 96 96 -cw 1 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 1, "feature_size": 24, "upsample_end": "False"}' -me 60
# train the UpscaleSwinUNETR model: feat: 12
# srun python single_train_run.py -d '../../data_final' -b 3 -spv 3 -u True -p 96 96 96 -cw 1 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 1, "feature_size": 12, "upsample_end": "False"}' -me 60

# UNET
# srun python single_train_run.py -b 3 -spv 4 -u True -p 60 60 60 -cw 1 1 -lr 0.001 -m 'UNET' -mp '{"in_channels": 1, "out_channels": 1, "feature_size": 16}' -me 60


# superresolution
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 0 1 -lr 0.0001 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 0 1 -lr 0.0002 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 0 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 0 1 -lr 0.0008 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 0 1 -lr 0.0016 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60

# srun python single_train_run.py -d '../../data_root_scale' -b 3 -spv 1 -u True -p 96 96 96 -cw 0 1 -lr 0.0001 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "False"}' -me 60
srun python single_train_run.py -d '../../data_root_scale' -b 3 -spv 1 -u True -p 96 96 96 -cw 0 1 -lr 0.0002 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "False"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 3 -spv 1 -u True -p 96 96 96 -cw 0 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "False"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 3 -spv 1 -u True -p 96 96 96 -cw 0 1 -lr 0.0008 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "False"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 3 -spv 1 -u True -p 96 96 96 -cw 0 1 -lr 0.0016 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "False"}' -me 60


# hyperparam - patch, feat
# srun python single_train_run.py -d '../../data_root_scale' -b 10 -spv 1 -u True -p 32 32 32 -cw 1 1 -lr 0.0008 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 48, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 10 -spv 1 -u True -p 32 32 32 -cw 1 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 10 -spv 1 -u True -p 32 32 32 -cw 1 1 -lr 0.0008 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 24, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 10 -spv 1 -u True -p 32 32 32 -cw 1 1 -lr 0.0008 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 12, "upsample_end": "True"}' -me 60

# srun python single_train_run.py -d '../../data_root_scale' -b 10 -spv 1 -u True -p 64 64 64 -cw 1 1 -lr 0.0008 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 48, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 10 -spv 1 -u True -p 64 64 64 -cw 1 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 10 -spv 1 -u True -p 64 64 64 -cw 1 1 -lr 0.0008 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 24, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 10 -spv 1 -u True -p 64 64 64 -cw 1 1 -lr 0.0008 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 12, "upsample_end": "True"}' -me 60

# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 1 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 48, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 1 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 9 -spv 1 -u True -p 96 96 96 -cw 1 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 24, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 10 -spv 1 -u True -p 96 96 96 -cw 1 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 12, "upsample_end": "True"}' -me 60

# srun python single_train_run.py -d '../../data_root_scale' -b 2 -spv 1 -u True -p 128 128 128 -cw 1 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 48, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 2 -spv 1 -u True -p 128 128 128 -cw 1 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 2 -spv 1 -u True -p 128 128 128 -cw 1 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 24, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 2 -spv 1 -u True -p 128 128 128 -cw 1 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 12, "upsample_end": "True"}' -me 60

# hyperparam - cw
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 1 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 1 2 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 1 3 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 1 4 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 1 5 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 1 6 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 1 10 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 1 20 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 0.1 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60

# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 1 1 -lr 0.0008 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 1 2 -lr 0.0008 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 1 3 -lr 0.0008 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 1 6 -lr 0.0008 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60

# hyperparam - batch size
# srun python single_train_run.py -d '../../data_root_scale' -b 1 -spv 1 -u True -p 96 96 96 -cw 0 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 1 -spv 7 -u True -p 96 96 96 -cw 0 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 2 -spv 3 -u True -p 96 96 96 -cw 0 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 3 -spv 2 -u True -p 96 96 96 -cw 0 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 0 1 -lr 0.0004 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60

# srun python single_train_run.py -d '../../data_root_scale' -b 1 -spv 7 -u True -p 96 96 96 -cw 1 1 -lr 0.0008 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 2 -spv 3 -u True -p 96 96 96 -cw 1 1 -lr 0.0008 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 3 -spv 2 -u True -p 96 96 96 -cw 1 1 -lr 0.0008 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
# srun python single_train_run.py -d '../../data_root_scale' -b 7 -spv 1 -u True -p 96 96 96 -cw 1 1 -lr 0.0008 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "True"}' -me 60
