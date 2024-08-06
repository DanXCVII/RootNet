#!/bin/bash -x
#SBATCH --account=visforai
#SBATCH --mail-user=d.weissen@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                  # Use the 4 GPUs available
#SBATCH --ntasks-per-node=1           # When using pl it should always be set to 4
#SBATCH --cpus-per-task=96            # Divide the number of cpus (96) by the number of GPUs (4)
#SBATCH --job-name=Root-Net-Testing
#SBATCH --output=output.%j
#SBATCH --error=error.%j
#SBATCH --time=00:30:00
#SBATCH --partition=dc-gpu # -devel

export CUDA_VISIBLE_DEVICES=0
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

cd /p/project/visforai/$USER
source root_net_venv/activate.sh # Now we finally use the fastai module

cd /p/project/visforai/$USER/RootNet/src/training

# model_test params:
# -m: path to the checkpoint directory of the trained model
# -d: data path

srun python model_test.py -m "superres/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0016-upsample_end_False" -d "../../data_final"
