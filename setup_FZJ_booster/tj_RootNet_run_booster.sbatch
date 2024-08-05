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

cd /p/project/visforai/$USER
source root_net_venv/activate.sh # Now we finally use the fastai module

cd /p/project/visforai/$USER/RootNet/src/training

# single_train_run params:
# -d: data path
# -b, -spv: b: number of MRIs used one batch, spv: number of samples per MRI -> batch size = b * spv
# -u: whether to use twice the resolution for the segmentation or not (depends on the NN if it applies superrsolution)
# -p: patch size (input size of the NN)
# -cw: class weights
# -lr: learning rate
# -m: model name
# -mp: model parameters (depends on the model)


srun python single_train_run.py -d '../../data_root_scale' -b 3 -spv 1 -u True -p 96 96 96 -cw 0 1 -lr 0.0002 -m 'UPSCALESWINUNETR' -mp '{"in_channels": 1, "out_channels": 2, "feature_size": 36, "upsample_end": "False"}' -me 60
