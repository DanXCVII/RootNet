import sys
import json

sys.path.append("../")

from pl_setup import MyUNETRWrapper
from mri_dataloader import MRIDataLoader

import torch
import pytorch_lightning as pl


model_name = "weight_1_4_Data_DICE_softmax_UPSCALESWINUNETR-img_shape_96-feat_36-upscale_True-out_channels_2-lr_0.0008-upsample_end_False"
checkpoint_dir = f"../../runs/{model_name}/"
# load the model params from the checkpoint dir (../../{model_name}/train_params.json) and store them in a dictionary
with open(f"{checkpoint_dir}train_params.json") as f:
    train_params = json.load(f)

my_dl = MRIDataLoader("../../data", 1, True, 1, (96, 96, 96))

model = MyUNETRWrapper.load_from_checkpoint(f"{checkpoint_dir}best_metric_model.ckpt")
model.to(torch.device("cuda"))
trainer = pl.Trainer(
    accelerator="gpu",
    strategy=pl.strategies.DDPStrategy(find_unused_parameters=False),
    default_root_dir=".tb_logs/runs",
)
trainer.test(model=model, dataloaders=my_dl)
