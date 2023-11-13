import sys

sys.path.append("../")
sys.path.append("data")

from data import DataLoaderSetup, MyMRIDataset

from monai.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import MyMonaiUNETR
from torch.utils.tensorboard import SummaryWriter


from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
)

from monai.metrics import DiceMetric

from monai.data import (
    decollate_batch,
)


import torch

num_workers = 4

class PyTorchTrainer:
    def __init__(
        self,
        model,
        post_label,
        post_pred,
        loss_function,
        optimizer,
        train_loader,
        val_loader,
        device,
        runs_dir,
    ):
        self.model = model
        self.post_label = post_label
        self.post_pred = post_pred
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.runs_dir = runs_dir
        self.global_step = 0


    def setup_device(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        torch.backends.cudnn.benchmark = True
        self.model.to(self.device)

    def validate(self, epoch_iterator_val):
        """
        Validate the model on the validation set

        Args:
        - epoch_iterator_val (torch.utils.data.DataLoader): DataLoader containing the validation data.
        """
        model.eval()
        with torch.no_grad():
            for batch in epoch_iterator_val:
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                val_outputs = sliding_window_inference(
                    val_inputs,
                    (128, 128, 128),
                    4,
                    model,
                )
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                epoch_iterator_val.set_description(
                    "Validate (%d / %d Steps)" % (self.global_step, 10.0)
                )
            mean_dice_val = dice_metric.aggregate().item()
            dice_metric.reset()
        return mean_dice_val

    def train_epoch(self, train_loader, writer):
        """
        Train the model for one epoch

        Args:
        - train_loader (torch.utils.data.DataLoader): DataLoader containing the training data.
        - writer (tensorboard.SummaryWriter): Tensorboard writer to log training loss.
        """
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(
            train_loader,
            desc="Training (X / X Steps) (loss=X.X)",
            dynamic_ncols=True,
        )

        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].cuda(), batch["label"].cuda())

            logit_map = model(x)
            loss = loss_function(logit_map, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)"
                % (self.global_step, max_iterations, loss)
            )
            # log training loss to tensorboard
            writer.add_scalar("training_loss", loss.item(), self.global_step)
            self.global_step += 1

        epoch_loss /= step

        return self.global_step, epoch_loss

    def train(self, max_iterations, eval_num):
        """
        Train the model for a given number of iterations and evaluate it for every eval_num iterations.

        Args:
        - max_iterations (int): Number of iterations to train the model.
        - eval_num (int): Number of iterations after which the model is evaluated.
        """
        self.setup_device()
        # create the tensorboard logger

        # Only the root process should handle logging and saving checkpoints
        writer = SummaryWriter(log_dir=self.runs_dir + "/logs")
        writer = SummaryWriter(log_dir=runs_dir + "/logs")
        epoch_loss_values = []
        metric_values = []

        while self.global_step < max_iterations:
            self.global_step, epoch_loss = self.train_epoch(
               train_loader, writer
            )
            epoch_loss_values.append(epoch_loss)

            if self.global_step % eval_num == 0 or self.global_step == max_iterations:
                epoch_iterator_val = tqdm(
                    val_loader,
                    desc="Validate (X / X Steps) (dice=X.X)",
                    dynamic_ncols=True
                )
              
                
                dice_val = self.validate(epoch_iterator_val)

                # Log validation metrics
                metric_values.append(dice_val)
                writer.add_scalar("Loss/epoch", epoch_loss, self.global_step)
                writer.add_scalar("Dice/val", dice_val, self.global_step)

                # Save the model if dice score improves
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = self.global_step
                    torch.save(
                        model.state_dict(),
                        os.path.join(runs_dir, "best_metric_model.pth"),
                    )
                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
        writer.close()
        return global_step_best, dice_val_best
    


# ====================
# Environment Setup
# ====================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================
# Model Configuration
# ====================
# Define the model
model = MyMonaiUNETR(
    in_channels=1,
    out_channels=1,
    img_size=(128, 128, 128),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    proj_type="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)

# ====================
# Training Setup
# ====================
# Load training data
train_dataset = MyMRIDataset(data="training")

# Training DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=num_workers, pin_memory=True)

# Loss function, optimizer, and metrics
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)


# Metrics and post-processing transforms
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
post_label = AsDiscrete(to_onehot=1)
post_pred = AsDiscrete(argmax=True, to_onehot=1)

# Validation data loader (assumes DataLoaderSetup is defined)
data_loader_setup = DataLoaderSetup()
val_loader = data_loader_setup.val_loader

# Save directory for model data
runs_dir = "../../runs"

# ====================
# Training Parameters
# ====================
max_iterations = 25000
eval_num = 500

# ====================
# Training Execution
# ====================
# Initialize and start the trainer
trainer = PyTorchTrainer(
    model,
    post_label,
    post_pred,
    loss_function,
    optimizer,
    train_loader,
    val_loader,
    device,
    runs_dir,
)

# Start the training process
trainer.train(max_iterations, eval_num)
