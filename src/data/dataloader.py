import sys

sys.path.append("../")

import os
import shutil
import tempfile
import json

import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.crop_transform import RandCropByPosNegLabeldWithResAdjust
from models import MyMonaiUNETR


from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


import torch

print_config()

# Setup data directory

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)

# Setup transforms for training and validation

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Spacingd( # TODO: ask if it should be used
        #     keys=["image", "label"], # different transform for image and label
        #     pixdim=(0.27, 0.27, 0.27), # maybe make z twice as big because the slices are always much bigger
        #     mode=("bilinear", "nearest"),
        # ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=30000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeldWithResAdjust(
            image_key="image",
            label_key="label",
            spatial_size=(128, 128, 128),
            pos=1,
            neg=1,
            num_samples=4,
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Spacingd(
        #     keys=["image"],
        #     pixdim=(0.27, 0.27, 0.27),
        #     mode=("bilinear", "nearest"),
        # ),
        # Spacingd(
        #     keys=["label"],
        #     pixdim=(0.5, 0.135, 0.135),
        #     mode=("bilinear", "nearest"),
        # ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=30000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)

# Download dataset and format in the folder.

data_config_path = "data_config.json"

with open(data_config_path, "r") as json_file:
    data_dict = json.load(json_file)

datasets = data_config_path
datalist = data_dict["training"]
val_files = data_dict["validation"]
train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=3,  # 24,
    cache_rate=1.0,
    num_workers=8,
)
train_loader = DataLoader(
    train_ds,
    batch_size=1,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)
val_ds = CacheDataset(
    data=val_files,
    transform=val_transforms,
    cache_num=3,  # 6,
    cache_rate=1.0,
    num_workers=4,
)
val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# Check data shape and visualize
case_num = 0
img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
img = val_ds[case_num]["image"]
label = val_ds[case_num]["label"]
# array_sum = np.sum(data)
tensor_sum = torch.sum(label)
print(f"tensor sum: {tensor_sum}")
# print average value

tensor_average = torch.mean(label)
tensor_max = torch.median(label)
print(f"tensor median: {tensor_max}")
print(f"tensor average: {tensor_average}")
unique_values = torch.unique(label)
print(f"unique values: {unique_values}")


# # plot a slice to see if the data is valid
# img_shape = img.shape
# label_shape = label.shape
# plt.figure("image", (18, 6))
# plt.subplot(1, 2, 1)
# plt.title("image")
# plt.imshow(img[0, :, :, 100].detach().cpu(), cmap="gray")
# plt.subplot(1, 2, 2)
# plt.title("label")
# label.astype("int16").tofile("test459x459x403" + ".raw")
# plt.imshow(label[0, :, :, 200].detach().cpu())
# plt.show()

# Create Model, Loss, Optimizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Execute a typical PyTorch training process


def validation(epoch_iterator_val):
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
                "Validate (%d / %d Steps)" % (global_step, 10.0)
            )
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        ### save image file
        batch["image"][0].astype("int16").tofile("test229x229x201" + ".raw")
        print(f"batch image shape: {batch['image'].shape}")
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        return

    #     logit_map = model(x)
    #     loss = loss_function(logit_map, y)
    #     loss.backward()
    #     epoch_loss += loss.item()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     epoch_iterator.set_description(
    #         "Training (%d / %d Steps) (loss=%2.5f)"
    #         % (global_step, max_iterations, loss)
    #     )
    #     if (
    #         global_step % eval_num == 0 and global_step != 0
    #     ) or global_step == max_iterations:
    #         epoch_iterator_val = tqdm(
    #             val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
    #         )
    #         dice_val = validation(epoch_iterator_val)
    #         epoch_loss /= step
    #         epoch_loss_values.append(epoch_loss)
    #         metric_values.append(dice_val)
    #         if dice_val > dice_val_best:
    #             dice_val_best = dice_val
    #             global_step_best = global_step
    #             torch.save(
    #                 model.state_dict(), os.path.join(root_dir, "best_metric_model.pth")
    #             )
    #             print(
    #                 "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
    #                     dice_val_best, dice_val
    #                 )
    #             )
    #         else:
    #             print(
    #                 "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
    #                     dice_val_best, dice_val
    #                 )
    #             )
    #     global_step += 1
    # return global_step, dice_val_best, global_step_best


max_iterations = 25000
eval_num = 500
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
global_step = 0
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []
print("Start Training...")
while global_step < max_iterations:
    global_step, dice_val_best, global_step_best = train(
        global_step, train_loader, dice_val_best, global_step_best
    )
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth")))
