import sys

sys.path.append("../utils")

import monai
from monai.transforms import (
    LoadImaged,
    RandBiasFieldd,
    SaveImaged,
    Compose,
    ScaleIntensityd,
    RandAffined,
    RandRotate90d,
    EnsureChannelFirstd,
    RandFlipd,
    RandAdjustContrastd,
    RandCoarseDropoutd,
    RandHistogramShiftd,
)
from monai.data import DataLoader, Dataset, CacheDataset
from crop_transform import RandCropByPosNegLabeldWithResAdjust
import os
import numpy as np
from MRI_operations import MRIoperations
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Define the path to your .otf font file
font_path = "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf"
# Create a FontProperties object with the .otf font
font_properties = FontProperties(fname=font_path)

# Set the font globally for all plots
plt.rcParams["font.family"] = font_properties.get_name()
plt.rcParams.update({"font.size": 9})


# Define the path to the input file and output directory
sand_mri_path = "/Users/daniel/Desktop/FZJ/CPlantBox/DUMUX/CPlantBox/tutorial/examples_segmentation/RootNet/src/plots/mris/my_Moraesetal_2020_day_5_res_237x237x151.nii.gz"
loam_mri_path = "/Users/daniel/Desktop/FZJ/CPlantBox/DUMUX/CPlantBox/tutorial/examples_segmentation/RootNet/src/plots/mris/my_Moraesetal_2020_day_9_res_237x237x161.nii.gz"
loam_label_path = "/Users/daniel/Desktop/FZJ/CPlantBox/DUMUX/CPlantBox/tutorial/examples_segmentation/RootNet/src/plots/mris/label_my_Moraesetal_2020_day_9_res_474x474x322.nii.gz"
output_dir = "./mris/"


# Create a dictionary with the file path

monai.utils.set_determinism(seed=44)  # 46 loam 44 sand


def perform_transform(image_path, label_path, output_dir):
    data_dicts = [{"image": image_path, "label": label_path}]

    # Define the transforms
    transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"], minv=0, maxv=1),
            # RandAffined(
            #     keys=["image", "label"],
            #     prob=1,
            #     # rotate_range=(1, 1, 1),
            #     scale_range=(1, 1, 1),
            #     mode=("bilinear", "nearest"),
            # ),
            EnsureChannelFirstd(keys=["image", "label"]),
            # RandAffined(
            #     keys=["image", "label"],
            #     prob=1,
            #     rotate_range=((-0.785, -0.785), (-0.785, -0.785), (-0.785, -0.785)),
            #     scale_range=((0.5, 0.5), (0.5, 0.5), (0.5, 0.5)),
            #     mode=("bilinear", "nearest"),
            # ),
            # RandBiasFieldd(keys=["image"], coeff_range=(0, 0.1), prob=1),
            # RandAdjustContrastd(keys=["image"], gamma=(0.5, 4.5), prob=1),
            # RandCoarseDropoutd(
            #     keys=["image", "label"],
            #     holes=16,
            #     fill_value=0,
            #     spatial_size=(50, 50, 50),
            #     prob=1,
            # ),
            # RandCropByPosNegLabeldWithResAdjust(
            #     image_key="image",
            #     label_key="label",
            #     spatial_size=(
            #         (96, 96, 96) if True else tuple(x // 2 for x in (96, 96, 96))
            #     ),
            #     pos=1,
            #     neg=1,
            #     num_samples=2,
            #     image_threshold=0,
            # ),
            # RandFlipd(
            #     keys=["image", "label"],
            #     spatial_axis=[2],
            #     prob=1,
            # ),
            # RandRotate90d(
            #     keys=["image", "label"],
            #     prob=1,
            #     max_k=3,
            # ),
            RandHistogramShiftd(keys=["image"], num_control_points=10, prob=1),
            ScaleIntensityd(keys=["image"], minv=0, maxv=1),
            SaveImaged(
                keys=["image", "label"],
                output_dir=output_dir,
                output_postfix="transformed",
                output_dtype=np.float32,
                resample=False,
            ),
        ]
    )

    # Create a dataset and a dataloader
    dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=1.0)
    dataloader = DataLoader(dataset, batch_size=1)

    # Apply the transform and save the image
    for batch in dataloader:
        pass  # The transform pipeline handles saving internally


def process_mri_slice(mri_data, z_slice, y_slice):
    mri_slice_hor = mri_data[:, :, z_slice]
    mri_slice_ver = mri_data[y_slice, :, :]
    # correct the orientation
    mri_slice_ver = np.swapaxes(mri_slice_ver, 0, 1)
    mri_slice_ver = np.flip(mri_slice_ver, axis=0)
    return mri_slice_hor, mri_slice_ver


def plot_mri_slices(mri_slices, titles, aspect_ratios):
    fig, axs = plt.subplots(1, len(mri_slices), figsize=(15, 3))
    for i, (slice_data, title, aspect) in enumerate(
        zip(mri_slices, titles, aspect_ratios)
    ):
        im = axs[i].imshow(slice_data, cmap="gray", aspect=aspect)
        axs[i].set_title(title)
        if i > 1:
            axs[i].set_xlabel("Y-axis")
            axs[i].set_ylabel("Z-axis")
        else:
            axs[i].set_xlabel("X-axis")
            axs[i].set_ylabel("Y-axis")
        axs[i].tick_params(axis="both", which="major")
        cbar = fig.colorbar(im, ax=axs[i], orientation="vertical")
        cbar.set_label("Intensity")
        cbar.ax.tick_params()


def plot_mri(image_file_path, label_file_path, soil_type):
    perform_transform(image_file_path, label_file_path, output_dir)

    mri_ops = MRIoperations()

    # extract the basename of the input file
    basename = os.path.basename(image_file_path)
    basename = os.path.splitext(basename)[0]
    basename = os.path.splitext(basename)[0]

    _, mri_trans = mri_ops.load_mri(f"mris/{basename}/{basename}_transformed.nii.gz")

    _, mri_old = mri_ops.load_mri(image_file_path)

    # Scale mri_old from 0 to 1
    div = np.max(mri_old) - np.min(mri_old)
    if div == 0:
        div = 1
    mri_old = (mri_old - np.min(mri_old)) / div

    z_slice = 70  # 105 for loam
    y_slice = 122

    # Process MRI slices
    mri_slice_hor_old, mri_slice_ver_old = process_mri_slice(mri_old, z_slice, y_slice)
    mri_slice_hor, mri_slice_ver = process_mri_slice(mri_trans, z_slice, y_slice)

    # Plot the slices
    plot_mri_slices(
        [mri_slice_hor_old, mri_slice_hor, mri_slice_ver_old, mri_slice_ver],
        [
            "Original MRI horizontal slice",
            "Transformed MRI horizontal slice",
            "Original MRI vertical slice",
            "Transformed MRI vertical slice",
        ],
        ["equal", "equal", 3.7, 3.7],
    )
    plt.tight_layout()

    # save the plot with text size 20
    plt.savefig(f"figures/mri_slices_{soil_type}.png")
    plt.savefig(f"figures/mri_slices_{soil_type}.svg")

    # plot the histogram of the whole mri_old and mri_trans side by side
    fig, axs = plt.subplots(1, 2, figsize=(11, 2.7))

    axs[0].hist(
        mri_old.flatten(), bins=300, color="blue", alpha=0.7, label="Original MRI"
    )
    axs[1].hist(
        mri_trans.flatten(), bins=300, color="red", alpha=0.7, label="Transformed MRI"
    )

    # axs[1].set_xlim(axs[0].get_xlim())
    # axs[1].set_ylim(axs[0].get_ylim())

    axs[0].set_title("Signal Intensity Distribution - Original MRI")
    axs[1].set_title("Signal Intensity Distribution - Transformed MRI")

    for ax in axs:
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Frequency")
        ax.tick_params(axis="both", which="major")
        ax.set_yscale("log")

    plt.tight_layout()
    # plot
    plt.savefig(f"figures/mri_histogram_{soil_type}.png")
    plt.savefig(f"figures/mri_histogram_{soil_type}.svg")

    fig, ax = plt.subplots(figsize=(5.25, 3))

    ax.hist(mri_old.flatten(), bins=60, color="blue", alpha=0.5, label="Original MRI")
    ax.hist(
        mri_trans.flatten(), bins=60, color="red", alpha=0.5, label="Transformed MRI"
    )
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="both", which="major")
    ax.set_yscale("log")
    ax.set_title("Signal Intensity Distribution \n(Original vs Transformed)")

    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.savefig(f"figures/mri_histogram_{soil_type}_combined.png", dpi=300)
    plt.savefig(f"figures/mri_histogram_{soil_type}_combined.svg")


plot_mri(loam_mri_path, loam_label_path, "loam")
plot_mri(sand_mri_path, loam_label_path, "sand")
