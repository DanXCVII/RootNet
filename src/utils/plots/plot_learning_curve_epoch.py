import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter

# Define the path to your .otf font file
font_path = "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf"
# Create a FontProperties object with the .otf font
font_properties = FontProperties(fname=font_path)

# Set the font globally for all plots
plt.rcParams["font.family"] = font_properties.get_name()
plt.rcParams.update({"font.size": 12})


def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df["Value"].values


def plot_performance(file_paths, learning_rates, config_name, title):
    plt.figure(figsize=(4.8, 3.8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(learning_rates)))

    for i, (file_path, lr) in enumerate(zip(file_paths, learning_rates)):
        values = read_csv_file(file_path)
        epochs = np.arange(1, len(values) + 1)  # Create epoch numbers
        formatted_lr = "{:.1e}".format(lr).replace("e-0", "e-")
        plt.plot(epochs, values, color=colors[i], label=f"lr={formatted_lr}")

    plt.xlabel("Epoch")
    plt.ylabel("Dice Loss")
    plt.title(f"{title} Loss - {config_name}")
    plt.legend(loc="upper right", fontsize="small")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()


def create_plots(file_paths_config1, file_paths_config2, learning_rates):
    plt.figure(figsize=(20, 8))
    # Plot for Configuration 1
    plot_performance(file_paths_config1, learning_rates, "Architecture 1")
    plt.savefig(
        "figures/performance_across_configs_1.png", bbox_inches="tight", dpi=300
    )
    plt.savefig("figures/performance_across_configs_1.svg", bbox_inches="tight")

    plt.figure(figsize=(20, 8))
    # Plot for Configuration 2
    plot_performance(file_paths_config2, learning_rates, "Architecture 2")

    plt.savefig(
        "figures/performance_across_configs_2.png", bbox_inches="tight", dpi=300
    )
    plt.savefig("figures/performance_across_configs_2.svg", bbox_inches="tight")


# Specify your file paths here
file_paths_config1 = [
    "./data/superres/deconv/train_loss/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0001-upsample_end_True_version_1.csv",
    "./data/superres/deconv/train_loss/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0002-upsample_end_True_version_1.csv",
    "./data/superres/deconv/train_loss/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0004-upsample_end_True_version_1.csv",
    "./data/superres/deconv/train_loss/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0008-upsample_end_True_version_1.csv",
    "./data/superres/deconv/train_loss/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0016-upsample_end_True_version_1.csv",
]

file_paths_config2 = [
    "./data/superres/extend_unet/train_loss/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0001-upsample_end_False_version_1.csv",
    "./data/superres/extend_unet/train_loss/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0002-upsample_end_False_version_1.csv",
    "./data/superres/extend_unet/train_loss/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0004-upsample_end_False_version_1.csv",
    "./data/superres/extend_unet/train_loss/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0008-upsample_end_False_version_1.csv",
    "./data/superres/extend_unet/train_loss/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0016-upsample_end_False_version_1.csv",
]

learning_rates = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016]

# Call the function to create the plots
create_plots(file_paths_config1, file_paths_config2, learning_rates, "Training")
