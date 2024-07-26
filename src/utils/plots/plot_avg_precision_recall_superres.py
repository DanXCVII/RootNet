import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import pandas as pd

# Define the path to your .otf font file
font_path = "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf"
# Create a FontProperties object with the .otf font
font_properties = FontProperties(fname=font_path)

# Set the font globally for all plots
plt.rcParams["font.family"] = font_properties.get_name()
plt.rcParams.update({"font.size": 12})


def plot_perf_vs_learning_rate(perf_config1, perf_config2, perf_metric):
    # Sample data (replace with your actual data)
    learning_rates = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016]

    # Create the plot
    plt.figure(figsize=(4.8, 3.8))
    plt.plot(learning_rates, perf_config1, marker="o", label="architecture 1")
    plt.plot(learning_rates, perf_config2, marker="s", label="architecture 2")

    # Customize the plot
    plt.xscale("log")  # Use log scale for learning rates
    plt.xlabel("Learning Rate")
    plt.ylabel(perf_metric)
    plt.title(f"Avg. Test {perf_metric} vs. Learning Rate")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Add value annotations
    for i, (p1, p2) in enumerate(zip(perf_config1, perf_config2)):
        plt.annotate(
            f"{p1:.4f}",
            (learning_rates[i], p1),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )
        plt.annotate(
            f"{p2:.4f}",
            (learning_rates[i], p2),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(f"figures/performance_vis_{perf_metric}.png", bbox_inches="tight")
    plt.savefig(f"figures/performance_vis_{perf_metric}.svg", bbox_inches="tight")


deconv_files = [
    "./data/superres/deconv/test_precision/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0001-upsample_end_True_version_1.csv",
    "./data/superres/deconv/test_precision/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0002-upsample_end_True_version_1.csv",
    "./data/superres/deconv/test_precision/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0004-upsample_end_True_version_1.csv",
    "./data/superres/deconv/test_precision/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0008-upsample_end_True_version_1.csv",
    "./data/superres/deconv/test_precision/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0016-upsample_end_True_version_1.csv",
]

# precision files
extend_unet_files = [
    "./data/superres/extend_unet/test_precision/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0001-upsample_end_False_version_1.csv",
    "./data/superres/extend_unet/test_precision/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0002-upsample_end_False_version_1.csv",
    "./data/superres/extend_unet/test_precision/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0004-upsample_end_False_version_1.csv",
    "./data/superres/extend_unet/test_precision/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0008-upsample_end_False_version_1.csv",
    "./data/superres/extend_unet/test_precision/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0016-upsample_end_False_version_1.csv",
]

# calculate the average precision for each learning rate
# read the deconv files as pandas dataframes
dfs = [pd.read_csv(file) for file in extend_unet_files]
avg_precisions_extend_unet = [df["Value"].mean() for df in dfs]

# read the extend_unet files as pandas dataframes
dfs = [pd.read_csv(file) for file in deconv_files]
avg_precisions_deconv = [df["Value"].mean() for df in dfs]

plot_perf_vs_learning_rate(
    avg_precisions_deconv, avg_precisions_extend_unet, "Precision"
)


deconv_files = [
    "./data/superres/deconv/test_recall/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0001-upsample_end_True_version_1.csv",
    "./data/superres/deconv/test_recall/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0002-upsample_end_True_version_1.csv",
    "./data/superres/deconv/test_recall/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0004-upsample_end_True_version_1.csv",
    "./data/superres/deconv/test_recall/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0008-upsample_end_True_version_1.csv",
    "./data/superres/deconv/test_recall/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0016-upsample_end_True_version_1.csv",
]

# precision files
extend_unet_files = [
    "./data/superres/extend_unet/test_recall/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0001-upsample_end_False_version_1.csv",
    "./data/superres/extend_unet/test_recall/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0002-upsample_end_False_version_1.csv",
    "./data/superres/extend_unet/test_recall/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0004-upsample_end_False_version_1.csv",
    "./data/superres/extend_unet/test_recall/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0008-upsample_end_False_version_1.csv",
    "./data/superres/extend_unet/test_recall/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0016-upsample_end_False_version_1.csv",
]


# calculate the average precision for each learning rate
# read the deconv files as pandas dataframes
dfs = [pd.read_csv(file) for file in extend_unet_files]
avg_precisions_extend_unet = [df["Value"].mean() for df in dfs]

# read the extend_unet files as pandas dataframes
dfs = [pd.read_csv(file) for file in deconv_files]
avg_precisions_deconv = [df["Value"].mean() for df in dfs]

plot_perf_vs_learning_rate(avg_precisions_deconv, avg_precisions_extend_unet, "Recall")
