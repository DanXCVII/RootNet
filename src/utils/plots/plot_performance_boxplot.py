import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_perf(files, title, file_name, ylabel):
    data = []
    learning_rates = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016]

    # Read data from each file and append to the data list
    for lr, file in zip(learning_rates, files):
        df = pd.read_csv(file)
        for i in range(4):
            data.append([lr, df.loc[i, "Value"], "Loam"])
        for i in range(4, 7):
            data.append([lr, df.loc[i, "Value"], "Sand"])

    df_combined = pd.DataFrame(data, columns=["Learning Rate", "Dice", "Soil Type"])

    # Create the plot
    plt.figure(figsize=(4.8, 3.8))
    print(df_combined)
    sns.boxplot(
        x="Learning Rate",
        y="Dice",
        hue="Soil Type",
        data=df_combined,
        width=0.7,
        palette="Pastel1",
    )
    strip = sns.stripplot(
        x="Learning Rate",
        y="Dice",
        hue="Soil Type",
        data=df_combined,
        dodge=True,
        color="green",
        jitter=True,
        size=4,
        alpha=0.6,
    )

    plt.title(title, fontsize=16)
    plt.xlabel("Learning Rate", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    handles, labels = strip.get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title="Soil Type")
    plt.savefig(f"figures/{file_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"figures/{file_name}.svg", dpi=300, bbox_inches="tight")


# Load the uploaded CSV files
deconv_files = [
    "./data/superres/deconv/test_surface_distance/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0001-upsample_end_True_version_1.csv",
    "./data/superres/deconv/test_surface_distance/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0002-upsample_end_True_version_1.csv",
    "./data/superres/deconv/test_surface_distance/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0004-upsample_end_True_version_1.csv",
    "./data/superres/deconv/test_surface_distance/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0008-upsample_end_True_version_1.csv",
    "./data/superres/deconv/test_surface_distance/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0016-upsample_end_True_version_1.csv",
]

plot_perf(
    deconv_files,
    "Test Surface Dist. - Architecture 1",
    "deconv_boxplot",
    "Surface Distance",
)

# replace the work deconv with extend_unet for all the deconv_files
extend_unet_files = [
    "./data/superres/extend_unet/test_surface_distance/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0001-upsample_end_False_version_1.csv",
    "./data/superres/extend_unet/test_surface_distance/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0002-upsample_end_False_version_1.csv",
    "./data/superres/extend_unet/test_surface_distance/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0004-upsample_end_False_version_1.csv",
    "./data/superres/extend_unet/test_surface_distance/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0008-upsample_end_False_version_1.csv",
    "./data/superres/extend_unet/test_surface_distance/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0016-upsample_end_False_version_1.csv",
]

plot_perf(
    extend_unet_files,
    "Test Surface Dist. - Architecture 2",
    "extend_unet_boxplot",
    "Surface Distance",
)
