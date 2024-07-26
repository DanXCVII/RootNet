import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors

# Define the path to your .otf font file
font_path = "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf"
# Create a FontProperties object with the .otf font
font_properties = FontProperties(fname=font_path)

# Set the font globally for all plots
plt.rcParams["font.family"] = font_properties.get_name()
plt.rcParams.update({"font.size": 12})


def plot_perf(files1, files2, title, file_name, ylabel):
    data = []
    learning_rates = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016]

    # Read data from each file and append to the data list
    for lr, file1, file2 in zip(learning_rates, files1, files2):
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)

        for i in range(4):
            data.append([lr, df1.loc[i, "value"], "Loam", "Archit. 1"])
        for i in range(4):
            data.append([lr, df2.loc[i, "value"], "Loam", "Archit. 2"])
        for i in range(4, 7):
            data.append([lr, df1.loc[i, "value"], "Sand", "Archit. 1"])
        for i in range(4, 7):
            data.append([lr, df2.loc[i, "value"], "Sand", "Archit. 2"])

    df_combined = pd.DataFrame(
        data,
        columns=["Learning Rate", "Dice", "Soil Type", "Archit."],
    )
    # combine Soil Type with Architecture
    df_combined["Soil Type"] = df_combined["Soil Type"] + " - " + df_combined["Archit."]

    # Create the plot
    plt.figure(figsize=(10.75, 3.8))
    print(df_combined)
    custom_colors = ["#F1B8B8", "#E53B2F", "#B9CCDD", "#1F81D9"]
    alpha = 0
    custom_palette = [mcolors.to_rgba(color, alpha) for color in custom_colors]

    sns.boxplot(
        x="Learning Rate",
        y="Dice",
        hue="Soil Type",
        data=df_combined,
        width=0.7,
        palette=custom_palette,
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
    plt.legend(handles[:4], labels[:4], title="Soil Type")
    plt.grid(True, linestyle="--", alpha=0.7)
    # set the y scale to log
    plt.yscale("log")

    plt.savefig(f"figures/{file_name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"figures/{file_name}.svg", dpi=300, bbox_inches="tight")


# Load the uploaded CSV files
deconv_files = [
    "./data/superres_new/simple_deconv/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0001-upsample_end_True/Test_surface_distance.csv",
    "./data/superres_new/simple_deconv/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0002-upsample_end_True/Test_surface_distance.csv",
    "./data/superres_new/simple_deconv/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0004-upsample_end_True/Test_surface_distance.csv",
    "./data/superres_new/simple_deconv/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0008-upsample_end_True/Test_surface_distance.csv",
    "./data/superres_new/simple_deconv/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0016-upsample_end_True/Test_surface_distance.csv",
]

# replace the work deconv with extend_unet for all the deconv_files
extend_unet_files = [
    "./data/superres_new/extend_unet/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0001-upsample_end_False/Test_surface_distance.csv",
    "./data/superres_new/extend_unet/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0002-upsample_end_False/Test_surface_distance.csv",
    "./data/superres_new/extend_unet/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0004-upsample_end_False/Test_surface_distance.csv",
    "./data/superres_new/extend_unet/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0008-upsample_end_False/Test_surface_distance.csv",
    "./data/superres_new/extend_unet/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0016-upsample_end_False/Test_surface_distance.csv",
]

# TODO: Run load_logs again to load the new files

plot_perf(
    deconv_files,
    extend_unet_files,
    "Test Symmetric Surface Distance",
    "superres_boxplot_symmetric_surface_distance",
    "Symmetric Surface Distance",
)
