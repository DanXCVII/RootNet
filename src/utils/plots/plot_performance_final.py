import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter

# plots the performance of the final model vs the best threshold

# Define the path to your .otf font file
font_path = "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf"
# Create a FontProperties object with the .otf font
font_properties = FontProperties(fname=font_path)

# Set the font globally for all plots
plt.rcParams["font.family"] = font_properties.get_name()
plt.rcParams.update({"font.size": 12})

# Define the root directory and configuration names
data_root = "./data/final_performance"
config1 = "data_root_scale_weight_0.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0004-upsample_end_True-batch_7-1"
config2 = "threshold_background_true"
config3 = "data_gmm_weight_0.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0004-upsample_end_True-batch_7-1"


# Function to read and process CSV file
def read_dice_csv(file_path):
    df = pd.read_csv(file_path)
    df["step"] = df["step"].astype(int)
    return df.set_index("step")["value"]


def plot_score(files, metric):
    # Read dice scores for both configurations
    dice_config1 = read_dice_csv(os.path.join(data_root, config1, files[0]))
    dice_config2 = read_dice_csv(os.path.join(data_root, config2, files[1]))
    dice_config3 = read_dice_csv(os.path.join(data_root, config3, files[2]))

    # Create the plot
    print("Metric: ", metric)
    print("FS-SV2+ values: ", dice_config1.values)
    print("Best threshold values: ", dice_config2.values)
    print("GMM-SV2+ Noise values: ", dice_config3.values)
    plt.figure(figsize=(4.8, 3.8))
    plt.scatter(
        dice_config1.index,
        dice_config1.values,
        label="FS-SV2+",
        marker="o",
    )
    plt.scatter(
        dice_config3.index,
        dice_config3.values,
        label="GMM-SV2+",
        marker="^",
    )
    plt.scatter(
        dice_config2.index,
        dice_config2.values,
        label="Best Threshold",
        marker="s",
    )

    # Add vertical line between Loam and Sand MRIs
    plt.axvline(x=3.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    plt.xlabel("Test Sample")

    # Customize the plot
    plt.title(f"Test {metric} Comparison")
    plt.ylabel(metric)
    plt.legend(fontsize="small")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Create custom x-axis labels
    x_labels = [
        "Loam MRI 1",
        "Loam MRI 2",
        "Loam MRI 3",
        "Loam MRI 4",
        "Sand MRI 1",
        "Sand MRI 2",
        "Sand MRI 3",
    ]

    # Set x-ticks and labels
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")

    # Adjust layout to prevent cutting off x-axis labels
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"figures/{metric}_scores_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"figures/{metric}_scores_comparison.svg", dpi=300, bbox_inches="tight")


plot_score(
    [
        "Test_dice.csv",
        "Test_root_f1.csv",
        "Test_dice.csv",
    ],
    "Dice",
)
plot_score(
    [
        "Test_root_precision.csv",
        "Test_root_precision.csv",
        "Test_root_precision.csv",
    ],
    "Precision",
)
plot_score(
    [
        "Test_root_recall.csv",
        "Test_root_recall.csv",
        "Test_root_recall.csv",
    ],
    "Recall",
)
plot_score(
    [
        "Test_surface_distance.csv",
        "Test_surface_distance.csv",
        "Test_surface_distance.csv",
    ],
    "Surface Distance",
)
