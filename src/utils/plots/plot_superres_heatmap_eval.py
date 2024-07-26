import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors


# Define the path to your .otf font file
font_path = "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf"
# Create a FontProperties object with the .otf font
font_properties = FontProperties(fname=font_path)

# Set the font globally for all plots
plt.rcParams["font.family"] = font_properties.get_name()
plt.rcParams.update({"font.size": 12})


# Function to extract dice score from a CSV file
def get_dice_score(file_path):
    df = pd.read_csv(file_path)
    return df["value"].mean()  # Assuming we want the best dice score


# Function to parse directory name and extract parameters
def parse_directory_name(dir_name):
    params = dir_name.split("-")
    patch_size = int(params[2].split("_")[-1])
    feat = int(params[3].split("_")[-1])
    lr = float(params[-2].split("_")[-1])
    return patch_size, feat, lr


# Directory containing the logs
logs_dir = "data/hyperparam_patch_feat/"

# Dictionary to store the best dice scores
best_scores = {}


def format_lr(lr):
    if lr == 0.0004:
        return "4e-4"
    elif lr == 0.0008:
        return "8e-4"
    else:
        return f"{lr:.0e}"  # This will handle other potential values


def get_text_color(background_color):
    rgb = mcolors.to_rgb(background_color)
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return "white" if luminance < 0.5 else "black"


def plot_perf_heatmap(filename, title, save_filename, lr_vals=None):
    best_scores = {}
    # Walk through the directory structure
    for root, dirs, files in os.walk(logs_dir):
        if filename in files:
            dir_name = os.path.basename(root)
            patch_size, feat, lr = parse_directory_name(dir_name)

            if lr_vals is not None and lr_vals[(patch_size, feat)] != format_lr(lr):
                continue

            dice_file = os.path.join(root, filename)
            dice_score = get_dice_score(dice_file)

            key = (patch_size, feat)
            if key not in best_scores or dice_score > best_scores[key][1]:
                best_scores[key] = (lr, dice_score)

    # Prepare data for the heatmap
    patch_sizes = sorted(set(k[0] for k in best_scores.keys()))
    feature_sizes = sorted(set(k[1] for k in best_scores.keys()))

    dice_scores = np.zeros((len(feature_sizes), len(patch_sizes)))
    if lr_vals is None:
        lr_values = {
            (patch_size, feat): format_lr(lr)
            for (patch_size, feat), (lr, _) in best_scores.items()
        }
    else:
        lr_values = lr_vals

    for i, feat in enumerate(feature_sizes):
        for j, patch in enumerate(patch_sizes):
            if (patch, feat) in best_scores:
                dice_scores[i, j] = best_scores[(patch, feat)][1]
                if lr_vals is not None:
                    lr_values[(patch, feat)] = format_lr(best_scores[(patch, feat)][0])

    # Create a DataFrame
    df = pd.DataFrame(dice_scores, index=feature_sizes, columns=patch_sizes)

    # Create the heatmap
    plt.figure(figsize=(4.8, 3.8))  # Increased figure size for better visibility
    ax = sns.heatmap(
        df,
        annot=True,
        cmap="YlGnBu",
        fmt=".3f",
        xticklabels=patch_sizes,
        yticklabels=feature_sizes,
    )

    # Get the color normalization
    norm = plt.Normalize(df.values.min(), df.values.max())

    # Add learning rate values to the corner of each cell
    for i in range(len(feature_sizes)):
        for j in range(len(patch_sizes)):
            # Get the background color
            bg_color = plt.get_cmap("YlGnBu")(norm(dice_scores[i, j]))
            # Determine text color based on background
            text_color = get_text_color(bg_color)

            text = ax.text(
                j + 0.05,
                i + 0.83,
                lr_values[(patch_sizes[i], feature_sizes[j])],
                ha="left",
                va="top",
                color=text_color,
                fontsize=8,
            )

    plt.title(title)
    plt.xlabel("Patch Size")
    plt.ylabel("Feature Size")

    # Save the figure
    plt.savefig(f"figures/{save_filename}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"figures/{save_filename}.svg", dpi=300, bbox_inches="tight")
    plt.close()

    # Print the DataFrame for reference
    print(df)

    return lr_values


lr_vals = plot_perf_heatmap(
    "Test_root_f1.csv",
    "Test Dice Heatmap",
    "test_dice_heatmap",
)

# save the lr_vals to a file
with open("lr_vals.txt", "w") as f:
    f.write(str(lr_vals))

plot_perf_heatmap(
    "Test_root_precision.csv",
    "Test Root Precision Heatmap",
    "test_root_precision_heatmap",
    lr_vals,
)

plot_perf_heatmap(
    "Test_root_precision.csv",
    "Test Root Precision Heatmap",
    "test_root_precision_heatmap",
    lr_vals,
)

plot_perf_heatmap(
    "Test_root_recall.csv",
    "Test Root Recall Heatmap",
    "test_root_recall_heatmap",
    lr_vals,
)

plot_perf_heatmap(
    "Test_surface_distance.csv",
    "Test Surface Distance Heatmap",
    "test_root_surface_distance_heatmap",
    lr_vals,
)
