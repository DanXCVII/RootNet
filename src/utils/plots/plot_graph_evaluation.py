import os
import json
import matplotlib.pyplot as plt
import ast
import re
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


def parse_file_content(content):
    # Remove the outer list and OrderedDict
    content = content.strip()[1:-1]  # Remove outer square brackets
    content = content[12:-2]  # Remove 'OrderedDict(' and '))'

    # Split into key-value pairs
    pairs = re.findall(r"\(\'(.*?)\',\s*(.*?)\)(?=,\s*\(|$)", content, re.DOTALL)

    result = {}
    for key, value in pairs:
        # Try to evaluate the value if it's a simple type
        try:
            value = eval(value)
        except:
            # If eval fails, keep it as a string
            pass
        result[key] = value

    return result


def read_total_length(file_path, key):
    with open(file_path, "r") as f:
        content = f.read()
    # Use ast.literal_eval to safely evaluate the string representation of the OrderedDict
    data = parse_file_content(content)
    # print the keys of the dictoinary
    if "All laterals:" in key:
        my_d = data["All laterals:"][1][1]
    else:
        my_d = data[key]
    return my_d


def get_lengths(base_path, method, key):
    lengths = []
    for i in range(7):  # 0 to 6
        file_path = os.path.join(base_path, method, str(i), "dict_2.txt")
        lengths.append(read_total_length(file_path, key))
    return lengths


# Base path
base_path = "/Users/daniel/Desktop/FZJ/Binary_output_FINAL"


def plot_performance(title, ylabel, key, filename):
    # Get lengths for each method
    threshold_lengths = get_lengths(base_path, "Threshold_segmentation", key)
    root_scale_lengths = get_lengths(base_path, "FS_test_prediction", key)
    gmm_lengths = get_lengths(base_path, "GMM_test_prediction", key)
    label_lengths = get_lengths(base_path, "label", key)
    # Create scatter plot
    plt.figure(figsize=(4.8, 4.0))
    samples = range(7)  # 0 to 6

    plt.scatter(samples, root_scale_lengths, label="FS NN", marker="o")
    plt.scatter(samples, gmm_lengths, label="GMM NN", marker="^")
    plt.scatter(samples, threshold_lengths, label="Best Threshold", marker="s")
    plt.scatter(samples, label_lengths, label="Label", marker="x")

    x_labels = [
        "Loam MRI 1",
        "Loam MRI 2",
        "Loam MRI 3",
        "Loam MRI 4",
        "Sand MRI 1",
        "Sand MRI 2",
        "Sand MRI 3",
    ]
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.axvline(x=3.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    plt.xlabel("Test Sample")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Add value labels
    for i, method_lengths in enumerate(
        [threshold_lengths, root_scale_lengths, gmm_lengths]
    ):
        for j, length in enumerate(method_lengths):
            continue
            plt.annotate(
                f"{length:.1f}",
                (samples[j], length),
                xytext=(5, 5),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(f"figures/{filename}.png", dpi=300)
    plt.savefig(f"figures/{filename}.svg")


plot_performance(
    "Total Root Length per Test Sample",
    "Total Root Length (mm)",
    "Total length [mm]",
    "total_root_length",
)
plot_performance(
    "Mean Root Depth per Test Sample",
    "Mean Root Depth [mm]",
    "Mean depth [mm]",
    "mean_root_depth",
)
plot_performance(
    "Mean Root Diameter per Test Sample",
    "Mean Root Diameter [mm]",
    "Mean diameter [mm]",
    "mean_root_diameter",
)
plot_performance(
    "Lateral Root Count",
    "Lateral Root Count",
    "All laterals:",
    "total_lateral_count",
)
