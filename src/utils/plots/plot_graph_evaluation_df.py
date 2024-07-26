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
    threshold_lengths = get_lengths(base_path, "threshold_output", key)
    root_scale_lengths = get_lengths(base_path, "root_scale_output", key)
    gmm_lengths = get_lengths(base_path, "gmm_output", key)
    label_lengths = get_lengths(base_path, "label", key)

    print(f"\n{title}")
    print("Threshold mean: ", sum(threshold_lengths) / len(threshold_lengths))
    print("Root scale mean: ", sum(root_scale_lengths) / len(root_scale_lengths))
    print("GMM mean: ", sum(gmm_lengths) / len(gmm_lengths))
    print("Label mean: ", sum(label_lengths) / len(label_lengths))

    print("Threshold values: ", threshold_lengths)
    print("Root scale values: ", root_scale_lengths)
    print("GMM values: ", gmm_lengths)
    print("Label values: ", label_lengths)


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
