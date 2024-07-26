import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.font_manager import FontProperties

# Define the path to your .otf font file
font_path = "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf"
# Create a FontProperties object with the .otf font
font_properties = FontProperties(fname=font_path)

# Set the font globally for all plots
plt.rcParams["font.family"] = font_properties.get_name()
plt.rcParams.update({"font.size": 12})


# Function to read CSV files
def read_csv(filename):
    return np.loadtxt(filename, delimiter=",")


# Get all x_values and dice_scores files
x_files = sorted(glob.glob("data/threshold/x_values_*.csv"))
dice_files = sorted(glob.glob("data/threshold/dice_scores_*.csv"))

# Create the plot
plt.figure(figsize=(9.6, 4.5))

# Define colors and labels for each line
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
labels = [
    "Loam MRI 1",
    "Loam MRI 2",
    "Loam MRI 3",
    "Loam MRI 4",
    "Sand MRI 1",
    "Sand MRI 2",
    "Sand MRI 3",
]

# Plot each pair of x_values and dice_scores
for i, (x_file, dice_file) in enumerate(zip(x_files, dice_files)):
    x_values = read_csv(x_file)
    dice_scores = read_csv(dice_file)

    # Plot line with dots
    plt.plot(
        x_values,
        dice_scores,
        label=labels[i],
        color=colors[i],
        marker="o",
        markersize=4,
    )

    # Find and annotate the max value point
    max_index = np.argmax(dice_scores)
    max_x = x_values[max_index]
    max_y = dice_scores[max_index]

    # Mark the maximum point with a star
    plt.plot(max_x, max_y, marker="*", markersize=15, color=colors[i])

    plt.text(
        max_x,
        max_y + 0.01,  # Slightly above the max point
        f"({max_x:.5f}, {max_y:.2f})",
        ha="center",  # Center align horizontally
        va="bottom",  # Align to the bottom of the text
        color=colors[i],  # Use the same color as the line
        fontweight="bold",
        fontsize=10,
    )

    # Annotate the maximum point
    # plt.annotate(
    #     f"Max: {max_y:.4f}",
    #     xy=(max_x, max_y),
    #     xytext=(10, 0),
    #     textcoords="offset points",
    #     ha="left",
    #     va="center",
    #     fontsize=15,
    #     color=colors[i],
    #     fontweight="bold",
    # )

# Customize the plot
plt.xlabel("Percentile Threshold")
plt.ylabel("Dice Scores")
plt.title("Dice Scores vs Threshold")
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.savefig("figures/threshold_dice.svg", bbox_inches="tight")
plt.savefig("figures/threshold_dice.png", bbox_inches="tight", dpi=300)
