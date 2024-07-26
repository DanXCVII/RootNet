import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Function to read CSV files
def read_csv(filename):
    return np.loadtxt(filename, delimiter=",")


# Read the data
x_values = []
dice_scores = []

for i in range(7):
    x_values.append(read_csv(f"data/x_values_{i}.csv"))
    dice_scores.append(read_csv(f"data/dice_scores_{i}.csv"))

# Flatten the lists
x_values = np.concatenate(x_values)
dice_scores = np.concatenate(dice_scores)


print(x_values)

# Create the plot
plt.figure(figsize=(12, 8))

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

# Plot each line
for i in range(7):
    start = i * 30
    end = (i + 1) * 30
    plt.plot(
        x_values[start:end],
        dice_scores[start:end],
        label=labels[i],
        color=colors[i],
        marker="o",
    )

    # Find and annotate the max value point
    max_index = np.argmax(dice_scores[start:end])
    max_x = x_values[start + max_index]
    max_y = dice_scores[start + max_index]
    plt.annotate(
        f"Max: {max_y:.4f}",
        xy=(max_x, max_y),
        xytext=(5, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

# Customize the plot
plt.xlabel("X Values")
plt.ylabel("Dice Scores")
plt.title("Dice Scores vs X Values")
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.savefig("figures/dice_scores_vs_x_values.png")
