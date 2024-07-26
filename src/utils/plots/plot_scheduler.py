import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.ticker import FuncFormatter

rc("font", **{"family": "serif", "serif": ["Latin Modern Roman"]})
plt.rcParams.update({"font.size": 10})

# Parameters
eta_max = 4e-4
eta_min = 1e-5
T_0 = 6
T_mult = 1
gamma = 0.9
num_cycles = 3
warmup_epochs = 5

# Generate learning rate schedule
lr = []
epochs = []
T_cur = 0
current_cycle = 0
total_epochs = warmup_epochs + sum(T_0 * T_mult**i for i in range(num_cycles))

for epoch in range(total_epochs):
    if epoch < warmup_epochs:
        # Linear warmup
        lr_t = (eta_max) * ((epoch + 1) / (warmup_epochs + 1))
    else:
        # Cosine Annealing Warm Restarts
        adjusted_epoch = epoch - warmup_epochs
        if T_cur >= T_0 * T_mult**current_cycle:
            T_cur = 0
            current_cycle += 1

        T_i = T_0 * T_mult**current_cycle
        max_lr = eta_max * gamma**current_cycle
        lr_t = eta_min + 0.5 * (max_lr - eta_min) * (1 + np.cos(np.pi * T_cur / T_i))

        T_cur += 1

    lr.append(lr_t)
    epochs.append(epoch)

# Create a plot with fig and ax
fig, ax = plt.subplots(figsize=(6, 2.5))

# Plot the overall learning rate line
ax.plot(epochs, lr, label="Learning Rate Schedule", color="green")

# Plot warmup phase dots
ax.scatter(
    epochs[: warmup_epochs + 1],
    lr[: warmup_epochs + 1],
    label="Warmup Phase",
    color="orange",
)

# Plot cosine annealing phase dots
ax.scatter(
    epochs[warmup_epochs:],
    lr[warmup_epochs:],
    label="Cosine Annealing Phase",
    color="blue",
)

# Set the labels, title, and limits
ax.set_ylim(0, 0.0005)
ax.set_xlim(0, 22)
ax.set_xlabel("Epochs")
ax.set_ylabel("Learning Rate")
ax.set_title(
    "Cosine Annealing Learning Rate Scheduler \n(Linear Warmup and Warm Restarts)"
)
ax.legend(fontsize=8)
ax.grid(True)

# Tight layout and save the plot
fig.tight_layout()
fig.savefig("figures/scheduler_lr_with_legend.pdf")
fig.savefig("figures/scheduler_lr_with_legend.png")
plt.show()

############################################################################################################

# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Example grid search results (random data for illustration purposes)
# channels = [36, 24, 12]
# patch_sizes = [32, 64, 96, 128]

# # Create a DataFrame with all combinations and two performance metrics
# data = []
# for ch in channels:
#     for ps in patch_sizes:
#         accuracy = np.random.rand()  # Random accuracy for illustration
#         loss = np.random.rand()  # Random loss for illustration
#         data.append([ch, ps, accuracy, loss])

# # Convert to DataFrame
# df = pd.DataFrame(data, columns=["channels", "patch_size", "accuracy", "loss"])

# # Pivot the DataFrame for heatmap using accuracy
# heatmap_data_accuracy = df.pivot(
#     index="patch_size", columns="channels", values="accuracy"
# )


# # Create a custom function to annotate each cell with multiple performance values
# def annotate_heatmap(data, df, ax, fmt="{:.2f}"):
#     for text in ax.texts:
#         x, y = text.get_position()
#         x = int(round(x))
#         y = int(round(y))
#         # Get the values based on the current cell's position in the heatmap
#         patch_size = data.index[y]
#         channel = data.columns[x]
#         accuracy = df[(df["patch_size"] == patch_size) & (df["channels"] == channel)][
#             "accuracy"
#         ].values[0]
#         loss = df[(df["patch_size"] == patch_size) & (df["channels"] == channel)][
#             "loss"
#         ].values[0]
#         text.set_text(f"Acc: {accuracy:.2f}\nLoss: {loss:.2f}")


# # Plot the heatmap for accuracy
# plt.figure(figsize=(12, 8))
# ax = sns.heatmap(
#     heatmap_data_accuracy,
#     annot=True,
#     fmt=".2f",
#     cmap="YlGnBu",
#     cbar_kws={"label": "Accuracy"},
# )

# # Annotate the heatmap with multiple metrics
# annotate_heatmap(heatmap_data_accuracy, df, ax)

# plt.title("Heatmap: Patch Size vs. Number of Channels with Accuracy and Loss")
# plt.xlabel("Number of Channels")
# plt.ylabel("Patch Size")
# plt.show()

############################################################################################################

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import rc

# # Configure matplotlib to use LaTeX and Latin Modern Roman font
# rc("font", **{"family": "serif", "serif": ["Latin Modern Roman"]})
# rc("text", usetex=True)

# # Generate an 8x8 image with values ranging from 0 to 1
# image = np.random.rand(8, 8)

# # Normalize the image so that the minimum value is 0 and the maximum value is 1
# image = (image - image.min()) / (image.max() - image.min())

# # Create the plot
# fig, ax = plt.subplots()
# fig.set_size_inches(5, 4)


# cax = ax.imshow(image, cmap="gray")
# cbar = fig.colorbar(cax)

# # Set the title, x-label, y-label, and other text elements to use the custom font with specified font size
# ax.set_xlabel("X-axis", fontsize=11)
# ax.set_ylabel("Y-axis", fontsize=11)

# # Add a legend

# # Set the colorbar label to use the custom font with specified font size
# cbar.set_label("Colorbar Label", fontsize=11)

# # Adjust tick parameters to change the font size of tick labels
# ax.tick_params(axis="both", which="major", labelsize=12)
# cbar.ax.tick_params(labelsize=12)

# plt.savefig("plot_sch_img.svg")
