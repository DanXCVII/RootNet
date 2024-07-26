import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Generate example data
np.random.seed(0)  # for reproducibility
patch_sizes = [32, 64, 96, 128]
feature_sizes = [12, 24, 36, 48]

data = []
for patch in patch_sizes:
    for feature in feature_sizes:
        precision = np.random.uniform(0.7, 0.9)
        recall = np.random.uniform(0.7, 0.9)
        dice = 2 * (precision * recall) / (precision + recall)
        data.append([patch, feature, dice])

# Create DataFrame
df = pd.DataFrame(data, columns=["Patch Size", "Feature Size", "Dice"])

# Create the pairplot
plt.figure(figsize=(15, 15))
sns.pairplot(
    df,
    vars=["Patch Size", "Feature Size", "Dice"],
    diag_kind="kde",
    plot_kws={"alpha": 0.6},
)

plt.suptitle("Neural Network Performance Metrics Pairplot", y=1.02)

# Save the figure
plt.savefig(
    "figures/neural_network_performance_pairplot.png", dpi=300, bbox_inches="tight"
)
plt.close()

# Print the first few rows of the DataFrame for reference
print(df.head())
