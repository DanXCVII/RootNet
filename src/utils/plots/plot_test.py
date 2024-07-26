import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data (replace with your actual data)
samples = [
    "Sample 1",
    "Sample 2",
    "Sample 3",
    "Sample 4",
    "Sample 5",
    "Sample 6",
    "Sample 7",
]
configurations = [
    "Config 1",
    "Config 2",
    "Config 3",
    "Config 4",
    "Config 5",
    "Config 6",
]
dice_scores = np.random.rand(7, 6)  # 7x6 matrix of random Dice scores

plt.figure(figsize=(12, 8))
sns.heatmap(
    dice_scores,
    annot=True,
    cmap="YlGnBu",
    xticklabels=configurations,
    yticklabels=samples,
)
plt.title("Neural Network Performance (Dice Score) Heatmap")
plt.xlabel("Configurations")
plt.ylabel("Samples")
plt.tight_layout()
plt.savefig("performance_heatmap.png", dpi=300)
