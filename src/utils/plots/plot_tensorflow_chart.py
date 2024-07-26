import numpy as np
import matplotlib.pyplot as plt

# Generate data for the first line
x = np.arange(250)
y = np.linspace(0, 0.4, 250) + np.random.normal(0, 0.1, 250)
y = np.maximum(y, 0)  # Ensure non-negative values

# Add some spikes
spike_indices = [50, 100, 150, 200]
for idx in spike_indices:
    y[idx] += 0.2

# Apply smoothing similar to TensorFlow's exponential moving average
smoothing_factor = 0.5
smoothed_y = np.zeros_like(y)
smoothed_y[0] = y[0]
for i in range(1, len(y)):
    smoothed_y[i] = smoothing_factor * y[i] + (1 - smoothing_factor) * smoothed_y[i - 1]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, color="#20B2AA", label="Original")
plt.plot(x, smoothed_y, color="#FF00FF", label="Smoothed")

# Customize the plot
plt.grid(True)
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Original vs Smoothed Data")
plt.legend()

# Set background color to dark
plt.gca().set_facecolor("#1C1C1C")
plt.gcf().set_facecolor("#1C1C1C")

# Adjust text color for better visibility
plt.title("Original vs Smoothed Data", color="white")
plt.xlabel("Time", color="white")
plt.ylabel("Value", color="white")
plt.tick_params(colors="white")
plt.legend(facecolor="#1C1C1C", edgecolor="white", labelcolor="white")

plt.savefig("figures/smoothing_example.png", dpi=300)
