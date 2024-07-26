import sys

sys.path.append("../utils")

from MRI_operations import MRIoperations
from fourier_synthesis import FourierSynthesis
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# Define the path to your .otf font file
font_path = "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf"
# Create a FontProperties object with the .otf font
font_properties = FontProperties(fname=font_path)

# Set the font globally for all plots
plt.rcParams["font.family"] = font_properties.get_name()
plt.rcParams.update({"font.size": 12})

mri_ops = MRIoperations()

_, loam_mri = mri_ops.load_mri(
    "/Users/daniel/Desktop/FZJ/CPlantBox/DUMUX/CPlantBox/tutorial/examples_segmentation/RootNet/src/plots/mris/scan_loam_43_138x139x51.raw"
)
_, sand_mri = mri_ops.load_mri(
    "/Users/daniel/Desktop/FZJ/CPlantBox/DUMUX/CPlantBox/tutorial/examples_segmentation/RootNet/src/plots/mris/scan_sand_38_142x136x49.raw"
)


def plot_histogram(mri, title, path):
    noise_gen = FourierSynthesis(mri)
    noise_volumes = noise_gen.generate_new_texture(mri, 1)

    # Generating some sample data
    real_soil_signal = mri.flatten()
    # get the 99.96% of the data
    # real_soil_signal = real_soil_signal[
    #     real_soil_signal < np.percentile(real_soil_signal, 99.0)
    # ]
    fourier_synthesis_signal = noise_volumes[0].flatten()
    print("real_soil_signal", real_soil_signal.shape)
    print("fourier_synthesis_signal", fourier_synthesis_signal.shape)

    # Define the bin edges based on the first histogram
    bin_count = 60
    hist, bin_edges = np.histogram(real_soil_signal, bins=bin_count)

    # Creating the histogram with the same bin edges
    plt.figure(figsize=(5.2, 4))
    plt.hist(real_soil_signal, bins=bin_edges, alpha=0.5, label="real soil signal")
    plt.hist(
        fourier_synthesis_signal,
        bins=bin_edges,
        alpha=0.5,
        label="fourier synthesis signal",
    )

    # Adding labels and legend
    plt.xlabel("signal strength")
    plt.ylabel("number voxels")
    plt.legend(loc="upper right", fontsize=10)
    plt.yscale("log")
    plt.title(title)
    plt.tight_layout()

    # Display the plot
    plt.savefig(path, dpi=300)
    plt.savefig(path.replace(".png", ".svg"))
    # clear plot
    plt.clf()


# Creating the histogram


plot_histogram(
    loam_mri,
    "Frequency distribution of signal strength\nReal soil vs Fourier synthesis - loam",
    "figures/histogram_loam.png",
)
plot_histogram(
    sand_mri,
    "Frequency distribution of signal strength\nReal soil vs Fourier synthesis - sand",
    "figures/histogram_sand.png",
)


# Display the plot
