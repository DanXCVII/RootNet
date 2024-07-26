import sys

sys.path.append("../utils")

from MRI_operations import MRIoperations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# Define the path to your .otf font file
font_path = "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf"
# Create a FontProperties object with the .otf font
font_properties = FontProperties(fname=font_path)

# Set the font globally for all plots
plt.rcParams["font.family"] = font_properties.get_name()
plt.rcParams.update({"font.size": 16})

loam_mris = [
    "/Users/daniel/Desktop/FZJ/Echte Daten/tobias_mri/III_Soil_3D_DAP15_scan_2_256x256x199.raw",
    "/Users/daniel/Desktop/FZJ/Echte Daten/tobias_mri/IV_Soil_1W_DAP9_scan_3_256x256x136.raw",
    "/Users/daniel/Desktop/FZJ/Echte Daten/tobias_mri/III_Soil_1W_DAP14_scan_1_256x256x186.raw",
    "/Users/daniel/Desktop/FZJ/Echte Daten/tobias_mri/IV_Soil_3D_DAP8_scan_4_256x256x193.raw",
]
sand_mris = [
    "/Users/daniel/Desktop/FZJ/Echte Daten/tobias_mri/III_Sand_3D_DAP14_scan_2_256x256x191.raw",
    "/Users/daniel/Desktop/FZJ/Echte Daten/tobias_mri/III_Sand_1W_DAP14_scan_1_256x256x131.raw",
    "/Users/daniel/Desktop/FZJ/Echte Daten/tobias_mri/IV_Sand_3D_DAP8_scan_3_256x256x192.raw",
]


def boxplot_data(data, soil_type, figsize, percentile):
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(data, widths=0.5)

    # Customizing the plot
    ax.set_title(f"Top {round(100 - percentile,2)}% of MRI loam scans")
    ax.set_xlabel("MRI data")
    ax.set_ylabel("Signal Intensity")

    # Adding the mean values to the plot
    means = []
    xticklabels = []
    for i, d in enumerate(data):
        xticklabels.append(f"{soil_type} MRI {i + 1}")
        means.append(np.mean(d))
    ax.set_xticklabels(xticklabels, rotation=10)

    for i, mean in enumerate(means):
        ax.text(
            i + 1,
            mean,
            f"{mean:.1f}",
            horizontalalignment="center",
            color="orange",
            weight="semibold",
        )

    plt.tight_layout()
    plt.savefig(f"figures/boxplot_{soil_type}.png", dpi=300)
    plt.savefig(f"figures/boxplot_{soil_type}.svg")


def create_threshold_array(mris, percentile):
    data_arr = []

    mri_ops = MRIoperations()

    for loam_mri in mris:
        _, loam_mri_data = mri_ops.load_mri(loam_mri)
        # select the top 0.04% of the MRI data
        loam_flat = loam_mri_data.flatten()
        percentile = np.percentile(loam_flat, 99.96)
        loam_mri_data = loam_mri_data[loam_mri_data >= percentile]

        data_arr.append(loam_mri_data)

    return data_arr


percentile = 99.96
boxplot_data(
    create_threshold_array(loam_mris, percentile), "loam", (7.2, 5), percentile
)
boxplot_data(create_threshold_array(sand_mris, percentile), "sand", (6, 5), percentile)
