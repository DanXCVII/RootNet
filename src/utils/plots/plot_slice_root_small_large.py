import sys

sys.path.append("../utils")

from MRI_operations import MRIoperations
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MaxNLocator

mri_ops = MRIoperations()

# Configure matplotlib to use LaTeX and Latin Modern Roman font
rc("font", **{"family": "serif", "serif": ["Latin Modern Roman"]})
rc("text", usetex=True)
plt.rcParams.update({"font.size": 15})


def plot_slice(mri, x_range, y_range, z, filename, swapped=True):
    # select the slice
    if swapped:
        my_slice = mri[x_range[0] : x_range[1], y_range[0] : y_range[1], z]
        range_x = y_range[1] - y_range[0]
        range_y = x_range[1] - x_range[0]
    else:
        my_slice = mri[z, y_range[0] : y_range[1], x_range[0] : x_range[1]]
        range_x = x_range[1] - x_range[0]
        range_y = y_range[1] - y_range[0]
    # normalize the slice
    my_slice = (my_slice - my_slice.min()) / (my_slice.max() - my_slice.min())

    # plot the slice
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    cax = ax.imshow(my_slice, cmap="gray")
    # add legend
    fig.colorbar(cax, ax=ax)
    # add ticks and labels
    if my_slice.shape[1] < 10:
        stepsize = 1
    else:
        stepsize = 2

    number_of_ticks = max(range_x, range_y) // stepsize

    ax.xaxis.set_major_locator(MaxNLocator(number_of_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(number_of_ticks))

    ax.set_xlabel("voxel count x-axis")
    ax.set_ylabel("voxel count y-axis")

    plt.tight_layout()
    plt.savefig(f"figures/{filename}")
    plt.close(fig)


_, my_real_mri = mri_ops.load_mri("./III_Soil_1W_DAP14_scan_1_256x256x186.raw")
_, my_synthetic_mri = mri_ops.load_mri(
    "./mris/my_Bench_lupin_day_6_res_237x237x171.nii.gz"
)

filetype = "svg"

plot_slice(
    my_real_mri,
    [65, 72],
    [208, 215],
    94,
    f"real_mri_slice_small.{filetype}",
    swapped=False,
)
plot_slice(
    my_synthetic_mri,
    [39, 47],
    [156, 164],
    94,
    f"synthetic_mri_slice_small.{filetype}",
)

plot_slice(
    my_real_mri,
    [164, 180],
    [104, 120],
    179,
    f"real_mri_slice_big.{filetype}",
    swapped=False,
)
plot_slice(
    my_synthetic_mri,
    [53, 68],
    [151, 167],
    113,
    f"synthetic_mri_slice_big.{filetype}",
)
