import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors


# Define the path to your .otf font file
font_path = "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf"
# Create a FontProperties object with the .otf font
font_properties = FontProperties(fname=font_path)

# Set the font globally for all plots
plt.rcParams["font.family"] = font_properties.get_name()
plt.rcParams.update({"font.size": 12})


def read_csv(file_path):
    return pd.read_csv(file_path)


def format_lr(lr):
    if lr == 0.0004:
        return "4e-4"
    elif lr == 0.0008:
        return "8e-4"
    else:
        return f"{lr:.0e}"


def process_experiment(exp_dir, lr_vals):
    dice = read_csv(os.path.join(exp_dir, "Test_root_f1.csv"))
    precision = read_csv(os.path.join(exp_dir, "Test_root_precision.csv"))
    recall = read_csv(os.path.join(exp_dir, "Test_root_recall.csv"))

    avg_dice = dice["value"].mean()
    avg_precision = precision["value"].mean()
    avg_recall = recall["value"].mean()

    # Extract patch size and feature size from directory name
    # if exp_dir doesn't contain thresh
    if "thresh" not in exp_dir:
        dir_name = os.path.basename(exp_dir)
        patch_size = re.search(r"patch_size_(\d+)", dir_name)
        feat_size = re.search(r"feat_(\d+)", dir_name)
        lr = re.search(r"lr_(\d+\.\d+)", dir_name)

        if lr_vals[(int(patch_size.group(1)), int(feat_size.group(1)))] == format_lr(
            float(lr.group(1))
        ):
            best = True
        else:
            best = False
        print(best)

        config = (
            f"P{patch_size.group(1)}_F{feat_size.group(1)}"
            if patch_size and feat_size
            else dir_name
        )
    else:
        config = "thresh_background_false"
        best = True

    return config, avg_dice, avg_precision, avg_recall, best


def collect_data(root_dir, lr_vals, include_thresh=True):
    results = []
    for subdir, _, _ in os.walk(root_dir):
        if "Test_dice.csv" in os.listdir(subdir):
            config, dice, precision, recall, best = process_experiment(subdir, lr_vals)
            if best:
                results.append(
                    {
                        "config": config,
                        "dice": dice,
                        "precision": precision,
                        "recall": recall,
                    }
                )
    if include_thresh:
        config, dice, precision, recall, best = process_experiment(
            "./data/threshold_background_true", lr_vals
        )
        results.append(
            {
                "config": config,
                "dice": dice,
                "precision": precision,
                "recall": recall,
            }
        )
    return pd.DataFrame(results)


def plot_results(df):
    plt.figure(figsize=(9.3, 3.8))

    scatter = plt.scatter(
        df["precision"],
        df["recall"],
        c=df["dice"],
        s=df["dice"] * 500,
        cmap="viridis",
        alpha=0.7,
    )

    plt.title("Test - Avg. Dice Scores for Avg. Precision and Recall")
    plt.xlabel("Average Precision")
    plt.ylabel("Average Recall")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Dice Score")

    for i, row in df.iterrows():
        plt.annotate(
            f"{row['config']}-{row['dice']:.3f}",
            (row["precision"], row["recall"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.7,
        )

    plt.tight_layout()
    plt.savefig("figures/precision_recall_dice.png", dpi=300)
    plt.savefig("figures/precision_recall_dice.svg", dpi=300)


with open("./data/lr_vals.txt", "r") as f:
    lr_vals = eval(f.read())

# Main execution
root_directory = "./data/hyperparam_patch_feat/"
results_df = collect_data(root_directory, lr_vals, include_thresh=True)
plot_results(results_df)
