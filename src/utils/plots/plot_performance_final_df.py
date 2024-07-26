import pandas as pd
import os

# Define the root directory and configuration names
data_root = "./data/final_performance"
config1 = "data_root_scale_weight_0.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0004-upsample_end_True-batch_7-1"
config2 = "threshold_background_true"
config3 = "data_gmm_weight_0.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0004-upsample_end_True-batch_7-1"


# Function to read and process CSV file
def read_dice_csv(file_path):
    df = pd.read_csv(file_path)
    df["step"] = df["step"].astype(int)
    return df.set_index("step")["value"]


def print_performance(files, metric):
    # Read scores for all configurations
    scores_config1 = read_dice_csv(os.path.join(data_root, config1, files[0]))
    scores_config2 = read_dice_csv(os.path.join(data_root, config2, files[1]))
    scores_config3 = read_dice_csv(os.path.join(data_root, config3, files[2]))

    # Print the performance
    print(f"\nMetric: {metric}")
    print("FS NN values:", ", ".join(f"{v:.4f}" for v in scores_config1.values))
    print(
        "Best threshold values:", ", ".join(f"{v:.4f}" for v in scores_config2.values)
    )
    print("GMM Noise values:", ", ".join(f"{v:.4f}" for v in scores_config3.values))

    # Print average scores
    # print(f"\nAverage scores for {metric}:")
    # print(f"FS NN: {scores_config1.mean():.4f}")
    # print(f"Best threshold: {scores_config2.mean():.4f}")
    # print(f"GMM Noise: {scores_config3.mean():.4f}")


# Call the function for each metric
print_performance(
    [
        "Test_dice.csv",
        "Test_root_f1.csv",
        "Test_dice.csv",
    ],
    "Dice",
)

print_performance(
    [
        "Test_root_precision.csv",
        "Test_root_precision.csv",
        "Test_root_precision.csv",
    ],
    "Precision",
)

print_performance(
    [
        "Test_root_recall.csv",
        "Test_root_recall.csv",
        "Test_root_recall.csv",
    ],
    "Recall",
)

print_performance(
    [
        "Test_surface_distance.csv",
        "Test_surface_distance.csv",
        "Test_surface_distance.csv",
    ],
    "Surface Distance",
)
