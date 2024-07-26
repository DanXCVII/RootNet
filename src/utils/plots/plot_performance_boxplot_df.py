import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter

# Define the path to your .otf font file
font_path = "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf"
# Create a FontProperties object with the .otf font
font_properties = FontProperties(fname=font_path)

# Set the font globally for all plots
plt.rcParams["font.family"] = font_properties.get_name()
plt.rcParams.update({"font.size": 12})


def load_data(files, architecture):
    data = []
    learning_rates = [0.0001, 0.0002, 0.0004, 0.0008, 0.0016]

    for lr, file in zip(learning_rates, files):
        df = pd.read_csv(file)
        for i in range(len(df)):
            soil_type = "Loam" if i < 4 else "Sand"
            data.append(
                {
                    "Learning Rate": lr,
                    "Architecture": architecture,
                    "Test Sample": i + 1,
                    "Soil Type": soil_type,
                    "Performance": df.loc[i, "value"],
                }
            )

    return data


# Load the uploaded CSV files
deconv_files = [
    "./data/superres_new/simple_deconv/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0001-upsample_end_True/Test_dice.csv",
    "./data/superres_new/simple_deconv/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0002-upsample_end_True/Test_dice.csv",
    "./data/superres_new/simple_deconv/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0004-upsample_end_True/Test_dice.csv",
    "./data/superres_new/simple_deconv/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0008-upsample_end_True/Test_dice.csv",
    "./data/superres_new/simple_deconv/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0016-upsample_end_True/Test_dice.csv",
]

# replace the work deconv with extend_unet for all the deconv_files
extend_unet_files = [
    "./data/superres_new/extend_unet/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0001-upsample_end_False/Test_dice.csv",
    "./data/superres_new/extend_unet/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0002-upsample_end_False/Test_dice.csv",
    "./data/superres_new/extend_unet/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0004-upsample_end_False/Test_dice.csv",
    "./data/superres_new/extend_unet/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0008-upsample_end_False/Test_dice.csv",
    "./data/superres_new/extend_unet/data_root_scale_weight_1.0-1.0_DICE_SOFTMAX_UPSCALESWINUNETR-patch_size_96-feat_36-upscale_True-out_channels_2-lr_0.0016-upsample_end_False/Test_dice.csv",
]

# Load data for both architectures
deconv_data = load_data(deconv_files, "Architecture 1")
extend_unet_data = load_data(extend_unet_files, "Architecture 2")

# Combine data from both architectures
all_data = deconv_data + extend_unet_data

# Create the final dataframe
df_combined = pd.DataFrame(all_data)

# Print the first few rows of the dataframe to verify
print(df_combined)
