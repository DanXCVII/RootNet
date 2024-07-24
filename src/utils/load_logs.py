import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os
import re

"""
Description:   This script was created to extract all scalar data from tensorboard logs and save it to CSV files for
                further analysis or plotting purposes. 
Usage: Uncomment the desired extraction method in the extract_tensorboard_data function.
       E.g. "uncomment for supperresolution" extracts the files in either a folder named simple_deconv or extend_unet depending 
       on the upsample_end_ boolean within the run_dir.
Example: python load_logs.py
"""

def extract_tensorboard_data(log_dir, output_base_dir):
    # Walk through all subdirectories
    for root, dirs, files in os.walk(log_dir):
        for run_dir in dirs:
            if run_dir.startswith("data_root_scale"):
                run_path = os.path.join(root, run_dir)

                # check if run_dir contains superres_new
                if "superres" not in run_path:
                    continue
                
                # uncomment for superresolution
                match = re.search(r'upsample_end_(True|False)', run_dir)

                if match:
                    upsample_end = match.group(1)
                    # Convert string to boolean
                    upsample_end_bool = upsample_end.lower() == 'true'
                    print(upsample_end_bool)  # This will print: False
                else:
                    print("upsample_end_ not found in the path")

                if upsample_end_bool:
                    output_dir = "simple_deconv"
                else:
                    output_dir = "extend_unet"

                # # uncomment for class weight
                # match = re.search(r'data_root_scale_weight_(\d+\.\d+)-(\d+\.\d+)', run_dir)
                # if match:
                #     weight1, weight2 = match.groups()
                
                # remove the decimals if they are 0
                # weight1 = weight1[:-2] if weight1.endswith('.0') else weight1
                # weight2 = weight2[:-2] if weight2.endswith('.0') else weight2

                # output_dir = f"weight_{weight1}-{weight2}"

                # # uncomment for batch size
                # result = run_dir.split("-batch_")[-1]

                # output_dir = f"batch_{result}"
                
                # uncomment for patch size and feature size
                # # Extract patch size from directory name
                # # match = re.search(r'patch_size_(\d+)', dir)
                # patch_size = re.search(r'patch_size_(\d+)', run_dir)
                # patch_size = int(patch_size.group(1)) if patch_size else None

                # # Extract feat
                # feat = re.search(r'feat_(\d+)', run_dir)
                # feat = int(feat.group(1)) if feat else None

                # if patch_size is None or feat is None:
                #     # print(f"Couldn't extract patch size from {dir}, skipping...")
                #     print(f"Couldn't extract patch size or feat from {run_dir}, skipping...")
                #     continue

                # output_dir = f"{patch_size}"

                

                # Create output directory
                output_dir = os.path.join(output_base_dir, output_dir, run_dir)
                os.makedirs(output_dir, exist_ok=True)

                # Load the event data
                subdirs = [d for d in os.listdir(run_path)]
                version_dirs = []
                for subdir in subdirs:
                    match = re.match(r'version_(\d+)', subdir)
                    if match:
                        version_num = int(match.group(1))
                        version_dirs.append((subdir, version_num))
                highest_version = max(version_dirs, key=lambda x: x[1])

                run_path = os.path.join(run_path, highest_version[0])

                ea = event_accumulator.EventAccumulator(run_path)
                ea.Reload()

                # Print available scalar tags for this run
                print(f"Processing {run_path}")
                print("Available scalar tags:")
                for tag in ea.Tags()['scalars']:
                    print(tag)

                # Process each scalar tag
                for tag in ea.Tags()['scalars']:
                    # Get the metric data
                    metric_data = ea.Scalars(tag)

                    # Convert to pandas DataFrame
                    df = pd.DataFrame(metric_data, columns=['wall_time', 'step', 'value'])
                    
                    # Save to CSV
                    output_file = os.path.join(output_dir, f"{tag.replace('/', '_')}.csv")
                    df.to_csv(output_file, index=False)
                    print(f"Data for {tag} saved to {output_file}")

# Example usage
log_dir = '/p/project/visforai/weissen1/RootNet/src/training/tb_logs'
output_base_dir = './logs/'
extract_tensorboard_data(log_dir, output_base_dir)


