import os
import shutil

"""
Description:    This script was created to delete all checkpoints, for which no tensorboard exists. 
                So when tiding up the logs, it simply deletes all corresponding checkpoints.
Usage: Adjust the paths to the checkpoints and tensorboards and run the script.
Example: python delete_checkpoints.py
"""

def delete_unmatched_directories(dir1, dir2):
    """
    Delete directories in dir2 that are not present in dir1. Useful, if tensorboard logs are 
    deleted and the corresponding checkpoints should be deleted as well.

    Args:
    - dir1: Path to the first directory.
    - dir2: Path to the second directory.
    """
    # List all entries in both directories
    entries_dir1 = os.listdir(dir1)
    entries_dir2 = os.listdir(dir2)

    # Filter to keep only directories
    dirs_dir1 = {entry for entry in entries_dir1 if os.path.isdir(os.path.join(dir1, entry))}
    dirs_dir2 = {entry for entry in entries_dir2 if os.path.isdir(os.path.join(dir2, entry))}

    # Delete directories in dir2 not present in dir1
    for dir in dirs_dir2:
        if dir not in dirs_dir1:
            shutil.rmtree(os.path.join(dir2, dir))
            print(f"Deleted directory: {dir}")

# Example usage
dir1 = '../training/tb_logs'
dir2 = '../../runs'
delete_unmatched_directories(dir1, dir2)