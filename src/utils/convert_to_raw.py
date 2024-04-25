import tkinter as tk
from tkinterdnd2 import DND_FILES, TkinterDnD
from MRI_operations import MRIoperations
import os
import numpy as np


def process_file_path(file_path):
    """
    Function to process the file path.
    Replace the content of this function with your desired file processing logic.
    """
    mri_ops = MRIoperations()

    # if the file_path doesn't end with .raw or .nii.gz, throw an error
    if not file_path.endswith(".raw") and not file_path.endswith(".nii.gz"):
        raise ValueError("The file must be a .raw or .nii.gz file.")

    _, my_mri = mri_ops.load_mri(file_path)

    # extract the directory, where the file is stored
    directory = os.path.dirname(file_path)
    # extract the file name
    file_name = os.path.basename(file_path)
    # extract the file name without the extension
    file_name_without_extension = file_name.split(".", 1)[0]
    # extract the file extension
    file_extension = os.path.splitext(file_name)[1]

    if file_extension == ".gz":
        print("Saving as .raw")
        my_mri = my_mri.swapaxes(0, 2)
        mri_ops.save_mri(f"{directory}/{file_name_without_extension}.raw", my_mri)
    else:
        print("Saving as .nii.gz")
        my_mri = my_mri.swapaxes(0, 2)
        mri_ops.save_mri(f"{directory}/{file_name_without_extension}.nii.gz", my_mri)


def on_drop(event):
    file_path = event.data
    if file_path:
        process_file_path(file_path)


# Create a TkinterDnD window
root = TkinterDnD.Tk()
root.title("Drag and Drop File Here")
root.geometry("400x200")

# Create a label to provide feedback to the user
label = tk.Label(root, text="Drag and drop a file here", pady=20, padx=20)
label.pack(expand=True, fill=tk.BOTH)

# Enable drag and drop on the label
label.drop_target_register(DND_FILES)
label.dnd_bind("<<Drop>>", on_drop)

# Run the application
root.mainloop()

# TODO: Fix the problem with the file extension
