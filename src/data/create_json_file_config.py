import os
import glob
import re
import json
from datetime import datetime


class CreateJsonFileConfig:
    def __init__(self, relative_data_path):
        """
        Searches for all files used for the training and testing of the DNN in the data directory and creates a config file.

        Args:
        - relative_data_path (str): Relative path to the data directory.
        """
        file_dir = os.path.dirname(__file__)  # Get the directory of the current script
        self.data_dir = os.path.abspath(os.path.join(file_dir, relative_data_path))
        self.training_dir = os.path.join(self.data_dir, "generated/training")
        self.validation_dir = os.path.join(self.data_dir, "generated/validation")
        self.test_dir = os.path.join(self.data_dir, "test")
        self.real_data = os.path.join(self.data_dir, "real")
        print("initiated CreateJsonFileConfig")

    def _find_files(self, root_path, extension=".nii.gz") -> list:
        """
        Finds all files with a given extension in a directory.

        Args:
        - root_path (str): Path to the root directory, where the search should start.
        - extension (str): Extension of the files to be found.

        Returns:
        - file_list (list): List of filenames.
        """
        file_list = []

        # Walk through directory recursively
        for dirpath, _, _ in os.walk(root_path):
            file_list.extend(glob.glob(os.path.join(dirpath, f"*{extension}")))

        return file_list

    def _create_input_label_dict(self, file_list, key, input_label_dict=None) -> dict:
        """
        Creates a dictionary with the input and label filename for the config file.

        Args:
        - file_list (list): List of filenames.
        - key (str): Key for the dictionary.
        - input_label_dict (dict): Dictionary to add the entries to.

        Returns:
        - input_label_dict (dict): Dictionary with the input and label filenames.
        """
        if input_label_dict is None:
            input_label_dict = {key: []}

        input_file_list = [s for s in file_list if "/label_" not in s]

        for file_path in input_file_list:
            # Get the filename with it's two subdirectories where it's in
            # Split the path into its components
            path_parts = file_path.split(os.sep)

            # Add the label prefix which identifies the label file
            path_parts[-1] = f"label_{path_parts[-1]}"

            # Combine them back into a string
            label_filepath = os.path.join(*path_parts)

            # Remove the extension
            label_path_filtered = re.sub(
                r"_res_\d+x\d+x\d+\.nii\.gz$",
                "",
                label_filepath,
            )

            # Find the label file
            label_files = [s for s in file_list if label_path_filtered in s]

            if len(label_files) == 1:
                # remove the data path of teh file_path and label_filepath
                # TODO: add back again
                # file_path = file_path.replace(self.data_dir, "")
                # label_filepath = label_filepath.replace(self.data_dir, "")

                input_label_dict[key].append(
                    {"image": file_path, "label": label_files[0]}
                )

        return input_label_dict

    def create_config(self):
        """
        Creates a config file for the deep learning model for which the necessary required config is found in the
        constructor.

        Returns:
        - filename (str): Filename of the config file.
        """
        training_files = self._find_files(
            self.training_dir,
            extension=".nii.gz",
        )
        validation_files = self._find_files(
            self.validation_dir,
            extension=".nii.gz",
        )
        test_files = self._find_files(
            self.test_dir,
            extension=".nii.gz",
        )
        test_files_real = self._find_files(
            self.real_data,
            extension=".nii.gz",
        )

        config_dict = {
            "description": "btcv yucheng",
            "labels": {
                "0": "background",
                "1": "root",
            },
            "modality": {"0": "CT"},
            "name": "btcv",
            "reference": "Vanderbilt University",
            "release": f"1.0 {datetime.now().strftime('%d/%m/%Y')}",
            "tensorImageSize": "3D",
        }

        train_config = self._create_input_label_dict(training_files, "training")
        val_config = self._create_input_label_dict(validation_files, "validation")
        test_config = self._create_input_label_dict(test_files, "test")

        config_dict.update(train_config)
        config_dict.update(val_config)
        config_dict.update(test_config)

        config_dict["numTraining"] = len(config_dict["training"])
        config_dict["numTest"] = len(config_dict["test"])

        filename = "data_config.json"
        with open(filename, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)

        return config_dict


# Example execution
config = CreateJsonFileConfig("../../data")
config.create_config()
