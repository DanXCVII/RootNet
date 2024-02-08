from monai.transforms import Randomizable
import numpy as np
import torch


class RandCropByPosNegLabeldWithResAdjust(Randomizable):
    def __init__(
        self, image_key, label_key, spatial_size, pos, neg, num_samples, image_threshold
    ):
        self.image_key = image_key
        self.label_key = label_key
        self.spatial_size = spatial_size
        self.pos = pos
        self.neg = neg
        self.num_samples = num_samples
        self.image_threshold = image_threshold

    def randomize(self, data):
        # Assuming data[image_key] is a PyTorch tensor with shape [batch, channel, depth, height, width]
        img_data = data[self.image_key]

        # Image dimensions without batch and channel
        depth, height, width = img_data.shape[-3:]

        # Choose positive or negative based on pos and neg values
        if torch.rand(1).item() < self.pos / (self.pos + self.neg):
            # Positive patch
            # Find all foreground voxel coordinates for the current batch and channel
            fg_coords = torch.nonzero(img_data[0] > self.image_threshold)

            # Randomly select one foreground voxel
            center_coord = fg_coords[torch.randint(0, len(fg_coords), (1,))][0]

        else:
            # Negative patch
            # Find all background voxel coordinates for the current batch and channel
            bg_coords = torch.nonzero(img_data[0] <= self.image_threshold)

            if len(bg_coords) == 0:
                # Get coordinates of non-zero elements
                coords = torch.nonzero(img_data)

                # Get the random coordinate

                center_coord = coords[torch.randint(0, len(coords), (1,))][0]
            else:
                # Randomly select one background voxel
                center_coord = bg_coords[torch.randint(0, len(bg_coords), (1,))][0]


        # Determine start and end coordinates based on the center voxel and spatial_size
        spatial_tensor = torch.tensor(self.spatial_size, dtype=torch.int64)
        start_coord = torch.clamp(
            center_coord - torch.div(spatial_tensor, 2, rounding_mode='floor'),
            torch.tensor([0, 0, 0]),
            torch.tensor([depth, height, width]),
        )
        end_coord = start_coord + spatial_tensor

        # Ensure end coordinates do not exceed image dimensions
        max_sizes = torch.tensor([depth, height, width], dtype=torch.int64)
        end_coord = torch.min(torch.max(end_coord, spatial_tensor), max_sizes)
        start_coord = end_coord - spatial_tensor

        self.start_coord = tuple(start_coord.tolist())
        self.end_coord = tuple(end_coord.tolist())

    def __call__(self, data):
        cropped_samples = []

        for _ in range(self.num_samples):
            # Call the randomize() function
            self.randomize(data)

            sample = data.copy()

            # Crop the image based on the randomized coordinates
            cropped_image = sample[self.image_key][
                ...,
                self.start_coord[0] : self.end_coord[0],
                self.start_coord[1] : self.end_coord[1],
                self.start_coord[2] : self.end_coord[2],
            ]

            # Calculate the corresponding coordinates for the label
            label_start_coord = tuple(coord * 2 for coord in self.start_coord)
            label_end_coord = tuple(coord * 2 for coord in self.end_coord)

            # Crop the label using the adjusted coordinates
            cropped_label = sample[self.label_key][
                ...,
                label_start_coord[0] : label_end_coord[0],
                label_start_coord[1] : label_end_coord[1],
                label_start_coord[2] : label_end_coord[2],
            ]

            sample[self.image_key] = cropped_image
            sample[self.label_key] = cropped_label

            cropped_samples.append(sample)

        return cropped_samples
