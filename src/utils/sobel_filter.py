from monai.transforms import Randomizable
import numpy as np
import torch
import cv2

import torch.nn.functional as F


class AppySobelFilter:
    def __init__(
        self,
        image_key,
    ):
        self.image_key = image_key

    def apply_sobel(self, data):
        sobel_x_kernel = (
            torch.tensor(
                [
                    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                    [[-2.0, 0.0, 2.0], [-4.0, 0.0, 4.0], [-2.0, 0.0, 2.0]],
                    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                ]
            )
            .view(1, 1, 3, 3, 3)
            .repeat(1, data.shape[1], 1, 1, 1)
        )

        sobel_y_kernel = (
            torch.tensor(
                [
                    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                    [[-2.0, -4.0, -2.0], [0.0, 0.0, 0.0], [2.0, 4.0, 2.0]],
                    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                ]
            )
            .view(1, 1, 3, 3, 3)
            .repeat(1, data.shape[1], 1, 1, 1)
        )

        sobel_z_kernel = (
            torch.tensor(
                [
                    [[-1.0, -2.0, -1.0], [-2.0, -4.0, -2.0], [-1.0, -2.0, -1.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                ]
            )
            .view(1, 1, 3, 3, 3)
            .repeat(1, data.shape[1], 1, 1, 1)
        )

        # Apply the Sobel filters
        sobel_x = F.conv3d(data, sobel_x_kernel, padding=1)
        sobel_y = F.conv3d(data, sobel_y_kernel, padding=1)
        sobel_z = F.conv3d(data, sobel_z_kernel, padding=1)

        # Calculate the magnitude of the gradients
        gradient_magnitude = torch.sqrt(sobel_x**2 + sobel_y**2 + sobel_z**2)

        return gradient_magnitude

    def gaussian_kernel_3d(self, kernel_size, sigma):
        """Creates a 3D Gaussian kernel using the specified parameters."""

        # Create a coordinate grid
        x = torch.arange(kernel_size).float() - kernel_size // 2
        y = torch.arange(kernel_size).float() - kernel_size // 2
        z = torch.arange(kernel_size).float() - kernel_size // 2
        x, y, z = torch.meshgrid(x, y, z)

        # Calculate the Gaussian function
        kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))

        # Normalize the kernel
        kernel /= torch.sum(kernel)

        return kernel.view(1, 1, *kernel.shape)

    def __call__(self, data):
        img_data = data[self.image_key].unsqueeze(0)
        print(img_data.shape)
        kernel_size = 5
        sigma = 2
        gaussian_kernel = self.gaussian_kernel_3d(kernel_size, sigma)
        blurred_data = F.conv3d(
            img_data,
            gaussian_kernel,
            padding=kernel_size // 2,
            groups=img_data.shape[1],
        )

        sobel_img = self.apply_sobel(blurred_data)

        data[self.image_key] = sobel_img[0]

        return data
