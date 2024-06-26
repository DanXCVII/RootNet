import matplotlib.pyplot as plt
import torch
from scipy.ndimage import zoom
import numpy as np

class Visualizations:
    def __init__(self):
        pass

    def plot_row(self, img, val_outputs, binary_output, slice_frac, filename, label=None):
        """
        plots the image, label, output and binary output for a given horizontal slice

        Args:
        - img: image tensor
        - label: label tensor
        - val_outputs: output tensor
        - slice_frac: fraction of the image, where the slice should be taken
        - filename: filename to save the plot
        """
        print("img shape: ", img.shape)
        print("val_outputs shape: ", val_outputs.shape)
        print("binary_output shape: ", binary_output.shape)
        
        total_imges = 0
        if label is None:
            total_imges = 3
        else:
            total_imges = 4
        plt.figure("check", (18, 6))

        img_shape = img.shape
        slice_num = int(img_shape[-1] * slice_frac)

        alpha_mask_pred = np.where(val_outputs.detach().cpu()[:, :, slice_num * 2] > 0.5, 0.9, 0)
        alpha_mask_label = np.where(label.cpu().numpy()[:, :, slice_num * 2] > 0.5, 1.0, 0)
        resized_input = zoom(img.cpu().numpy()[:, :, slice_num], zoom=2, order=3)

        # plot data image
        plt.subplot(1, total_imges, 1)
        plt.title("image upscaled")
        plt.imshow(resized_input, cmap="gray")
        
        # plot prediction of NN
        plt.subplot(1, total_imges, 2)
        plt.title("output")
        plt.imshow(val_outputs.detach().cpu()[:, :, slice_num * 2])
        plt.imshow(resized_input, cmap="gray")
        plt.imshow(val_outputs.detach().cpu()[:, :, slice_num * 2], alpha=0.5)

        # plot binary prediction of NN
        plt.subplot(1, total_imges, 3)
        plt.title("binary output")
        plt.imshow(binary_output[:, :, slice_num * 2])

        # plot the ground truth if existent
        if label is not None:
            plt.subplot(1, total_imges, 4)
            plt.title("label")
            plt.imshow(resized_input, cmap="gray")
            plt.imshow(label.cpu().numpy()[:, :, slice_num * 2], alpha=0.5)

        plt.savefig(f"./{filename}.png")