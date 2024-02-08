import matplotlib.pyplot as plt
import torch


class Visualizations:
    def __init__(self):
        pass

    def plot_row(
        self, img, val_outputs, binary_output, slice_frac, filename, label=None
    ):
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

        # plot data image
        plt.subplot(1, total_imges, 1)
        plt.title("image")
        plt.imshow(img.cpu().numpy()[:, :, slice_num], cmap="gray")

        # plot prediction of NN
        plt.subplot(1, total_imges, 2)
        plt.title("output")
        plt.imshow(val_outputs.detach().cpu()[:, :, slice_num * 2])

        # plot binary prediction of NN
        plt.subplot(1, total_imges, 3)
        plt.title("binary output")
        plt.imshow(binary_output[:, :, slice_num * 2])

        # plot the ground truth if existent
        if label is not None:
            plt.subplot(1, total_imges, 4)
            plt.title("label")
            plt.imshow(label.cpu().numpy()[:, :, slice_num * 2])

        plt.savefig(f"./{filename}.png")
