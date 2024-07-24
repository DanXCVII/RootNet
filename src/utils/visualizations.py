# import matplotlib.pyplot as plt
# import torch
# from scipy.ndimage import zoom
# import numpy as np
# from MRI_operations import MRIoperations
# from enum import Enum

# class Visualizations:
#     def __init__(self):
#         pass

#     def plot_row(self, img, val_outputs, binary_output, cut_slice, filename, label=None, threshold_mri=None):
#         """
#         plots the image, label, output and binary output for a given horizontal slice

#         Args:
#         - img: image tensor
#         - label: label tensor
#         - val_outputs: output tensor
#         - cut_slice: fraction of the image, where the slice should be taken
#         - filename: filename to save the plot
#         """
#         print("img shape: ", img.shape)
#         print("val_outputs shape: ", val_outputs.shape)
#         print("binary_output shape: ", binary_output.shape)

#         if isinstance(img, torch.Tensor):
#             img = img.cpu().numpy()
#         if isinstance(val_outputs, torch.Tensor):
#             val_outputs = val_outputs.detach().cpu()
#         if label is not None and isinstance(label, torch.Tensor):
#             label = label.cpu().numpy()
        
#         total_imges = 0
#         if label is None:
#             total_imges = 3
#         else:
#             if threshold_mri is not None:
#                 total_imges = 5
#             else:
#                 total_imges = 4
#         plt.figure("check", (18, 6))

#         img_shape = img.shape
#         if cut_slice < 1:
#             slice_num = int(img_shape[-1] * cut_slice)
#         else:
#             slice_num = cut_slice

#         print("slice_num: ", slice_num)
#         if val_outputs.shape[-1] > img_shape[-1] * 2:
#             slice_num_adjusted = slice_num * 2 + (val_outputs.shape[-1] - img_shape[-1] * 2) // 2
#         else:
#             slice_num_adjusted = slice_num * 2

#         alpha_mask_pred = np.where(val_outputs[:, :, slice_num_adjusted] > 0.5, 0.9, 0)
#         alpha_mask_label = np.where(label[:, :, slice_num_adjusted] > 0.5, 1.0, 0)
#         resized_input = zoom(img[:, :, slice_num], zoom=2, order=3)

#         slice_num_treshold = slice_num_adjusted + 1
#         # save the visualized slices
#         # create the folder if it does not exist
#         import os
#         if not os.path.exists(f"./slice_{cut_slice}"):
#             os.makedirs(f"./slice_{cut_slice}")
#         np.save(f"./slice_{cut_slice}/label_slice.npy", label[:, :, slice_num_adjusted])
#         np.save(f"./slice_{cut_slice}/output_slice.npy", val_outputs[:, :, slice_num_adjusted])
#         np.save(f"./slice_{cut_slice}/binary_output_slice.npy", binary_output[:, :, slice_num_adjusted])
#         np.save(f"./slice_{cut_slice}/img_slice.npy", img[:, :, slice_num])
#         np.save(f"./slice_{cut_slice}/threshold_slice.npy", threshold_mri[:, :, slice_num_treshold])
#         filename = f"./slice_{cut_slice}/{filename}"

#         # plot data image
#         plt.subplot(1, total_imges, 1)
#         plt.title("image upscaled")
#         plt.imshow(resized_input, cmap="gray")
        
#         # plot prediction of NN
#         plt.subplot(1, total_imges, 2)
#         plt.title("output")
#         plt.imshow(val_outputs[:, :, slice_num_adjusted])
#         plt.imshow(resized_input, cmap="gray")
#         plt.imshow(val_outputs[:, :, slice_num_adjusted], alpha=1)

#         # plot binary prediction of NN
#         plt.subplot(1, total_imges, 3)
#         plt.title("binary output")
#         plt.imshow(binary_output[:, :, slice_num_adjusted])

#         # plot the ground truth if existent
#         if label is not None:
#             plt.subplot(1, total_imges, 4)
#             plt.title("label")
#             plt.imshow(resized_input, cmap="gray")
#             plt.imshow(label[:, :, slice_num_adjusted], alpha=0.95)

#         if threshold_mri is not None:
#             plt.subplot(1, total_imges, 5)
#             plt.title("threshold_mri")
#             plt.imshow(threshold_mri[:, :, slice_num_treshold])

#         plt.savefig(f"./{filename}.png")


# # test
# vis = Visualizations()
# mri_ops = MRIoperations()

# # create an enum class with threshold, root_scaled, gmm
# class MRI(Enum):
#     threshold = 0
#     root_scaled = 1
#     gmm = 2

# def my_plot(my_slice, mri_num, mri_type=MRI.root_scaled):
#     if mri_num == 0:
#         test_mri = "/p/project1/visforai/weissen1/RootNet/data_final/test/real/soil/III_Soil_3D_DAP15_res_256x256x199.nii.gz"
#     elif mri_num == 1:
#         test_mri = "/p/project1/visforai/weissen1/RootNet/data_final/test/real/soil/IV_Soil_1W_DAP9_res_256x256x136.nii.gz"
#     elif mri_num == 2:
#         test_mri = "/p/project1/visforai/weissen1/RootNet/data_final/test/real/soil/III_Soil_1W_DAP14_res_256x256x186.nii.gz"
#     elif mri_num == 3:
#         test_mri = "/p/project1/visforai/weissen1/RootNet/data_final/test/real/soil/IV_Soil_3D_DAP8_res_256x256x193.nii.gz"
#     elif mri_num == 4:
#         test_mri = "/p/project1/visforai/weissen1/RootNet/data_final/test/real/sand/III_Sand_3D_DAP14_res_256x256x191.nii.gz"
#     elif mri_num == 5:
#         test_mri = "/p/project1/visforai/weissen1/RootNet/data_final/test/real/sand/III_Sand_1W_DAP14_res_256x256x131.nii.gz"
#     elif mri_num == 6:
#         test_mri = "/p/project1/visforai/weissen1/RootNet/data_final/test/real/sand/IV_Sand_3D_DAP8_res_256x256x192.nii.gz"

#     if mri_type == MRI.threshold:
#         mri_dir = "threshold_output"
#     elif mri_type == MRI.root_scaled:
#         mri_dir = "root_scale_best_output"
#     elif mri_type == MRI.gmm:
#         mri_dir = "gmm_output"

#     _, img = mri_ops.load_mri(test_mri)
#     _, label = mri_ops.load_mri(f"/p/project/visforai/weissen1/RootNet/src/training/{mri_dir}/{mri_num}/label_512x512x402.nii.gz")
#     _, val_outputs = mri_ops.load_mri(f"/p/project/visforai/weissen1/RootNet/src/training/{mri_dir}/{mri_num}/output_512x512x402.nii.gz")
#     _, binary_output = mri_ops.load_mri(f"/p/project/visforai/weissen1/RootNet/src/training/{mri_dir}/{mri_num}/binary_output_512x512x402.nii.gz")
#     _, threshold_mri = mri_ops.load_mri(f"/p/project/visforai/weissen1/RootNet/src/training/threshold_output/{mri_num}/binary_output_512x512x402.nii.gz")

#     vis.plot_row(img, val_outputs, binary_output, my_slice, f"{mri_type}_test", label=label, threshold_mri=threshold_mri)

# my_plot(82, 5, MRI.root_scaled)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! old version (When code above not needed anymore, remove top code and uncomment bottom) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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