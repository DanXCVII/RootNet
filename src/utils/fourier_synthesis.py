import numpy as np
import matplotlib.pyplot as plt
from .MRI_operations import MRIoperations


class FourierSynthesis:
    """
    Class for applying the "Fourier synthesis" technique to generate new samples from a given volume. This
    is done by first performing a FFT on the original volume, and then manipulating the Fourier
    transform by randomizing the phase of the components. An inverse Fourier transform is then applied to
    obtain the new volume.
    """
    def __init__(self, volume):
        """
        Initialize the Fourier synthesis.

        - volume: 3D numpy array representing the original volume.
        """
        self.f_transform = None
        self.f_shift = None

        self.learn_new_params(volume)

    def learn_new_params(self, volume):
        """
        Perform a Fourier analysis on a given 3D numpy array.

        - volume: 3D numpy array representing the original volume.
        """
        # Fourier transform of the original volume
        self.f_transform = np.fft.fftn(volume)
        self.f_shift = np.fft.fftshift(self.f_transform)

    def visualize_spectrun(self, shift=None, name="original"):
        if shift is None:
            print("shift is None")
            my_shift = self.f_transform
        else:
            print("shift is not None")
            my_shift = shift

        phase_spectrum = np.angle(my_shift)

        plt.imshow(phase_spectrum[my_shift.shape[0] // 2], cmap="hsv")
        plt.colorbar()
        plt.title(f"Phase Spectrum of a Central Slice")
        plt.savefig(f"{name}_phase.png")
        plt.close()

        # Compute the magnitude spectrum
        magnitude_spectrum = np.abs(my_shift)
        log_magnitude = np.log1p(magnitude_spectrum)  # +1 to avoid log(0)

        # Visualize a central slice of the log magnitude spectrum
        # For a 3D volume, you might choose a slice along one of the dimensions
        plt.imshow(log_magnitude[log_magnitude.shape[0] // 2], cmap="gray")
        plt.title("Log Magnitude Spectrum of a Central Slice")
        plt.colorbar()
        plt.savefig(f"{name}_log_magnitude.png")
        plt.close()

    def generate_new_texture(self, original_array, num_images) -> list:
        """
        Generate new samples (amount: num_images) with the same shape as the original volume.

        - original_array: 3D numpy array representing the original volume.
        - num_images: number of new samples to be generated.
        """

        # Manipulate the Fourier transform for synthesis
        amplitude = np.abs(self.f_shift)
        phase = []
        for i in range(num_images):
            fraction_random = 0  # fraction of random phases all the components
            mask = np.random.choice(
                [True, False],
                size=self.f_shift.shape,
                p=[1 - fraction_random, fraction_random],
            )
            random_values = np.random.uniform(0, 2 * np.pi, self.f_shift.shape)

            result = np.where(mask, random_values, np.angle(self.f_shift))

            phase.append(np.exp(1j * result))

        # Initialize a new, larger array for the Fourier transform
        new_f_transform = np.zeros(self.f_shift.shape, dtype=complex)

        # Calculate the center slice positions
        original_center = [s // 2 for s in self.f_shift.shape]
        new_center = [s // 2 for s in self.f_shift.shape]

        # Calculate slicing ranges
        slices_from = [max(nc - oc, 0) for nc, oc in zip(new_center, original_center)]
        slices_to = [sf + s for sf, s in zip(slices_from, self.f_shift.shape)]
        slices = tuple(slice(sf, st) for sf, st in zip(slices_from, slices_to))

        # Place the original Fourier transform in the center of the new array
        volumens = []
        for i in range(num_images):
            new_f_transform[slices] = amplitude * phase[i]

            # Inverse Fourier transform to get the new volume
            new_f_transform_shifted = np.fft.ifftshift(new_f_transform)
            new_volume = np.fft.ifftn(new_f_transform_shifted)
            new_volume = np.abs(new_volume)

            volumens.append(new_volume)

        return volumens


# example usage
# mri_ops = MRIoperations()
# _, my_mri = mri_ops.load_mri(
#     "/Users/daniel/Desktop/FZJ/CPlantBox/DUMUX/CPlantBox/tutorial/examples_segmentation/RootNet/data_assets/noise/soil_noise/scan_sand_38_142x136x49.raw"
# )

# # Create a FourierSynthesis object
# fourier_synthesis = FourierSynthesis(my_mri)

# # Visualize the spectrum
# fourier_synthesis.visualize_spectrun(name="original_spectrum.png")

# # Generate new textures
# new_textures = fourier_synthesis.generate_new_texture(my_mri, 1)
# mri_ops.save_mri(
#     f"new_texture_{new_textures[0].shape[2]}x{new_textures[0].shape[1]}x{new_textures[0].shape[0]}.raw",
#     new_textures[0],
# )
