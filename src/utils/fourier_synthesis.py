import numpy as np
import matplotlib.pyplot as plt


class FourierSynthesis:
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

    def visualize_spectrun(self, shift=None):
        if shift is None:
            print("shift is None")
            my_shift = self.f_transform
        else:
            my_shift = shift

        print("visualize shift:", my_shift[-1:])
        # Compute the magnitude spectrum
        print("my_shift mean:", np.mean(my_shift))
        magnitude_spectrum = np.abs(my_shift)
        log_magnitude = np.log(magnitude_spectrum + 1)  # +1 to avoid log(0)

        print("log_m mean:", np.mean(log_magnitude))

        # Visualize a central slice of the log magnitude spectrum
        # For a 3D volume, you might choose a slice along one of the dimensions
        plt.imshow(log_magnitude[log_magnitude.shape[0] // 2], cmap="gray")
        plt.title("Log Magnitude Spectrum of a Central Slice")
        plt.colorbar()
        plt.show()

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
            phase.append(
                np.exp(1j * np.random.uniform(0, 2 * np.pi, self.f_shift.shape))
            )

        # Initialize a new, larger array for the Fourier transform
        new_f_transform = np.zeros(original_array.shape, dtype=complex)

        # Calculate the center slice positions
        original_center = [s // 2 for s in self.f_shift.shape]
        new_center = [s // 2 for s in original_array.shape]

        # Calculate slicing ranges
        slices_from = [max(nc - oc, 0) for nc, oc in zip(new_center, original_center)]
        slices_to = [sf + s for sf, s in zip(slices_from, original_array.shape)]
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
