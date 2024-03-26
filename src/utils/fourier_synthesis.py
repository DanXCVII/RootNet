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


# class FourierSynthesis:
#     def __init__(self, volume):
#         """
#         Initialize the Fourier synthesis.

#         - volume: 3D numpy array representing the original volume.
#         """
#         self.f_transform = None
#         self.f_shift = None

#         self.learn_new_params(volume)

#     def learn_new_params(self, volume):
#         """
#         Perform a Fourier analysis on a given 3D numpy array.

#         - volume: 3D numpy array representing the original volume.
#         """
#         # Fourier transform of the original volume
#         self.f_transform = np.fft.fftn(volume)
#         self.f_shift = np.fft.fftshift(self.f_transform)

#     def visualize_spectrun(self, shift=None):
#         if shift is None:
#             print("shift is None")
#             my_shift = self.f_transform
#         else:
#             my_shift = shift

#         print("visualize shift:", my_shift[-1:])
#         # Compute the magnitude spectrum
#         print("my_shift mean:", np.mean(my_shift))
#         magnitude_spectrum = np.abs(my_shift)
#         log_magnitude = np.log(magnitude_spectrum + 1)  # +1 to avoid log(0)

#         print("log_m mean:", np.mean(log_magnitude))

#         # Visualize a central slice of the log magnitude spectrum
#         # For a 3D volume, you might choose a slice along one of the dimensions
#         plt.imshow(log_magnitude[log_magnitude.shape[0] // 2], cmap="gray")
#         plt.title("Log Magnitude Spectrum of a Central Slice")
#         plt.colorbar()
#         plt.show()

#     def generate_new_texture(
#         self, original_array, num_images, variation_strength=0.5
#     ) -> list:
#         """
#         Generate new samples (amount: num_images) with the same shape as the original volume.

#         - original_array: 3D numpy array representing the original volume.
#         - num_images: number of new samples to be generated.
#         """
#         new_textures = []
#         for _ in range(num_images):
#             # Introduce random variations in the Fourier domain
#             random_phase_shift = np.exp(
#                 2j * np.pi * np.random.rand(*self.f_shift.shape) * variation_strength
#             )

#             # Get amplitude and phase
#             amplitude = np.abs(self.f_transform)
#             phase = np.angle(self.f_transform)

#             # Introduce randomness to the phase
#             # adjusted_phase = phase

#             ########

#             randomness_coefficient = 1

#             random_phase = np.random.uniform(0, np.pi, phase.shape)
#             adjusted_phase = (
#                 1 - randomness_coefficient
#             ) * phase + randomness_coefficient * random_phase

#             # noise_level = 0.05  # Adjust the noise level to your preference
#             # random_noise = np.random.normal(0, noise_level, self.f_transform.shape)
#             # new_f_transform = self.f_transform + random_noise
#             ########

#             new_ft_data = amplitude * np.exp(1j * adjusted_phase)

#             # Inverse Fourier transform to get back to spatial domain
#             modified_f_transform = np.fft.ifftshift(new_ft_data)
#             modified_volume = np.fft.ifftn(modified_f_transform).real

#             print("visualize_spectrun")
#             print("shift:", modified_f_transform[-1:])
#             print("shift:", self.f_shift[-1:])
#             self.visualize_spectrun(shift=new_ft_data)

#             # Create the histogram
#             plt.hist(modified_volume.flatten(), bins=30, alpha=0.75)

#             # Add titles and labels
#             plt.title("Histogram of 3D Array Values")
#             plt.xlabel("Value")
#             plt.ylabel("Frequency")

#             # Show the plot
#             plt.show()

#             # Optionally, clip or normalize modified_volume to maintain a valid range
#             # For example, to keep the pixel values in the same range as the original
#             modified_volume = np.clip(
#                 modified_volume, np.min(original_array), np.max(original_array)
#             )

#             new_textures.append(modified_volume)

#         return new_textures

#     # def generate_new_texture(self, original_array, num_images, variation_strength=0.1):
#     #     """
#     #     Generate new samples (amount: num_images) with the same shape as the original volume,
#     #     introducing only slight variations to the original noise.

#     #     - original_array: 3D numpy array representing the original volume.
#     #     - num_images: number of new samples to be generated.
#     #     - variation_strength: controls the amount of variation from the original (0 to 1).
#     #     """

#     #     # Extract the original phase and amplitude from the Fourier transform
#     #     amplitude = np.abs(self.f_shift)
#     #     original_phase = np.angle(self.f_shift)

#     #     # Generate new textures with slight variations
#     #     volumes = []
#     #     for _ in range(num_images):
#     #         # Generate a random phase variation
#     #         random_phase_variation = np.exp(
#     #             1j * np.random.uniform(0, 2 * np.pi, self.f_shift.shape)
#     #         )

#     #         # Blend the original phase with the random phase variation
#     #         # The variation_strength parameter controls the blend ratio
#     #         blended_phase = np.exp(
#     #             1j
#     #             * (
#     #                 (1 - variation_strength) * original_phase
#     #                 + variation_strength * np.angle(random_phase_variation)
#     #             )
#     #         )

#     #         # Construct the new Fourier transform with the blended phase and original amplitude
#     #         new_f_transform = amplitude * blended_phase

#     #         # Perform the inverse Fourier transform to get back to spatial domain
#     #         new_f_transform_shifted = np.fft.ifftshift(new_f_transform)
#     #         new_volume = np.fft.ifftn(new_f_transform_shifted)
#     #         new_volume = np.abs(new_volume)

#     #         volumes.append(new_volume)

#     #     return volumes
