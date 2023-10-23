import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from virtual_mri_generation import Virtual_MRI, RootSystemSimulation, SoilWaterSimulation



from noise import pnoise3


def load_mri(path='../../../data/generated/virtual_mri/Anagallis_femina_Leitner_2010_day_3_SNR_3_res_222x222x200.raw', shape=(200, 222, 222), reshape=True):
    with open(path, 'rb') as f:
        if reshape:
            mri_reloaded = np.fromfile(f, dtype=np.int16).reshape(shape[0], shape[1], shape[2])
        else:
            mri_reloaded = np.fromfile(f, dtype=np.int16)

        return mri_reloaded

def save_mri(grid):
    filename = "../../../data/generated/virtual_mri/test_{}x{}x{}.raw".format(grid.shape[2], grid.shape[1], grid.shape[0])
    grid.astype('int16').tofile(filename)


def _add_gaussian_noise(image, sigma, intensity=1):
    """
    adds gaussian noise to the image with the given mean and variance

    Args:
    image (numpy array): image to which the noise is added
    mean (float): mean of the gaussian distribution
    sigma (float): variance of the gaussian distribution
    """
    row,col,ch= image.shape
    gauss = np.random.normal(0,sigma,(row,col,ch))

    gauss = gauss.reshape(row,col,ch)
    gauss = intensity * gauss
    print("gauss", gauss.max())
    noisy = image + gauss

    
    return noisy

def _add_noise_to_grid(grid, snr, intensity_g=0.3, intensity_p=0.3):
    """
    add Gaussian noise to the image and scaling it according to the intensity of the root and the SNR

    Args:
    grid: 3d (numpy) array of the root container with dimensions (nx, ny, nz)
    """
    ps = np.sum((grid - np.mean(grid)) ** 2) / (grid).size
    pn = 8*100000000
    sigma = np.sqrt(pn)
    print("sigma", sigma)

    # meanC = np.mean(grid[grid>0])
    # print("meanC", meanC)
    # n_var = meanC/self.snr

    # TODO: Replace 0.5 with water_intensity_grid
    grid_noise = _add_gaussian_noise(grid, sigma, intensity=intensity_g)
    grid_noise = np.clip(grid_noise, 0, 25000).astype(np.int16)

    grid_noise = add_perlin_noise(grid_noise, intensity=intensity_p)
    grid_noise = np.clip(grid_noise, 0, 25000).astype(np.int16)

    return grid_noise

def add_perlin_noise(image, intensity=1):
    """
    Add Perlin noise to a 3D image.
    
    Parameters:
    - image: Input 3D image (numpy array).
    - intensity: Intensity of the noise (multiplier for noise values).
    
    Returns:
    - noisy_image: 3D image with added Perlin noise.
    """
    noise_array = _generate_perlin_noise_3d(image.shape, 5, 5, 15)
    
    # Scale noise to fit the desired intensity
    noise_array = intensity * ((50000 * noise_array))
    print("intensity", intensity)
    print("noise_array", noise_array.max())
    # Add noise to the image
    noisy_image = image + noise_array
    
    # Clip values to ensure they remain in the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 25000).astype(np.int16)
    
    return noisy_image

def load_mri_noisy_part():
    with open('../../../../tmp/III_Sand_1W_DAP14_256x256x131.raw', 'rb') as f:
        mri_reloaded = np.fromfile(f, dtype=np.int16).reshape(131,256,256)

        my_slice = mri_reloaded[62]
        my_slice.astype('int16').tofile('../../../../tmp/'+'my_slice_256x256x1.raw')
        rectangular_region = my_slice[48:212, 82:200]
        print(rectangular_region.max())
        rectangular_region.astype('int16').tofile('../../../../tmp/'+'rectanglular_reg_118x164x1.raw')

        return rectangular_region
    
def load_mri_root_part():
    with open('../../../../tmp/III_Sand_1W_DAP14_256x256x131.raw', 'rb') as f:
        mri_reloaded = np.fromfile(f, dtype=np.int16).reshape(131,256,256)
        print("mri_reloaded max", mri_reloaded.max())

        my_slice = mri_reloaded[62]
        my_slice.astype('int16').tofile('../../../../tmp/'+'my_slice_256x256x1.raw')
        print(my_slice.max())
        #rectangular_region.astype('int16').tofile('../../../../tmp/'+'rectanglular_reg_2x3x1.raw')


        #return rectangular_region

def estimate_noise_and_generate_image():
    # Simulate a noisy image and a clean image
    noise = load_mri_noisy_part()  # Gaussian noise as an example

    # Estimate the noise using a Gaussian mixture model
    gmm = GaussianMixture(n_components=3).fit(noise.reshape(-1, 1))
    print("gmm.means_", gmm.means_)
    print("gmm.covariances_", gmm.covariances_)

    # generate a new image with the estimated noise
    gmm_noise, _ = gmm.sample(6553600)
    np.random.shuffle(gmm_noise)
    print("gmm_noise", gmm_noise[:10])
    gmm_noise = gmm_noise.reshape(100, 256, 256)

    gmm_noise.astype('int16').tofile('../../../../tmp/'+'noise_256x256x100.raw')


def _generate_perlin_noise_3d(shape, scale_x, scale_y, scale_z):
    """
    Generate a 3D numpy array of Perlin noise.
    
    :param shape: The shape of the generated array (tuple of 3 ints).
    :param scale: The scale of noise.
    :return: 3D array of values in the range [-1, 1].
    """
    noise = np.zeros(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                noise_value = pnoise3(i / scale_x, 
                                         j / scale_y, 
                                         k / scale_z, 
                                         octaves=4, 
                                         persistence=2, 
                                         lacunarity=2, 
                                         repeatx=256, 
                                         repeaty=256, 
                                         repeatz=200, 
                                         base=0)
                # Normalize to [0, 1]
                # noise_value = (noise_value + 1) / 2
               
                noise[i][j][k] = noise_value

    return noise

def add_noise():
    # ""
    my_mri = load_mri()#'/Users/daniel/Desktop/FZJ/Echte Daten/tobiasBean/tobiasBean_12_300x300x577.raw', (577, 300, 300), reshape=False)
    array_flat = my_mri.ravel()
    print(array_flat.max())
    print(array_flat.min())
    plt.hist(array_flat, bins=50, facecolor='blue', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of 3D Array Values')
    plt.grid(True)
    plt.savefig('histogram.png')

    #save_mri(my_mri)

dict = {'root_model_name': 'Bench_lupin', 'soil_type': 'sand', 'root_growth_days': 8, 'initial': -0, 'sim_time': 8, 'perlin_noise_intensity': 0.9}
# add_noise()
plant_path = "/Users/daniel/Desktop/FZJ/CPlantBox/DUMUX/CPlantBox/tutorial/examples_segmentation/final/NN/data/generated/Bench_lupin/loam/sim_days_10-initial_-2300-noise_0.6/Bench_lupin_day_10.rsml"
my_soil_sim = SoilWaterSimulation("../../data_assets/meshes/cylinder_r_0.032_d_-0.21_res_0.01.msh", plant_path, "../../data/generated", "sand", sim_time=8, initial=-300)
my_soil_sim.run()


# load_mri_noisy_part()

# # Example usage:
# shape = (100, 256, 256)
# scale = 2
# perlin_3d = generate_perlin_noise_3d(shape, scale)

# print(perlin_3d[1,:,:])

# perlin_3d.astype('int16').tofile('tmp/'+'perlin_256x256x100.raw')