"""simulates a root system, which is rasterized to a given resolution. To mimic an MRI image, Gaussian noise is additionally added"""
import sys
sys.path.append("/Users/daniel/Desktop/FZJ/CPlantBox/DUMUX/CPlantBox")
sys.path.append("/Users/daniel/Desktop/FZJ/CPlantBox/DUMUX/CPlantBox/experimental/parametrisation/")
sys.path.append("/Users/daniel/Desktop/FZJ/CPlantBox/DUMUX/CPlantBox/src")

import plantbox as pb
import visualisation.vtk_plot as vp
import numpy as np
import functional.bresenham3D as bres3D
import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK
import rsml_reader as rsml
import scipy
from .root_growth_simulation import RootSystemSimulation as RSSim
from scipy import ndimage
from .interpolate_water_sim import VTUInterpolatedGrid
from noise import pnoise3

# import pygorpho as pg
import skimage
from skimage.morphology import ball
import math
import nibabel as nib
import random
import os
import time


def get_min_max_numpy(array):
    """
    since the min max function of numpy does 
    """
    non_nan_indices = np.argwhere(~np.isnan(array.flat))
    first_non_nan_value = array.flat[non_nan_indices[0][0]]

    new = np.nan_to_num(array, nan=first_non_nan_value)

    min = new.min()
    max = new.max()

    return min, max



class Virtual_MRI:
    def __init__(self, seganalyzer, rsml_path, soil_type, vtu_path, perlin_noise_intensity, width=3, depth=20, resolution=[0.027, 0.027, 0.1], snr=3):
        """
        creates a virtual MRI for the given root system with simulating noise based on the water content of the soil.
        - rootsystem: root system object (pb.RootSystem)
        - rsml_path: path to the rsml file of the root system # currently not used
        - vtu_path: path to the vtu file of the water simulation
        - perlin_noise_intensity: intensity of how much perlin noise should be added (value between 0 and 1)
        - width: width of the soil container
        - depth: depth of the soil container
        - resolution: resolution of the MRI
        - snr: signal to noise ratio
        """
        self.resx = resolution[0]
        self.resy = resolution[1]
        self.resz = resolution[2]
        self.width = width
        self.depth = depth
        self.rsml_path = rsml_path
        self.vtu_path = vtu_path
        self.soil_type = soil_type
        self.perlin_noise_intensity = perlin_noise_intensity
        self.segana = seganalyzer
        self.snr = snr
        self.water_intensity_grid = None

        self.nx = int(self.width*2 / self.resx)
        self.ny = int(self.width*2 / self.resy)
        self.nz = int(self.depth / self.resz)

        self.max_signal_intensity = 30000
        self.max_root_signal_intensity = 30000

    def _add_gaussian_noise(self, image, sigma, water_intensity_grid):
        """
        adds gaussian noise to the image with the given mean and variance

        Args:
        image (numpy array): image to which the noise is added
        mean (float): mean of the gaussian distribution
        sigma (float): variance of the gaussian distribution
        """
        row,col,ch = image.shape
        gauss = np.random.normal(0, sigma,(row,col,ch))
        
        gauss_water_scaled = np.multiply(gauss, water_intensity_grid)

        noisy_plus_image = image + gauss_water_scaled
        
        return noisy_plus_image
    
    def _add_perlin_noise(self, image, water_intensity_grid):
        """
        Add Perlin noise to a 3D image.
        
        Parameters:
        - image: Input 3D image (numpy array).
        - intensity: Intensity of the noise (multiplier for noise values).
        
        Returns:
        - noisy_image: 3D image with added Perlin noise.
        """
        noise_array = self._generate_perlin_noise_3d(image.shape, 5, 5, 15)
        
        # Scale noise to fit the desired intensity
        noise_array_scaled = self.perlin_noise_intensity * water_intensity_grid * ((self.max_signal_intensity * 2 * noise_array))
        
        # Add noise to the image
        image_plus_noise = image + noise_array_scaled
        
        # Clip values to ensure they remain in the valid range [0, self.max_signal_strength]
        noisy_image = np.clip(image_plus_noise, 0, self.max_signal_intensity).astype(np.int16)
        
        return noisy_image
    
    def _generate_perlin_noise_3d(self, shape, scale_x, scale_y, scale_z):
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

    def _isConsecutive(self, A):
        """
        checks if the array only contains consecutive values like [1,2,3,4,5], nothing else is accepted
        """
        if len(A) <= 1:
            return True
    
        minimum = min(A)
        maximum = max(A)

        if maximum - minimum != len(A) - 1:
            return False

        visited = set()
        for i in A:
            if i in visited:
                return False
            visited.add(i)
        return True
    
    # def _get_root_data_from_rsml(self):
    #     rsml.read_rsml(self.rsml_path)
    #     polylines, props, functions = rsml.read_rsml(self.rsml_path)

    #     nodes, segments = rsml.get_segments(polylines, props)
    #     keys = list(props.keys())
    #     print("keys", keys)
    #     radius = [props["diameter"][i]/2 for i in range(len(props["diameter"]))]

    #     segCTs = np.zeros(len(radius))
        

    #     ana = pb.SegmentAnalyser(nodes, segments, segCTs, radius)
    #     # nodes = ana.nodes
    #     # segments = ana.segments
    #     # radius = ana.getParameter("radius")

    #     # print("nodes", nodes[:3])
    #     # print("segments", segments[:3])
    #     # print("radius", radius[:3])

    #     segments = [[int(item) for item in sublist] for sublist in segments]

    #     # list keys of props
        
    #     # devide the values for "diameter" by 2 and save them in the list radius
    #     radius = [props["diameter"][i]/2 for i in range(len(props["diameter"]))]
    #     length = props["length"]

    #     # print("nodes", nodes[:3])
    #     # print("segments", segments[-3:-1])
    #     # print("radius", radius[:3])
    #     # print("length", length[:3])

    #     # print("nodes", len(nodes))
    #     # print("segments", len(segments))
    #     # print("radius", len(radius))
    #     # print("length", len(length))

    #     return nodes, segments, radius, length
    
    def _get_dimensions_container_array(self, X, Y, Z):
        """
        creates the dimensions of the container array, which must consist of consecutive values for each pixel
        representing a gray intensity value for the MRI

        Args:
        X, Y, Z: respective coordinate in the root container for which we want to have the gray value
        """
        x_int = np.ceil(X[:-1]/self.resx)
        y_int = np.ceil(Y[:-1]/self.resy)
        z_int = np.round(Z[:-1]/self.resz)
        

        if not (self._isConsecutive(x_int)):
            x_int = np.linspace(np.min(x_int), np.max(x_int), int(abs(np.min(x_int)))+int(np.max(x_int))+1)
        if not (self._isConsecutive(y_int)):
            y_int = np.linspace(np.min(y_int), np.max(y_int), abs(int(np.min(y_int)))+int(np.max(y_int))+1)
        if not (self._isConsecutive(z_int)):
            z_int = np.linspace(0, np.min(z_int), len(z_int)+1)

        return x_int, y_int, z_int

    def _get_root_segment_idx(self, segment, nodes, x_int, y_int, z_int):
        """
        calculates the indexes in the mri grid, where the root segment is located

        Args:
        segment: root segment
        nodes: nodes of the root system
        x_int, y_int, z_int: coordinates of the mri grid
        """
        n1, n2 = nodes[segment.x], nodes[segment.y]
        (x1, y1, z1) = [np.around(n1.x/self.resx), np.around(n1.y/self.resy), np.around(n1.z/self.resz)]
        (x2, y2, z2) = [np.around(n2.x/self.resx), np.around(n2.y/self.resy), np.around(n2.z/self.resz)]
        # contains all points on the segment
        ListOfPoints = np.array(bres3D.Bresenham3D(x1, y1, z1, x2, y2, z2))

        allidx_ = []
        # searches the points in the 3d structure, which correspond to root segments
        for j in range(0, len(ListOfPoints)):
            # ListOfPoints[j,0] is the same as ListOfPoints[j][0]
            xidx = np.where(x_int==ListOfPoints[j,0])
            yidx = np.where(y_int==ListOfPoints[j,1])
            zidx = np.where(z_int==ListOfPoints[j,2])
            # if the point of the segment is in the 3d structure
            if (xidx[0].size > 0 and yidx[0].size > 0 and zidx[0].size > 0):
                a = [int(xidx[0][0]),int(yidx[0][0]),int(zidx[0][0])]
                allidx_.append(a)

        return np.array(allidx_)
    
    def ellipsoid(self, radius_x, radius_y, radius_z, dtype=np.int16):
        """
        Create a 3D ellipsoidal structuring element.
        
        Parameters:
        radius_x, radius_y, radius_z: Radii along the x, y, and z axes.
        dtype: Desired data type of the output (default is np.int16).
            
        Returns:
            - A 3D numpy array representing the ellipsoidal structuring element.
        """
        # Create a 3D grid of coordinates
        x, y, z = np.ogrid[-radius_x:radius_x+1, -radius_y:radius_y+1, -radius_z:radius_z+1]
        
        # Evaluate the ellipsoidal equation for each coordinate
        ellipsoid = (x/radius_x)**2 + (y/radius_y)**2 + (z/radius_z)**2 <= 1
        
        return ellipsoid.astype(dtype)
    
    def _get_binary_delation_root_segment_idx(self, grid, radius):
        """
        Expands the coordinates where the root segment is located by its radius and returns the indexes of the
        expanded coordinates

        Args:
        array: 3d array of the root container with dimensions (nx, ny, nz)
        radius: radius of the root segment
        """
        # if radius < 0.1: # TODO: Check if required
        #     radius = 0.1
        width = int(np.around(radius/self.resx))
        height = int(np.around(radius/self.resz))
        selem = self.ellipsoid(width, width, height)
        # Czero1 contains only 0 and 1 and binary dilation takes the struct and performs dilation on the 1s which
        # are on in Czero1. So in this case a ball is put around all ones and the zeros are set to one, when they
        # are in the ball. 
        grid = ndimage.binary_dilation(grid, structure = selem).astype(grid.dtype)
        grid = np.reshape(grid, (self.nx*self.ny*self.nz))
        idx = np.where(grid==1)

        return idx
    
    def _add_noise_to_grid(self, grid, water_intensity_grid):
        """
        Adds gaussian and perlin noise to the MRI and scales it according to the water saturation of the soil

        Args:
        grid: 3d (numpy) array of the root container with dimensions (nx, ny, nz)
        water_intensity_grid: 3d (numpy) array of the water intensity (nx, ny, nz)
        """
        ##### Gaussian Noise #####
        # calculation of the power signal but since all MRIs are very similar, calculating it, doesn't
        # really add value and just makes it more complicated, so the Power noise (pn) is just set to 
        # a fixed value
        # ps = np.sum((grid - np.mean(grid)) ** 2) / (grid).size
        pn = 8*100000000
        sigma = np.sqrt(pn)

        # gaussian noise is added to the grid and then the grid cut off at min and max grey value intensity
        grid_noise = self._add_gaussian_noise(grid, sigma, water_intensity_grid)
        grid_noise = np.clip(grid_noise, 0, self.max_signal_intensity).astype(np.int16)

        ##### Perlin Noise #####
        grid_noise = self._add_perlin_noise(grid_noise, water_intensity_grid)
        grid_noise = np.clip(grid_noise, 0, self.max_signal_intensity).astype(np.int16)

        return grid_noise
    
    
    def _get_grid_water_content(self, X, Y, Z):
        """
        scales the color intensity of the grid by the water content of the soil

        Args:
        X, Y, Z: respective coordinate in the root container for which we want the water simulation value
        grid_values: 3d (numpy) array of the root container with dimensions (nx, ny, nz)
        """
        
        interpolator = VTUInterpolatedGrid(self.vtu_path, resolution=[self.resx, self.resy, self.resz])

        _, grid_data = interpolator.process_and_visualize(interpolating_coords = [X[:-1], Y[:-1], Z[:-1]])

        water_intensity_grid = np.nan_to_num(grid_data, nan=0)
        water_intensity_grid = water_intensity_grid.reshape(self.nx, self.ny, self.nz)

        # previously used to check, if the water intensity grid scaling is working because the difference
        # in the saturation of the soil with water is very small
        # scaled_arr = (grid_data - min_gd) / (max_gd - min_gd)

        return water_intensity_grid
    
    def _print_progress_bar(self, iteration, total, info="", bar_length=50):
        """
        prints a progress bar to the console, which is updated with each iteration

        Args:
        iteration: current iteration
        total: total number of iterations
        info: additional information to be printed
        bar_length: length of the progress bar
        """
        progress = (iteration / total)
        arrow = '=' * int(round(progress * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        
        sys.stdout.write(f"\rProgress: [{arrow + spaces}] {int(progress*100)}% {info}")
        sys.stdout.flush()  # This is important to ensure the progress is updated
    
    def _add_root_to_v_mri(self, mri_grid, xx, yy, zz, root_signal_intensity):
        """
        adds the root to the MRI grid by adding a white color, where the root is located and a light
        grey depending on how much of a root segment is present in a cell. 

        Args:
        mri_grid: 3d (numpy) array of the root container with dimensions (nx, ny, nz)
        xx, yy, zz: respective coordinate in the root container
        """

        nodes = self.segana.nodes
        segs = self.segana.segments
        radius = self.segana.getParameter("radius")

        idxrad = np.argsort(radius)

        cellvol = self.resx*self.resy*self.resz

        iteration = 1
        total_segs = len(segs)
        for k, _ in enumerate(segs):
            # The list allidx will eventually contain the discretized 3D indices in the grid for all the points 
            # along the segment. This part of simulation/visualization is discretizing the root system into a 3D grid, 
            # and allidx_ is helping you keep track of which grid cells are occupied by the segment.
            allidx = self._get_root_segment_idx(segs[idxrad[k]], nodes, xx, yy, zz)

            self._print_progress_bar(iteration, len(segs), info="Adding root segment {} of {}".format(iteration, total_segs))

            mri_grid_zero = np.zeros((self.nx, self.ny, self.nz))
            # checks if the diameter is greater than the resolution
            if (np.round(radius[idxrad[k]]*2/self.resx)>1):
                if (len(allidx)):
                    # set the element of the root to 1, indicating that it is present
                    mri_grid_zero[allidx[:,0], allidx[:,1], allidx[:,2]] = 1
                mri_grid_segment = mri_grid_zero
                # idx contains the indices of the binary dilation across the root segment
                idx = self._get_binary_delation_root_segment_idx(mri_grid_segment, radius[idxrad[k]])

                mri_grid[idx[0]] = root_signal_intensity
            else:
                estlen = self.resx # estimated segment length within voxel: very rough estimation
                rootvol = radius[idxrad[k]]**2*math.pi*estlen
                frac = rootvol/cellvol
                if frac > 1:
                    frac = 1
                if (len(allidx)):
                    mri_grid_zero[allidx[:,0], allidx[:,1], allidx[:,2]] = 1

                # set root voxels to the appropriate value
                mri_grid_zero = np.reshape(mri_grid_zero, (self.nx*self.ny*self.nz))
                idx = np.where(mri_grid_zero==1)
                mri_grid[idx[0]] = int(np.floor(frac*self.max_signal_intensity))

            iteration += 1

        print("\n" + "\033[33m" + # Green text
          "====================================================" + "\n" +
          "||        Adding root segments: COMPLETE!         ||" + "\n" +
          "====================================================" +
          "\033[0m")  # Reset text color

        
    def _get_avg_numpy_array_non_zero(self, array):
        """
        calculates the average of the non zero values of a numpy array

        Args:
        array: numpy array
        """
        non_zero = array[array != 0]
        if non_zero.size == 0:
            return 0
        else:
            return np.mean(non_zero)

    def create_virtual_root_mri(self, mri_output_path, add_noise=True):
        """
        creates a virtual MRI for the given root system with simulating noise based on the water content of the soil.

        Args:
        mri_output_path: path to the folder, where the generated virtual MRI should be saved
        """

        X = np.linspace(-1*self.width, -1*self.width+self.nx*self.resx, self.nx+1)
        Y = np.linspace(-1*self.width, -1*self.width+self.ny*self.resy, self.ny+1)
        Z = np.linspace(0, 0-self.nz*self.resz, self.nz+1)

        water_grid = self._get_grid_water_content(X, Y, Z)
        avg_water_content = self._get_avg_numpy_array_non_zero(water_grid)
        avg_water_percent = int(avg_water_content*100)

        # set the root signal strength depending on the water content of the soil but with some random noise
        # maximum value is 30000 and min 5000
        random_value = random.uniform(0, 0.2)

        # water_scale = (random.randint(avg_water_percent, avg_water_percent+20))/100
        # root_signal_strength = water_scale * self.max_root_signal_intensity
        # root_signal_strength = max(root_signal_strength, 5000)
        # root_signal_strength = min(root_signal_strength, 30000)
        # print("root_signal_strength", root_signal_strength)

        xx, yy, zz = self._get_dimensions_container_array(X, Y, Z)
        mri_grid = np.zeros((self.nx*self.ny*self.nz))

        self._add_root_to_v_mri(mri_grid, xx, yy, zz, self.max_signal_intensity)

        self.root_idx = np.where(mri_grid != 0, 1, 0)

        root_idx_grid = np.reshape(self.root_idx, (self.nx, self.ny, self.nz))
        mri_grid = np.reshape(mri_grid, (self.nx, self.ny, self.nz))
        #np.multiply(mri_grid, water_grid+random_value, out=mri_grid)

        # add noise scaled by the water content in the soil
        if add_noise:
            mri_final_grid = self._add_noise_to_grid(mri_grid, water_grid)
        else:
            mri_final_grid = mri_grid
        
        # mri_grid_noise = np.full((self.nx, self.ny, self.nz), 255)
        
        # save the mri grid as a raw file (same as the original MRIs)
        mri_final_grid = np.swapaxes(mri_final_grid, 0, 2)
        mri_final_grid = mri_final_grid[::-1]
        root_system_name = self.rsml_path.split('/')[-1].split('.')[0]
        filename = mri_output_path+'/'+root_system_name+'_SNR_'+str(self.snr)+'_res_'+str(self.nx)+'x'+str(self.ny)+'x'+str(self.nz)
        mri_final_grid.astype('int16').tofile(filename+".raw")
        print(filename)

        # create a file containing the root idx (label where the root is present in the original MRI)
        root_idx_grid = np.swapaxes(root_idx_grid, 0, 2)
        root_idx_grid = root_idx_grid[::-1]
        root_idx_grid.astype('int16').tofile(f"{filename}_root_idx"+".raw")

        return mri_final_grid, filename, root_idx_grid

# Example usage:
# rssim = RSSim("Anagallis_femina_Leitner_2010", "../../../../data/generated/root_systems", 3, 20)
# anas, filenames = rssim.run_simulation([10, 11])

# for i in range(len(anas)):
#     my_vi = Virtual_MRI(anas[i], "../../../data/generated/{}".format(filenames[i]), "../../../../soil_simulation_data/generated_20_-90.8_7.1.vtu", "../../../data/generated/virtual_mri")
#     my_vi.create_virtual_root_mri()
