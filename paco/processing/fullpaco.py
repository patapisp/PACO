from .. import core
from paco.util.util import *
from .paco import PACO
import matplotlib.pyplot as plt
import sys

import multiprocessing as mp

class FullPACO(PACO):
    def __init__(self,                 
                 patch_size = 5,
                 file_name = None,
                 directory = None):
        self.filename = file_name
        self.directory = directory
        self.FitsInput = None
        self.im_stack = []
        self.k = int(patch_size) # defaults to paper value
        self.T = -1
        self.scale = 1
        self.angles = []
        self.x_grid = None
        self.y_grid = None
        self.model_name = ""
        return
    """
    Algorithm Functions
    """
    
    def PACO(self,angles, scale = 1,model_name=gaussian2d_model):
        """
        PACO

        This function wraps the actual PACO algorithm, setting up the pixel coordinates 
        that will be iterated over. The output will probably be changes to output the
        true SNR map.

        :angles: Array of angles from frame rotation
        :resolution: Amount of oversampling of image to improve positioning of PSF (don't use yet)
        """
        print("Running PACO...")
        if scale != 1:
            self.rescale_image_sequence(scale)
        # Setup pixel coordinates
        self.x_grid,self.y_grid = np.meshgrid(np.arange(0,int(scale * self.im_stack.shape[1])),
                                              np.arange(0,int(scale * self.im_stack.shape[2])))

        phi0s = np.column_stack((self.x_grid.flatten(),self.y_grid.flatten()))

        self.angles = angles
        self.scale = scale
        self.model_name = model_name
        npx = len(phi0s)  # Number of pixels in an image
        try:
            assert npx == self.im_stack.shape[1]**2
        except AssertionError:
            print("Position grid does not match pixel grid.")
            
        # Multiprocessing
        pool = mp.Pool(4)
        result = np.array(pool.map(self.PACO_calc,phi0s))
        print(result)
        print("Done")
        a = result[:,0]
        b = result[:,1]
        
        # Reshape into a 2D image, with the same dimensions as the input images
        a = np.reshape(a,(self.im_stack.shape[1],self.im_stack.shape[2]))
        b = np.reshape(b,(self.im_stack.shape[1],self.im_stack.shape[2]))
        return a,b

    def PACO_calc(self, p0):
        """
        PACO_calc
        
        This function iterates of a list of test points (phi0) and a list
        of angles between frames to produce 'a' and b', which can be used to
        generate a signal to noise map where SNR = b/sqrt(a) at each pixel.

        :phi0s: Array of pixel locations to estimate companion position
        :angles: Array of angles from frame rotation
        :model_name: Name of the template for the off-axis PSF
        """
        N = self.im_stack.shape[1] # Length of an image axis (assume a square image)                
        dim = (N/2)
        T = self.T
        k = self.k * self.scale
        
        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patch = np.zeros((T,T,2*k,2*k)) # a patch is a small, 2d selection of pixels around a given point
        m     = np.zeros((T,2*k,2*k)) # the mean of a temporal column of patches at each pixel
        Cinv  = np.zeros((T,4*k*k,4*k*k)) # the inverse covariance matrix at each point
        h_template = self.model_function(2*self.k,self.model_name,sigma=5)
        h = np.zeros((T,2*k,2*k)) # The off axis PSF at each point

        # Angle Calculations
        phi0 = np.array([self.x_grid[p0[0], p0[1]], self.y_grid[p0[0], p0[1]]])
        angles_px = self.find_angles_px(phi0,dim)
        
        # Iterate over each temporal frame/each angle
        # Same as iterating over phi_l
        for i,ang in enumerate(angles_px):
            patch[i] = self.get_patch(ang, k) # Get the column of patches at this point
            m[i] = np.mean(patch[i], axis=0) # Calculate the mean of the column
            # Calculate the covariance matrix
            S = self.sample_covariance(patch[i], m[i], T)
            rho = self.shrinkage_factor(S, T) 
            F = self.diag_sample_covariance(S)
            C = self.covariance(rho, S, F)
            Cinv[i] = np.linalg.inv(C)
            if self.scale!=1:
                h[i] = resizeImage(h_template,self.scale)
            else:
                h[i] = h_template
                
        # Calculate a and b values for this pixel.
        a = np.sum(self.al(h, Cinv),axis=0)
        b = np.sum(self.bl(h, Cinv, patch, m), axis=0)
        return np.array([a,b])

    def find_angles_px(self,phi0,dim):
        # Convert to polar coordinates
        rphi0 = cart_to_pol(phi0)
        angles_rad = rphi0[1] - np.array([a*np.pi/180 for a in self.angles]) 
        
        # Rotate the polar coordinates by each frame angle
        angles_ind = [[rphi0[0],phi] for phi in angles_rad]
        angles_pol = np.array(list(zip(*angles_ind)))
        
        # Convert from polar to cartesian and pixel coordinates
        angles_px = np.array(grid_pol_to_cart(angles_pol[0], angles_pol[1]))+dim
        angles_px = angles_px.T
        angles_px = np.fliplr(angles_px)
        return angles_px
