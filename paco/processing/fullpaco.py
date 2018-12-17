from .. import core
from paco.util.util import *
from .paco import PACO
import matplotlib.pyplot as plt
import sys

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
        return
    """
    Algorithm Functions
    """
    
    def PACO(self,angles):
        """
        PACO

        This function wraps the actual PACO algorithm, setting up the pixel coordinates 
        that will be iterated over. The output will probably be changes to output the
        true SNR map.

        :angles: Array of angles from frame rotation
        """

        # Setup pixel coordinates
        x,y = np.meshgrid(np.arange(0,self.im_stack.shape[1]),np.arange(0,self.im_stack.shape[2]))
        phi0s = np.column_stack((x.flatten(),y.flatten()))
        # Compute a,b
        a,b,c = self.PACO_calc(np.array(phi0s),angles)

        # Reshape into a 2D image, with the same dimensions as the input images
        a = np.reshape(a,(self.im_stack.shape[1],self.im_stack.shape[2]))
        b = np.reshape(b,(self.im_stack.shape[1],self.im_stack.shape[2]))
        c = np.reshape(c,(self.im_stack.shape[1],self.im_stack.shape[2]))
        return a,b,c
    
    def PACO_calc(self, phi0s, angles, model_name=gaussian2d_model):
        """
        PACO_calc
        
        This function iterates of a list of test points (phi0) and a list
        of angles between frames to produce 'a' and b', which can be used to
        generate a signal to noise map where SNR = b/sqrt(a) at each pixel.

        :phi0s: Array of pixel locations to estimate companion position
        :angles: Array of angles from frame rotation
        :model_name: Name of the template for the off-axis PSF
        """
        N = np.shape(self.im_stack[0])[0] # Length of an image axis (assume a square image)
        npx = len(phi0s)  # Number of pixels in an image
        a = np.zeros(npx) # Setup output arrays
        b = np.zeros(npx)
        c = np.zeros(npx)
        
        T = len(self.im_stack) # Number of temporal frames
        k = self.k             # Half-width of a patch

        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patch = np.zeros((npx,T,T,2*k,2*k)) # a patch is a small, 2d selection of pixels around a given point
        m     = np.zeros((npx,T,2*k,2*k)) # the mean of a temporal column of patches at each pixel
        Cinv  = np.zeros((npx,T,4*k*k,4*k*k)) # the inverse covariance matrix at each point
        h     = np.zeros((npx,T,2*k,2*k)) # The off axis PSF at each point
        print("Running PACO...")
        # Set up coordinates so 0 is at the center of the image                   
        dim = (N/2)
        x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
        r,theta = grid_cart_to_pol(x,y)
        theta = (theta + np.pi)
        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        for i,p0 in enumerate(phi0s):
            if(i%1000 == 0):
                print(str(i/100) + "%")
            # Current pixel
            phi0 = np.array([x[p0[0], p0[1]], y[p0[0], p0[1]]])
            # Convert to polar coordinates
            rphi0 = cart_to_pol(phi0)
            rphi0[1] = rphi0[1]

            # Convert the input angles from degrees to radians
            angles_rad = rphi0[1] - np.array([a*np.pi/180 for a in angles]) 
    
            # Rotate the polar coordinates by each frame angle
            angles_ind = [[rphi0[0],phi] for phi in angles_rad]
            angles_pol = np.array(list(zip(*angles_ind)))

            # Convert from polar to cartesian and pixel coordinates
            angles_px = np.array(grid_pol_to_cart(angles_pol[0], angles_pol[1]))+dim
            #plt.imshow(self.im_stack[0])
            #plt.plot(angles_px[0], angles_px[1], 'rx')
            #sys.exit(1)
            # Transpose to tuples
            angles_px = angles_px.T

            # Iterate over each temporal frame/each angle
            # Same as iterating over phi_l
            for l,ang in enumerate(angles_px):
                patch[i][l] = self.get_patch(ang, k) # Get the column of patches at this point
                m[i][l] = np.mean(patch[i][l], axis=0) # Calculate the mean of the column

                # Calculate the covariance matrix
                S = self.sample_covariance(patch[i][l], m[i][l], T) 
                rho = self.shrinkage_factor(S, T) 
                F = self.diag_sample_covariance(S)
                C = self.covariance(rho, S, F)
                Cinv[i][l] = np.linalg.inv(C)
                # Setup the model
                h[i][l] = self.model_function(2*k,model_name,sigma=5)

            # At this location, calculate a and b    
            a[i] = np.sum(self.al(h[i], Cinv[i]),axis=0)
            b[i] = np.sum(self.bl(h[i], Cinv[i], patch[i], m[i]), axis=0)
            c[i] = i
        print("Done")
        return a,b,c

    def plotting(self,patch, vmax = 5):
        # Just a temp function, ignore
        print(patch.shape)
        dim = len(patch)
        n = int(np.ceil(np.sqrt(dim)))
        m = int(np.ceil(dim/n))
        fig, ax = plt.subplots(nrows=m, ncols=n, figsize=(8,6))
        ax = ax.flatten()
        #ax = ax[:dim+1]
        for i in range(dim):
            ax[i].imshow(patch[i], vmax=vmax)
        return    
