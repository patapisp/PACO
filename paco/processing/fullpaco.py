"""
This module implements ALGORITHM 1 from Flasseur et al.
It uses patch covariance to determine the signal to noise
ratio of a signal within ADI image data.
"""
from .. import core
from paco.util.util import *
from .paco import PACO
import matplotlib.pyplot as plt
import sys

class FullPACO(PACO):
    def __init__(self,                 
                 patch_size = 49,
                 file_name = None,
                 directory = None):
        self.filename = file_name
        self.directory = directory
        self.FitsInput = None
        self.im_stack = []
        self.p_size = int(patch_size) # Number of pixels in a patch
        self.psf_rad = int(np.ceil(np.sqrt(patch_size/np.pi))) # width of a patch
        return
    
    """
    Algorithm Functions
    """   
    def PACO(self,angles, params, scale = 1, model_name=gaussian2d_model):
        """
        PACO
        This function wraps the actual PACO algorithm, setting up the pixel coordinates 
        that will be iterated over. The output will probably be changes to output the
        true SNR map.
        :angles: Array of angles from frame rotation
        :resolution: Amount of oversampling of image to improve positioning of PSF (don't use yet)
        """
        if scale != 1:
            print("Rescaling")
            self.rescale_image_sequence(scale)
            print(self.im_stack.shape)
        # Setup pixel coordinates
        x,y = np.meshgrid(np.arange(0,int(self.im_stack.shape[1])),
                          np.arange(0,int(self.im_stack.shape[2])))
        phi0s = np.column_stack((x.flatten(),y.flatten()))
        # Compute a,b
        a,b = self.PACO_calc(np.array(phi0s),angles, params, scale, model_name)
        # Reshape into a 2D image, with the same dimensions as the input images
        a = np.reshape(a,(self.im_stack.shape[1],self.im_stack.shape[2]))
        b = np.reshape(b,(self.im_stack.shape[1],self.im_stack.shape[2]))
        return a,b
    
    def PACO_calc(self, phi0s, angles, params, scale = 1, model_name=gaussian2d_model):
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
        npx = len(phi0s)  # Number of pixels in an image
        dim = (N/2)
        try:
            assert npx == N**2
        except AssertionError:
            print("Position grid does not match pixel grid.")
            sys.exit(1)
        
        a = np.zeros(npx) # Setup output arrays
        b = np.zeros(npx)
        T = len(self.im_stack) # Number of temporal frames
        k = int(2*np.ceil(scale * self.psf_rad ) + 2) # Width of a patch, just for readability

        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patch = np.zeros((T,T,self.p_size*scale**2)) # 2d selection of pixels around a given point
        mask =  createCircularMask((k,k),radius = self.psf_rad*scale)
        
        m     = np.zeros((N,N,self.p_size*scale**2)) # the mean of a temporal column of patches at each pixel
        Cinv  = np.zeros((N,N,self.p_size*scale**2,self.p_size*scale**2)) # the inverse covariance matrix at each point

        h_template = self.model_function(k,model_name, params)
        h_mask = createCircularMask(h_template.shape,radius = self.psf_rad*scale)
        h = np.zeros((N,N,self.p_size*scale**2)) # The off axis PSF at each point
        print("Running PACO...")
        
        # Set up coordinates so 0 is at the center of the image                     
        x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
        
        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        for i,p0 in enumerate(phi0s):
            if(i%1000 == 0):
                print(str(i/100) + "%")
                
            # Get list of pixels for each rotation angle
            angles_px = getRotatedPixels(x,y,p0,angles)
            
            # Iterate over each temporal frame/each angle
            # Same as iterating over phi_l
            for l,ang in enumerate(angles_px):
                patch[l] = self.get_patch(ang, k, mask) # Get the column of patches at this point
                m[l] = np.mean(patch[l], axis=0) # Calculate the mean of the column
                # Calculate the covariance matrix
                S = self.sample_covariance(patch[l], m[l], T)
                rho = self.shrinkage_factor(S, T)
                F = self.diag_sample_covariance(S)
                C = self.covariance(rho, S, F)
                Cinv[l] = np.linalg.inv(C)
                if scale!=1:
                    h[l] = resizeImage(h_template,scale)[h_mask]
                else:
                    h[l] = h_template[h_mask]

            # Calculate a and b, matrices
            a[i] = self.al(h, Cinv)
            b[i] = max(self.bl(h, Cinv, patch, m),0.0)
        print("Done")
        return a,b
  
