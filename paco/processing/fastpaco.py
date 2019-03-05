"""
This module implements ALGORITHM 2 from the PACO paper. 
Covariance and mean statistics are computed and stored 
before running the algorithm, but do not have subpixel
accuracy.
"""
from .. import core
from paco.util.util import *
from .paco import PACO
from multiprocessing import Process
import matplotlib.pyplot as plt
import sys

class FastPACO(PACO):
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
    def PACO(self,angles, params,scale = 1, model_name=gaussian2d_model):
        """
        PACO
        This function wraps the actual PACO algorithm, setting up the pixel coordinates 
        that will be iterated over. The output will probably be changes to output the
        true SNR map.
        :angles: Array of angles from frame rotation
        :resolution: Amount of oversampling of image to improve positioning of PSF (don't use yet)
        """
        if scale != 1:
            self.rescale_image_sequence(scale)
        # Setup pixel coordinates
        x,y = np.meshgrid(np.arange(0,int(scale * self.im_stack.shape[1])),
                          np.arange(0,int(scale * self.im_stack.shape[2])))
        phi0s = np.column_stack((x.flatten(),y.flatten()))
        # Compute a,b
        a,b = self.PACO_calc(np.array(phi0s),angles, params,scale, model_name)
        # Reshape into a 2D image, with the same dimensions as the input images
        a = np.reshape(a,(self.im_stack.shape[1],self.im_stack.shape[2]))
        b = np.reshape(b,(self.im_stack.shape[1],self.im_stack.shape[2]))
        return a,b


    def compute_statistics(self, phi0s, params, scale = 1, model_name=gaussian2d_model):
        """
        compute_statistics
        
        This function computes the mean and inverse covariance matrix for
        each patch in the image stack.

        :phi0s: Array of pixel locations to estimate companion position
        :model_name: Name of the template for the off-axis PSF
        """
    
        print("Precomputing Statistics...")
        N = self.im_stack.shape[1] # Length of an image axis (assume a square image)
        npx = len(phi0s)           # Number of pixels in an image      
        dim = int(N/2)
        T = len(self.im_stack)            # Number of temporal frames
        k = int(np.ceil(scale * self.k )) # Half-width of a patch, just for readability
        
        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patch = np.zeros((T,2*k,2*k)) # 2d selection of pixels around a given point
        m     = np.zeros((N,N,2*k,2*k)) # the mean of a temporal column of patches at each pixel
        Cinv  = np.zeros((N,N,4*k*k,4*k*k)) # the inverse covariance matrix at each point
        h_template = self.model_function(2*k,model_name, params)
        h = np.zeros((N,N,2*k,2*k)) # The off axis PSF at each point
        
        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        for i,p0 in enumerate(phi0s):
            if(i%1000 == 0):
                print(str(i/100) + "%")
            # Current pixel
            patch = self.get_patch(p0, k) # Get the column of patches at this point
            if patch is None:
                continue
            m[p0[0]][p0[1]] = np.mean(patch, axis=0) # Calculate the mean of the column

            # Calculate the covariance matrix
            S = self.sample_covariance(patch, m[p0[0]][p0[1]], T)
            rho = self.shrinkage_factor(S, T) 
            F = self.diag_sample_covariance(S)
            C = self.covariance(rho, S, F)
            
            Cinv[p0[0]][p0[1]] = np.linalg.inv(C)
            if scale!=1:
                h[p0[0]][p0[1]] = resizeImage(h_template,scale)
            else:
                h[p0[0]][p0[1]] = h_template
        return Cinv,m,h

    def PACO_calc(self, phi0s, angles,  params,  scale = 1, model_name=gaussian2d_model):
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
        k = int(np.ceil(scale * self.k ))            # Half-width of a patch, just for readability

        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patch = np.zeros((T,T,2*k,2*k))#a patch is a small, 2d selection of pixels around a given point
        h_template = self.model_function(2*k,model_name,  params)
        h = np.zeros((T,2*k,2*k)) # The off axis PSF at each point 
        # Set up coordinates so 0 is at the center of the image                   
        x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
        Cinv,m,h = self.compute_statistics(phi0s,  params, scale = 1, model_name=model_name)
        
        print("Running PACO...")
        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        for i,p0 in enumerate(phi0s):
            if(i%1000 == 0):
                print(str(i/100) + "%")

            # Get Angles
            angles_px = GetRotatedPixels(x,y,p0,angles)

            # Ensure within image bounds
            if(int(np.max(angles_px.flatten()))>=N or int(np.min(angles_px.flatten()))<0):
                a[i] = np.nan
                b[i] = np.nan
                continue

            # Extract relevant patches and statistics
            Cinlst = []
            mlst = []
            hlst = []
            for l,ang in enumerate(angles_px):
                Cinlst.append(Cinv[int(ang[0])][int(ang[1])])
                mlst.append(m[int(ang[0])][int(ang[1])])
                hlst.append(h[int(ang[0])][int(ang[1])])
                patch[l] = self.get_patch(ang, k)
            Cinlst = np.array(Cinlst)
            mlst   = np.array(mlst)
            hlst   = np.array(hlst)

            #print(Cinlst.shape,mlst.shape,hlst.shape,a.shape,patch.shape)
            # Calculate a and b, matrices
            a[i] = np.sum(self.al(hlst, Cinlst), axis=0)
            b[i] = max(np.sum(self.bl(hlst, Cinlst, patch, mlst), axis=0),0.0)
        print("Done")
        return a,b
  
