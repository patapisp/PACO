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
    
    def PACO(self,angles, scale = 1, model_name=gaussian2d_model):
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
        a,b = self.PACO_calc(np.array(phi0s),angles,scale, model_name)
        # Reshape into a 2D image, with the same dimensions as the input images
        a = np.reshape(a,(self.im_stack.shape[1],self.im_stack.shape[2]))
        b = np.reshape(b,(self.im_stack.shape[1],self.im_stack.shape[2]))
        return a,b
    
    def PACO_calc(self, phi0s, angles, scale = 1, model_name=gaussian2d_model):
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
        #try:
        #    assert npx == N**2
        #except AssertionError:
        #    print("Position grid does not match pixel grid.")
        #    sys.exit(1)
        
        a = np.zeros(npx) # Setup output arrays
        b = np.zeros(npx)
        T = len(self.im_stack) # Number of temporal frames
        k = int(np.ceil(scale * self.k ))            # Half-width of a patch, just for readability

        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patch = np.zeros((T,T,2*k,2*k)) # a patch is a small, 2d selection of pixels around a given point
        m     = np.zeros((T,2*k,2*k)) # the mean of a temporal column of patches at each pixel
        Cinv  = np.zeros((T,4*k*k,4*k*k)) # the inverse covariance matrix at each point
        h_template = self.model_function(2*k,model_name,sigma=3)
        h = np.zeros((T,2*k,2*k)) # The off axis PSF at each point
        print("Running PACO...")
        # Set up coordinates so 0 is at the center of the image                   
        dim = (N/2)
        x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        for i,p0 in enumerate(phi0s):
            if(i%1000 == 0):
                print(str(i/100) + "%")
            # Current pixel
            phi0 = np.array([x[p0[0], p0[1]], y[p0[0], p0[1]]])
            # Convert to polar coordinates
            rphi0 = cart_to_pol(phi0)
            angles_rad = rphi0[1] - np.array([a*np.pi/180 for a in angles]) 
    
            # Rotate the polar coordinates by each frame angle
            angles_ind = [[rphi0[0],phi] for phi in angles_rad]
            angles_pol = np.array(list(zip(*angles_ind)))

            # Convert from polar to cartesian and pixel coordinates
            angles_px = np.array(grid_pol_to_cart(angles_pol[0], angles_pol[1]))+dim
            angles_px = angles_px.T
            angles_px = np.fliplr(angles_px)

            # Iterate over each temporal frame/each angle
            # Same as iterating over phi_l
            for l,ang in enumerate(angles_px):
                patch[l] = self.get_patch(ang, k) # Get the column of patches at this point
                m[l] = np.mean(patch[l], axis=0) # Calculate the mean of the column
                # Calculate the covariance matrix

                #mfig,mfax = plt.subplots(ncols = 2,figsize = (12,8))
                #mfax = mfax.flatten()
                #mimg = mfax[0].imshow(m[l])
                #mimg1 = mfax[1].imshow(patch[l][l])
                #mfig.colorbar(mimg,ax = mfax[0])
                #mfig.colorbar(mimg1,ax = mfax[1])
                #mfig.suptitle("Patch and mean plots")
                S = self.sample_covariance(patch[l], m[l], T)
                rho = self.shrinkage_factor(S, T)
                #fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(12,8))
                #fig.suptitle("Covariance Plots l = " + str(l) + ", rho= " + str(float(rho)))
                #ax = ax.flatten()
                #im = ax[0].imshow(S)

                F = self.diag_sample_covariance(S)
                C = self.covariance(rho, S, F)
                Cinv[l] = np.linalg.inv(C)
                #fig.colorbar(im,ax = ax[0])
                #im0 = ax[1].imshow(F)
                #fig.colorbar(im0,ax = ax[1])
                #im1 = ax[2].imshow(C)
                #fig.colorbar(im1,ax = ax[2])
                #im2 = ax[3].imshow(Cinv[l])
                #fig.colorbar(im2,ax = ax[3])
                #print(np.min(Cinv[l]))
                if scale!=1:
                    h[l] = resizeImage(h_template,scale)
                else:
                    h[l] = h_template

            # Calculate a and b, matrices
            a[i] = np.sum(self.al(h, Cinv),axis=0)
            b[i] = max(np.sum(self.bl(h, Cinv, patch, m), axis=0),0.0)
        print("Done")
        return a,b
  
