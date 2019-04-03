"""
This module implements ALGORITHM 1 from Flasseur et al.
It uses patch covariance to determine the signal to noise
ratio of a signal within ADI image data.
"""
from paco.util.util import *
from .paco import PACO
import matplotlib.pyplot as plt
import sys

class FullPACO(PACO):
    def __init__(self,                 
                 image_stack = None
                 angles = None
                 psf = None
                 patch_size = 49):   
        """
        FastPACO Parent Class Constructor
        Parameters
        -----------------------------
        image_stack : arr
            Array of 2D science frames taken in pupil tracking/ADI mode
        angles : arr
            List of differential angles between each frame.
        psf : arr
            2D PSF image
        patch_size : int
            Number of pixels contained in a circular patch. Typical  values 13,49,113
        """
        self.m_im_stack = np.array(image_stack)
        self.m_nFrames = 0
        self.m_width = 0
        self.m_height = 0
        if image_stack:
            self.m_nFrames = self.m_im_stack.shape[0]
            self.m_width = self.m_im_stack.shape[2]
            self.m_height = self.m_im_stack.shape[1]
            
        self.m_angles = angles
        self.m_psf = psf
        
        self.m_p_size = patch_size # Number of pixels in a patch
        self.m_psf_rad = np.ceil(np.sqrt(patch_size/np.pi)) # width of a patch
        return
    
    """
    Algorithm Functions
    """     
    def PACOCalc(self, phi0s, params, scale = 1, model_name=gaussian2d_model, cpu = 1):
        """
        PACO_calc
        
        This function iterates of a list of test points (phi0) and a list
        of angles between frames to produce 'a' and b', which can be used to
        generate a signal to noise map where SNR = b/sqrt(a) at each pixel.
        
        phi0s : int arr
            Array of pixel locations to estimate companion position
        params: dict
            Dictionary of parameters about the psf, containing either the width
            of a gaussian distribution, or a label 'psf_template'
        scale : float
            Resolution scaling
        model_name: str
            Name of the template for the off-axis PSF
        cpu : int >= 1
            Number of cores to use for parallel processing. Not yet implemented.
        """  
        npx = len(phi0s)  # Number of pixels in an image
        dim = self.m_width/2
        try:
            assert npx == self.m_width*self.m_height
        except AssertionError:
            print("Position grid does not match pixel grid.")
            sys.exit(1)
        
        a = np.zeros(npx) # Setup output arrays
        b = np.zeros(npx)
        T = len(self.m_im_stack) # Number of temporal frames
        k = int(2*np.ceil(scale * self.m_psf_rad ) + 2) # Width of a patch, just for readability

        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        # 2d selection of pixels around a given point
        patch = np.zeros((self.m_nFrames,self.m_nFrames,self.m_p_size*scale**2))
        self.m_mask =  createCircularMask((k,k),radius = self.m_psf_rad*scale)

        # the mean of a temporal column of patches at each pixel
        m     = np.zeros((self.m_nFrames,self.m_p_size*scale**2))
        # the inverse covariance matrix at each point
        Cinv  = np.zeros((self.m_nFrames,self.m_p_size*scale**2,self.p_size*scale**2))

        if self.m_psf:
            h_template = self.m_psf
        else:
            h_template = self.modelFunction(k, model_name, params)
        h_mask = createCircularMask(h_template.shape,radius = self.m_psf_rad*scale)
        h = np.zeros((self.m_nFrames,self.m_p_size*scale**2)) # The off axis PSF at each point
        print("Running PACO...")
        
        # Set up coordinates so 0 is at the center of the image                     
        x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
        
        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        for i,p0 in enumerate(phi0s):
            if(i%(npx/10) == 0):
                print(str(i/100) + "%")
                
            # Get list of pixels for each rotation angle
            angles_px = getRotatedPixels(x,y,p0,angles)
            
            # Iterate over each temporal frame/each angle
            # Same as iterating over phi_l
            for l,ang in enumerate(angles_px):
                patch[l] = self.getPatch(ang, k, self.m_mask) # Get the column of patches at this point
                m[l],Cinv[l] = self.pixelCalc(self.m_nFrames,self.m_p_size)
                if scale!=1:
                    h[l] = resizeImage(h_template,scale)[h_mask]
                else:
                    h[l] = h_template[h_mask]

            # Calculate a and b, matrices
            a[i] = self.al(h, Cinv)
            b[i] = max(self.bl(h, Cinv, patch, m),0.0)
        print("Done")
        return a,b
  
