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
    """
    Algorithm Functions
    """     
    def PACOCalc(self,
                 phi0s,
                 cpu = 1):
        """
        PACOCalc
        
        This function iterates of a list of test points (phi0) and a list
        of angles between frames to produce 'a' and b', which can be used to
        generate a signal to noise map where SNR = b/sqrt(a) at each pixel.
        
        phi0s : int arr
            Array of pixel locations to estimate companion position
        cpu : int >= 1
            Number of cores to use for parallel processing. Not yet implemented.
        """  
        npx = len(phi0s)  # Number of pixels in an image
        dim = self.m_width/2
        
        a = np.zeros(npx) # Setup output arrays
        b = np.zeros(npx)
        T = len(self.m_im_stack) # Number of temporal frames
        mask = createCircularMask(self.m_psf.shape,radius = self.m_psf_rad)
        if self.m_p_size != len(mask[mask]):
            self.m_p_size = len(mask[mask])      
        h = np.zeros((self.m_nFrames,self.m_psf_area)) # The off axis PSF at each point
        
        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        # 2d selection of pixels around a given point
        patch = np.zeros((self.m_nFrames,self.m_nFrames,self.m_psf_area))

        # the mean of a temporal column of patches at each pixel
        m = np.zeros((self.m_nFrames,self.m_psf_area))
        # the inverse covariance matrix at each point
        Cinv = np.zeros((self.m_nFrames,self.m_psf_area,self.m_psf_area))


        print("Running Full PACO...")
        
        # Set up coordinates so 0 is at the center of the image                     
        x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
        
        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        for i,p0 in enumerate(phi0s):
            # Get list of pixels for each rotation angle
            angles_px = getRotatedPixels(x,y,p0,self.angles)
            
            # Iterate over each temporal frame/each angle
            # Same as iterating over phi_l
            for l,ang in enumerate(angles_px):
                patch[l] = self.getPatch(ang, self.m_pwidth, mask) # Get the column of patches at this point
                m[l],Cinv[l] = pixelCalc(patch[l])
                h[l] = self.m_psf[mask]

            # Calculate a and b, matrices
            a[i] = self.al(h, Cinv)
            b[i] = self.bl(h, Cinv, patch, m)
        print("Done")
        return a,b
  
