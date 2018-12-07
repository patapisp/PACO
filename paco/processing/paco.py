"""
This file will implement AGORITHM 1 from the PACO paper
"""
from .. import core.ReadInFitsFile
from ..util import *

class PACO:
    def __init__(self,
                 file_name = None,
                 directory = None):
        self.FitsInput
        self.im_stack = []
        self.k = 49 # defaults to paper value
        return

    """
    Utility Functions
    """    
    def set_patch_size(self,npx):
        self.k = npx

    def set_image_sequence(self,imgs):
        self.im_stack = imgs

    def get_image_sequence(self):
        return self.im_stack

    def open_fits_images(self,file_name, directory):
        """
        Read in the fits file
        """
        self.FitsInput = ReadInFitsFile.ReadInFitsFile(file_name,directory)
        self.FitsInput.open_one_fits(file_name)
        self.im_stack = FitsInput.images
        
    def create_circular_mask(self,w, h, center=None, radius=None):
        """
        Returns a 2D boolean mask given some radius and location
        :w: width, number of x pixels
        :h: height, number of y pixels
        :center: [x,y] pair of pixel indices denoting the center of the mask
        :radius: radius of mask
        """
        if center is None: 
            center = [int(w/2), int(h/2)]
        if radius is None:
            radius = min(center[0], center[1], w-center[0], h-center[1])
        X, Y = np.ogrid[:w, :h]
        dist2 = (X - center[0])**2 + (Y-center[1])**2
        mask = d2 <= radius**2
        return mask

    def get_patch(self,px, k):
        """
        gets patch at given pixel px with size k for the current img sequence
        """
        radius = int(np.sqrt(k))
        nx, ny = np.shape(self.im_stack[0])[:2]
        #if px[0]+k > nx or px[0]-k < 0 or px[1]+k > ny or px[1]-k < 0:
        #    print("pixel out of range")
        #    return None
        #patch = [self.im_stack[i][px[0]-k:px[0]+k, px[1]-k:px[1]+k] for i in range(len(self.im_stack))]
        mask = self.create_circular_mask(nx,ny,px,radius)        
        patch= [self.im_stack[i][np.where(mask)] for i in range(len(self.im_stack))]    
        return patch

    """
    Math Functions
    """
    def model_function(self,n, model, **kwargs):
        """
        h_θ

        n: mean
        model: numpy statistical model
        **kwargs: additional arguments for model
        """
        return model(n, **kwargs)

    def background_covariance(self,rho, S, F):
        """
        Ĉ

        Shrinkage covariance matrix
        rho: shrinkage weight
        S: Sample covariance matrix
        F: Diagonal of sample covariance matrix
        """
        return (1-rho)*S + rho*F
    
    def sample_covariance(self,r, m, T):
        """
        Ŝ

        Sample covariance matrix
        r: obcserved intensity at position θk and time tl
        m: mean of all background patches at position θk
        T: number of temporal frames
        """
        return (1/T)*np.sum([np.dot((p-m),(p-m).T) for p in r], axis=0)
    
    def diag_sample_covariance(self,S):
        """
        F

        Diagonal elements of the sample covariance matrix
        S: Sample covariance matrix
        """
        return np.diag(np.diag(S))

    def shrinkage_factor(self,S, T):
        """
        ρ

        Shrinkage factor to regularize covariant matrix
        S: Sample covariance matrix
        T: Number of temporal frames
        """
        top = (np.trace(np.dot(S,S)) + np.trace(S)**2 - 2*np.sum(np.array([d**2 for d in np.diag(S)])))
        bot = ((T+1)*(np.trace(np.dot(S,S))**2-np.sum(np.array([d**2 for d in np.diag(S)]))))
        return top/bot

    def al(self,hfl, Cfl_inv):
        """
        a_l
        """

        return np.dot(hfl.T, np.dot(Cfl_inv, hfl))
    
    def bl(self,hfl, Cfl_inv, r_fl, m_fl):
        """
        b_l
        """
        return np.dot(self,hfl.T, np.dot(Cfl_inv, (r_fl-m_fl)))


