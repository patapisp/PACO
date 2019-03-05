"""
This file will implement ALGORITHM 1 from the PACO paper
"""
import paco.core.ReadInFitsFile
import matplotlib.pyplot as plt
from paco.util.util import *
import sys
class PACO:
    def __init__(self,
                 patch_size = 49,
                 file_name = None,
                 directory = None):
        self.FitsInput = None
        self.im_stack = []
        self.k = patch_size # defaults to paper value
        self.T = -1
        return

    """
    Utility Functions
    """    
    def set_patch_size(self,npx):
        self.k = npx

    def set_image_sequence(self,imgs):
        self.im_stack = imgs
        self.T = len(imgs)

    def get_image_sequence(self):
        return self.im_stack

    def rescale_image_sequence(self,scale):
        new_stack = []
        for i,img in enumerate(self.im_stack):
            new_stack.append(resizeImage(img,scale))
        self.im_stack =  np.array(new_stack)
    
    def set_angles(self,angles = []):
        if len(angles) == 0:
            #do stuff
            return
        else:
            return angles
                
    def get_patch_size(self):
        return self.k
    
    def open_fits_images(self,file_name, directory):
        """
        Read in the fits file
        """
        self.FitsInput = ReadInFitsFile.ReadInFitsFile(file_name,directory)
        self.FitsInput.open_one_fits(file_name)
        self.im_stack = FitsInput.images
        
    def create_circular_mask(self,w, h,radius=None, center=None):
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
        mask = dist2 <= radius**2
        return mask

    def get_patch(self,px, k):
        """
        gets patch at given pixel px with size k for the current img sequence
        
        Patch returned will be a square of dim 2k x 2k
        A circular mask will be created separately to select
        which pixels to examine within a patch.
        """
        nx, ny = np.shape(self.im_stack[0])[:2]
        if px[0]+k > nx or px[0]-k < 0 or px[1]+k > ny or px[1]-k < 0:
            #print("pixel out of range")
            return None
        patch = np.array([self.im_stack[i][int(px[0])-k:int(px[0])+k, int(px[1])-k:int(px[1])+k] for i in range(len(self.im_stack))])
        return patch

    """
    Math Functions
    """
    def model_function(self,n, model, params):
        """
        h_θ

        n: mean
        model: numpy statistical model (need to import numpy module for this)
        **kwargs: additional arguments for model
        """
        if model.__name__ == "psftemplate_model":
            try:
                return model(n, params)
            except ValueError:
                print("Fix template size")            
        return model(n, params)

    def covariance(self,rho, S, F):
        """
        Ĉ

        Shrinkage covariance matrix
        rho: shrinkage weight
        S: Sample covariance matrix
        F: Diagonal of sample covariance matrix
        """
        C = (1.0-rho)*S + rho*F
        #fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,8))
        #im1 = ax.imshow(C)
        #fig.colorbar(im1,ax = ax)
        return C
    
    def sample_covariance(self,r, m, T):
        """
        Ŝ

        Sample covariance matrix
        r: observed intensity at position θk and time tl
        m: mean of all background patches at position θk
        T: number of temporal frames
        """
        #print(m.shape,r.shape)
        #S =  (1.0/T)*np.sum([np.outer((p-m).ravel(),(p-m).ravel().T) for p in r], axis=0)
        mv = m.flatten()
        S = (1.0/T)*np.sum([np.cov(np.stack((p.ravel(),mv)), rowvar = False, bias = False) for p in r],axis = 0)
        return S
    
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
        top = (np.trace(np.dot(S,S)) + np.trace(S)**2 - 2.0*np.sum(np.array([d**2.0 for d in np.diag(S)])))
        bot = ((T+1.0)*(np.trace(np.dot(S,S))-np.sum(np.array([d**2.0 for d in np.diag(S)]))))
        return top/bot

    def al(self,hfl, Cfl_inv):
        """
        a_l
        """
      

        #ht = np.array([h.T for h in hfl])
        #fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,8))
        #ax = ax.flatten()
        #im1 = ax[0].imshow(hfl[0])
        #ax[1].imshow(ht[0])
        #im2 =ax.imshow(Cfl_inv[0])
        #fig.colorbar(im2,ax = ax)
        #ht = np.reshape(ht,(hfl.shape[0],hfl.shape[1]*hfl.shape[2]))
        #hfl = np.reshape(hfl,(hfl.shape[0],hfl.shape[1]*hfl.shape[2]))
        #hflT = np.reshape(hflT,(hflT.shape[0],hflT.shape[1]*hflT.shape[2]))
        a = np.array([np.dot(hfl[i].ravel(), np.dot(Cfl_inv[i], hfl[i].ravel()).T) for i in range(len(hfl))])
        #print(a)
        return a
        
    
    def bl(self,hfl, Cfl_inv, r_fl, m_fl):
        """
        b_l
        """
        #print("Calculating b")
        #fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,8))
        #ax = ax.flatten()
        #ax[0].imshow(hfl[0])
        #ax[1].imshow(hfl[4])
        #ax[2].imshow(r_fl[1][1])
        #ax[3].imshow(r_fl[3][3])
        #ax[4].imshow(m_fl[0])
        #im1 = ax.imshow(Cfl_inv[4])
        #fig.colorbar(im1,ax = ax)
        #hflT = np.zeros(hfl.shape)
        #for j,h in enumerate(hfl):
        #    hflT[j] = h.T
        #ht = np.array([h.T for h in hfl])
        #ht = np.reshape(ht,(hfl.shape[0],hfl.shape[1]*hfl.shape[2]))
        #hfl = np.reshape(hfl,(hfl.shape[0],hfl.shape[1]*hfl.shape[2]))
        #hflT = np.reshape(hflT,(hflT.shape[0],hflT.shape[1]*hflT.shape[2]))
        #r_fl = np.reshape(r_fl,(r_fl.shape[0],r_fl.shape[1],r_fl.shape[2]*r_fl.shape[3]))
        #m_fl = np.reshape(m_fl,(m_fl.shape[0],m_fl.shape[1]*m_fl.shape[2]))
        b = np.array([np.dot(np.dot(Cfl_inv[i], hfl[i].ravel()).T,(r_fl[i][i]-m_fl[i]).ravel()) for i in range(len(hfl))])
        #print(b)
        return b


