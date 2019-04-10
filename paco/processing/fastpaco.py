"""
This module implements ALGORITHM 2 from the PACO paper. 
Covariance and mean statistics are computed and stored 
before running the algorithm, but do not have subpixel
accuracy.
"""
from paco.util.util import *
from .paco import PACO
from multiprocessing import Pool

import matplotlib.pyplot as plt
import sys,os
import time

#Pacito
class FastPACO(PACO):
    def __init__(self,                 
                 image_stack = None,
                 angles = None,
                 psf = None,
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
        if image_stack is not None:
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
    def PACOCalc(self,
                 phi0s,
                 params,
                 scale = 1,
                 model_name=gaussian2dModel,
                 cpu = 1):
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
            Number of cores to use for parallel processing
        """      
        npx = len(phi0s)  # Number of pixels in an image
        dim = int((self.m_width*scale)/2)
        
        try:
            assert npx == int(self.m_width*self.m_height*scale**2)
        except AssertionError:
            print("Position grid does not match pixel grid.")
            sys.exit(1)
        
        a = np.zeros(npx) # Setup output arrays
        b = np.zeros(npx)
        k = int(2*np.ceil(scale * self.m_psf_rad )) # Width of a patch, just for readability
        

        if cpu == 1:
            Cinv,m,h = self.computeStatistics(phi0s, params, scale = scale, model_name = model_name)
        else:
            Cinv,m,h = self.computeStatisticsParallel(phi0s, params, scale = scale, cpu = cpu, model_name = model_name)

            # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patch = np.zeros((self.m_nFrames,self.m_nFrames,self.m_p_size)) # 2d selection of pixels around a given point
        mask =  createCircularMask((k,k),radius = int(self.m_psf_rad*scale))

        x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))    

        print("Running PACO...")
        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        for i,p0 in enumerate(phi0s):
            #if(i%(npx/10) == 0):
            #    print(str(i/100) + "%")

            # Get Angles
            angles_px = getRotatedPixels(x,y,p0,self.m_angles)

            # Ensure within image bounds
            if(int(np.max(angles_px.flatten()))>=self.m_width or int(np.min(angles_px.flatten()))<0):
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
                patch[l] = self.getPatch(ang, k, mask)
            Cinv_arr = np.array(Cinlst)
            m_arr   = np.array(mlst)
            hl   = np.array(hlst)

            #print(Cinlst.shape,mlst.shape,hlst.shape,a.shape,patch.shape)
            # Calculate a and b, matrices
            a[i] = self.al(hlst, Cinlst)
            b[i] = self.bl(hlst, Cinlst, patch, mlst)
        print("Done")
        return a,b

    def computeStatistics(self, phi0s, params, scale, model_name):
        """    
        This function computes the mean and inverse covariance matrix for
        each patch in the image stack in Serial.
        Parameters
        ---------------
        phi0s : int arr
            Array of pixel locations to estimate companion position
        params: dict
            Dictionary of parameters about the psf, containing either the width
            of a gaussian distribution, or a label 'psf_template'
        scale : float
            Resolution scaling
        model_name: str
            Name of the template for the off-axis PSF
        """
    
        print("Precomputing Statistics...")
        npx = len(phi0s)           # Number of pixels in an image      
        dim = int((self.m_width*scale)/2)
        k = int(2*np.ceil(scale * self.m_psf_rad)) # Width of a patch, just for readability
        
        # Create arrays needed for storage
        # PSF Template
        if self.m_psf is not None:
            h_template = self.m_psf
        else:
            h_template = self.modelFunction(k, model_name, params)
        h_mask = createCircularMask((k,k),radius = int(self.m_psf_rad*scale))
        if self.m_p_size != len(h_mask[h_mask]):
            self.m_p_size = len(h_mask[h_mask])
        # The off axis PSF at each point
        h = np.zeros((self.m_height,self.m_width,self.m_p_size)) 

        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patch = np.zeros((self.m_nFrames,self.m_p_size)) # 2d selection of pixels around a given point
        mask =  createCircularMask((k,k),radius = int(self.m_psf_rad*scale))

        # the mean of a temporal column of patches centered at each pixel
        m     = np.zeros((self.m_height,self.m_width,self.m_p_size)) 
        # the inverse covariance matrix at each point
        Cinv  = np.zeros((self.m_height,self.m_width,self.m_p_size,self.m_p_size)) 

        # *** SERIAL ***
        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        for p0 in phi0s:
            apatch = self.getPatch(p0,k,mask)
            m[p0[0]][p0[1]],Cinv[p0[0]][p0[1]] = pixelCalc(apatch)           
            if scale!=1:
                h[p0[0]][p0[1]] = resizeImage(h_template,scale)[h_mask]
            else:
                h[p0[0]][p0[1]] = h_template[h_mask]
        return Cinv,m,h

    def computeStatisticsParallel(self, phi0s, params, scale, cpu, model_name):
        """    
        This function computes the mean and inverse covariance matrix for
        each patch in the image stack in Serial.
        Parameters
        ---------------
        phi0s : int arr
            Array of pixel locations to estimate companion position
        params: dict
            Dictionary of parameters about the psf, containing either the width
            of a gaussian distribution, or a label 'psf_template'
        scale : float
            Resolution scaling
        model_name: str
            Name of the template for the off-axis PSF
        cpu : int
            Number of processors to use
            
        NOTES:
        This function currently seems slower than computing in serial...
        """
    
        print("Precomputing Statistics using %d Processes...",cpu)
        npx = len(phi0s)           # Number of pixels in an image      
        dim = int((self.m_width*scale)/2)
        k = int(2*np.ceil(scale * self.m_psf_rad ) + 2) # Width of a patch, just for readability
        
        # Create arrays needed for storage
        # PSF Template
        if self.m_psf is not None:
            h_template = self.m_psf
        else:
            h_template = self.modelFunction(k, model_name, params)
        h_mask = createCircularMask(h_template.shape,radius =int( self.m_psf_rad*scale))
        if self.m_p_size != len(h_mask[h_mask]):
            self.m_p_size = len(h_mask[h_mask])
        h = np.zeros((self.m_height,self.m_width,self.m_p_size)) # The off axis PSF at each point

        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patches = []
        patch = np.zeros((self.m_nFrames,self.m_p_size)) # 2d selection of pixels around a given point
        mask =  createCircularMask((k,k),radius = self.m_psf_rad)
                
        # the mean of a temporal column of patches at each pixel
        m     = np.zeros((self.m_height*self.m_width*self.m_p_size)) 
        # the inverse covariance matrix at each point
        Cinv  = np.zeros((self.m_height*self.m_width*self.m_p_size*self.m_p_size)) 
        for p0 in phi0s:
            if scale!=1:
                h[p0[0]][p0[1]] = resizeImage(h_template,scale)[h_mask]
            else:
                h[p0[0]][p0[1]] = h_template[h_mask]

                
        # *** Parallel Processing ***
        #start = time.time()
        arglist = [np.copy(np.array(self.getPatch(p0, k, mask))) for p0 in phi0s]
        p = Pool(processes = cpu)
        data = p.map(pixelCalc, arglist, chunksize = int(npx/16))
        p.close()
        p.join()
        ms,cs = [],[]
        for d in data:
            ms.append(d[0])
            cs.append(d[1])
        ms = np.array(ms)
        cs = np.array(cs)  
        m = ms.reshape((self.m_height,self.m_width,self.m_p_size))
        Cinv = cs.reshape((self.m_height,self.m_width,self.m_p_size,self.m_p_size))
        #end = time.time()
        #print("Parallel elapsed",end-start)
        return Cinv,m,h
