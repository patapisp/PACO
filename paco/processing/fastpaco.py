"""
This module implements ALGORITHM 2 from the PACO paper. 
Covariance and mean statistics are computed and stored 
before running the algorithm, but do not have subpixel
accuracy.
"""
from paco.util.util import *
from .paco import PACO
from multiprocessing import Process,Pool,sharedctypes,Lock,Array,Queue,Manager
import ctypes

import matplotlib.pyplot as plt
import sys,os
import time

class FastPACO(PACO):
    def __init__(self,                 
                 patch_size = 49,
                 file_name = None,
                 directory = None):       
        self.m_im_stack = []
        self.m_nFrames = 0
        self.m_width = 0
        self.m_height = 0
        self.m_p_size = int(patch_size) # Number of pixels in a patch
        self.m_psf_rad = int(np.ceil(np.sqrt(patch_size/np.pi))) # width of a patch
        self.m_mask = None
        self.m_psf = None
        return

    """
    Algorithm Functions
    """   
    
    def PACOCalc(self,
                 phi0s,
                 angles,
                 params,
                 scale = 1,
                 model_name=gaussian2d_model,
                 cpu = 1):
        """
        PACO_calc
        
        This function iterates of a list of test points (phi0) and a list
        of angles between frames to produce 'a' and b', which can be used to
        generate a signal to noise map where SNR = b/sqrt(a) at each pixel.
        
        phi0s : int arr
            Array of pixel locations to estimate companion position
        angles : int arr
            Array of angles from frame rotation
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
        dim = self.m_width/2
        
        try:
            assert npx == self.m_width*self.m_height
        except AssertionError:
            print("Position grid does not match pixel grid.")
            sys.exit(1)
        
        a = np.zeros(npx) # Setup output arrays
        b = np.zeros(npx)
        k = int(2*np.ceil(scale * self.m_psf_rad ) + 2) # Width of a patch, just for readability
        
        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patch = np.zeros((self.m_nFrames,self.m_nFrames,self.m_p_size)) # 2d selection of pixels around a given point
        self.m_mask =  createCircularMask((k,k),radius = self.m_psf_rad)

        if cpu == 1:
            Cinv,m,h = self.computeStatistics(phi0s, params, scale = scale, model_name = model_name)
        else:
            Cinv,m,h = self.computeStatisticsParallel(phi0s, params, scale = scale, cpu = cpu, model_name = model_name)

        x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))    

        print("Running PACO...")
        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        for i,p0 in enumerate(phi0s):
            if(i%(npx/10) == 0):
                print(str(i/100) + "%")

            # Get Angles
            angles_px = getRotatedPixels(x,y,p0,angles)

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
                patch[l] = self.getPatch(ang, k, self.m_mask)
            Cinv_arr = np.array(Cinlst)
            m_arr   = np.array(mlst)
            hl   = np.array(hlst)

            #print(Cinlst.shape,mlst.shape,hlst.shape,a.shape,patch.shape)
            # Calculate a and b, matrices
            a[i] = self.al(hlst, Cinlst)
            b[i] = max(self.bl(hlst, Cinlst, patch, mlst),0.0)
        print("Done")
        return a,b

    def computeStatistics(self, phi0s, params, scale, model_name):
        """
        compute_statistics
        
        This function computes the mean and inverse covariance matrix for
        each patch in the image stack.

        :phi0s: Array of pixel locations to estimate companion position
        :model_name: Name of the template for the off-axis PSF
        """
    
        print("Precomputing Statistics...")
        npx = len(phi0s)           # Number of pixels in an image      
        dim = int(N/2)
        k = int(2*np.ceil(scale * self.m_psf_rad ) + 2) # Width of a patch, just for readability
        
        # Create arrays needed for storage
        # PSF Template
        h_template = self.modelFunction(k, model_name, params)
        h_mask = createCircularMask(h_template.shape,radius = self.m_psf_rad*scale)
        h = np.zeros((self.m_height,self.m_width,self.m_p_size*scale**2)) # The off axis PSF at each point

        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patches = []
        patch = np.zeros((self.m_nFrames,self.m_p_size*scale**2)) # 2d selection of pixels around a given point

        # the mean of a temporal column of patches centered at each pixel
        m     = np.zeros((self.m_height,self.m_width,self.m_p_size*scale**2)) 
        # the inverse covariance matrix at each point
        Cinv  = np.zeros((self.m_height,self.m_width,self.m_p_size*scale,self.m_p_size*scale)) 

        # *** SERIAL ***
        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        # Do h in serial so I don't have to pass too many arguments
        for p0 in phi0s:
            apatch = self.getPatch(p0,k,self.m_mask)
            m[p0[0]][p0[1]],Cinv[p0[0]][p0[1]] = self.pixelCalc(patch,self.m_nFrames,self.m_p_size)
            if scale!=1:
                h[p0[0]][p0[1]] = resizeImage(h_template,scale)[h_mask]
            else:
                h[p0[0]][p0[1]] = h_template[h_mask]
        return Cinv,m,h

    def computeStatisticsParallel(self, phi0s, params, scale, cpu, model_name):
        """
        compute_statistics
        
        This function computes the mean and inverse covariance matrix for
        each patch in the image stack.

        :phi0s: Array of pixel locations to estimate companion position
        :model_name: Name of the template for the off-axis PSF
        """
    
        print("Precomputing Statistics using %d Processes...",cpu)
        npx = len(phi0s)           # Number of pixels in an image      
        dim = int(N/2)
        k = int(2*np.ceil(scale * self.m_psf_rad ) + 2) # Width of a patch, just for readability
        
        # Create arrays needed for storage
        # PSF Template
        h_template = self.modelFunction(k,model_name, params)
        h_mask = createCircularMask(h_template.shape,radius = self.m_psf_rad*scale)
        h = np.zeros((self.m_height,self.m_width,self.m_p_size*scale**2)) # The off axis PSF at each point

        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patches = []
        patch = np.zeros((self.m_nFrames,self.m_p_size*scale**2)) # 2d selection of pixels around a given point

        # the mean of a temporal column of patches at each pixel
        m     = np.zeros((self.m_height*self.m_width*self.m_p_size*scale**2)) 
        m_C = np.ctypeslib.as_array(m)
        #N*N*self.p_size*scale**2

        # the inverse covariance matrix at each point
        Cinv  = np.zeros((self.m_height*self.m_width*self.m_p_size*self.m_p_size*scale**2)) 
        Cinv_C  = np.ctypeslib.as_array(Cinv)
        #N*N*self.p_size*self.p_size*scale**2

        for p0 in phi0s:
            if scale!=1:
                h[p0[0]][p0[1]] = resizeImage(h_template,scale)[h_mask]
            else:
                h[p0[0]][p0[1]] = h_template[h_mask]

                
        # *** Parallel Processing ***
        start = time.time()
        #print(arglist[0])
        # Create a pool
        #shared_m = sharedctypes.RawArray('d', m_C)
        #shared_m = sharedctypes.Array(ctypes.c_double,m_C,lock = False)
        #shared_Cinv = sharedctypes.RawArray('d',Cinv_C)
        #shared_Cinv = sharedctypes.Array(ctypes.c_double,Cinv_C,lock = False)
        # pynpoint processing/limits.py
        
        #queue = Queue(100000000)

        #patches = [np.copy(np.array(self.get_patch(p0, k, self.mask))) for p0 in phi0s]
        arglist = [(np.copy(np.array(self.getPatch(p0, k, self.m_mask))),self.m_nFrames,self.m_p_size) for p0 in phi0s]
        '''
        jobs = []
        result = []
        for apatch in patches:
            process = Process(target = pixel_calc,
                              args = (apatch,
                                      T,
                                      self.p_size,
                                      queue))
            jobs.append(process)
            
        for count,job in enumerate(jobs):
            job.start()
            #print(job,count)
            #print("started")
            if (count+1)%cpu == 0:
                for njob in jobs[(count+1-cpu):(count+1)]:
                    njob.join(timeout = 0.5)
            elif (count+1) == len(jobs) and (count+1)%cpu != 0:
                for njob in jobs[count + 1 - (count + 1)%cpu:]:
                    njob.join()
        '''
        p = Pool(processes = cpu)
        data = p.starmap(self.pixelCalc, arglist, chunksize = int(npx/16))
        p.close()
        p.join()
       '''
        queue.put(None)
        while True:
            item = queue.get()

            if item is None:
                break
            else:
                result.append(item)        
        '''
        ms = np.array([d[0] for d in data])
        cs = np.array([d[1] for d in data])
        m = ms.reshape((self.m_height,self.m_width,self.m_p_size*scale**2))
        Cinv = cs.reshape((self.m_height,self.m_width,self.m_p_size,self.m_p_size*scale**2))
        end = time.time()
        print("Parallel elapsed",end-start)
        return Cinv,m,h
