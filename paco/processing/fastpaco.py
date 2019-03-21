"""
This module implements ALGORITHM 2 from the PACO paper. 
Covariance and mean statistics are computed and stored 
before running the algorithm, but do not have subpixel
accuracy.
"""
from .. import core
from paco.util.util import *
from .paco import PACO
from multiprocessing import Process,Pool,sharedctypes


import matplotlib.pyplot as plt
import sys
import concurrent.futures as cf
import time

class FastPACO(PACO):
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
        self.mask = None
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
            self.rescale_image_sequence(scale)
        # Setup pixel coordinates
        x,y = np.meshgrid(np.arange(0,int(scale * self.im_stack.shape[1])),
                          np.arange(0,int(scale * self.im_stack.shape[2])))
        phi0s = np.column_stack((x.flatten(),y.flatten()))
        # Compute a,b
        a,b = self.PACO_calc(np.array(phi0s),angles, params, scale, model_name)
        # Reshape into a 2D image, with the same dimensions as the input images
        a = np.reshape(a,(self.im_stack.shape[1],self.im_stack.shape[2]))
        b = np.reshape(b,(self.im_stack.shape[1],self.im_stack.shape[2]))
        return a,b


    def compute_statistics(self, phi0s, params, scale, model_name):
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
        k = int(2*np.ceil(scale * self.psf_rad ) + 2) # Width of a patch, just for readability
        
        # Create arrays needed for storage
        # PSF Template
        h_template = self.model_function(k,model_name, params)
        h_mask = createCircularMask(h_template.shape,radius = self.psf_rad*scale)
        h = np.zeros((N,N,self.p_size*scale**2)) # The off axis PSF at each point

        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patches = []
        patch = np.zeros((T,self.p_size*scale**2)) # 2d selection of pixels around a given point

        # the mean of a temporal column of patches at each pixel
        m     = np.zeros((N*N*self.p_size*scale**2)) 
        m_C = np.ctypeslib.as_array(m)
        #N*N*self.p_size*scale**2

        # the inverse covariance matrix at each point
        Cinv  = np.zeros((N*N*self.p_size*self.p_size*scale**2)) 
        Cinv_C  = np.ctypeslib.as_array(Cinv)
        #N*N*self.p_size*self.p_size*scale**2


        
        # *** SERIAL ***
        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        # Do h in serial so I don't have to pass too many arguments
        for p0 in phi0s:
            if scale!=1:
                h[p0[0]][p0[1]] = resizeImage(h_template,scale)[h_mask]
            else:
                h[p0[0]][p0[1]] = h_template[h_mask]

        # *** Parallel Processing ***
        start = time.time()     
        arglist = [(self.get_patch(p0, k, self.mask),p0,k,T,count) for count,p0 in enumerate(phi0s)]
        
        # Create a pool
        print(m_C.shape)
        shared_m = sharedctypes.RawArray('d', m_C)
        shared_Cinv = sharedctypes.RawArray('d',Cinv_C)
        
        p = Pool(processes = 2,initializer = init_array,initargs = (shared_m,shared_Cinv))
        data = p.map_async(pixel_calc, arglist, chunksize = int(npx/4))
        p.close()
        p.join()

        print(shared_m)
        print(m_C)
        m = np.frombuffer(shared_m).reshape((N,N,self.p_size*scale**2))
        print(m)
        Cinv = np.frombuffer(shared_Cinv).reshape((N,N,self.p_size,self.p_size*scale**2))
        end = time.time()
        
        print("Parallel elapsed",end-start)
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
        k = int(2*np.ceil(scale * self.psf_rad ) + 2) # Width of a patch, just for readability
        
        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patch = np.zeros((T,T,self.p_size)) # 2d selection of pixels around a given point
        self.mask =  createCircularMask((k,k),radius = self.psf_rad)
        
        Cinv,m,h = self.compute_statistics(phi0s, params, scale = scale, model_name = model_name)
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
                patch[l] = self.get_patch(ang, k, self.mask)
            Cinv_arr = np.array(Cinlst)
            m_arr   = np.array(mlst)
            hl   = np.array(hlst)

            #print(Cinlst.shape,mlst.shape,hlst.shape,a.shape,patch.shape)
            # Calculate a and b, matrices
            a[i] = self.al(hlst, Cinlst)
            b[i] = max(self.bl(hlst, Cinlst, patch, mlst),0.0)
        print("Done")
        return a,b

        
