"""
This file will implement ALGORITHM 1 from the PACO paper
"""
from paco.util.util import *

class PACO:
    def __init__(self,
                 image_stack = None
                 angles = None
                 psf = None
                 patch_size = 49):
        """
        PACO Parent Class Constructor
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

    def PACO(self,
             model_params,
             scale = 1,
             model_name=gaussian2d_model,
             cpu = 1):
        """
        PACO
        This function wraps the actual PACO algorithm, setting up the pixel coordinates 
        that will be iterated over. The output will probably be changes to output the
        true SNR map.
        Parameters
        ------------    
        scale : float
            Amount of oversampling of image to improve positioning of PSF
        model_params : dict
            Dictionary describing the analytic template model, or containing the PSF
        model_name : method
            A function to produce a 2D template, or the keyword psfTemplateModel
        cpu : int>=1
            Number of processers to use        
        """
        if scale != 1:
            self.rescaleImageSequence(scale)
        # Setup pixel coordinates
        x,y = np.meshgrid(np.arange(0,int(scale * self.m_height)),
                          np.arange(0,int(scale * self.m_width)))
        phi0s = np.column_stack((x.flatten(),y.flatten()))
        # Compute a,b
        a,b = self.PACOCalc(np.array(phi0s), params, scale, model_name,cpu = cpu)
        # Reshape into a 2D image, with the same dimensions as the input images
        a = np.reshape(a,(self.m_height,self.m_width))
        b = np.reshape(b,(self.m_height,self.m_height))
        return a,b
    
    """
    Utility Functions
    """    
    # Set number of pixels in a patch
    def setPatchSize(self,npx):
        self.m_p_size = npx
    def getPatchSize(self):
        return self.m_p_size
    
    # Set the image stack to be processed
    def setImageSequence(self,imgs):
        self.m_im_stack = np.array(imgs)
        self.m_nFrames = self.m_im_stack.shape[0]
        self.m_width = self.m_im_stack.shape[2]
        self.m_height = self.m_im_stack.shape[1]
    def getImageSequence(self):
        return self.m_im_stack
    def rescaleImageSequence(self,scale):
        new_stack = []
        for i,img in enumerate(self.m_im_stack):
            new_stack.append(resizeImage(img,scale))
        self.m_im_stack =  np.array(new_stack)
        
    # Set the template PSF
    def setPSF(self,psf):
        self.m_psf = psf
    def getPSF(self):
        return self.m_psf
    
    # Set parallactic angles
    def setAngles(self,angles):
        self.m_angles = angles
                
    # Select a masked patch of pixels surrounding a given pixel
    def getPatch(self,px, width, mask = None):
        """
        Gets patch at given pixel px with size k for the current img sequence       
        Parameters
        --------------
        px : (int,int)
            Pixel coordinates for center of patch
        width : int
            width of a square patch to be masked
        mask : arr
            Circular mask to select a round patch of pixels, which is then flattened into a 1D array.        
        """
        k = int(width/2)
        nx, ny = np.shape(self.m_im_stack[0])[:2]
        if px[0]+k > nx or px[0]-k < 0 or px[1]+k > ny or px[1]-k < 0:
            #print("pixel out of range")
            return None
        if mask is not None:
            patch = np.array([self.m_im_stack[i][int(px[0])-k:int(px[0])+k, int(px[1])-k:int(px[1])+k][mask] for i in range(len(self.m_im_stack))])
        else:
            patch = np.array([self.m_im_stack[i][int(px[0])-k:int(px[0])+k, int(px[1])-k:int(px[1])+k] for i in range(len(self.m_im_stack))])
        return patch



    """
    Math Functions
    """
    def pixelCalc(self,patch):
        """
        Calculate the mean and inverse covariance within a patch
        Parameters
        -------------
        patch : arr
            Array of circular (flattened) patches centered on the same physical pixel vertically throughout the image stack
        """
        T = patch.shape[0]
        size = patch.shape[1]
        #def pixel_calc(_args):
        #patch = _args[0]
        #T = _args[1]
        #size = _args[2]
        #queue = _args[3]
        if patch is None:
            return np.asarray([np.zeros(size),np.zeros(size**2)])
        if len(patch.shape) != 2:
            return np.asarray([np.zeros(size),np.zeros(size**2)])
        m = np.mean(patch,axis = 0) # Calculate the mean of the column

        # Calculate the covariance matrix
        S = sampleCovariance(patch, m, T)
        rho = shrinkageFactor(S, T) 
        F = diagSampleCovariance(S)
        C = covariance(rho, S, F)    
        Cinv = np.linalg.inv(C).flatten()
        return (m,Cinv)
    
    def modelFunction(self, n, model, params):
        """
        h_θ

        n : mean
        model : numpy statistical model (need to import numpy module for this)
        **kwargs: additional arguments for model
        """    
        if self.m_psf:
            return self.m_psf
        else:
            if model.__name__ == "psftemplate_model":
                try:
                    return model(n, params)
                except ValueError:
                    print("Fix template size")
            return model(n, params)



    def al(self, hfl, Cfl_inv):
        """
        a_l

        The sum of a_l is the inverse of the variance of the background at the given pixel
        """
        a = np.sum(np.array([np.dot(hfl[i], np.dot(Cfl_inv[i], hfl[i]).T) for i in range(len(hfl))]),axis = 0)
        return a
        
    
    def bl(self, hfl, Cfl_inv, r_fl, m_fl):
        """
        b_l

        The sum of b_l is the flux estimate at the given pixel. 
        """
        b = np.sum(np.array([np.dot(np.dot(Cfl_inv[i], hfl[i]).T,(r_fl[i][i]-m_fl[i])) for i in range(len(hfl))]),axis = 0)
        return b


    """
    FluxPACO
    """
    def fluxEstimate(self, p0, eps, params, initial_est = 9999.0, scale = 1, model_name=gaussian2d_model):
        """
        Unbiased estimate of the flux of a source located at p0
        Parameters
        ------------
        p0 : arr
            List of locations of sources to compute unbiased flux estimate
        eps : float
            Precision requirement for iteration (0,1)
        params : dict
            Dictionary describing the analytic template model, or containing the PSF
        initial_est : float
            Initial estimate of the flux at p0
        scale : float
            Resolution scaling
        model_name: method
            Name of the template for the off-axis PSF
        """
        npx = len(phi0s)  # Number of pixels in an image
        dim = (self.m_width/2)
        try:
            assert npx == self*m_width*self.m_height
        except AssertionError:
            print("Position grid does not match pixel grid.")
            sys.exit(1)

        # Unbiased flux estimation
        a = 0.0
        b = 0.0
        ahat = initial_est
        aprev = 0.0
        k = int(2*np.ceil(scale * self.m_psf_rad ) + 2) # Width of a patch, just for readability

        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patch = np.zeros((self.m_nFrames,self.m_nFrames,self.m_p_size*scale**2))
        mask =  createCircularMask((k,k),radius = self.m_psf_rad*scale)

        # the mean of a temporal column of patches at each pixel
        m  = np.zeros((self.m_nFrames,self.m_p_size*scale**2))
        
         # the inverse covariance matrix at each point
        Cinv  = np.zeros((self.m_nFrames,self.m_p_size*scale**2,self.m_p_size*scale**2))

        h_template = self.modelFunction(k,model_name, params)
        h_mask = createCircularMask(h_template.shape,radius = self.m_psf_rad*scale)
        h = np.zeros((self.p_size*scale**2)) # The off axis PSF at each point
        if scale!=1:
            h = resizeImage(h_template,scale)[h_mask]
        else:
            h = h_template[h_mask]

        print("Computing unbiased flux estimate...")
        x,y = np.meshgrid(np.arange(0,int(self.m_height)),
                          np.arange(0,int(self.m_width)))
        angles_px = getRotatedPixels(x,y,p0,self.m_angles)
        patch = self.getPatch(ang, k, mask) # Get the column of patches at this point
        while np.abs(ahat - aprev) > eps*ahat:
            for l,ang in enumerate(angles_px):
                m[l], Cinv[l] = self.iterStep(p0,ahat,patch[l],h)
            a = self.al(h, Cinv)
            b = max(self.bl(h, Cinv, patch, m),0.0)
            aprev = ahat
            ahat = b/a
        print("Done")
        return ahat
  

    def iterStep(self, p0, est, patch, model):
        """
        Compute the iterative estimates for the mean and inverse covariance
        Parameters
        ----------
        p0 : (int,int)
            Pixel coordinates
        est : float
            Current estimate for the magnitude of the flux
        patch : arr
            Column of patches about p0
        model : arr
            Template for PSF
        """
        T = patch.shape[0]
        unbiased = np.array([apatch- est*model for apatch in patch])
        m = np.mean(unbiased,axis = 0)
        S = sampleCovariance(unbiased, m)
        rho = self.shrinkageFactor(S, T)
        F = self.diagSampleCovariance(S)
        C = self.covariance(rho, S, F)
        Cinv = np.linalg.inv(C)
        return m,Cinv
