"""
This will implement ALGORITHM 3 from the PACO paper
"""
def cost_function(a):
    return
class FluxPACO(PACO):
    def __init__(self,                 
                 patch_size = 49,
                 precision = 0.1
                 file_name = None,
                 directory = None):
        self.filename = file_name
        self.directory = directory
        self.FitsInput = None
        self.im_stack = []
        self.p_size = int(patch_size) # Number of pixels in a patch
        self.psf_rad = int(np.ceil(np.sqrt(patch_size/np.pi))) # width of a patch
        self.epsilon = precision
        return
    """
    Algorithm Functions
    """
    
    def fluxEstimate(self, p0, angles, params, initial_est = 9999.0, scale = 1, model_name=gaussian2d_model):
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

        # Unbiased flux estimation
        a = 0.0
        b = 0.0
        ahat = initial_est
        aprev = 0.0
        
        T = self.im_stack.shape[0] # Number of temporal frames
        k = int(2*np.ceil(scale * self.psf_rad ) + 2) # Width of a patch, just for readability

        # Create arrays needed for storage
        # Store for each image pixel, for each temporal frame an image
        # for patches: for each time, we need to store a column of patches
        patch = np.zeros((T,T,self.p_size*scale**2)) # 2d selection of pixels around a given point
        mask =  createCircularMask((k,k),radius = self.psf_rad*scale)

        # the mean of a temporal column of patches at each pixel
        m     = np.zeros((T,self.p_size*scale**2))
        
         # the inverse covariance matrix at each point
        Cinv  = np.zeros((T,self.p_size*scale**2,self.p_size*scale**2))

        h_template = self.modelFunction(k,model_name, params)
        h_mask = createCircularMask(h_template.shape,radius = self.psf_rad*scale)
        h = np.zeros((self.p_size*scale**2)) # The off axis PSF at each point
        if scale!=1:
            h = resizeImage(h_template,scale)[h_mask]
        else:
            h = h_template[h_mask]

        print("Running PACO...")
        # Loop over all pixels
        # i is the same as theta_k in the PACO paper
        x,y = np.meshgrid(np.arange(0,int(self.im_stack.shape[1])),
                          np.arange(0,int(self.im_stack.shape[2])))
        angles_px = getRotatedPixels(x,y,p0,angles)
        patch = self.getPatch(ang, k, mask) # Get the column of patches at this point
        while np.abs(ahat - aprev) > self.epsilon*ahat:
            # Get list of pixels for each rotation angle            
            # Iterate over each temporal frame/each angle
            # Same as iterating over phi_l
            for l,ang in enumerate(angles_px):
                m[l], Cinv[l] = self.iterStep(p0,ahat,patch[l],h)
            # Calculate a and b, matrices
            a = self.al(h, Cinv)
            b = max(self.bl(h, Cinv, patch, m),0.0)
            aprev = ahat
            ahat = b/a
        print("Done")
        return a,b
  

    def iterStep(self, p0, est, patch, model):
        T = patch.shape[0]
        unbiased = np.array([apatch- est*model for apatch in patch])
        m = np.mean(unbiased,axis = 0)
        S = sampleCovariance(unbiased, m)
        rho = self.shrinkageFactor(S, T)
        F = self.diagSampleCovariance(S)
        C = self.covariance(rho, S, F)
        Cinv = np.linalg.inv(C)
        return m,Cinv
