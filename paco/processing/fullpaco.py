from .. import core
from paco.util.util import *
from .paco import PACO

class FullPACO(PACO):
    def __init__(self,
                 patch_size = 5,
                 file_name = None,
                 directory = None):
        self.filename = file_name
        self.directory = directory
        self.FitsInput = None
        self.im_stack = []
        self.k = patch_size # defaults to paper value
        return
    """
    Algorithm Functions
    """
    def PACO_test(self,angles, phi0):
        """
        ALGORITHM 1 from PACO paper

        angles: list of angles of rotation of each image in stack"
        phi0: initial position of source
        """
        N = np.shape(self.im_stack[0])[0]
        dim = (N/2)
        x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
        phi0 = (x[phi0[0], phi0[1]], y[phi0[0], phi0[1]])
        
        r, theta = cart2pol(x,y)
        r0, theta0 = cart2pol(phi0[0], phi0[1])
        
        
        angles_rad = np.array([a*np.pi/180 for a in angles])+theta0
        
        angles_ind = [[r.flatten()[(np.abs(r.flatten() - r0)).argmin()],
                       theta.flatten()[(np.abs(theta.flatten() - phi)).argmin()]] for phi in angles_rad]
        angles_pol = np.array(list(zip(*angles_ind)))
        angles_px = np.array(pol2cart(angles_pol[0], angles_pol[1]))+dim
        return self.im_stack[-1]
    
    def PACO(self, phi0s, model_name=gaussian2d_model):
        # Generate list of initial points to test (how many, where?)
        N = np.shape(self.im_stack[0])
        a = np.zeros(N)
        b = np.zeros(N)
        T = len(self.im_stack)
        print(N)
        for p0 in phi0s:
        #       for l in range(T):
            patch = self.get_patch(p0, self.k)
            m = np.mean(patch, axis=0)
            S = self.sample_covariance(patch, m, T)
            rho = self.shrinkage_factor(S, T)
            F = self.diag_sample_covariance(S)
            C = self.background_covariance(rho, S, F)
            Cinv = np.linalg.inv(C)
            h = self.model_function(2*self.k,model_name,sigma=5)
            print("Shapes")
            print(m.shape,S.shape,patch.shape,Cinv.shape,h.shape)
            
            a = np.sum([self.al(h, Cinv) for r in patch],axis=0)
            b = np.sum([self.bl(h, Cinv, p, m) for p in patch], axis=0)
            print(a.shape)
            print(b.shape)
        return b/np.sqrt(a)

