from .. import core
from paco.util.util import *
from .paco import PACO
import matplotlib.pyplot as plt
import sys

class FullPACO(PACO):
    def __init__(self,                 
                 patch_size = 5,
                 file_name = None,
                 directory = None):
        self.filename = file_name
        self.directory = directory
        self.FitsInput = None
        self.im_stack = []
        self.k = int(patch_size) # defaults to paper value
        return
    """
    Algorithm Functions
    """
    
    def run(self,angles):
        #do error checking
        phi0s = []
        for i in range(self.im_stack.shape[1]):
            for j in range(self.im_stack.shape[2]):
                phi0s.append(np.array([i,j]))

        
        a,b = self.PACO(np.array(phi0s),angles)
        a = np.reshape(a,(self.im_stack.shape[1],self.im_stack.shape[2]))
        b = np.reshape(b,(self.im_stack.shape[1],self.im_stack.shape[2]))
        return a,b
    
    def PACO(self, phi0s, angles, model_name=gaussian2d_model):
        # Generate list of initial points to test (how many, where?)
        N = np.shape(self.im_stack[0])[0]
        npx = len(phi0s)
        a = np.zeros(npx)
        b = np.zeros(npx)
        T = len(self.im_stack)

        
        k = self.k
        
        patch = np.zeros((npx,T,T,2*k,2*k))
        m     = np.zeros((npx,T,2*k,2*k))
        Cinv  = np.zeros((npx,T,4*k*k,4*k*k))
        h     = np.zeros((npx,T,2*k,2*k))
        print("Running PACO...")
        for i,p0 in enumerate(phi0s):
            if(i%1000 == 0):
                print(i)
            dim = (N/2)
            x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
            phi0 = np.array([x[p0[0], p0[1]], y[p0[0], p0[1]]])
            rphi0 = cart_to_pol(phi0)
            r,theta = grid_cart_to_pol(x,y)
            angles_rad = np.array([a*np.pi/180 for a in angles])-rphi0[1]  
            angles_ind = [[-rphi0[0],rphi0[0]-phi] for phi in angles_rad]
            angles_pol = np.array(list(zip(*angles_ind)))
            angles_px = np.array(grid_pol_to_cart(angles_pol[0], angles_pol[1]))+dim
            angles_px = angles_px.T
            for j,ang in enumerate(angles_px):
                patch[i][j] = self.get_patch(ang, k)
                m[i][j] = np.mean(patch[i][j], axis=0)
                S = self.sample_covariance(patch[i][j], m[i][j], T)
                rho = self.shrinkage_factor(S, T)
                F = self.diag_sample_covariance(S)
                C = self.covariance(rho, S, F)
                Cinv[i][j] = np.linalg.inv(C)
                h[i][j] = self.model_function(2*k,model_name,sigma=5)
            #self.plotting(patch[i][0])
            #self.plotting(C[i])
            a[i] = np.sum(self.al(h[i], Cinv[i]),axis=0)
            b[i] = np.sum(self.bl(h[i], Cinv[i], patch[i][j], m[i]), axis=0)
        print("Done")
        return a,b

    def plotting(self,patch, vmax = 5):       
        print(patch.shape)
        dim = len(patch)
        n = int(np.ceil(np.sqrt(dim)))
        m = int(np.ceil(dim/n))
        fig, ax = plt.subplots(nrows=m, ncols=n, figsize=(8,6))
        ax = ax.flatten()
        #ax = ax[:dim+1]
        for i in range(dim):
            ax[i].imshow(patch[i], vmax=vmax)
        return    
