from .. import core
from ..util import *

class FullPACO(PACO):
    def __init__():
        return
    """
    Algorithm Functions
    """
    def PACO(angles, phi0):
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
