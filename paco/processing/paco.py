"""
This file will implement AGORITHM 1 from the PACO paper
"""
from .. import core
from ..util import *

class PACO:
    def __init__():
        return__()
    def get_patch(images, px, k):
        """
        gets patch at given pixel with size k for the given img sequence
        """
        nx, ny = np.shape(images[0])[:2]
        if px[0]+k > nx or px[0]-k < 0 or px[1]+k > ny or px[1]-k < 0:
            print("pixel out of range")
            return None
        patch = [images[i][px[0]-k:px[0]+k, px[1]-k:px[1]+k] for i in range(len(images))]
        return patch

    def PACO(images, angles, phi0):
        N = np.shape(images[0])[0]
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
        return

    def h_theta(n, model, **kwargs):
        return model(n, **kwargs)

    def Chat(rho, S, F):
        return (1-rho)*S + rho*F
    
    def Shat(r, m, T):
        return (1/T)*np.sum([np.dot((p-m),(p-m).T) for p in r], axis=0)
    
    def Fhat(S):
        return np.diag(np.diag(S))

    def rho_hat(S, T):
        top = (np.trace(np.dot(S,S)) + np.trace(S)**2 - 2*np.sum(np.array([d**2 for d in np.diag(S)])))
        bot = ((T+1)*(np.trace(np.dot(S,S))**2-np.sum(np.array([d**2 for d in np.diag(S)]))))
        return top/bot

    def al(hfl, Cfl_inv):
        return np.dot(hfl.T, np.dot(Cfl_inv, hfl))
    
    def bl(hfl, Cfl_inv, r_fl, m_fl):
        return np.dot(hfl.T, np.dot(Cfl_inv, (r_fl-m_fl)))
