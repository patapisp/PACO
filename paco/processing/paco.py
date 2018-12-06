"""
This file will implement AGORITHM 1 from the PACO paper
"""
from .. import core
from ..util import *

class PACO:
    def __init__():
        self.im_stack = []
        self.k = -1
        return

    """
    Utility Functions
    """
    def set_patch_size(npx):
        self.k = npx
    
    def get_patch(px, k):
        """
        gets patch at given pixel with size k for the given img sequence
        """
        r2 = k
        nx, ny = np.shape(self.im_stack[0])[:2]
        if px[0]+k > nx or px[0]-k < 0 or px[1]+k > ny or px[1]-k < 0:
            print("pixel out of range")
            return None
        patch = [self.im_stack[i][px[0]-k:px[0]+k, px[1]-k:px[1]+k] for i in range(len(self.im_stack))]
        return patch


    """
    Math Functions
    """
    def model_function(n, model, **kwargs):
        """
        h_θ

        n: mean
        model: numpy statistical model
        **kwargs: additional arguments for model
        """
        return model(n, **kwargs)

    def background_covariance(rho, S, F):
        """
        Ĉ

        Shrinkage covariance matrix
        rho: shrinkage weight
        S: Sample covariance matrix
        F: Diagonal of sample covariance matrix
        """
        return (1-rho)*S + rho*F
    
    def sample_covariance(r, m, T):
        """
        Ŝ

        Sample covariance matrix
        r: obcserved intensity at position θk and time tl
        m: mean of all background patches at position θk
        T: number of temporal frames
        """
        return (1/T)*np.sum([np.dot((p-m),(p-m).T) for p in r], axis=0)
    
    def diag_sample_covariance(S):
        """
        F

        Diagonal elements of the sample covariance matrix
        S: Sample covariance matrix
        """
        return np.diag(np.diag(S))

    def shrinkage_factor(S, T):
        """
        ρ

        Shrinkage factor to regularize covariant matrix
        S: Sample covariance matrix
        T: Number of temporal frames
        """
        top = (np.trace(np.dot(S,S)) + np.trace(S)**2 - 2*np.sum(np.array([d**2 for d in np.diag(S)])))
        bot = ((T+1)*(np.trace(np.dot(S,S))**2-np.sum(np.array([d**2 for d in np.diag(S)]))))
        return top/bot

    def al(hfl, Cfl_inv):
        """
        a_l
        """

        return np.dot(hfl.T, np.dot(Cfl_inv, hfl))
    
    def bl(hfl, Cfl_inv, r_fl, m_fl):
        """
        b_l
        """
        return np.dot(hfl.T, np.dot(Cfl_inv, (r_fl-m_fl)))



