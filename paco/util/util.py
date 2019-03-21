"""
This module contains basic utility functions

- Rotations
- Coordinate transformations
- Any other commonly used calculations
"""

import numpy as np
import cv2


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)
    return result

def getRotatedPixels(x,y,p0,angles):
    # Current pixel
    phi0 = np.array([x[p0[0], p0[1]], y[p0[0], p0[1]]])
    # Convert to polar coordinates
    rphi0 = cart_to_pol(phi0)
    angles_rad = rphi0[1] - np.array([a*np.pi/180 for a in angles]) 
    
    # Rotate the polar coordinates by each frame angle
    angles_ind = [[rphi0[0],phi] for phi in angles_rad]
    angles_pol = np.array(list(zip(*angles_ind)))
    
    # Convert from polar to cartesian and pixel coordinates
    angles_px = np.array(grid_pol_to_cart(angles_pol[0], angles_pol[1]))+int(x.shape[0]/2)
    angles_px = angles_px.T
    angles_px = np.fliplr(angles_px)
    return angles_px

def createCircularMask(shape, radius=4, center=None):
        """
        Returns a 2D boolean mask given some radius and location
        :w: width, number of x pixels
        :h: height, number of y pixels
        :center: [x,y] pair of pixel indices denoting the center of the mask
        :radius: radius of mask
        """
        w = shape[0]
        h = shape[1]
        if center is None: 
            center = [int(w/2), int(h/2)]
        if radius is None:
            radius = min(center[0], center[1], w-center[0], h-center[1])
        X, Y = np.ogrid[:w, :h]
        dist2 = (X - center[0])**2 + (Y-center[1])**2
        mask = dist2 <= radius**2
        return mask
    
def resizeImage(image, scaleFactor):
    return cv2.resize(image,(0,0), fx = scaleFactor, fy = scaleFactor, interpolation = cv2.INTER_NEAREST)

def gaussian2d(x,y,A, sigma):
    return A*np.exp(-(x**2+y**2)/(2*sigma**2))

def gaussian2d_model(n,params):
    sigma = params["sigma"]
    dim = int(n/2)
    x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
    return 1.0/(2.0*np.pi*sigma**2) * np.exp(-((x+0.5)**2+(y+0.5)**2)/(2*sigma**2))     

def psftemplate_model(n, params):
    """
    Model using a psf template directly from the data. 
    Template should be normalized such that the sum equals 1.
    
    If model needs rescaling it is done here
    """
    psf_template = params["psf_template"]
    print("PSF template shape", np.shape(psf_template))
    dim = int(n)
    m = np.shape(psf_template)[0]
    #if m != dim:
    #    raise ValueError("PSF template dimension not equal patch size")
        
    if np.sum(psf_template) != 1:
        print("Normalizing PSF template to sum = 1")
        psf_template = psf_template/np.sum(psf_template)        
    return psf_template

def cart_to_pol(coords):
    """
    Takes cartesian (2D) coordinates and transforms them into polar.
    """
    if len(coords.shape) == 1:
        rho = np.sqrt(coords[0]**2 + coords[1]**2)
        phi = np.arctan2(coords[1], coords[0])
        return np.array((rho,phi))
    else:
        rho = np.sqrt(coords[:,0]**2 + coords[:,1]**2)
        phi = np.arctan2(coords[:,1], coords[:,0])
        return np.column_stack(rho,phi)

def pol_to_cart(coords):
    if len(coords.shape) == 1:
        x = coords[0]*np.cos(coords[1])
        y = coords[0]*np.sin(coords[0])
        return np.array((x,y))
    else:
        x = coords[:,0]*np.cos(coords[:,1])
        y = coords[:,0]*np.sin(coords[:,1])
        return np.column_stack(x,y)

def int_pol_to_cart(coords):
    if len(coords.shape) == 1:
        x = int(coords[0]*np.cos(coords[1]))
        y = int(coords[0]*np.sin(coords[0]))
        return np.array((x,y))
    else:
        x = coords[:,0]*np.cos(coords[:,1]).astype(int)
        y = coords[:,0]*np.sin(coords[:,1]).astype(int)
        return np.column_stack(x,y)

def grid_cart_to_pol(x,y):
    """
    Takes cartesian (2D) coordinates and transforms them into polar.
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def grid_pol_to_cart(r, phi):
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return (x,y)



shared_arr1 = 0
shared_arr2 = 0
## Single funcs for parallel processing
def init_array(arr1,arr2):
    global shared_arr1
    shared_arr1 = arr1
    global shared_arr2
    shared_arr2 = arr2
    
def pixel_calc(_args):
    patch = _args[0]
    p0 = _args[1]
    k = _args[2]
    T = _args[3]
    count = args[4]
    if patch is None:
        m = None,
        Cinv = None
        return
    
    m = np.mean(patch,axis = 0) # Calculate the mean of the column
    nc = 0
    print("here")
    for i in range(count*m.shape,(count+1)*m.shape):
        shared_arr1[i] = m[nc]
        print(m[nc],shared_arr1[i])
        nc += 1
    nc = 0
    # Calculate the covariance matrix
    S = sample_covariance(patch, _args[4], T)
    rho = shrinkage_factor(S, T) 
    F = diag_sample_covariance(S)
    C = covariance(rho, S, F)    
    cinv = np.linalg.inv(C)
    for i in range(count*cinv.shape,(count+1)*cinv.shape):
        shared_arr2[i] = cinv[nc]
        nc+=1
    #print("here for px",p0)
    #return np.array([p0,m,Cinv])


def covariance(rho, S, F):
    """
    Ĉ
    
    Shrinkage covariance matrix
    rho: shrinkage weight
    S: Sample covariance matrix
    F: Diagonal of sample covariance matrix
    """
    C = (1.0-rho)*S + rho*F
    #fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,8))
    #im1 = ax.imshow(C)
    #fig.colorbar(im1,ax = ax)
    return C

def sample_covariance(r, m, T):
    """
    Ŝ
    
    Sample covariance matrix
    r: observed intensity at position θk and time tl
    m: mean of all background patches at position θk
    T: number of temporal frames
    """
    #print(m.shape,r.shape)
    #S =  (1.0/T)*np.sum([np.outer((p-m).ravel(),(p-m).ravel().T) for p in r], axis=0)
    S = (1.0/T)*np.sum([np.cov(np.stack((p, m)), rowvar = False, bias = False) for p in r],axis = 0)
    return S
    
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
    top = (np.trace(np.dot(S,S)) + np.trace(S)**2 - 2.0*np.sum(np.array([d**2.0 for d in np.diag(S)])))
    bot = ((T+1.0)*(np.trace(np.dot(S,S))-np.sum(np.array([d**2.0 for d in np.diag(S)]))))
    return top/bot
