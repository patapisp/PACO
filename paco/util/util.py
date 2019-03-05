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

def GetRotatedPixels(x,y,p0,angles):
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

def resizeImage(image, scaleFactor):
    return cv2.resize(image,(0,0), fx = scaleFactor, fy = scaleFactor, interpolation = cv2.INTER_NEAREST)

def gaussian2d(x,y,A, sigma):
    return A*np.exp(-(x**2+y**2)/(2*sigma**2))

def gaussian2d_model(n,params):
    sigma = params["sigma"]
    dim = int(n/2)
    x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
    return np.exp(-((x+0.5)**2+(y+0.5)**2)/(2*sigma**2))  

def psftemplate_model(n, params):
    """
    Model using a psf template directly from the data. 
    Template should be normalized such that the sum equals 1.
    
    If model needs rescaling it is done here
    """
    psf_template = params["psf_template"]
    print("PSF template shape", np.shape(psf_template))
    dim = int(n)
    print("dim", dim)
    m = np.shape(psf_template)[0]
    if m != dim:
        raise ValueError("PSF template dimension not equal patch size")
        
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

