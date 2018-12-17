"""
This file will contain basic utility functions

- Rotations
- Coordinate transformations
- Any other commonly used calculations
"""

import numpy as np
import cv2


def rotateImage(image, angle):
    
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def gaussian2d(x,y,A, sigma):
    return A*np.exp(-(x**2+y**2)/(2*sigma**2))

def gaussian2d_model(n,sigma):
    dim = int(n/2)
    x, y = np.meshgrid(np.arange(-dim, dim), np.arange(-dim, dim))
    return np.exp(-((x+0.5)**2+(y+0.5)**2)/(2*sigma**2))     

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

