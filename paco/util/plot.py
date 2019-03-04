"""
This module will include a variety of plotting functions for PACO
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_patches(patch, vmax=5):
    dim = len(patch)
    n = int(np.ceil(np.sqrt(dim)))
    m = int(np.ceil(dim/n))
    fig, ax = plt.subplots(nrows=m, ncols=n, figsize=(8,6))
    ax = ax.flatten()
    #ax = ax[:dim+1]
    for i in range(dim):
        ax[i].imshow(patch[i], vmax=vmax)
    return

