import numpy as np
import os
from astropy.io import fits
import matplotlib.pyplot as plt

import hcipy as hci
import vip_hci as vip

# Telescope and Wavelength parameters
D_tel = 6.5 # meter
wavelength = 2.5e-6 # meter

# Generate pupil and focal plane grids, and the wavefront propogator
pupil_grid = hci.make_pupil_grid(1024,D_tel)
focal_grid = hci.make_focal_grid(pupil_grid, 4, 8, wavelength=wavelength)
prop = hci.FraunhoferPropagator(pupil_grid, focal_grid)

# Build the pupil aperture

# Circular Aperture
#aperture = hci.circular_aperture(D_tel)
#aperture = hci.evaluate_supersampled(aperture, pupil_grid, 8)

# Magellan Aperture
aperture = hci.make_magellan_aperture()

# Adaptive Optics
F_mla = 0.03
N_mla = 8
D_mla = 1.0 / N_mla
#x = np.arange(-1,1,D_mla)
#mla_grid = hci.CartesianGrid(SeparatedCoords((x,x)))
#mla_shape = hci.rectangular_aperture(D_mla)
#microlens_array = hci.MicroLensArray(pupil_grid, mla_grid, F_mla * D_mla, mla_shape)
#sh_prop = hci.FresnelPropagator(pupil_grid, F_mla * D_mla)
im_stack = []
psfs = []
nFrames = 150
tInt = 1.5
# Generate the coronagraph
coro = hci.VortexCoronagraph(pupil_grid, charge=2, levels=8)
lyot_stop = hci.Apodizer(hci.circular_aperture(0.99*D_tel)(pupil_grid))

for i in range(nFrames):
    # Generate a wavefront
    #wf = hci.Wavefront(aperture,wavelength)
    wf = hci.Wavefront(aperture(pupil_grid),wavelength)
    wf.total_power = 100000

    # Atmospheric Distortion Layers
    fried_parameter = 5.5 # meter
    outer_scale = 20 # meter
    velocity = 1 # meter/sec

    # Single Layer
    Cn_squared = hci.Cn_squared_from_fried_parameter(fried_parameter, wavelength)
    layer = hci.InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)
    layer.t = i * tInt
    wf2 = layer(wf)

    # Multi Layer
    #spectral_noise_factory = hci.SpectralNoiseFactoryFFT(kolmogorov_psd, pupil_grid, 8)
    #turbulence_layers = hci.make_standard_multilayer_atmosphere(fried_parameter, wavelength=wavelength)
    #atmospheric_model = hci.AtmosphericModel(spectral_noise_factory, turbulence_layers)
    #wf2 = atmospheric_model(wf)
    #sci_img = prop(wf2).intensity
    #wf3 = microlens_array(wf2)
    #wfs_img = sh_prop(wf3).intensity

    # Generate surface aberration
    aberration = hci.SurfaceAberration(pupil_grid, 0.25*wavelength, D_tel)
    ab_wf = aberration(wf2)
    
    # Lyot Plane
    lyot_wf = coro(ab_wf)

    # Add a Lyot Stop
    lyot_img = prop(lyot_stop(lyot_wf))
    img = lyot_img
    img_ref = prop(wf)
    # Build a Detector
    flat_field = 0.1
    dark = 10
    detector = hci.NoisyDetector(focal_grid, dark_current_rate=dark, flat_field=flat_field)
    detector.integrate(lyot_img, tInt)
    image = detector.read_out()
    im_stack.append(image)
    img = prop(aberration(wf))
    detector.integrate(img, 1.5)
    psf = detector.read_out()
    psfs.append(psf)

psfs = np.asarray(psfs).reshape((nFrames,64,64))
hdu = fits.PrimaryHDU(np.asarray(im_stack).reshape((nFrames,64,64)))
hdu.writeto("testData/SimImages.fits")
hdu = fits.PrimaryHDU(np.asarray(im_stack).reshape((nFrames,64,64)))
hdu.writeto("testData/SimImages.fits")
