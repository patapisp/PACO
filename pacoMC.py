import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import paco.processing.paco as p
import paco.processing.fullpaco as f_paco
import paco.processing.fastpaco as fastPACO

from paco.util.util import *
import cv2 as cv2
import numpy as np
import pandas as pd

import multiprocessing
import multiprocessing.pool

# MC Parameters
nFrames = 5
angle = 60
angles = np.linspace(0,angle,nFrames)
psig = [(30,30)]
nTrials = 2
nProcess = min(nTrials,8)
np.random.seed(4096)

OUTPUT_DIR = "output/MC_V1/"
def GenerateImageStack(nFrames,angles,signalStrength,noiseLevel,dim = 100):  
    # Hardcoded source location
    p0 = (30,30)
    mean = 0

    images = [np.reshape(np.random.normal(mean, noiseLevel, dim**2), (dim,dim)) for j in range(nFrames)]
    X,Y = np.meshgrid(np.arange(-dim/2, dim/2),np.arange(-dim/2, dim))
    xx, yy = np.meshgrid(np.arange(-30, dim-30),np.arange(-30, dim-30))
    s = gaussian2d(xx,yy,signalStrength/np.sqrt(nFrames), 2)

    #images_signal = [i + s for i in images]
    rot_noise = np.array([rotateImage(images[j], angles[j]) for j in range(nFrames)])
    rot_sigs = np.array([rotateImage(s, angles[j]) for j in range(nFrames)])
    rot_images = np.array([rot_noise[j] + rot_sigs[j] for j in range(nFrames)])
    return rot_images

def GetImPatch(im,px,width):
        k = int(width/2)
        nx, ny = np.shape(im.shape)[:2]
        if px[0]+k > nx or px[0]-k < 0 or px[1]+k > ny or px[1]-k < 0:
            #print("pixel out of range")
            return None
        patch = im[i][int(px[0])-k:int(px[0])+k, int(px[1])-k:int(px[1])+k]
        return patch

def pacoTrial(im_stack):
    #im_stack = GenerateImageStack(nFrames,angles,5.0,1.0)
    fp = fastPACO.FastPACO(image_stack = im_stack,
                           angles = angles)

    a,b = fp.PACO(cpu = 1,
                  model_params={"sigma":2.0},
                  model_name = gaussian2dModel)
    est = fp.fluxEstimate(phi0s = psig,
                          eps = 0.05,
                          initial_est = 0.0)
    
    return (a,b,est)


trials = [GenerateImageStack(nFrames,angles,5.0,1.0) for i in range(nTrials)]

#for t in trials:
#    print(t[0])
pool = multiprocessing.Pool(nProcess)
data = pool.map(pacoTrial,trials)
pool.close()
pool.join()
alist,blist,flux = [],[],[]
for d in data:
    alist.append(d[0])
    blist.append(d[1])
    flux.append(d[2])
alist = np.array(alist)
blist = np.array(blist)
print(flux)
flux = np.array(flux).flatten()
if(not os.path.isdir(os.getcwd() + '/' +OUTPUT_DIR)):
    os.mkdir(os.getcwd() + '/' +OUTPUT_DIR)
np.save(OUTPUT_DIR + "a_mc.npy",alist)
np.save(OUTPUT_DIR + "b_mc.npy",blist)
np.save(OUTPUT_DIR + "est_mc.npy",flux)

var = []
peak = []
snr = []
sig = []
var_full = []
#Should do this with numpy slicing...
for i in range(nTrials):
    var.append(alist[i][30][30])
    peak.append(blist[i][30][30]/ alist[i][30][30])
    snr.append(blist[i][30][30]/ np.sqrt(alist[i][30][30]))
    sig.append(blist[i][30][30])
    var_full.append(np.var(blist[i]/alist[i]))
var = np.array(var)
peak = np.array(peak)
snr = np.array(snr)
var_full = np.array(var_full)

df = pd.DataFrame()
df['flux'] = flux
df['peak_flux'] = peak
df['peak_var'] = var
df['frame_var'] = var_full

df.to_csv(OUTPUT_DIR + "mc_stats.csv")
