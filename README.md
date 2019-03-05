# PACO Implementation
 
This package implements the algorithms developed by [Flasseur et. al 2018 [1]](https://www.aanda.org/articles/aa/abs/2018/10/aa32745-18/aa32745-18.html)

## Authors
Polychronis Patapis, _ETH Zurich_

Evert Nasedkin, _ETH Zurich_

email: evertn@student.ethz.ch

## Usage
Currently, the Example and Data_from_gabriele notebooks provide the best overview of usage. Creation of simulated data or FITS file IO is still handled by the user, with improvements planned.

To use PACO, import one of the the processing modules (Fast or Full PACO are currently available). Fast PACO is recommended, as the loss of accuracy in the SNR is small, while the computation time is much, much lower.

Create an instance of the class, specifying the patch size:

```
import paco.processing.fastpaco as fastPACO
fp = fastPACO.fastPACO(patch_size = 5)
```

Set the stack of frames to be processed:

```fp.set_image_sequence(image_sequence)```

Supplying the list of rotation angles between frames, and the pixel scaling, run PACO:

```a,b = fp.PACO(angles = angle_list, scale = scale)```

This returns 2D maps for a and b, the the inverse variance and flux estimate respectively. The signal to noise can be computed as b/sqrt(a).

## Directory Structure
* paco: Contains python modules that implement the PACO algorithm and various IO and utility functions.
 - core: File IO handline
 - processing: Implementation of PACO algorithms and ADI preprocessing
 - util: Utility functions, including rotations, coordinate transformations, distributions and models

* testData: Contains toy dataset used in testing.
* output: Location of output files (or dirs? not sure yet...)

Run the Example.ipynb notebook to use. Currently this notebook generates a toy dataset, and runs the FastPACO algorithm to produce a signal-to-noise (SNR) map of the data.

## Algorithms
fullpaco - Algorithm 1 from [1].
fastpaco - Algorithm 2 from [1]. Adds preprocessing of statistics to reduce computation time at expense of accuracy.
fluxpaco - Algorithm 3 from [1]. Calculates unbiased flux of the detected source.

## Requirements
* Python 3.0
* pipenv

#### Environment
This package includes a pipenv environment file to include the necessary packages (listeted in the Pipfile)
