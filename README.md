# PACO Implementation
 
This package implements the algorithms developed by [Flasseur et. al 2018 [1]](https://www.aanda.org/articles/aa/abs/2018/10/aa32745-18/aa32745-18.html)

## Authors
Polychronis Patapis, _ETH Zurich_

Evert Nasedkin, _ETH Zurich_

## Directory Structure
* paco: Contains python modules that implement the PACO algorithm and various IO and utility functions.
 - core: File IO handline
 - processing: Implementation of PACO algorithms, and ADI preprocessing
 - util: Utility functions, including rotations, coordinate transformations, distributions and models

* testData: Contains toy dataset used in testing.
* output: Location of output files (or dirs? not sure yet...)

Run the Example.ipynb notebook to use. Currently this notebook generates a toy dataset, and runs the FullPACO algorithm to produce a signal-to-noise (SNR) map of the data.

## Algorithms
fullpaco - Algorithm 1 from [1].
fastpaco - Algorithm 2 from [1]. Adds preprocessing of statistics to reduce computation time at expense of accuracy.
fluxpaco - Algorithm 3 from [1]. Calculates unbiased flux of the detected source.

## Requirements
* Python 3.7
* pipenv
#### Environment
This package includes a pipenv environment file to include the necessary packages (listeted in the Pipfile)
