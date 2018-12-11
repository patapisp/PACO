"""
Implementation of FITS file input

Reads in a single or series of images from an ADI fits file
Based on Pynpoint FitsReading module, so that the PACO package
can easily be transitioned into pynpoint in the future.
"""
import os
import sys
import warnings

from astropy.io import fits


class ReadInFitsFile():
    def __init__(self,
                 name_in = None,
                 dir_in = None):
        """
        ReadInFitsFile Constructor

        :filename: List of names of a fits files
        :directory: Location of the fits files
        :hdulist: input from reading in a fits file
        :images: Individual frames from the fits file
        :n_images: Number of individual frames
        """
        self.filename = name_in
        self.directory = dir_in
        self.hdulist = None
        self.images = None
        self.n_images = None
        return
    def __del__(self):
        """
        ReadInFitsFile Destructor      
        Ensure that the fits file is closed
        """
        self.hdulist.close()
        
    def set_filename(self, file_name):
        """
        file_name: filename ending in .fits
        """
        if not file_name.endswith(".fits"):
            raise ValueError("Input 'file_name' requires the FITS extension.")
        self.filename = file_name

    def set_directory(self, directory):
        """
        directory: location of fits files
        """
        if not directory.endswith("/"):
            directory = directory + "/"
        self.directory = directory

    def open_one_fits(self, file_name):
        """
        file_name: fits filename
        
        Reads in the header and data from a single fits file
        """
        if not file_name.endswith(".fits"):
            raise ValueError("Please enter a filename")
        if not self.directory.endswith("/"):
            raise ValueError("Please enter a directory")
        self.hdulist = fits.open(self.directory + file_name)
        # From Pynpoint,not sure why byteswapping
        self.images = hdulist[0].data.byteswap().newbyteorder()
        
        return self.hdulist[0].header, self.images.shape


    def run(self):
        """
        Reads in all fits files in self.directory
        
        TODO:
        Implement central database storage, as in pynpoint.
        For now, only use open_one_fits
        """
        self.directory = os.path.join(self.directory, "")
        files = []
        for filename in os.listdir(self.directory):
            if filename.endswith('.fits'):
                files.append(filename)

        files.sort()
        assert(files), 'No FITS files found in %s.' % self.m_input_location

        for i, fits_file in enumerate(files):
            header, shape = self.open_one_fits(fits_file)
            # Store some attributes

        return
    
