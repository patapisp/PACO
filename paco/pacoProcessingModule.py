import paco as paco
from paco.util.util import *
from pynpoint.core.processing import ProcessingModule
import numpy as np

class PACOModule(ProcessingModule):
    def __init__(self,
                 name_in = "paco",
                 image_in_tag = "im_arr",
                 psf_in_tag = None,
                 snr_out_tag = "paco_snr",
                 patch_size = 49,
                 scaling = 1.0,
                 algorithm = "fastpaco",
                 angles = None,
                 flux_calc = False,
                 psf_params = None,
                 psf_model = None,
                 cpu_limit = 1
    ):
        """
        Constructor of PACOModule.
        Parameters
        ----------
        name_in : str
            Unique name of the module instance.
        image_in_tag : str
            Tag of the database entry that contains the stack with images.
        psf_in_tag : str
            Tag of the database entry that contains the reference PSF that is used as fake planet.
            Can be either a single image (2D) or a cube (3D) with the dimensions equal to
            *image_in_tag*.
        snr_out_tag : str
            Tag of the database entry that contains the SNR map and unbiased flux estimation
            computed using one of the PACO algorithms
        patch_size : int
            Number of pixels in a circular patch in which the patch covariance is computed
        algorithm : str
            One of 'fastpaco' or 'fullpaco', depending on which PACO algorithm is to be run
        flux_calc : bool
            True if  fluxpaco is to be run, computing the unbiased flux estimation of 
            a set of companions.

        """
        super(PACOModule,self).__init__(name_in)
        self.m_image_in_port = self.add_input_port(image_in_tag)
        if psf_in_tag is not None:
            if psf_in_tag == image_in_tag:
                self.m_psf_in_port = self.m_image_in_port
            else:
                self.m_psf_in_port = self.add_input_port(psf_in_tag)
        else:
            self.m_psf_in_port = None
        self.m_snr_out_port = self.add_output_port(snr_out_tag)
        self.m_algorithm = algorithm
        self.m_patch_size = patch_size
        self.m_angles = angles
        self.m_flux_calc = flux_calc
        self.m_scale = scaling
        self.m_psf_params = psf_params
        self.m_cpu_lim = cpu_limit
        self.m_model_function = psf_model
        
    def run(self):
        """
        Run function for PACO
        """
        # Hardware settings
        cpu = self._m_config_port.get_attribute("CPU")
        if cpu>self.m_cpu_lim:
            cpu = self.m_cpu_lim
        # Read in science frames and psf model
        # Should add existance checks
        images = self.m_image_in_port.get_all()
        
        # Read in parallactic angles, and use the first frame as the 0 reference.
        if self.m_angles is not None:
            angles = self.m_angles
        else:
            angles = self.m_image_in_port.get_attribute("PARANG")
        angles = angles - angles[0]
        # Setup PACO
        if self.m_algorithm == "fastpaco":
            fp = paco.processing.fastpaco.FastPACO(angles = angles,
                                                   patch_size = self.m_patch_size)
        elif self.m_algorithm == "fullpaco":
            fp = paco.processing.fullpaco.FullPACO(angles = angles,
                                                   patch_size = self.m_patch_size)
        else:
            print("Please input either 'fastpaco' or 'fullpaco' for the algorithm")
        fp.setImageSequence(images)

        
        if self.m_psf_in_port is not None:
            psf = self.m_psf_in_port.get_all()
            fp.setPSF(psf)     
        elif self.m_psf_params is not None:
            psf = None
        
        # Run PACO
        # SNR = b/sqrt(a)
        # Flux estimate = b/a
        a,b  = fp.PACO(scale = self.m_scale,
                       model_params = self.m_psf_params,
                       model_name = self.m_model_function,
                       cpu = cpu)

        # Iterative, unbiased flux estimation
        if self.m_flux_calc:
            p0 = [0,0]
            eps = 0.1
            ests = [9999.0]
            fp.fluxEstimate(p0s,angles,eps,params,ests,scale)
        
        # Output
        snr = b/np.sqrt(a)
        self.m_snr_out_port.set_all(snr, data_dim=2)
        self.m_snr_out_port.close_port()
