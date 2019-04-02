import paco as paco
from pynpoint.core import ProcessingModule

class PACOModule(ProcessingModule):
    def __init__(self,
                 name_in = "paco",
                 image_in_tag = "im_arr",
                 psf_in_tag = "im_psf",
                 snr_out_tag = "paco_snr",
                 patch_size = 49,
                 scaling = 1.0
                 algorithm = "fastpaco",
                 flux_calc = False
                 psf_params = {"psf_template":s}
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
        if psf_in_tag == image_in_tag:
            self.m_psf_in_port = self.m_image_in_port
        else:
            self.m_psf_in_port = self.add_input_port(psf_in_tag)

        self.m_snr_out_port = self.add_output_port(snr_out_tag)
        self.m_algorithm = algorithm
        self.m_flux_calc = flux_calc
        self.m_scale = scaling
        self.m_psf_params = psf_params

    def run(self):
        images = self.m_image_in_port.get_all()
        psf = self.m_psf_in_port.get_all()
        cpu = self._m_config_port.get_attribute("CPU")
        angles = self.m_image_in_port.get_attribute("PARANG")
        if self.algorithm == "fastpaco":
            fp = paco.processing.fastpaco.FastPACO(patch_size = self.m_patch_size)
        elif self.algorithm == "fullpaco":
             fp = paco.processing.fullpaco.FullPACO(patch_size = self.m_patch_size)

        fp.set_image_sequence(images)
        fp.setPSF(psf)
        a,b  = fp.PACO(angles,self.m_scale,psf,params = params=self.m_psf_params)

        if self.m_flux_calc:
            p0 = [0,0]
            eps = 0.1
            ests = [9999.0]
            fp.fluxEstimate(p0s,angles,eps,params,ests,scale)
        snr = b/sqrt(a)
        self.m_snr_output_port.set_all(snr, data_dim=2)
        self.m_snr_output_port.close()
