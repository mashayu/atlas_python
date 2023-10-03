from fw_model import Fw_model
from probe import Probe
from headvol import Headvol


class AtlasViewer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize the contained classes
            cls._instance.probe = Probe()
            cls._instance.headvol = Headvol()
            cls._instance.fw_model = Fw_model(cls._instance.headvol)
            cls._instance.working_dir = None
            cls._instance.atlas_dir = None
            cls._instance.anatomy_dir = None
            cls._instance.probe_dir = None
            cls._instance.binary_vol_path = None
            cls._instance.binary_vol_t_path = None
        return cls._instance

    def reset_probe(self):
        """Reset the state of the Probe object."""
        self.probe.reset()

    def reset_fwmodel(self):
        """Reset the state of the FWModel object."""
        self.fwmodel.reset()

    def reset_all(self):
        """Reset the state of both Probe and FWModel."""
        self.reset_probe()
        self.reset_fwmodel()
