from temp_shared_globals import refpts
import scipy.io


class Probe:
    def __init__(self):
        self.optpos = None
        self.path_probe = None
        self.meas_list = None
        self.nsrc = None
        self.optpos_dict = None
        self.reg_optpos = None

    def reset():
        pass

    def load_probe_from_nirs(self, filename=None):
        """
        Loads probe data from a .nirs file.

        Args:
            filename (str): Path to the .nirs file. Defaults to None.
        """
        if not filename:
            raise ValueError("A valid filename must be provided.")

        self.path_probe = filename
        filedata = scipy.io.loadmat(self.path_probe)
        SD = filedata["SD"]
        self.meas_list = SD[0]["MeasList"][0]
        self.nsrc = SD[0]["nSrcs"][0][0][0]

    def set_optpos(self, optpos):
        """Sets the optpos attribute.

        Args:
            optpos: The value to set for optpos.
        """
        self.optpos = optpos

    def read_refpoints_from_digpts(self, digpts_file_path) -> dict:
        """
        Retrieves reference digitized points.

        Args:
            digpts_file_path (str): Path to the digitized points file.
        """
        points_of_interest = ["nz", "ar", "al", "cz", "iz"]
        coordinates_dict = {}
        with open(digpts_file_path, "r") as file:
            for line in file:
                key, values = line.split(": ")
                if key in points_of_interest:
                    coordinates_dict[key] = list(map(float, values.split()))

        self.digpts_ref = coordinates_dict
        return self.digpts_ref

    def read_optpoints_from_digpts(self, optpts_file_path) -> dict:
        """
        Reads optpoints from a digitized points file.

        Args:
            optpts_file_path (str): Path to the optpoints file. Defaults to "../digpts.txt".
        """
        coordinates = {}
        start_reading = False

        with open(optpts_file_path, "r") as f:
            for line in f:
                if line.startswith("s1:"):
                    start_reading = True
                if start_reading:
                    label, coords_str = line.strip().split(":")
                    coordinates[label] = [float(x) for x in coords_str.split()]

        self.optpos_dict = coordinates
        return self.optpos_dict
