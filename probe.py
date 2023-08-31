from temp_shared_globals import optpos, refpts, tiss_prop

import scipy.io

class Probe:
    def __init__(self):
        self.optpos = None
        self.path_probe = None
        self.meas_list = None
        self.nsrc = None

    def load_probe(self, filename = r'C:\Users\User\Documents\masha\alex_mrtim\2022-04-25_001.nirs'):
        self.path_probe = filename
        filedata = scipy.io.loadmat(self.path_probe)
        SD = filedata["SD"]
        self.meas_list = SD[0]['MeasList'][0]
        self.nsrc = SD[0]['nSrcs'][0][0][0]
        return self.meas_list, self.nsrc
    
    def set_optpos(self, optpos):
        self.optpos = optpos

    def get_ref_digpoints(self, digpts_file_path='../digpts.txt'):
        with open(digpts_file_path, "r") as file:
            content = file.read()

        # Define the points of interest
        points_of_interest = ['nz', 'ar', 'al', 'cz', 'iz']

        # Initialize the dictionary to store coordinates
        coordinates_dict = {}

        # Process the lines and extract coordinates
        lines = content.strip().split('\n')

        for line in lines:
            key, values = line.split(': ')
            if key in points_of_interest:
                coordinates_dict[key] = list(map(float, values.split()))

        self.digpts_ref = coordinates_dict
        print(self.digpts_ref)
        return
    
    def get_optpoints(self, optpts_file_path='../digpts.txt'):
        with open(optpts_file_path, 'r') as f:
            lines = f.readlines()

        # Initialize the dictionary to store the coordinates
        coordinates = {}

        # Flag to indicate when to start reading coordinates
        start_reading = False

        # Loop through the lines and extract the coordinates
        for line in lines:
            if line.startswith('s1:'):
                start_reading = True
            
            if start_reading:
                parts = line.strip().split(':')
                label = parts[0].strip()
                coords = [float(x) for x in parts[1].strip().split()]
                coordinates[label] = coords

        self.optpos_dict = coordinates
        # Print the coordinates
        for label, coords in coordinates.items():
            print(f"{label}: {coords}")

        return

