import numpy as np
import os
from load_atlas.load_tiss_prop import get_tiss_prop
from helpers import *


class Headvol:
    def __init__(self, filename=None):
        self.volume = None
        self.tiss_prop = None
        self.T_2digpts = None
        self.T_2mc = None
        self.T_2ras = None
        self.T_2ref = None
        self.orientation = None
        self.orientationOrig = None
        if filename is not None:
            self.load(filename)

        self.brain_vol = None
        self.scalp_vol = None
        self.center = None
        self.ref_pts = None

    def find_center(self):
        volsurf = self.volume
        bbox, _, _ = gen_bbox(volsurf)
        c = find_region_centers([bbox])
        self.center = c[0]
        return c[0]

    def load(self, filename, dims=None):
        with open(filename, "rb") as fid:
            self.img = np.fromfile(fid, dtype=np.uint8)

        i = filename.rfind(".vox")
        if dims is None:
            filename_img_dims = filename[:i] + "_dims.txt"
            if os.path.exists(filename_img_dims):
                dims = np.loadtxt(filename_img_dims, dtype=int)
        if dims is not None:
            self.img = np.uint8(np.reshape(self.img, dims))

        filename_img_tiss = filename[:i] + "_tiss_type.txt"
        if os.path.exists(filename_img_tiss):
            self.tiss_prop = get_tiss_prop(filename_img_tiss)
        # print(self.tiss_prop)

        filename_2digpts = filename[:i] + "2digpts.txt"
        if os.path.exists(filename_2digpts):
            self.T_2digpts = np.loadtxt(filename_2digpts)

        filename_2mc = filename[:i] + "2mc.txt"
        if os.path.exists(filename_2mc):
            self.T_2mc = np.loadtxt(filename_2mc)

        filename_2ras = filename[:i] + "2ras.txt"
        if os.path.exists(filename_2ras):
            self.T_2ras = np.loadtxt(filename_2ras)

        filename_2ref = filename[:i] + "2ref.txt"
        if os.path.exists(filename_2ref):
            self.T_2ref = np.loadtxt(filename_2ref)

        filename_orientation = filename[:i] + "_orientation.txt"
        if os.path.exists(filename_orientation):
            with open(filename_orientation, "r") as fd:
                orientation = fd.readline().strip()
                if orientation:
                    self.orientation = orientation
                    self.orientationOrig = orientation

    def set_ref_pts(self, ref_pts):
        self.ref_pts = ref_pts

    def set_brain_and_scalp_volumes(self, brain_vol, scalp_vol):
        """
        Set the brain and scalp volumes for the head volume.

        :param brain_vol: Numpy array representing the brain volume
        :param scalp_vol: Numpy array representing the scalp volume
        """
        self.brain_vol = brain_vol
        self.scalp_vol = scalp_vol

    def set_and_save_full_volume(self, volume_path, volume_data):
        """
        Set the full volume for the head volume and save it to the provided path.

        :param volume_path: Path where the volume should be saved
        :param volume_data: Numpy array representing the full volume data
        """
        self.volume = volume_data
        writevolbin2(self.volume, volume_path)
