from helpers import printProgressBar, find_region_centers, gen_bbox
import trimesh
from skimage import measure
import numpy as np
from headvol import Headvol
import os
import nibabel as nib


class Fw_model:
    class Mesh:
        def __init__(self, mesh=None, reduced_mesh=None):
            self.vertices = None
            self.faces = None

            self.reduced_vertices = None
            self.reduced_faces = None

            if mesh is not None:
                self.vertices = mesh.vertices
                self.faces = mesh.faces
            if reduced_mesh is not None:
                self.reduced_vertices = reduced_mesh.vertices
                self.reduced_faces = reduced_mesh.faces

        def set_faces(self, faces):
            self.faces = faces

        def set_vertices(self, vertices):
            self.vertices = vertices

        def set_faces_and_vertices(self, faces, vertices):
            self.set_faces(faces)
            self.set_vertices(vertices)

        def reduce_mesh(self, num_of_faces=60000, filename=None):
            reduced_mesh = trimesh.Trimesh(
                self.vertices,
                self.faces).simplify_quadratic_decimation(num_of_faces)

            self.reduced_vertices = reduced_mesh.vertices
            self.reduced_faces = reduced_mesh.faces

            if filename is not None:
                reduced_mesh.export(filename)

            return reduced_mesh.vertices, reduced_mesh.faces

        def set_reduced_mesh(self, reduced_mesh):
            self.reduced_vertices = reduced_mesh.vertices
            self.reduced_faces = reduced_mesh.faces

        # def save_mesh(self):

    def __init__(self, headvol=None):

        self.headvol = headvol
        # self.brain_vol = None
        # self.scalp_vol = None
        # self.headvol_center = None

        self.mesh_orig = self.Mesh()
        self.mesh_scalp_orig = self.Mesh()
        self.Adot = None
        self.Adot_scalp = None
        self.path_projVoltoMesh_brain = None
        self.path_projVoltoMesh_scalp = None

        self.vol_shape = None
        self.tiss_prop = None

        # self.tiss_prop = [
        #     {
        #         "absorption": [0.0191, 0.0191],
        #         "scattering": [0.66, 0.66],
        #         "anisotropy": [0.001],
        #         "refraction": [1.0],
        #     },
        #     {
        #         "absorption": [0.0136, 0.0136],
        #         "scattering": [0.86, 0.86],
        #         "anisotropy": [0.001],
        #         "refraction": [1.0],
        #     },
        #     {
        #         "absorption": [0.0026, 0.0026],
        #         "scattering": [0.01, 0.01],
        #         "anisotropy": [0.001],
        #         "refraction": [1.0],
        #     },
        #     {
        #         "absorption": [0.0186, 0.0186],
        #         "scattering": [1.1, 1.1],
        #         "anisotropy": [0.001],
        #         "refraction": [1.0],
        #     },
        #     {
        #         "absorption": [0.0186, 0.0186],
        #         "scattering": [1.1, 1.1],
        #         "anisotropy": [0.001],
        #         "refraction": [1.0],
        #     },
        # ]

        # self.tiss_prop2 = [
        #     {
        #         "absorption": [0.0191],
        #         "scattering": [0.66],
        #         "anisotropy": [0.001],
        #         "refraction": [1.0],
        #     },
        #     {
        #         "absorption": [0.0136],
        #         "scattering": [0.86],
        #         "anisotropy": [0.001],
        #         "refraction": [1.0],
        #     },
        #     {
        #         "absorption": [0.0026],
        #         "scattering": [0.01],
        #         "anisotropy": [0.001],
        #         "refraction": [1.0],
        #     },
        #     {
        #         "absorption": [0.0186],
        #         "scattering": [1.1],
        #         "anisotropy": [0.001],
        #         "refraction": [1.0],
        #     },
        #     {
        #         "absorption": [0.0186],
        #         "scattering": [1.1],
        #         "anisotropy": [0.001],
        #         "refraction": [1.0],
        #     },
        # ]

    def reset():
        pass

    def _marching_cubes(self, volume, filename, isovalue=0.9):
        vertices, faces, _, _ = measure.marching_cubes(volume, isovalue)
        return vertices, faces

    def load_masks(
            self,
            atlas_viewer,
            anatomy_folder=r"C:\Users\User\Documents\masha\Alex\atlas_python\icbm_seg",
            binary_vol_t_path=None,
            working_directory=None):
        new_working_directory = atlas_viewer.working_dir
        if not os.path.exists(new_working_directory):
            os.makedirs(new_working_directory)
        os.chdir(new_working_directory)

        img = nib.load(os.path.join(anatomy_folder, "gm.nii"))
        gm = img.get_fdata()[:]
        gm[gm != 0] = 1

        img = nib.load(os.path.join(anatomy_folder, "scalp.nii"))
        skin = img.get_fdata()[:]
        skin[skin != 0] = 1

        mask_paths = {
            "scalp": os.path.join(anatomy_folder, "scalp.nii"),
            "skull": os.path.join(anatomy_folder, "skull.nii"),
            "csf": os.path.join(anatomy_folder, "csf.nii"),
            "gray_matter": os.path.join(anatomy_folder, "gm.nii"),
            "white_matter": os.path.join(anatomy_folder, "wm.nii"),
        }

        mask_values = {}
        updated_masks = {}
        mask_value = 1
        for mask_name, mask_path in mask_paths.items():
            mask = nib.load(mask_path).get_fdata()[:]
            mask[mask != 0] = 1
            updated_mask = self.update_mask(mask, mask_value)
            updated_masks[mask_name] = updated_mask
            mask_values[mask_name] = mask_value
            mask_value += 1
            # print(mask_name)
            # print(np.unique(updated_mask[updated_mask != 0]))

        full_volume = np.zeros(updated_mask.shape, dtype=int)
        for updated_mask in updated_masks.values():
            full_volume += updated_mask

        self.headvol.set_brain_and_scalp_volumes(gm, skin)
        atlas_viewer.binary_vol_path = os.path.join(atlas_viewer.working_dir,
                                                    "myvolume.bin")
        self.headvol.set_and_save_full_volume(atlas_viewer.binary_vol_path,
                                              full_volume)
        self.headvol.vol_shape = full_volume.shape

    def build_pial_surf(self, filename="bGM.nii"):
        vertices, faces = self._marching_cubes(self.headvol.brain_vol,
                                               filename)
        self.mesh_orig.set_faces_and_vertices(faces, vertices)
        return vertices, faces

    def build_head_surf(self, filename="skin.nii"):
        vertices, faces = self._marching_cubes(self.headvol.scalp_vol,
                                               filename)
        self.mesh_scalp_orig.set_faces_and_vertices(faces, vertices)
        return vertices, faces

    def projVoltoMesh_brain(self,
                            filepath,
                            filename="projVoltoMesh_brain.npy"):

        if os.path.exists(os.path.join(filepath, filename)):
            self.path_projVoltoMesh_brain = os.path.join(filepath, filename)
            return

        if not hasattr(self.mesh_orig, 'reduced_vertices') or not hasattr(
                self.mesh_orig,
                'reduced_faces'):  # Check for attribute existence
            raise ValueError("mesh_orig is missing required attributes.")

        nodeX = self.mesh_orig.reduced_vertices
        elem = self.mesh_orig.reduced_faces

        if not hasattr(self.headvol,
                       'brain_vol'):  # Check for attribute existence
            raise ValueError("headvol is missing required attributes.")

        if not os.path.isdir(
                filepath):  # Check if the filepath directory exists
            raise ValueError(f"The directory {filepath} does not exist.")

        nNode = np.size(nodeX, 0)

        i_headvol = np.where(self.headvol.brain_vol != 0)
        i_headvol_flat = np.ravel_multi_index(i_headvol,
                                              self.headvol.brain_vol.shape)

        nC = np.size(i_headvol_flat, 0)
        Amap = np.zeros((nC, 1), dtype=np.int64)
        mapMesh2Vox = np.ones((nNode, 4000), dtype=np.int64)
        NVoxPerNode = np.zeros((nNode, 1), dtype=np.int64)

        # Unpack the indices along each dimension
        x_indices, y_indices, z_indices = i_headvol

        # Convert indices to x, y, z coordinates
        x_coordinates = x_indices
        y_coordinates = y_indices
        z_coordinates = z_indices

        # Display the x, y, z coordinates
        ii = int(0)
        nmiss = 0
        h = 15  # with 15 we miss < 1% of total number of cortex voxels

        printProgressBar(0,
                         nC,
                         prefix="projVoltoMesh_brain progress:",
                         suffix="Complete",
                         length=50)
        # print("nC", nC)
        update_interval = int(np.ceil(nC / 100))
        for x, y, z in zip(x_coordinates, y_coordinates, z_coordinates):
            # print(f"Voxel at coordinates ({x}, {y}, {z}) is gray matter.")
            if ii % update_interval == 1:
                printProgressBar(
                    ii,
                    nC,
                    prefix="projVoltoMesh_brain progress:",
                    suffix="Complete",
                    length=50,
                )
                # print(f'{ii} of {nC}')
            i_nX = np.where((nodeX[:, 0] > x - h)
                            & (nodeX[:, 0] < x + h)
                            & (nodeX[:, 1] > y - h)
                            & (nodeX[:, 1] < y + h)
                            & (nodeX[:, 2] > z - h)
                            & (nodeX[:, 2] < z + h))[0]
            if i_nX.size > 0:
                # rsep: get the distances from [x y z] to all the points in nodeX(i_nX,:).
                rsep = np.linalg.norm(nodeX[i_nX, :] - np.ones(
                    (len(i_nX), 1)) * [x, y, z],
                    axis=1)
                imin = np.argmin(rsep)
                Amap[ii] = i_nX[imin]
                NVoxPerNode[Amap[ii]] += 1  # might be useful?
                # print('NVoxPerNode', np.shape(NVoxPerNode))
                # print('Amap', np.shape(Amap))

                mapMesh2Vox[Amap[ii],
                            NVoxPerNode[Amap[ii]] - 1] = i_headvol_flat[ii]
                ii += 1
            else:
                nmiss += 1  # temporary var, delete when everything works

        # print("miss", nmiss)
        ciao = np.where(mapMesh2Vox == 0)
        mapMesh2Vox[ciao] = 1

        np.save(self.path_projVoltoMesh_brain, mapMesh2Vox)

        return mapMesh2Vox, NVoxPerNode, Amap

    def projVoltoMesh_scalp(self,
                            filepath,
                            filename="projVoltoMesh_scalp.npy"):

        if os.path.exists(os.path.join(filepath, filename)):
            self.path_projVoltoMesh_scalp = os.path.join(filepath, filename)
            return

        nodeX = self.mesh_scalp_orig.reduced_vertices
        elem = self.mesh_scalp_orig.reduced_faces
        nNode = np.size(nodeX, 0)

        i_headvol = np.where(self.headvol.scalp_vol != 0)
        i_headvol_flat = np.ravel_multi_index(i_headvol,
                                              self.headvol.scalp_vol.shape)

        nx = self.headvol.scalp_vol.shape[0]
        ny = self.headvol.scalp_vol.shape[1]
        nz = self.headvol.scalp_vol.shape[2]

        nmiss = 0
        nxy = nx * ny
        nC = np.size(i_headvol_flat, 0)

        Amap = np.zeros((nC, 1), dtype=np.int64)
        mapMesh2Vox = np.ones((nNode, 4000), dtype=np.int64)
        NVoxPerNode = np.zeros((nNode, 1), dtype=np.int64)

        # Unpack the indices along each dimension
        x_indices, y_indices, z_indices = i_headvol

        # Convert indices to x, y, z coordinates
        x_coordinates = x_indices
        y_coordinates = y_indices
        z_coordinates = z_indices

        h = 15
        ii = int(0)

        update_interval = int(np.ceil(nC / 100))
        printProgressBar(0,
                         nC,
                         prefix="projVoltoMesh_scalp progress:",
                         suffix="Complete",
                         length=50)
        for x, y, z in zip(x_coordinates, y_coordinates, z_coordinates):
            # print(f"Voxel at coordinates ({x}, {y}, {z}) is gray matter.")
            if ii % update_interval == 1:
                printProgressBar(
                    ii,
                    nC,
                    prefix="projVoltoMesh_scalp progress:",
                    suffix="Complete",
                    length=50,
                )
            i_nX = np.where((nodeX[:, 0] > x - h)
                            & (nodeX[:, 0] < x + h)
                            & (nodeX[:, 1] > y - h)
                            & (nodeX[:, 1] < y + h)
                            & (nodeX[:, 2] > z - h)
                            & (nodeX[:, 2] < z + h))[0]
            if i_nX.size > 0:
                # rsep: get the distances from [x y z] to all the points in nodeX(i_nX,:).
                rsep = np.linalg.norm(nodeX[i_nX, :] - np.ones(
                    (len(i_nX), 1)) * [x, y, z],
                    axis=1)
                imin = np.argmin(rsep)
                Amap[ii] = i_nX[imin]
                NVoxPerNode[Amap[ii]] += 1  # might be useful?
                # print('NVoxPerNode', np.shape(NVoxPerNode))
                # print('Amap', np.shape(Amap))

                mapMesh2Vox[Amap[ii],
                            NVoxPerNode[Amap[ii]] - 1] = i_headvol_flat[ii]
                ii += 1
            else:
                nmiss += 1  # temporary var, delete when everything works

        ciao = np.where(mapMesh2Vox == 0)
        mapMesh2Vox[ciao] = 1

        self.path_projVoltoMesh_scalp = os.path.join(filepath, filename)
        np.save(self.path_projVoltoMesh_scalp, mapMesh2Vox)

        return mapMesh2Vox, NVoxPerNode, Amap

    def get_projVolToMesh_brain(self):
        mapMesh2Vox = np.load(self.path_projVoltoMesh_brain)
        return mapMesh2Vox

    def get_projVolToMesh_scalp(self):
        mapMesh2Vox = np.load(self.path_projVoltoMesh_scalp)
        return mapMesh2Vox
