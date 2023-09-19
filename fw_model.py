from helpers import printProgressBar, writevolbin, find_region_centers, gen_bbox
import trimesh
from skimage import measure
import numpy as np


class Fw_model:

    class Headvol:
        def __init__(self):
            self.volume = None
            self.brain_vol = None
            self.scalp_vol = None
            self.center = None

        def find_center(self):
            volsurf = self.volume
            bbox,_,_ = gen_bbox(volsurf)
            print('bbox', bbox)
            c = find_region_centers([bbox])
            self.center = c[0]
            return c[0]

    class Mesh:
        def __init__(self, mesh = None, reduced_mesh = None):
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


        def reduce_mesh(self, num_of_triangles = 40000, filename = None):
            reduced_mesh = trimesh.Trimesh(self.vertices, self.faces).simplify_quadratic_decimation(num_of_triangles)

            self.reduced_vertices = reduced_mesh.vertices
            self.reduced_faces = reduced_mesh.faces

            if filename is not None:
                reduced_mesh.export(filename+'.stl')

            return reduced_mesh.vertices, reduced_mesh.faces
        
        def set_reduced_mesh(self, reduced_mesh):
            self.reduced_vertices = reduced_mesh.vertices
            self.reduced_faces = reduced_mesh.faces

    def __init__(self):
        self.headvol = self.Headvol()
        #self.brain_vol = None
        #self.scalp_vol = None
        #self.headvol_center = None


        self.mesh_orig = self.Mesh()
        self.mesh_scalp_orig = self.Mesh()
        self.Adot = None
        self.Adot_scalp = None
        self.path_projVoltoMesh_brain= None
        self.path_projVoltoMesh_scalp = None

        self.vol_shape = None

        self.tiss_prop = [
                {"absorption": [0.0191,0.0191], "scattering": [0.66,0.66], "anisotropy": [0.001], "refraction": [1.0]},
                {"absorption": [0.0136,0.0136], "scattering": [0.86,0.86], "anisotropy": [0.001], "refraction": [1.0]},
                {"absorption": [0.0026,0.0026], "scattering": [0.01,0.01], "anisotropy": [0.001], "refraction": [1.0]},
                {"absorption": [0.0186,0.0186], "scattering": [1.1,1.1], "anisotropy": [0.001], "refraction": [1.0]},
                {"absorption": [0.0186,0.0186], "scattering": [1.1,1.1], "anisotropy": [0.001], "refraction": [1.0]},
            ]
        
        self.tiss_prop2 = [
                {"absorption": [0.0191], "scattering": [0.66], "anisotropy": [0.001], "refraction": [1.0]},
                {"absorption": [0.0136], "scattering": [0.86], "anisotropy": [0.001], "refraction": [1.0]},
                {"absorption": [0.0026], "scattering": [0.01], "anisotropy": [0.001], "refraction": [1.0]},
                {"absorption": [0.0186], "scattering": [1.1], "anisotropy": [0.001], "refraction": [1.0]},
                {"absorption": [0.0186], "scattering": [1.1], "anisotropy": [0.001], "refraction": [1.0]}
            ]

    def set_volume(self, brain_vol, scalp_vol):
        self.headvol.brain_vol = brain_vol
        self.headvol.scalp_vol = scalp_vol

    def set_volume2(self, volume_path, volume_file):
        #vol = nib.load(volume_file)
        #self.volume = vol.get_fdata()[:]
        self.headvol.volume = volume_file
        writevolbin(self.headvol.volume, volume_path)

    
    def build_pial_surf(self, filename = 'bGM.nii'):
        isovalue = 0.9 #questionable parameter
        #img = nib.load(filename)
        #gm = img.get_fdata()[:]
        vertices, faces, normals, values = measure.marching_cubes(self.headvol.brain_vol, isovalue)
        self.mesh_orig.vertices = vertices
        self.mesh_orig.faces = faces
        return vertices, faces
    
    def build_head_surf(self, filename = 'skin.nii'):
        isovalue = 0.9 #questionable parameter
        #img = nib.load(filename)
        #gm = img.get_fdata()[:]
        vertices, faces, normals, values = measure.marching_cubes(self.headvol.scalp_vol, isovalue)
        self.mesh_scalp_orig.vertices = vertices
        self.mesh_scalp_orig.faces = faces
        return vertices, faces

    
    def projVoltoMesh_brain(self):
        nodeX = self.mesh_orig.reduced_vertices
        elem  = self.mesh_orig.reduced_faces
        nNode = np.size(nodeX,0)

        i_headvol = np.where(self.headvol.brain_vol != 0)
        i_headvol_flat = np.ravel_multi_index(i_headvol, self.headvol.brain_vol.shape)

        nC = np.size(i_headvol_flat,0)
        Amap = np.zeros((nC, 1), dtype=np.int64)
        mapMesh2Vox = np.ones((nNode, 2000), dtype=np.int64)
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

        printProgressBar(0, nC, prefix = 'projVoltoMesh_brain progress:', suffix = 'Complete', length = 50)
        print("nC",nC)
        update_interval = int(np.ceil(nC / 100))
        for x, y, z in zip(x_coordinates, y_coordinates, z_coordinates):
            #print(f"Voxel at coordinates ({x}, {y}, {z}) is gray matter.")
            if ii % update_interval == 1:
                printProgressBar(ii, nC, prefix = 'projVoltoMesh_brain progress:', suffix = 'Complete', length = 50)
                #print(f'{ii} of {nC}')
            i_nX = np.where((nodeX[:, 0] > x - h) & (nodeX[:, 0] < x + h) &
                            (nodeX[:, 1] > y - h) & (nodeX[:, 1] < y + h) &
                            (nodeX[:, 2] > z - h) & (nodeX[:, 2] < z + h))[0]
            if i_nX.size > 0:
                # rsep: get the distances from [x y z] to all the points in nodeX(i_nX,:).
                rsep = np.linalg.norm(nodeX[i_nX, :] - np.ones((len(i_nX), 1)) * [x, y, z], axis=1)
                imin = np.argmin(rsep)
                Amap[ii] = i_nX[imin]
                NVoxPerNode[Amap[ii]] += 1  # might be useful?
                #print('NVoxPerNode', np.shape(NVoxPerNode))
                #print('Amap', np.shape(Amap))

                
                
                mapMesh2Vox[Amap[ii], NVoxPerNode[Amap[ii]] - 1] = i_headvol_flat[ii]
                ii+=1
            else:
                nmiss += 1  # temporary var, delete when everything works
                print('miss', nmiss)
        print(nmiss)
        ciao = np.where(mapMesh2Vox == 0)
        mapMesh2Vox[ciao] = 1
        np.save('projVoltoMesh_brain.npy', mapMesh2Vox)
        self.path_projVoltoMesh_brain = 'projVoltoMesh_brain.npy'

        return mapMesh2Vox, NVoxPerNode, Amap
    
    def projVoltoMesh_scalp(self):

        nodeX = self.mesh_scalp_orig.reduced_vertices
        elem  = self.mesh_scalp_orig.reduced_faces 
        nNode = np.size(nodeX,0)

        i_headvol = np.where(self.headvol.scalp_vol != 0)
        i_headvol_flat = np.ravel_multi_index(i_headvol, self.headvol.scalp_vol.shape)

        nx = self.headvol.scalp_vol.shape[0]
        ny = self.headvol.scalp_vol.shape[1]
        nz = self.headvol.scalp_vol.shape[2]

        nmiss = 0
        nxy = nx * ny
        nC = np.size(i_headvol_flat,0)

        Amap = np.zeros((nC, 1), dtype=np.int64)
        mapMesh2Vox = np.ones((nNode, 2000), dtype=np.int64)
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
        printProgressBar(0, nC, prefix = 'projVoltoMesh_scalp progress:', suffix = 'Complete', length = 50)
        for x, y, z in zip(x_coordinates, y_coordinates, z_coordinates):
            #print(f"Voxel at coordinates ({x}, {y}, {z}) is gray matter.")
            if ii % update_interval == 1:
                printProgressBar(ii, nC, prefix = 'projVoltoMesh_scalp progress:', suffix = 'Complete', length = 50)
            i_nX = np.where((nodeX[:, 0] > x - h) & (nodeX[:, 0] < x + h) &
                            (nodeX[:, 1] > y - h) & (nodeX[:, 1] < y + h) &
                            (nodeX[:, 2] > z - h) & (nodeX[:, 2] < z + h))[0]
            if i_nX.size > 0:
                # rsep: get the distances from [x y z] to all the points in nodeX(i_nX,:).
                rsep = np.linalg.norm(nodeX[i_nX, :] - np.ones((len(i_nX), 1)) * [x, y, z], axis=1)
                imin = np.argmin(rsep)
                Amap[ii] = i_nX[imin]
                NVoxPerNode[Amap[ii]] += 1  # might be useful?
                #print('NVoxPerNode', np.shape(NVoxPerNode))
                #print('Amap', np.shape(Amap))
                
                
                mapMesh2Vox[Amap[ii], NVoxPerNode[Amap[ii]] - 1] = i_headvol_flat[ii]
                ii+=1
            else:
                nmiss += 1  # temporary var, delete when everything works
        
        ciao = np.where(mapMesh2Vox == 0)
        mapMesh2Vox[ciao] = 1
        np.save('projVoltoMesh_scalp.npy', mapMesh2Vox)
        self.path_projVoltoMesh_scalp = 'projVoltoMesh_scalp.npy'
        return mapMesh2Vox, NVoxPerNode, Amap
    
    def get_projVolToMesh_brain(self):
        mapMesh2Vox = np.load(self.path_projVoltoMesh_brain)
        return mapMesh2Vox
    
    def get_projVolToMesh_scalp(self):
        mapMesh2Vox = np.load(self.path_projVoltoMesh_scalp)
        return mapMesh2Vox