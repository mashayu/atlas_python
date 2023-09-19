import numpy as np
import nibabel as nib
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#from skimage import measure
import trimesh
import numpy as np
import os

from probe import Probe
from fw_model import Fw_model
from mcx import calculate_fluence
from temp_shared_globals import refpts
from helpers import *


##########################

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#############################



def plot_point_cloud(data):
    # Find coordinates of points where the value equals 1
    indices = np.argwhere(data == 1)

    indices = indices[::30]

    # Extract the x, y, and z coordinates from the indices
    x_coords = indices[:, 0]
    y_coords = indices[:, 1]
    z_coords = indices[:, 2]

    # Create a 3D scatter plot for the point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()



def load_subject(anatomy_folder = None, probe_folder = None):

    probe = Probe()
    meas_list, nsrc =probe.load_probe(probe_folder)
    print("measlist", meas_list)
    #probe.set_optpos(optpos=optpos)

    fwmodel = Fw_model()
    load_masks(fwmodel, anatomy_folder)

    #print("center", find_center(fwmodel.volume))
    #fwmodel.headvol_center = find_center(fwmodel.volume)[0]

    fwmodel.headvol.find_center()
    return fwmodel, probe

def register_to_digpts(probe, fwmodel):

    probe.get_ref_digpoints()
    probe.read_optpoints_from_digpts()
    p1 = refpts.values()
    #print('p1', list(p1))
    T_2digpts = gen_xform_from_pts(np.array(list(refpts.values())), np.array(list(probe.digpts_ref.values())))
    print("T_2digpts", T_2digpts)
    vol_transf, digpts_T_2mc = xform_apply_vol_smooth(fwmodel.headvol.volume, T_2digpts)
    #writevolbin(vol_transf, "myvolume_t")
    writevolbin2(vol_transf, "myvolume_t")
    fwmodel.headvol.vol_t_shape = vol_transf.shape
    fwmodel.headvol.volume_t = vol_transf
    print("digpts_T_2mc", digpts_T_2mc)
    print("vol_transf shape", vol_transf.shape)

    transformed_optpos = xform_apply(np.array(list(probe.optpos_dict.values())), digpts_T_2mc)


    
    headvol_T_2mc = np.dot(digpts_T_2mc, T_2digpts)
    fwmodel.headvol.T_2mc = headvol_T_2mc
    print('before transformation',probe.optpos_dict)
    print('transformed_optpos',transformed_optpos)

    print("old center", fwmodel.headvol.center)
    fwmodel.headvol.center = xform_apply(np.array([fwmodel.headvol.center]), fwmodel.headvol.T_2mc)[0]
    print("new center", fwmodel.headvol.center)


    Rx = fwmodel.headvol.center[0]
    Ry = fwmodel.headvol.center[1]
    Rz = fwmodel.headvol.center[2]

    new_columns = np.zeros((len(transformed_optpos), 3))

    for i in range(transformed_optpos.shape[0]):
        x = transformed_optpos[i, 0]
        y = transformed_optpos[i, 1]
        z = transformed_optpos[i, 2]
        r = ((Rx - x)**2 + (Ry - y)**2 + (Rz - z)**2)**0.5

        # Calculate the normalized vector components
        vx = (Rx - x) / r
        vy = (Ry - y) / r
        vz = (Rz - z) / r

        new_columns[i] = [vx, vy, vz]

    new_transformed_optpos = np.concatenate((transformed_optpos, new_columns), axis=1)
    print(new_transformed_optpos)

    probe.reg_optpos = new_transformed_optpos
    return 

def find_center(x):
    volsurf = x
    bbox,_,_ = gen_bbox(volsurf)
    print('bbox', bbox)
    c = find_region_centers([bbox])
    return c



def load_masks(fwmodel, anatomy_folder = r'C:\Users\User\Documents\masha\Alex\atlas_python\icbm_seg'):
    #anatomy_folder = r'C:\Users\User\Documents\masha\Alex\atlas_python\icbm_seg'
    new_working_directory = os.path.join(anatomy_folder, 'wd')
    if not os.path.exists(new_working_directory):
        # Create the directory
        os.makedirs(new_working_directory)
    os.chdir(new_working_directory)
    
    img = nib.load(os.path.join(anatomy_folder, "gm.nii"))
    gm = img.get_fdata()[:]
    gm[gm != 0] = 1

    img = nib.load(os.path.join(anatomy_folder, "scalp.nii"))
    skin = img.get_fdata()[:]
    skin[skin != 0] = 1

    mask_paths = {
        'scalp': "",
        'skull': "",
        'csf': "",
        'gray_matter': "",
        'white_matter': ""
        
    }

    mask_paths['scalp'] = os.path.join(anatomy_folder, 'scalp.nii')
    mask_paths['skull'] = os.path.join(anatomy_folder, 'skull.nii')
    mask_paths['csf'] = os.path.join(anatomy_folder, 'csf.nii')
    mask_paths['gray_matter'] = os.path.join(anatomy_folder, 'gm.nii')
    mask_paths['white_matter'] = os.path.join(anatomy_folder, 'wm.nii')

    mask_values = {}

    updated_masks = {}

    mask_value = 1
    for mask_name, mask_path in mask_paths.items():
        if mask_path:
            mask = nib.load(mask_path).get_fdata()[:]
            mask[mask != 0] = 1
            updated_mask = update_mask(mask, mask_value)
            updated_masks[mask_name] = updated_mask
            mask_values[mask_name] = mask_value
            mask_value +=1
            print(mask_name)
            print(np.unique(updated_mask[updated_mask != 0]))

    print(mask_values)

    full_volume = np.zeros(updated_mask.shape, dtype=np.float64)
    for updated_mask in updated_masks.values():
        full_volume += updated_mask
    
    print(np.unique(full_volume))

    print('volume shape', full_volume.shape)

    fwmodel.set_volume(gm, skin) 
    volume_file_name = "myvolume"
    fwmodel.set_volume2(volume_file_name, full_volume)

    fwmodel.headvol.vol_shape = full_volume.shape


def plot_vol(volume):
    test = volume[:,:,59]
    plt.imshow(test)
    plt.show()


def save_nifti(volume, filename):
    # Create a NIfTI image object
    nifti_img = nib.Nifti1Image(volume, affine=None)  # The affine matrix can be provided if needed

    # Specify the output filename (e.g., "output.nii.gz")
    output_filename = filename+".nii"

    # Save the NIfTI image to a file
    nib.save(nifti_img, output_filename)

def check_unique_values(vol_file, data_type):
    with open(vol_file, 'rb') as file:
        array_1d = np.fromfile(file, dtype=data_type)
    return np.unique(array_1d)


if __name__=='__main__':
    
    volume_file_name = "myvolume_t"#matlab_volume

    anatomy_folder = r'C:\Users\User\Documents\masha\Alex\atlas_python\icbm_seg'
    probe_folder =  r'C:\Users\User\Documents\masha\alex_mrtim\2022-04-25_001.nirs'

    fwmodel, probe = load_subject(anatomy_folder, probe_folder)

    #plot_vol(fwmodel.headvol.volume)
    save_nifti(fwmodel.headvol.volume, "volume_1")
    
    plot_point_cloud(fwmodel.headvol.volume)
    register_to_digpts(probe, fwmodel)
    #fwmodel.set_volume('tissues_cat12_volatlas.nii'
    save_nifti(fwmodel.headvol.volume_t, "volume_1t")
    plot_point_cloud(fwmodel.headvol.volume_t)




    #probe.reg_optpos[0] = [155.345811492738, 184.316432551622, 155.868323109028, -0.703023648717975, -0.476906863214703, -0.527558141972975]

    ###########################################
    ######load volume##########################
    ###########################################
    #full_volume_path = ""
    #full_volume = nib.load(full_volume_path).get_fdata()[:]
    ###########################################


    #fwmodel.set_volume(gm, skin)

    fwmodel.build_head_surf()
    fwmodel.build_pial_surf()

    #fwmodel.mesh_orig.reduce_mesh(filename="brainmesh_reduced")
    #fwmodel.mesh_scalp_orig.reduce_mesh(filename="scalpmesh_reduced")

    reduced_scalp = trimesh.load_mesh('scalpmesh_reduced.stl')
    reduced_brain = trimesh.load_mesh('brainmesh_reduced.stl')
    fwmodel.mesh_orig.set_reduced_mesh(reduced_brain)
    fwmodel.mesh_scalp_orig.set_reduced_mesh(reduced_scalp)

    #plot_mesh(fwmodel.mesh_scalp_orig.reduced_vertices,fwmodel.mesh_scalp_orig.reduced_faces)
    #print(gm.shape)

    #fwmodel.projVoltoMesh_brain()
    #fwmodel.projVoltoMesh_scalp()

    fwmodel.path_projVoltoMesh_brain = 'projVoltoMesh_brain.npy'
    fwmodel.path_projVoltoMesh_scalp = 'projVoltoMesh_scalp.npy'
    mapMesh2Vox = fwmodel.get_projVolToMesh_brain()
    mapMesh2Vox_scalp = fwmodel.get_projVolToMesh_scalp()



    print("check values", check_unique_values(volume_file_name+'.bin', int))
    calculate_fluence(fwmodel=fwmodel, probe=probe, volume_file=volume_file_name+'.bin')


    

