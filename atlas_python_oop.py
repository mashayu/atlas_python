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
from temp_shared_globals import optpos, refpts
from helpers import *

def load_subject(anatomy_folder = None, probe_folder = None):

    probe = Probe()
    probe.load_probe(probe_folder)
    probe.set_optpos(optpos=optpos)

    fwmodel = Fw_model()

    load_masks(fwmodel, anatomy_folder)
    register_to_digpts(probe)

    return fwmodel, probe

def register_to_digpts(probe):

    probe.get_ref_digpoints()
    probe.get_optpoints()
    p1 = refpts.values()
    print('p1', list(p1))
    T_2digpts = gen_xform_from_pts(np.array(list(refpts.values())), np.array(list(probe.digpts_ref.values())))
    print("T_2digpts", T_2digpts)
    vol_transf, digpts_T_2mc = xform_apply_vol_smooth(fwmodel.volume, T_2digpts)
    print("digpts_T_2mc", digpts_T_2mc)
    print("vol_transf shape", vol_transf.shape)

    transformed_srcpos = xform_apply(np.array(list(probe.optpos_dict.values())), digpts_T_2mc)

    headvol_T_2mc = np.dot(digpts_T_2mc, T_2digpts)
    print('transformed_srcpos',transformed_srcpos)
    #transformed_detpos = xform_apply(np.array(list(probe.digpts_ref.values())), digpts_T_2mc)

    #Rx = fwmodel.headvol.center[0]
    #Ry = fwmodel.headvol.center[1]
    #Rz = fwmodel.headvol.center[2]

    #for i in range(optpos.shape[0]):
    #    x = optpos[i, 0]
    #    y = optpos[i, 1]
    #    z = optpos[i, 2]
    #    r = ((Rx - x)**2 + (Ry - y)**2 + (Rz - z)**2)**0.5

        # Calculate the normalized vector components
    #    vx = (Rx - x) / r
    #    vy = (Ry - y) / r
    #    vz = (Rz - z) / r

        # Store the normalized vector components in the list
    #    optpos[i] = np.append(optpos[i], [vx, vy, vz])

    return 


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
        'gray_matter': "",
        'white_matter': "",
        'csf': ""
    }

    mask_paths['scalp'] = os.path.join(anatomy_folder, 'scalp.nii')
    mask_paths['skull'] = os.path.join(anatomy_folder, 'skull.nii')
    mask_paths['gray_matter'] = os.path.join(anatomy_folder, 'gm.nii')
    mask_paths['white_matter'] = os.path.join(anatomy_folder, 'wm.nii')
    mask_paths['csf'] = os.path.join(anatomy_folder, 'csf.nii')

    mask_values = {}

    updated_masks = {}

    mask_value = 1
    for mask_name, mask_path in mask_paths.items():
        if mask_path:
            mask = nib.load(mask_path).get_fdata()[:]
            mask[mask != 0] = 1
            updated_mask = update_mask(mask, mask_value)
            mask_value +=1
            updated_masks[mask_name] = updated_mask
            mask_values[mask_name] = mask_value

    full_volume = np.zeros(updated_mask.shape, dtype=np.float64)
    for updated_mask in updated_masks.values():
        full_volume += updated_mask

    print('volume shape', full_volume.shape)

    fwmodel.set_volume(gm, skin) 
    volume_file_name = "myvolume"
    fwmodel.set_volume2(volume_file_name, full_volume)

    fwmodel.vol_shape = full_volume.shape





if __name__=='__main__':
    
    volume_file_name = "myvolume"

    anatomy_folder = r'C:\Users\User\Documents\masha\Alex\atlas_python\icbm_seg'
    probe_folder =  r'C:\Users\User\Documents\masha\alex_mrtim\2022-04-25_001.nirs'

    fwmodel, probe = load_subject(anatomy_folder, probe_folder)

    #fwmodel.set_volume('tissues_cat12_volatlas.nii')



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




    calculate_fluence(fwmodel=fwmodel, probe=probe, volume_file=volume_file_name+'.bin')


    

