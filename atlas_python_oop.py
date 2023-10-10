import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mcx import calculate_fluence
from temp_shared_globals import refpts_icbm
from helpers import *
from atlas import AtlasViewer
from load_atlas.load_atlas import load_atlas
from load_atlas.load_tiss_prop import get_tiss_prop


def load_probe_from_folder(atlas_viewer: AtlasViewer):
    """
    Load the probe configuration from the probe directory.

    Args:
        - atlas_viewer (AtlasViewer): AtlasViewer object.

    Returns:

    """
    probe = atlas_viewer.probe
    nirs_file = find_nirs_file(atlas_viewer.probe_dir)
    probe.load_probe_from_nirs(nirs_file)
    return probe


def load_fw_model(atlas_viewer: AtlasViewer):
    """
    Load the forward model from the anatomy directory.

    Args:
        - atlas_viewer (AtlasViewer): AtlasViewer object.

    Returns:

    """

    fw_model = atlas_viewer.fw_model
    fw_model.load_masks(atlas_viewer)
    fw_model.headvol.find_center()
    return fw_model


def load_subject(atlas_viewer: AtlasViewer):
    """
    Load subject-specific anatomical data and probe configuration.

    Args:
        - atlas_viewer (AtlasViewer): AtlasViewer object.

    Returns:
        - tuple of ForwardModel and Probe objects.
    """

    print("Loading subject-specific anatomical data and probe configuration...")
    probe = load_probe_from_folder(atlas_viewer)
    fw_model = load_fw_model(atlas_viewer)
    # probe.set_optpos(optpos=optpos)
    # return fw_model, probe


def read_digpts(digpts_file_path):
    """
    Read digitized points from a text file.

    Args:
        - digpts_file_path (str): Path to the digitized points text file.

    Returns:
        - tuple of dicts: digitized points reference data and optode positions.
    """

    # Obtain reference digitized points from the probe
    digpts_ref = probe.read_refpoints_from_digpts(digpts_file_path)
    # Read optpoints from the digitized reference points
    optpos = probe.read_optpoints_from_digpts(digpts_file_path)
    return digpts_ref, optpos


def extract_ordered_values(digpts_ref: dict, ref: dict):
    """
    Extract values from two dictionaries in an order defined by their keys.

    Args:
    - digpts_ref (dict): Dictionary containing digitized points reference data.
    - ref (dict): Dictionary containing reference data.

    Returns:
    - tuple of lists: values from the first and second dictionary ordered by their keys.
    """

    # Mapping of key aliases
    key_aliases = {
        "ar": ["ar", "RPA"],
        "cz": ["cz", "Cz"],
        "nz": ["nz", "Nz"],
        "al": ["al", "LPA"],
        "iz": ["iz", "Iz"]
    }

    # Normalize and map the keys
    def get_normalized_key(key):
        for main_key, aliases in key_aliases.items():
            if key in aliases:
                return main_key
        return key  # Return the original key if no mapping found

    normalized_digpts_ref = {
        get_normalized_key(key): value
        for key, value in digpts_ref.items()
    }
    normalized_ref = {
        get_normalized_key(key): value
        for key, value in ref.items()
    }

    # Ensure keys are identical in both dictionaries
    if set(normalized_digpts_ref.keys()) != set(normalized_ref.keys()):
        raise ValueError("Dictionaries have different keys.")

    # Extract values based on sorted keys
    keys_sorted = normalized_digpts_ref.keys()
    values_digpts_ref = [normalized_digpts_ref[key] for key in keys_sorted]
    values_ref = [normalized_ref[key] for key in keys_sorted]

    return values_digpts_ref, values_ref


def register_to_digpts(atlas_viewer: AtlasViewer):
    """
    Register the probe's digital points to the head anatomy.

    Args:
        - atlas_viewer (AtlasViewer): AtlasViewer object.

    Returns:

    """

    print("Registering to digitized points...")
    print("It will take a while...")

    fw_model = atlas_viewer.fw_model
    probe = atlas_viewer.probe
    headvol = atlas_viewer.headvol
    headsurf = fw_model.mesh_orig
    pialsurf = fw_model.mesh_scalp_orig

    if probe.digpts_ref is None:
        raise ValueError("No digitized points reference data found.")
    if headvol.ref_pts is None:
        raise ValueError("No reference data found.")

    ###########################################
    # 1 Transform the reference points to get transformation matrix
    ###########################################
    values_digpts_ref, values_ref = extract_ordered_values(
        probe.digpts_ref, headvol.ref_pts)
    headvol.T_2digpts = gen_xform_from_pts(
        np.array(values_ref), np.array(values_digpts_ref))
    ###########################################
    # 2 Apply the volume transformation and obtain the transformation matrix from digitized points to head volume coordinates
    # Register headvol to digpts
    ###########################################
    headvol_transf, digpts_T_2mc = xform_apply_vol_smooth(
        headvol.volume, headvol.T_2digpts)

    save_nifti(headvol_transf, os.path.join(
        atlas_viewer.working_dir, "volume_t.nii"))

    # Calculate transformation matrix from digitized points to Monte Carlo coordinates
    headvol.T_2mc = np.dot(digpts_T_2mc, headvol.T_2digpts)
    headvol.center = xform_apply(np.array([headvol.center]), headvol.T_2mc)[0]

    # Update fwmodel's head volume shape and volume data with transformed data
    headvol.vol_t_shape = headvol_transf.shape
    headvol.volume_t = headvol_transf

    # Transform optical positions using the transformation matrix from digitized points to head volume coordinates
    transformed_optpos = xform_apply(
        np.array(list(probe.optpos_dict.values())), digpts_T_2mc)

    # print("transformed_optpos", transformed_optpos)

    ###########################################
    # 3 Calculate the normalized vector components of each transformed optical position relative to the head volume's center
    ###########################################
    Rx, Ry, Rz = fw_model.headvol.center
    new_columns = np.zeros((len(transformed_optpos), 3))
    for i in range(transformed_optpos.shape[0]):
        x, y, z = transformed_optpos[i]
        r = ((Rx - x)**2 + (Ry - y)**2 + (Rz - z)**2)**0.5

        # Calculate the normalized vector components
        vx, vy, vz = (Rx - x) / r, (Ry - y) / r, (Rz - z) / r
        new_columns[i] = [vx, vy, vz]

    # Concatenate the transformed optical positions with their corresponding normalized vectors
    probe.reg_optpos = np.concatenate((transformed_optpos, new_columns),
                                      axis=1)

    # probe.reg_optpos[0] = [220, 220,
    #                       220, -0.587456391631848, -0.658452231495843,	-0.470463225735056]

    ###########################################
    # 4 Move surfaces to monte carlo space
    ###########################################
    headsurf.vertices = xform_apply(headsurf.vertices, headvol.T_2mc)
    pialsurf.vertices = xform_apply(pialsurf.vertices, headvol.T_2mc)

    # TODO MOVE REFPTS TO MONTE CARLO SPACE
    # TODO MOVE DIGPTS TO MONTE CARLO SPACE

    ###########################################
    # % Save the transformed volume data to a binary file
    ###########################################
    atlas_viewer.binary_vol_t_path = os.path.join(atlas_viewer.working_dir,
                                                  "myvolume_t.bin")
    writevolbin2(headvol_transf, atlas_viewer.binary_vol_t_path)

    return


def head_anatomy_pipeline(atlas_viewer: AtlasViewer, refpts):
    """
    Load subject-specific anatomical data and probe configuration.

    Args:
        - atlas_viewer (AtlasViewer): AtlasViewer object.

    Returns:

    """

    # Load subject-specific anatomical data and probe configuration
    load_subject(atlas_viewer)

    atlas_viewer.headvol.set_ref_pts(refpts)

    fw_model = atlas_viewer.fw_model
    probe = atlas_viewer.probe

    # TODO: extract tiss_prop_labels automatically
    tiss_prop_labels = ["skin", "skull", "csf", "gm", "wm"]
    fw_model.tiss_prop = get_tiss_prop(tiss_prop_labels)

    # Save the head volume to a NIFTI file
    save_nifti(fw_model.headvol.volume, os.path.join(
        atlas_viewer.working_dir, "volume_1.nii"))

    digpts_file = os.path.join(atlas_viewer.probe_dir, 'digpts.txt')
    _, _ = read_digpts(digpts_file)

    # Build surface models for the head and pial (brain surface)
    fw_model.build_head_surf()
    fw_model.build_pial_surf()

    # Register the probe's digital points to the head anatomy
    register_to_digpts(atlas_viewer)

    # Save the transformed volume to another NIFTI file
    save_nifti(fw_model.headvol.volume_t, os.path.join(
        atlas_viewer.working_dir, "volume_1.nii"))

    # Visualize the transformed volume as a point cloud
    # plot_point_cloud(fw_model.headvol.volume_t)

    scalpmesh_reduced_path = os.path.join(
        atlas_viewer.working_dir, "scalpmesh_reduced.stl")
    brainmesh_reduced_path = os.path.join(
        atlas_viewer.working_dir, "brainmesh_reduced.stl")

    fw_model.mesh_orig.reduce_mesh(filename=brainmesh_reduced_path)
    fw_model.mesh_scalp_orig.reduce_mesh(filename=scalpmesh_reduced_path)

    # Load the reduced mesh versions of the head and brain surfaces
    # scalpmesh_reduced_path = trimesh.load_mesh(os.path.join(
    #    atlas_viewer.working_dir, "scalpmesh_reduced.stl"))
    # brainmesh_reduced_path = trimesh.load_mesh(os.path.join(
    #    atlas_viewer.working_dir, "brainmesh_reduced.stl"))

    # plot_mesh(fwmodel.mesh_scalp_orig.reduced_vertices,fwmodel.mesh_scalp_orig.reduced_faces)
    # print(gm.shape)

    # Set paths for volume-to-mesh projections for brain and scalp
    fw_model.path_projVoltoMesh_brain = os.path.join(
        atlas_viewer.working_dir, "projVoltoMesh_brain.npy")
    fw_model.path_projVoltoMesh_scalp = os.path.join(
        atlas_viewer.working_dir, "projVoltoMesh_scalp.npy")

    mapMesh2Vox = fw_model.projVoltoMesh_brain(
        fw_model.path_projVoltoMesh_brain)
    mapMesh2Vox_scalp = fw_model.projVoltoMesh_scalp(
        fw_model.path_projVoltoMesh_scalp)

    # Compute fluence based on the model and probe configuration
    Adot, Adot_scalp = calculate_fluence()
    # print(Adot)


def load_atlas_pipeline(atlas_viewer: AtlasViewer):
    """
    Load the atlas and probe configuration.


    Args:
        - atlas_viewer (AtlasViewer): AtlasViewer object.

    Returns:

    """

    fw_model = atlas_viewer.fw_model
    load_atlas(atlas_viewer, atlas_path=atlas_viewer.atlas_dir)
    load_probe_from_folder(atlas_viewer)
    digpts_file = os.path.join(atlas_viewer.probe_dir, 'digpts.txt')

    _, _ = read_digpts(digpts_file)
    register_to_digpts(atlas_viewer)

    brainmesh_reduced_path = os.path.join(atlas_viewer.working_dir,
                                          'brainmesh_reduced.stl')

    fw_model.mesh_orig.reduce_mesh(filename=brainmesh_reduced_path)

    scalpmesh_reduced_path = os.path.join(atlas_viewer.working_dir,
                                          'scalpmesh_reduced.stl')
    fw_model.mesh_scalp_orig.reduce_mesh(filename=scalpmesh_reduced_path)

    # Set paths for volume-to-mesh projections for brain and scalp
    fw_model.path_projVoltoMesh_brain = os.path.join(
        atlas_viewer.working_dir, "projVoltoMesh_brain.npy")
    fw_model.path_projVoltoMesh_scalp = os.path.join(
        atlas_viewer.working_dir, "projVoltoMesh_scalp.npy")

    fw_model.projVoltoMesh_brain(filepath=fw_model.path_projVoltoMesh_brain)

    fw_model.projVoltoMesh_scalp(filepath=fw_model.path_projVoltoMesh_scalp)

    Adot, Adot_scalp = calculate_fluence()


if __name__ == "__main__":
    atlas_viewer = AtlasViewer()
    fw_model = atlas_viewer.fw_model
    probe = atlas_viewer.probe

    base_directory = os.path.dirname(os.path.abspath(__file__))
    print(base_directory)

    # path to the atlas
    # atlas_viewer.atlas_dir = r"C:\Users\User\Documents\masha\Alex\atlas_python\atlas_python\atlas_data\Colin"
    atlas_viewer.atlas_dir = os.path.join(
        base_directory, r"demo_data\atlas_data\Colin")

    # path to the anatomy
    # atlas_viewer.anatomy_dir = r"C:\Users\User\Documents\masha\Alex\atlas_python\icbm_seg"
    atlas_viewer.anatomy_dir = os.path.join(
        base_directory, r"demo_data\anatomy_data")

    # path to the probe
    # atlas_viewer.probe_dir = r"C:\Users\User\Documents\masha\Alex\atlas_python\icbm_seg"
    atlas_viewer.probe_dir = os.path.join(
        base_directory, r"demo_data\probe_data")

    # atlas_pipeline
    atlas_viewer.working_dir = os.path.join(
        atlas_viewer.probe_dir, "wd_atlas")
    load_atlas_pipeline(atlas_viewer)

    # head_anatomy_pipeline
    atlas_viewer.working_dir = os.path.join(
        atlas_viewer.probe_dir, "wd")
    head_anatomy_pipeline(atlas_viewer, refpts=refpts_icbm)
