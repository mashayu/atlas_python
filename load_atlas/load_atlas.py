import trimesh
from skimage import measure
from headvol import Headvol
from fw_model import Fw_model
from atlas import AtlasViewer
from scipy.io import loadmat
from .load_tiss_prop import get_tiss_prop
from helpers import plot_mesh, plot_point_cloud
import os
import numpy as np

# from read_surf import read_surf, read_surf2
import json
import sys

sys.path.append("..")


def load_mesh_data(file_path):
    data = loadmat(file_path)
    vertices = data["vertex_coords"]
    faces = [[item - 1 for item in sublist] for sublist in data["faces"]]
    return vertices, faces


def load_mesh_data_ttt(file_path):
    # data = loadmat(file_path)["mesh"]
    # print(data)
    # vertices = data["vertices"]
    # print()
    # faces = [[item - 1 for item in sublist] for sublist in data["faces"]]
    # return vertices, faces

    data = loadmat(file_path)
    print(data)

    # Extract the 'mesh' structure
    mesh = data['mesh'][0, 0]  # This extracts the structured array from 'mesh'

    # Extract 'faces' and 'vertices'
    faces = mesh['faces']
    vertices = mesh['vertices']

    # Convert to numpy arrays (if required)
    faces = faces  # This should give you a numpy array of faces
    faces = [[item - 1 for item in sublist] for sublist in faces]
    print(faces)
    vertices = vertices  # This should give you a numpy array of vertices
    print(vertices)
    return vertices, faces


def load_mesh_data_ttt2(file_path):
    # data = loadmat(file_path)["mesh"]
    # print(data)
    # vertices = data["vertices"]
    # print()
    # faces = [[item - 1 for item in sublist] for sublist in data["faces"]]
    # return vertices, faces

    data = loadmat(file_path)
    print(data)

    # Extract the 'mesh' structure
    mesh = data['mesh_scalp'][
        0, 0]  # This extracts the structured array from 'mesh'

    # Extract 'faces' and 'vertices'
    faces = mesh['faces']
    vertices = mesh['vertices']

    # Convert to numpy arrays (if required)
    faces = faces  # This should give you a numpy array of faces
    faces = [[item - 1 for item in sublist] for sublist in faces]
    print(faces)
    vertices = vertices  # This should give you a numpy array of vertices
    print(vertices)
    return vertices, faces


def set_fw_model(fw_model, volume, volume_file_name, pialsurf_v, pialsurf_f,
                 headsurf_v, headsurf_f, gm, scalp, tiss_prop):
    fw_model.headvol.set_and_save_full_volume(volume_file_name, volume)
    fw_model.headvol.set_brain_and_scalp_volumes(gm, scalp)
    fw_model.mesh_orig.set_faces_and_vertices(pialsurf_f, pialsurf_v)
    fw_model.mesh_scalp_orig.set_faces_and_vertices(headsurf_f, headsurf_v)
    fw_model.headvol.find_center()
    fw_model.tiss_prop = tiss_prop


def compose_landmark_dict(landmarks, landmarks_labels):
    """
    Composes a dictionary using landmarks and their corresponding labels.

    Args:
    - landmarks (list): List containing the landmark coordinates.
    - landmarks_labels (list): List containing the labels for each landmark.

    Returns:
    - dict: Dictionary with labels as keys and coordinates as values.
    """

    # Ensure the landmarks and labels lists have the same length
    if len(landmarks) != len(landmarks_labels):
        raise ValueError(
            "Lengths of landmarks and landmarks_labels lists must be the same."
        )

    # If landmarks_labels are stored as numpy arrays within the main array, convert them to regular strings
    if isinstance(landmarks_labels[0], np.ndarray):
        landmarks_labels = [label[0] for label in landmarks_labels]

    # Compose dictionary
    landmark_dict = {
        label: landmark
        for label, landmark in zip(landmarks_labels, landmarks)
    }

    return landmark_dict


def get_tissue_indices(tiss_prop_list: list):
    """
    Get indices of each tissue type.

    Args:
    - tissue_list (list): List of dictionaries containing tissue properties.

    Returns:
    - dict: Dictionary with tissue names as keys and their indices in the list as values.
    """

    return {
        tissue['name']: index + 1
        for index, tissue in enumerate(tiss_prop_list)
    }


def load_atlas(
        atlas_viewer: AtlasViewer,
        atlas_path="C:/Users/nirx/Documents/AtlasViewer-2.44.0/Data/Colin"):
    """
    Load atlas data and update the provided atlas_viewer instance.

    Args:
    - atlas_viewer (AtlasViewer): An instance of the AtlasViewer class to be updated.
    - atlas_path (str): Path to the atlas data folder.
    """

    # Load the head volume
    # headvol = Headvol(os.path.join(atlas_path, "anatomical", "headvol.vox"))
    # atlas_viewer.headvol = headvol

    # Ensure the working directory exists, if not, create it and change to it
    new_working_directory = atlas_viewer.working_dir
    if not os.path.exists(new_working_directory):
        os.makedirs(new_working_directory)
    os.chdir(new_working_directory)

    # Load mesh data for pial and head surfaces
    pialsurf_v, pialsurf_f = load_mesh_data(
        os.path.join(atlas_path, "pialmesh_data.mat"))
    headsurf_v, headsurf_f = load_mesh_data(
        os.path.join(atlas_path, "headmesh_data.mat"))

    # Load landmark data and its associated labels
    landmarks = loadmat(os.path.join(atlas_path,
                                     "landmarks.mat"))["landmark"]
    landmark_labels = loadmat(
        os.path.join(atlas_path,
                     "landmark_labels.mat"))["landmark_labels"][0]

    # Create a dictionary to associate landmarks with their labels
    landmark_dict = compose_landmark_dict(landmarks, landmark_labels)
    atlas_viewer.headvol.ref_pts = landmark_dict

    # Print the landmark dictionary for debugging purposes
    print(landmark_dict)

    # Load volume data
    vol2 = loadmat(os.path.join(atlas_path,
                                "headvol.mat"))["headvol_img"]

    # Define the binary volume path
    atlas_viewer.binary_vol_path = os.path.join(atlas_viewer.working_dir,
                                                "myvolume.bin")

    tiss_prop = get_tiss_prop(
        os.path.join(atlas_path,  "headvol_tiss_type.txt"))

    print(tiss_prop)

    # For projecting to mesh
    tissue_indices = get_tissue_indices(tiss_prop)
    print("tissue", tissue_indices)
    gm = np.zeros_like(vol2)
    gm[vol2 == tissue_indices["gm"]] = 1
    scalp = np.zeros_like(vol2)
    scalp[vol2 == tissue_indices["scalp"]] = 1
    csf = np.zeros_like(vol2)
    csf[vol2 == tissue_indices["csf"]] = 1
    wm = np.zeros_like(vol2)
    wm[vol2 == tissue_indices["wm"]] = 1

    # vertices, faces, _, _ = measure.marching_cubes(gm, 0.1)
    # reduced_mesh = trimesh.Trimesh(
    #    vertices, faces)  #.simplify_quadratic_decimation(100000)
    # reduced_mesh.fill_holes()
    # print(reduced_mesh.is_watertight)
    # print(len(pialsurf_f))
    # reduced_mesh.show()

    # reduced_vertices = reduced_mesh.vertices
    # reduced_faces = reduced_mesh.faces

    # Set up the forward model
    set_fw_model(atlas_viewer.fw_model, vol2, atlas_viewer.binary_vol_path,
                 pialsurf_v, pialsurf_f, headsurf_v, headsurf_f, gm, scalp, tiss_prop)

    # plot_mesh(headsurf_v, headsurf_f)
    # plot_mesh(pialsurf_v, pialsurf_f)
    # print(landmarks)
    # plot_point_cloud(headvol.img)
    plot_point_cloud(vol2)

    # print("scalp")
    # plot_point_cloud(scalp)
    # print("csf")
    # plot_point_cloud(csf)
    # print("wm")
    # plot_point_cloud(wm)
    # print("gm")
    # plot_point_cloud(gm)

    # plot_mesh(reduced_vertices, reduced_faces)

    # print(pialsurf_f)


def save_mesh_npz(vertices, faces, filename):
    np.savez(filename, vertices=vertices, faces=faces)


def load_mesh_npz(filename):
    data = np.load(filename)
    return data['vertices'], data['faces']
