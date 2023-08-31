import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection




def plot_mesh(v,f):
    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(v[f])
    #print(v[f[:20]])
    mesh.set_facecolors('g')
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.set_xlabel("x-axis: a = 6 per ellipsoid")
    ax.set_ylabel("y-axis: b = 10")
    ax.set_zlabel("z-axis: c = 16")
    ax.set_xlim(0, 230)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(0, 300)  # b = 10
    ax.set_zlim(0, 240)  # c = 16
    plt.tight_layout()
    plt.show()


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()



def update_mask(volume, number:int):
    ones_mask = volume == 1
    updated_mask = ones_mask * number
    new_volume = np.zeros_like(volume)
    new_volume += updated_mask
    return new_volume


def gen_xform_from_pts(p1, p2):
    """
    Calculate the affine transformation matrix T that transforms p1 to p2.

    Parameters:
    p1 (numpy.ndarray): Source points (p x m) where p is the number of points and m is the number of dimensions.
    p2 (numpy.ndarray): Target points (p x m) where p is the number of points and m is the number of dimensions.

    Returns:
    numpy.ndarray: Affine transformation matrix T.
    """

    T = np.eye(p1.shape[1] + 1)
    p, m = p1.shape
    q, n = p2.shape

    if p != q:
        print("Number of points for p1 and p2 must be the same")
        return None

    if m != n:
        print("Number of dimensions for p1 and p2 must be the same")
        return None

    if p < n:
        print(f"Cannot solve transformation with fewer anchor points ({p}) than dimensions ({n}).")
        return None

    A = np.hstack((p1, np.ones((p, 1))))
   
    for ii in range(n):
        x = np.dot(pinv(A), p2[:, ii])
        T[ii, :] = x.T
      
    return T

def xform_apply(v, T):
    if v.size == 0:
        return np.array([])

    n = v.shape[0]
    v = v.T

    print(T.shape)

    if T.shape == (4, 4) or T.shape == (3, 4):
        A = T[:3, :3]
        b = T[:3, 3]
    elif T.shape == (3, 3) or T.shape == (2, 3):
        A = T[:2, :2]
        b = T[:2, 2]

    v1 = np.dot(A, v)
    for i in range(n):
        v1[:, i] = v1[:, i] + b

    return v1.T

def gen_bbox(volume):
    min_coords = np.array([np.inf, np.inf, np.inf])
    max_coords = np.array([-np.inf, -np.inf, -np.inf])
    non_zero_coords = np.argwhere(volume != 0)
    for coord in non_zero_coords:
        min_coords = np.minimum(min_coords, coord)
        max_coords = np.maximum(max_coords, coord)
    return np.vstack((min_coords, max_coords))

def xform_apply_vol_smooth(v1, T1, mode='slow'):
    Dx_v1, Dy_v1, Dz_v1 = v1.shape

    # Get bounding boxes
    bbox_v1 = gen_bbox(v1)
    bbox_v1_grid = np.array([
        [1, 1, 1],
        [1, 1, Dz_v1],
        [1, Dy_v1, 1],
        [1, Dy_v1, Dz_v1],
        [Dx_v1, 1, 1],
        [Dx_v1, 1, Dz_v1],
        [Dx_v1, Dy_v1, 1],
        [Dx_v1, Dy_v1, Dz_v1]
    ])
    bbox_v2 = np.round(np.dot(bbox_v1, T1[:3, :3].T) + T1[:3, 3])
    bbox_v2_grid0 = np.round(np.dot(bbox_v1_grid, T1[:3, :3].T) + T1[:3, 3])

    maxx, minx = np.max(bbox_v2[:, 0]), np.min(bbox_v2[:, 0])
    maxy, miny = np.max(bbox_v2[:, 1]), np.min(bbox_v2[:, 1])
    maxz, minz = np.max(bbox_v2[:, 2]), np.min(bbox_v2[:, 2])

    if (
        (maxx - minx) < Dx_v1 and minx > 1 and
        (maxy - miny) < Dy_v1 and miny > 1 and
        (maxz - minz) < Dz_v1 and minz > 1
    ) or mode == 'gridsamesize':
        Dx_v2, Dy_v2, Dz_v2 = Dx_v1, Dy_v1, Dz_v1
    else:
        tx, ty, tz = 1 - np.min(bbox_v2_grid0[:, 0]), 1 - np.min(bbox_v2_grid0[:, 1]), 1 - np.min(bbox_v2_grid0[:, 2])
        T_vert2vox = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])
        if mode.lower() == 'fast':
            return v2, T_vert2vox
        bbox_v2_grid = np.dot(bbox_v2_grid0, T_vert2vox[:3, :3].T) + T_vert2vox[:3, 3]
        v2_gridsize = np.max(bbox_v2_grid, axis=0) - 1
        Dx_v2, Dy_v2, Dz_v2 = v2_gridsize.astype(int)

    v2 = np.zeros((Dx_v2, Dy_v2, Dz_v2))
    T1 = np.linalg.inv(np.dot(T_vert2vox, T1))

    v2 = xform_apply_vol_smooth_64bit(v1, v2, T1)

    return v2, T_vert2vox


def xform_apply_vol_smooth_64bit(v1, v2, T):
    if np.all(T == np.eye(4)):
        return v2

    Dx_v1, Dy_v1, Dz_v1 = v1.shape
    Dx_v2, Dy_v2, Dz_v2 = v2.shape

    int_val = round((Dx_v2 * Dy_v2 * Dz_v2) / 50)
    N = Dx_v2 * Dy_v2 * Dz_v2
    for kk in range(Dz_v2):
        for jj in range(Dy_v2):
            for ii in range(Dx_v2):
                # Display progress
                #if math.isclose(((kk - 1) * Dx_v2 * Dy_v2 + (jj - 1) * Dx_v2 + (ii - 1)) % int_val, 0):
                #    print(f"Progress: {((kk - 1) * Dx_v2 * Dy_v2 + (jj - 1) * Dx_v2 + (ii - 1)) / N:.2%}")

                # Look up the voxel value (i, j, k) in the destination volume (v2)
                # by inversely transforming it to the source volume (v1) equivalent
                # and assigning the value to v2[i, j, k].

                i = np.round(T[:, 0] * ii + T[:, 1] * jj + T[:, 2] * kk + T[:, 3]).astype(int)

                if (1 <= i[0] <= Dx_v1) and (1 <= i[1] <= Dy_v1) and (1 <= i[2] <= Dz_v1):
                    if v1[i[0] - 1, i[1] - 1, i[2] - 1] == 0:
                        continue
                    v2[ii, jj, kk] = v1[i[0] - 1, i[1] - 1, i[2] - 1]

    return v2



def writevolbin(vol,filename):
	fname = filename+'.bin'
	f = open(fname, 'wb')
	dmedium = np.ascontiguousarray(vol,int)
	f.write(dmedium)
	f.close()
        