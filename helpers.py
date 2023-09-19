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

def gen_bbox2(volume):
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
    bbox_v1 = gen_bbox2(v1)
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

    #TODO check indexing 
    for kk in range(1, Dz_v2 + 1):
        for jj in range(1, Dy_v2 + 1):
            for ii in range(1, Dx_v2 + 1):

                i = (np.dot(T[:, :3], np.array([ii, jj, kk])) + T[:, 3]).astype(int)

                if (1 <= i[0] <= Dx_v1) and (1 <= i[1] <= Dy_v1) and (1 <= i[2] <= Dz_v1):
                    if v1[i[0] - 1, i[1] - 1, i[2] - 1] == 0:
                        continue
                    v2[ii - 1, jj - 1, kk - 1] = v1[i[0] - 1, i[1] - 1, i[2] - 1]

    return v2



def writevolbin(vol,filename):
	fname = filename+'.bin'
	f = open(fname, 'wb')
	dmedium = np.ascontiguousarray(vol,int)
	f.write(dmedium)
	f.close()
        

def writevolbin2(vol, filename, dtype=int):
    fname = filename+'.bin'
    print("file " + fname + " is written with shape " + str(vol.shape))
    with open(fname, "wb") as file:
        vol.astype(dtype).tofile(file)


def find_region_centers(region_vertices):
    print(region_vertices[0])
    n = len(region_vertices)
    region_centers = np.zeros((n, 3), dtype=int)

    for i in range(n):
        vert = region_vertices[i]
        region_size = vert.shape[0]

        if region_size > 0:
            xmin, ymin, zmin = np.min(vert, axis=0)
            xmax, ymax, zmax = np.max(vert, axis=0)

            region_centers[i, 0] = round(xmin + ((xmax - xmin) / 2))
            region_centers[i, 1] = round(ymin + ((ymax - ymin) / 2))
            region_centers[i, 2] = round(zmin + ((zmax - zmin) / 2))

    return region_centers

def gen_bbox(obj, paddingsize=None):
    bbox_vert = []
    bbox_mp = []
    bbox_vol = []

    if obj.ndim == 3:
        vol = obj
        i = np.where(vol != 0)
        x, y, z = i[0], i[1], i[2]
        print('x', x.shape)
        vert = []
    elif obj.ndim == 2:
        vert = obj
        x, y, z = vert[:, 0], vert[:, 1], vert[:, 2]
        vol = []
    elif obj.ndim == 1:
        x, y, x_delta, y_delta = obj[0], obj[1], obj[2], obj[3]
        bbox_vert = np.array([[x, y], [x + x_delta, y], [x, y + y_delta], [x + x_delta, y + y_delta]])
        bbox_vol = []
        return bbox_vert, bbox_mp, bbox_vol

    if x.size == 0 or y.size == 0 or z.size == 0:
        return bbox_vert, bbox_mp, bbox_vol

    maxx, maxy, maxz = np.max(x), np.max(y), np.max(z)
    print('max', maxx, maxy, maxz)
    minx, miny, minz = np.min(x), np.min(y), np.min(z)

    if paddingsize is not None:
        padding_x, padding_y, padding_z = paddingsize, paddingsize, paddingsize
    else:
        padding_x, padding_y, padding_z = 0, 0, 0

    minx_p = minx - padding_x
    maxx_p = maxx + padding_x
    miny_p = miny - padding_y
    maxy_p = maxy + padding_y
    minz_p = minz - padding_z
    maxz_p = maxz + padding_z

    bbox_vert = np.array([
        [minx_p, miny_p, minz_p], [minx_p, miny_p, maxz_p], [minx_p, maxy_p, minz_p], [minx_p, maxy_p, maxz_p],
        [maxx_p, miny_p, minz_p], [maxx_p, miny_p, maxz_p], [maxx_p, maxy_p, minz_p], [maxx_p, maxy_p, maxz_p]
    ])

    bbox_mp = np.array([
        [minx_p, miny_p + (maxy_p - miny_p) / 2, minz_p + (maxz_p - minz_p) / 2],
        [maxx_p, miny_p + (maxy_p - miny_p) / 2, minz_p + (maxz_p - minz_p) / 2],
        [minx_p + (maxx_p - minx_p) / 2, miny_p, minz_p + (maxz_p - minz_p) / 2],
        [minx_p + (maxx_p - minx_p) / 2, maxy_p, minz_p + (maxz_p - minz_p) / 2],
        [minx_p + (maxx_p - minx_p) / 2, miny_p + (maxy_p - miny_p) / 2, minz_p],
        [minx_p + (maxx_p - minx_p) / 2, miny_p + (maxy_p - miny_p) / 2, maxz_p]
    ])

    T = np.array([[1, 0, 0, -minx_p + 1], [0, 1, 0, -miny_p + 1], [0, 0, 1, -minz_p + 1], [0, 0, 0, 1]])
    bbox_vol = np.round(xform_apply(bbox_vert, T))

    if vol.size != 0:
        vol = np.zeros_like(vol, dtype=np.uint8)
        vol[bbox_vol[0, 0]:bbox_vol[-1, 0] + 1, bbox_vol[0, 1]:bbox_vol[-1, 1] + 1, bbox_vol[0, 2]:bbox_vol[-1, 2] + 1] = 1

    print("bbox_vol", bbox_vol)
    return bbox_vert, bbox_mp, bbox_vol

