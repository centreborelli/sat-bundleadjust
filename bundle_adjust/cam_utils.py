"""
A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery
author: Roger Mari <roger.mari@ens-paris-saclay.fr>
year: 2021

This script consists of a series of functions dedicated to handle camera models:
composition and decomposition, local approximation of a projection matrix from a RPC model
and other secondary tasks. The considered models are: RPC and perspective or affine camera
"""


import matplotlib.pyplot as plt
import numpy as np

from bundle_adjust import geo_utils


def decompose_perspective_camera(P):
    """
    Decomposition of the perspective camera matrix as P = KR[I|-C] = K [R | vecT] (Hartley and Zissermann 6.2.4)
    Let  P = [M|T]. Compute internal and rotation as [K,R] = rq(M). Fix the sign so that diag(K) is positive.
    Camera center is computed with the formula C = -M^-1 T

    Args:
        P: 3x4 perspective projection matrix

    Returns:
        K: 3x3 calibration matrix
        R: 3x3 rotation matrix
        vecT: 3x1 translation vector
        oC: 3x1 optical center
    """
    from scipy import linalg

    # rq decomposition of M gives rotation and calibration
    M, T = P[:, :-1], P[:, -1]
    K, R = linalg.rq(M)
    # fix sign of the scale params
    R = np.diag(np.sign(np.diag(K))).dot(R)
    K = K.dot(np.diag(np.sign(np.diag(K))))
    # optical center
    oC = -((np.linalg.inv(M)).dot(T))
    # translation vector of the camera
    vecT = (R @ -oC[:, np.newaxis]).T[0]
    # fix sign of the scale params
    R = np.diag(np.sign(np.diag(K))).dot(R)
    K = K.dot(np.diag(np.sign(np.diag(K))))
    return K, R, vecT, oC


def compose_perspective_camera(K, R, oC):
    """
    Compose perspective camera matrix as P = KR[I|-C]
    Args:
        K: 3x3 calibration matrix
        R: 3x3 rotation matrix
        oC: 3x1 optical center
    Returns:
        P: 3x4 perspective projection matrix
    """
    P = K @ R @ np.hstack((np.eye(3), -oC[:, np.newaxis]))
    return P


def get_perspective_optical_center(P):
    """
    Extract the optical center of a perspective projection matrix

    Args:
        P: 3x4 perspective projection matrix

    Returns:
        oC: 3x1 optical center of P
    """
    _, _, _, oC = decompose_perspective_camera(P)
    return oC


def decompose_affine_camera(P):
    """
    Decomposition of the affine camera matrix

    Args:
        P: 3x4 perspective projection matrix

    Returns:
        K: 2x2 calibration matrix
        R: 3x3 rotation matrix
        vecT: 2x1 translation vector
    """
    M, vecT = P[:2, :3], np.array([P[:2, -1]])
    MMt = M @ M.T
    fy = np.sqrt(MMt[1, 1])
    s = MMt[1, 0] / fy
    fx = np.sqrt(MMt[0, 0] - s ** 2)
    K = np.array([[fx, s], [0, fy]])
    # check that the solution of K is valid:
    # print(np.allclose(np.identity(2), np.linalg.inv(K) @ MMt @ np.linalg.inv(K.T)  ))
    R = np.linalg.inv(K) @ M
    r1 = np.array([R[0, :]]).T
    r2 = np.array([R[1, :]]).T
    r3 = np.cross(r1, r2, axis=0)
    R = np.vstack((r1.T, r2.T, r3.T))
    return K, R, vecT


def compose_affine_camera(K, R, vecT):
    """
    Compose affine camera matrix as P = KR[I|-C]

    Args:
        K: 3x3 calibration matrix
        R: 3x3 rotation matrix
        oC: optical center

    Returns:
        P: 3x4 perspective projection matrix
    """
    return np.vstack((np.hstack((K @ R[:2, :], vecT.T)), np.array([[0, 0, 0, 1]], dtype=np.float32)))


def approx_rpc_as_affine_projection_matrix(rpc, x, y, z, offset={"col0": 0.0, "row0": 0.0}):
    """
    Compute the first order Taylor approximation of an RPC projection function

    Args:
        rpc: instance of the rpc_model.RPCModel class
        x: ECEF x-coordinate of the 3d point where we want to locally approximate the rpc
        y: ECEF y-coordinate of the 3d point where we want to locally approximate the rpc
        z: ECEF y-coordinate of the 3d point where we want to locally approximate the rpc
        offset (optional): dictionary containing a translation (useful when working with crops of big geotiffs)

    Returns:
        P: 3x4 affine projection matrix
    """
    import ad

    p = ad.adnumber([x, y, z])
    lat, lon, alt = geo_utils.ecef_to_latlon_custom_ad(*p)
    q = rpc.projection(lon, lat, alt)
    J = ad.jacobian(q, p)
    A = np.zeros((3, 4))
    A[:2, :3] = J
    A[:2, 3] = np.array(q) - np.dot(J, p)
    A[2, 3] = 1
    P_img = A.copy()
    offset_translation = np.array([[1.0, 0.0, -offset["col0"]], [0.0, 1.0, -offset["row0"]], [0.0, 0.0, 1.0]])
    P_crop = offset_translation @ P_img  # use offset_translation to set (0,0) to the top-left corner of the crop
    P = P_crop / P_crop[2, 3]
    return P


def approx_rpc_as_perspective_projection_matrix(rpc, offset):
    """
    Approximate the RPC projection function as a 3x4 perspective projection matrix P
    P is found via resectioning, using a set of correspondences between a grid of 3d points and their 2d projections

    Args:
        rpc: instance of the rpc_model.RPCModel class
        x: ECEF x-coordinate of the 3d point where we want to locally approximate the rpc
        y: ECEF y-coordinate of the 3d point where we want to locally approximate the rpc
        z: ECEF y-coordinate of the 3d point where we want to locally approximate the rpc
        offset (optional): dictionary containing a translation (useful when working with crops of big geotiffs)

    Returns:
        P: 3x4 affine projection matrix
        mean_err: the average reprojection error associated to P with respect to the RPC model
    """
    x, y, w, h, alt = offset["col0"], offset["row0"], offset["width"], offset["height"], rpc.alt_offset
    P_img, mean_err = approx_rpc_as_proj_matrix(rpc, [x, x + w, 10], [y, y + h, 10], [alt - 100, alt + 100, 10])
    offset_translation = np.array([[1.0, 0.0, -x], [0.0, 1.0, -y], [0.0, 0.0, 1.0]])
    P_crop = offset_translation @ P_img  # use offset_translation to set (0,0) to the top-left corner of the crop
    P = P_crop / P_crop[2, 3]
    return P, mean_err


def apply_projection_matrix(P, pts3d):
    """
    Use a projection matrix to project a set of 3d points

    Args:
        P: 3x4 projection matrix
        pts3d: Nx3 array of 3d points in ECEF coordinates

    Returns:
        pts2d: Nx2 array containing the 2d projections of pts3d given by P
    """
    proj = P @ np.hstack((pts3d, np.ones((pts3d.shape[0], 1)))).T
    pts2d = (proj[:2, :] / proj[-1, :]).T
    return pts2d


def apply_rpc_projection(rpc, pts3d):
    """
    Use rpc model to project a set of 3d points

    Args:
        rpc: RPC model
        pts3d: Nx3 array of 3d points in ECEF coordinates

    Returns:
        pts2d: Nx2 array containing the 2d projections of pts3d given by the RPC model
    """
    lat, lon, alt = geo_utils.ecef_to_latlon_custom(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])
    col, row = rpc.projection(lon, lat, alt)
    pts2d = np.vstack((col, row)).T
    return pts2d


def approx_rpc_as_proj_matrix(rpc_model, col_range, lin_range, alt_range, verbose=False):
    """
    Returns a least-square approximation of the RPC functions as a projection
    matrix. The approximation is optimized on a sampling of the 3D region
    defined by the altitudes in alt_range and the image tile defined by
    col_range and lin_range. The average reprojection error of the least-square
    fit is also returned as output (mean_err)
    """
    ### step 1: generate cartesian coordinates of 3d points used to fit the
    ###         best projection matrix
    # get mesh points and convert them to geodetic then to geocentric
    # coordinates
    cols, lins, alts = generate_point_mesh(col_range, lin_range, alt_range)
    lons, lats = rpc_model.localization(cols, lins, alts)
    x, y, z = geo_utils.latlon_to_ecef_custom(lats, lons, alts)

    ### step 2: estimate the camera projection matrix from corresponding
    # 3-space and image entities
    world_points = np.vstack([x, y, z]).T
    image_points = np.vstack([cols, lins]).T
    P = camera_matrix(world_points, image_points)

    ### step 3: for debug, test the approximation error
    # compute the projection error made by the computed matrix P, on the
    # used learning points
    proj = P @ np.hstack((world_points, np.ones((world_points.shape[0], 1)))).T
    image_pointsPROJ = (proj[:2, :] / proj[-1, :]).T
    colPROJ, linPROJ = image_pointsPROJ[:, 0], image_pointsPROJ[:, 1]
    d_col, d_lin = cols - colPROJ, lins - linPROJ
    mean_err = np.mean(np.linalg.norm(image_points - image_pointsPROJ, axis=1))

    if verbose:
        _, f = plt.subplots(1, 2, figsize=(10, 3))
        f[0].hist(np.sort(d_col), bins=40)
        f[0].title.set_text("col diffs")
        f[1].hist(np.sort(d_lin), bins=40)
        f[1].title.set_text("row diffs")
        plt.show()

        print("approximate_rpc_as_projective: (min, max, mean)")
        print("distance on cols:", np.min(d_col), np.max(d_col), np.mean(d_col))
        print("distance on rows:", np.min(d_lin), np.max(d_lin), np.mean(d_lin))

    return P, mean_err


def generate_point_mesh(col_range, row_range, alt_range):
    """
    Generates image coordinates (col, row, alt) of 3D points located on the grid
    defined by col_range and row_range, at uniformly sampled altitudes defined
    by alt_range.
    Args:
        col_range: triplet (col_min, col_max, n_col), where n_col is the
            desired number of samples
        row_range: triplet (row_min, row_max, n_row)
        alt_range: triplet (alt_min, alt_max, n_alt)

    Returns:
        3 lists, containing the col, row and alt coordinates.
    """
    # input points in col, row, alt space
    cols, rows, alts = [np.linspace(v[0], v[1], v[2]) for v in [col_range, row_range, alt_range]]

    # make it a kind of meshgrid (but with three components)
    # if cols, rows and alts are lists of length 5, then after this operation
    # they will be lists of length 5x5x5
    cols, rows, alts = (
        (cols + 0 * rows[:, np.newaxis] + 0 * alts[:, np.newaxis, np.newaxis]).reshape(-1),
        (0 * cols + rows[:, np.newaxis] + 0 * alts[:, np.newaxis, np.newaxis]).reshape(-1),
        (0 * cols + 0 * rows[:, np.newaxis] + alts[:, np.newaxis, np.newaxis]).reshape(-1),
    )

    return cols, rows, alts


def camera_matrix(X, x):
    """
    Estimates the camera projection matrix from corresponding 3-space and image
    entities.

    Arguments:
        X: 2D array of size Nx3 containing the coordinates of the 3-space
            points, one point per line
        x: 2D array of size Nx2 containing the pixel coordinates of the imaged
            points, one point per line
            These two arrays are supposed to have the same number of lines.

    Returns:
        the estimated camera projection matrix, given by the Direct Linear
        Transformation algorithm, as described in Hartley & Zisserman book.
    """
    # normalize the input coordinates
    X, U = normalize_3d_points(X)
    x, T = normalize_2d_points(x)

    # make a linear system A*P = 0 from the correspondances, where P is made of
    # the 12 entries of the projection matrix (encoded in a vector P). This
    # system corresponds to the concatenation of correspondance constraints
    # (X_i --> x_i) which can be written as:
    # x_i x P*X_i = 0 (the vectorial product is 0)
    # and lead to 2 independent equations, for each correspondance. The system
    # is thus of size 2n x 12, where n is the number of correspondances. See
    # Zissermann, chapter 7, for more details.

    A = np.zeros((len(x) * 2, 12))
    for i in range(len(x)):
        A[2 * i + 0, 4:8] = -1 * np.array([X[i, 0], X[i, 1], X[i, 2], 1])
        A[2 * i + 0, 8:12] = x[i, 1] * np.array([X[i, 0], X[i, 1], X[i, 2], 1])
        A[2 * i + 1, 0:4] = np.array([X[i, 0], X[i, 1], X[i, 2], 1])
        A[2 * i + 1, 8:12] = -x[i, 0] * np.array([X[i, 0], X[i, 1], X[i, 2], 1])

    # the vector P we are looking for minimizes the norm of A*P, and satisfies
    # the constraint \norm{P}=1 (to avoid the trivial solution P=0). This
    # solution is obtained as the singular vector corresponding to the smallest
    # singular value of matrix A. See Zissermann for details.
    # It is the last line of matrix V (because np.linalg.svd returns V^T)
    W, S, V = np.linalg.svd(A)
    P = V[-1, :].reshape((3, 4))

    # denormalize P
    # implements P = T^-1 * P * U
    P = np.dot(np.dot(np.linalg.inv(T), P), U)
    return P


def normalize_2d_points(pts):
    """
    Translates and scales 2D points.

    The input points are translated and scaled such that the output points are
    centered at origin and the mean distance from the origin is sqrt(2). As
    shown in Hartley (1997), this normalization process typically improves the
    condition number of the linear systems used for solving homographies,
    fundamental matrices, etc.

    References:
        Richard Hartley, PAMI 1997
        Peter Kovesi, MATLAB functions for computer vision and image processing,

    Args:
        pts: 2D array of dimension Nx2 containing the coordinates of the input
            points, one point per line

    Returns:
        new_pts, T: coordinates of the transformed points, together with
            the similarity transform
    """
    # centroid
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])

    # shift origin to centroid
    new_x = pts[:, 0] - cx
    new_y = pts[:, 1] - cy

    # scale such that the average distance from centroid is \sqrt{2}
    mean_dist = np.mean(np.sqrt(new_x ** 2 + new_y ** 2))
    s = np.sqrt(2) / mean_dist
    new_x = s * new_x
    new_y = s * new_y

    # matrix T           s     0   -s * cx
    # is given     T  =  0     s   -s * cy
    # by                 0     0    1
    T = np.eye(3)
    T[0, 0] = s
    T[1, 1] = s
    T[0, 2] = -s * cx
    T[1, 2] = -s * cy

    return np.vstack([new_x, new_y]).T, T


def normalize_3d_points(pts):
    """
    Translates and scales 3D points.

    The input points are translated and scaled such that the output points are
    centered at origin and the mean distance from the origin is sqrt(3).

    Args:
        pts: 2D array of dimension Nx3 containing the coordinates of the input
            points, one point per line

    Returns:
        new_pts, U: coordinates of the transformed points, together with
            the similarity transform
    """
    # centroid
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    cz = np.mean(pts[:, 2])

    # shift origin to centroid
    new_x = pts[:, 0] - cx
    new_y = pts[:, 1] - cy
    new_z = pts[:, 2] - cz

    # scale such that the average distance from centroid is \sqrt{3}
    mean_dist = np.mean(np.sqrt(new_x ** 2 + new_y ** 2 + new_z ** 2))
    s = np.sqrt(3) / mean_dist
    new_x = s * new_x
    new_y = s * new_y
    new_z = s * new_z

    # matrix U             s     0      0    -s * cx
    # is given             0     s      0    -s * cy
    # by this        U  =  0     0      s    -s * cz
    # formula              0     0      0     1

    U = np.eye(4)
    U[0, 0] = s
    U[1, 1] = s
    U[2, 2] = s
    U[0, 3] = -s * cx
    U[1, 3] = -s * cy
    U[2, 3] = -s * cz

    return np.vstack([new_x, new_y, new_z]).T, U
