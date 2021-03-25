"""
Bundle Adjustment for 3D Reconstruction from Multi-Date Satellite Images
This script implements all functions necessary to handle the different camera models considered in this project
The considered cameras are defined by perspective or affine projection matrices or rpc models
by Roger Mari <roger.mari@ens-paris-saclay.fr>
"""

import numpy as np


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
    return K @ R @ np.hstack((np.eye(3), -oC[:, np.newaxis]))


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
        P_affine: 3x4 affine projection matrix
    """
    import ad

    from bundle_adjust.geotools import ecef_to_latlon_custom_ad

    p = ad.adnumber([x, y, z])
    lat, lon, alt = ecef_to_latlon_custom_ad(*p)
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
        P_perspective: 3x4 affine projection matrix
    """
    from bundle_adjust.rpc_utils import approx_rpc_as_proj_matrix

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
        rpc: rpc model
        pts3d: Nx3 array of 3d points in ECEF coordinates
    Returns:
        pts2d: Nx2 array containing the 2d projections of pts3d given by the rpc model
    """
    from bundle_adjust.geotools import ecef_to_latlon_custom

    lat, lon, alt = ecef_to_latlon_custom(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])
    col, row = rpc.projection(lon, lat, alt)
    pts2d = np.vstack((col, row)).T
    return pts2d

