"""
A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery
author: Roger Mari <roger.mari@ens-paris-saclay.fr>
year: 2021

This script implements functions dedicated to initialize the coordinates of 3d points
according to their feature tracks observations, which is done by triangulation
"""

import numpy as np
import cv2

from bundle_adjust import geo_utils
from bundle_adjust.loader import flush_print
from s2p.triangulation import stereo_corresp_to_xyz


def linear_triangulation_multiple_pts(P1, P2, pts1, pts2):
    """
    Linear triangulation of a list of pairwise correspondences using 3x4 projection matrices

    Args:
        P1: array of size 3x4 corresponding to the projection matrix of the first image
        P2: array of size 3x4 corresponding to the projection matrix of the second image
        pts1: array of size Nx2 containing the image coordinates (col, row) of N points in the first image
        pts2: array of size Nx2 corresponding to the image coordinates of pts1 as seen in a second image

    Returns:
        pts3d: array of size Nx3 with the 3d coordinates (x, y, z) corresponding to each pairwise correspondence
    """
    X = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    X = X[:3, :] / X[-1, :]
    pts3d = X.T
    return pts3d


def rpc_triangulation(rpc1, rpc2, pts1, pts2):
    """
    Triangulate a list of pairwise correspondences using Rational Polynomial Camera models

    Args:
        rpc1: RPC model of the first image
        rpc2: RPC model of the second image
        pts1: array of size Nx2 containing the image coordinates (col, row) of N points in the first image
        pts2: array of size Nx2 corresponding to the image coordinates of pts1 as seen in a second image

    Returns:
        pts3d: array of size Nx3 with the 3d coordinates (x, y, z) corresponding to each pairwise correspondence
        err: array of size Nx1 with the error associated to each triangulated point
    """
    lonlatalt, err = stereo_corresp_to_xyz(rpc1, rpc2, pts1, pts2)
    x, y, z = geo_utils.latlon_to_ecef_custom(lonlatalt[:, 1], lonlatalt[:, 0], lonlatalt[:, 2])
    pts3d = np.vstack((x, y, z)).T
    return pts3d, err


def init_pts3d(C, cameras, cam_model, pairs_to_triangulate, verbose=False):
    """
    Initialize the 3d point corresponding to each feature track
    How? Pick the average value of all possible triangulated points within each track

    Args:
        C: array of size 2CxT, correspondence matrix describing T feature tracks across C cameras
        cameras: a list containing the camera models (either RPCs or projection matrices)
        cam_model: string, stating the camera model used in cameras: "affine", "perspective" or "rpc"
        pairs_to_triangulate: a list of pairs, where each pair is represented by a tuple of image indices
                              the pairs in this list are considered as suitable for triangulation purposes
        verbose (optional): boolean, set to True to print some information about the process

    Returns:
        avg_pts3d: array of size Tx3 with the 3d coordinates (x, y, z) corresponding to each feature track
    """

    def update_avg_pts3d(avg, count, new_v, t):
        # t = indices of the points 3d to update
        count[t] += 1.0
        avg[t] = ((count[t, np.newaxis] - 1.0) * avg[t] + new_v[t]) / count[t, np.newaxis]
        return avg, count

    import time

    t0 = time.time()
    last_print = time.time()

    n_pts, n_cam = C.shape[1], C.shape[0] // 2
    avg_pts3d = np.zeros((n_pts, 3), dtype=np.float32)
    n_pairs = np.zeros(n_pts, dtype=np.float32)
    n_triangulation_pairs = len(pairs_to_triangulate)
    mask = ~np.isnan(C[::2])

    if verbose:
        flush_print("Computing {} points 3d from feature tracks...".format(n_pts))

    for pair_idx, (c_i, c_j) in enumerate(pairs_to_triangulate):

        # get all track observations in cam_i with an equivalent observation in cam_j
        if c_i < n_cam and c_j < n_cam:
            pt_indices = np.where(mask[c_i] & mask[c_j])[0]
            obs2d_i = C[2 * c_i : 2 * c_i + 2, pt_indices].T
            obs2d_j = C[2 * c_j : 2 * c_j + 2, pt_indices].T

            if pt_indices.shape[0] == 0:
                continue

            # triangulate
            if cam_model in ["affine", "perspective"]:
                new_pts3d = linear_triangulation_multiple_pts(cameras[c_i], cameras[c_j], obs2d_i, obs2d_j)
            else:
                new_pts3d, _ = rpc_triangulation(cameras[c_i], cameras[c_j], obs2d_i, obs2d_j)

            # update average 3d point coordinates
            new_values = np.zeros((n_pts, 3), dtype=np.float32)
            new_values[pt_indices] = new_pts3d
            avg_pts3d, n_pairs = update_avg_pts3d(avg_pts3d, n_pairs, new_values, pt_indices)

            if verbose and ((time.time() - last_print) > 10 or pair_idx == n_triangulation_pairs - 1):
                to_print = [pair_idx + 1, n_triangulation_pairs, time.time() - t0]
                flush_print("...{}/{} triangulation pairs done in {:.2f} seconds".format(*to_print))
                last_print = time.time()

    if verbose:
        flush_print("done!")

    return avg_pts3d
