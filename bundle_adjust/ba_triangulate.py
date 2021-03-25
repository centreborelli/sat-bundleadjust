"""
* Bundle Adjustment (BA) for 3D Reconstruction from Multi-Date Satellite Images
* This script contains tools to triangulate 3d points from stereo correspondences
* by Roger Mari <roger.mari@ens-paris-saclay.fr>
"""

import numpy as np
import cv2

from bundle_adjust import geotools
from bundle_adjust.loader import flush_print


def linear_triangulation_multiple_pts(P1, P2, pts1, pts2):
    """
    Linear triangulation of multiple stereo correspondences
    pts1, pts2 are 2d arrays of shape Nx2
    P1, P2 are projection matrices of shape 3x4
    X are the output pts3d, an array of shape Nx3
    """
    X = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    X = X[:3, :] / X[-1, :]
    return X.T


def rpc_triangulation(rpc_im1, rpc_im2, pts2d_im1, pts2d_im2):

    from s2p.triangulation import stereo_corresp_to_xyz

    lonlatalt, err = stereo_corresp_to_xyz(rpc_im1, rpc_im2, pts2d_im1, pts2d_im2)
    x, y, z = geotools.latlon_to_ecef_custom(lonlatalt[:, 1], lonlatalt[:, 0], lonlatalt[:, 2])
    pts3d = np.vstack((x, y, z)).T
    return pts3d, err


def init_pts3d(C, cameras, cam_model, pairs_to_triangulate, verbose=False):
    """
    Initialize the 3D point corresponding to each feature track.
    How? Pick the average value of all possible triangulated points within each track.
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
