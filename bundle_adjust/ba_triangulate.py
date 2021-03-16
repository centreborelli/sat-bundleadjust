"""
* Bundle Adjustment (BA) for 3D Reconstruction from Multi-Date Satellite Images
* This script contains tools to triangulate 3d points from stereo correspondences
* by Roger Mari <roger.mari@ens-paris-saclay.fr>
"""

import numpy as np
import cv2

from bundle_adjust import ba_core, ba_utils, geotools


def linear_triangulation_single_pt_multiview(pts2d, projection_matrices):
    """
    pts2d = Nx2M array, where each row stands for the 2d observations of a 3d point. N 3d points, M cameras
    projection_matrices = list containing the M projection matrices
    A will have shape 2Mx4N
    """
    A = np.vstack(
        [
            np.array([pts2d[2 * i] * P[2, :] - P[0, :], pts2d[2 * i + 1] * P[2, :] - P[1, :]])
            for i, P in enumerate(projection_matrices)
        ]
    )
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    pt_3d = vh.T[:3, -1] / vh.T[-1, -1]
    return pt_3d


def linear_triangulation_single_pt(P1, P2, pt1, pt2):
    """
    Linear triangulation of a single stereo correspondence (does the same as triangulate points from OpenCV)
    """
    x1, y1, x2, y2 = pt1[0], pt1[1], pt2[0], pt2[1]
    A = np.array(
        [
            x1 * P1[2, :] - P1[0, :],
            y1 * P1[2, :] - P1[1, :],
            x2 * P2[2, :] - P2[0, :],
            y2 * P2[2, :] - P2[1, :],
        ]
    )
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    pt_3d = vh.T[:3, -1] / vh.T[-1, -1]
    # print(np.allclose(A, u @ np.diag(s) @ vh))  # to check that svd is applied properly
    # pt_3d_opencv = cv2.triangulatePoints(P1,P2,pt1,pt2)[:3,0]/cv2.triangulatePoints(P1,P2,pt1,pt2)[-1,0]
    # print(np.allclose(pt_3d, pt_3d_opencv)) # to check that linear triangulation works properly
    return pt_3d


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


def dist_between_proj_rays(pt1, pt2, P1, P2):
    """
    Input: two 2D correspondences (pt1, pt2) and two projection matrices (P1, P2)
    Output: the distance between the projection rays that go from the optical center of the cameras to the points pt1, pt2

    If the camera calibration and correspondence are both good, the rays should intersect and the distance should be 0
    This is why this distance gives an idea about the triangulation error

    Inspired by https://math.stackexchange.com/questions/2213165/find-shortest-distance-between-lines-in-3d
    """
    K, R, vecT, C1 = ba_core.decompose_perspective_camera(P1)
    dir_vec_ray1 = np.linalg.inv(K @ R) @ np.expand_dims(np.hstack((pt1, np.ones(1))), axis=1)
    K, R, vecT, C2 = ba_core.decompose_perspective_camera(P2)
    dir_vec_ray2 = np.linalg.inv(K @ R) @ np.expand_dims(np.hstack((pt2, np.ones(1))), axis=1)
    n = np.cross(dir_vec_ray1, dir_vec_ray2, axis=0)
    d = np.dot((C2 - C1), n) / np.linalg.norm(n)
    return abs(d[0])


def init_pts3d_multiview(C, cameras, verbose=False):
    import time

    t0 = time.time()
    last_print = time.time()

    n_pts, n_cam = C.shape[1], int(C.shape[0] / 2)
    pts_3d = np.zeros((n_pts, 3))

    true_where_obs = np.invert(np.isnan(C))  # (i,j)=True if j-th point seen in i-th image
    for pt_idx in range(n_pts):
        projection_matrices = [cameras[cam_idx] for cam_idx in np.where(true_where_obs[::2, pt_idx])]
        pts2d = C[true_where_obs[:, pt_idx], pt_idx]
        pts_3d[pt_idx, :] = linear_triangulation_single_pt_multiview(pts2d, projection_matrices)

        if verbose and ((time.time() - last_print) > 10 or pt_idx == n_pts - 1):
            args = [pt_idx + 1, n_pts, time.time() - t0]
            print("Computing points 3d from feature tracks... {}/{} done in {:.2f} seconds".format(*args), flush=True)
            last_print = time.time()
    return pts_3d


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

    # if cam_model == 'perspective':
    #    return init_pts3d_multiview(C, cameras, verbose=verbose)

    def update_avg_pts3d(avg, count, new_v, t):
        # t = indices of the points 3d to update
        count[t] += 1.0
        avg[t] = ((count[t, np.newaxis] - 1.0) * avg[t] + new_v[t]) / count[t, np.newaxis]
        return avg, count

    import time

    t0 = time.time()
    last_print = time.time()

    n_pts, n_cam = C.shape[1], int(C.shape[0] / 2)
    avg_pts3d = np.zeros((n_pts, 3), dtype=np.float32)
    n_pairs = np.zeros(n_pts, dtype=np.float32)
    n_triangulation_pairs = len(pairs_to_triangulate)

    if verbose:
        print("Computing {} points 3d from feature tracks...".format(n_pts), flush=True)

    for pair_idx, (c_i, c_j) in enumerate(pairs_to_triangulate):

        if c_i >= n_cam or c_j >= n_cam:
            continue

        # get all track observations in cam_i with an equivalent observation in cam_j
        pt_indices = np.arange(n_pts)[np.logical_and(~np.isnan(C[c_i * 2, :]), ~np.isnan(C[c_j * 2, :]))]
        obs2d_i = C[(c_i * 2) : (c_i * 2 + 2), pt_indices].T
        obs2d_j = C[(c_j * 2) : (c_j * 2 + 2), pt_indices].T

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
            args = [pair_idx + 1, n_triangulation_pairs, time.time() - t0]
            print(
                "...{}/{} triangulation pairs done in {:.2f} seconds".format(*args),
                flush=True,
            )
            last_print = time.time()

    if verbose:
        print("done!", flush=True)

    return avg_pts3d
