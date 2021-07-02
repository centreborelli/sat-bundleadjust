"""
A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery
author: Roger Mari <roger.mari@ens-paris-saclay.fr>
year: 2021

This script implements a series of functions dedicated to the suppression of
outlier feature track observations according to reprojection error
"""

import matplotlib.pyplot as plt
import numpy as np


def get_elbow_value(err, max_outliers_percent=20, verbose=False):
    """
    Compute the elbow value of an input function that is expected to follow a L-shape
    We compute the elbow value as the point furthest away between the segment going from min to max values
    Inspired by https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve

    Args:
        err: vector of values (i.e. input function)
        max_outliers_percent: the maximum percentage of outliers that is expected in the upper part of the function
        verbose (optional): boolean, a plot will be displayed if True

    Returns:
        elbow_value: scalar with the elbow value of the function
        success: success is False if elbow_value falls bellow the i-th percentile,
                 where i = 100 - max_outliers_percent; otherwise it is True
                 success = False implies that the input function is likely to not follow an L-shape
    """

    values = np.sort(err).tolist()
    n_pts = len(values)
    all_coord = np.vstack((range(n_pts), values)).T

    # get vector between first and last point - this is the line
    line_vec = all_coord[-1] - all_coord[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec ** 2))

    # find the distance from each point to the line:
    vec_from_first = all_coord - all_coord[0]
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_pts, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))

    # the elbow point is the point with max distance value
    elbow_value = values[np.argmax(dist_to_line)]
    success = False if (elbow_value < np.percentile(err, 100 - max_outliers_percent)) else True

    if verbose:
        plt.figure(figsize=(10, 5))
        plt.plot(values)
        plt.axhline(y=elbow_value, color="r", linestyle="-")
        plt.title("Elbow value is {:.3f}. Success: {}".format(elbow_value, success))
        plt.show()

    return elbow_value, success


def reset_ba_params_after_outlier_removal(C_new, p, verbose=True):
    """
    Update the bundle adjustment parameters after removing some 2d observations from a correspondence matrix

    Args:
        C_new: correspondence matrix where some 2d observations have been removed
        p: the bundle adjustment parameters associated to the correspondence matrix, which need to be updated

    Returns:
        new_p: output bundle adjustment parameters, which are now coherent with C_new
    """

    # count the updated number of obs per track and keep those tracks with 2 or more observations
    obs_per_track = np.sum(1 * np.invert(np.isnan(C_new)), axis=0)
    tracks_to_preserve_1 = obs_per_track >= 4
    C_new, pts3d_new = C_new[:, tracks_to_preserve_1], p.pts3d[tracks_to_preserve_1, :]

    # remove matches found in pairs with short baseline that were not extended to more images
    from .feature_tracks.ft_utils import filter_C_using_pairs_to_triangulate

    tracks_to_preserve_2 = filter_C_using_pairs_to_triangulate(C_new, p.pairs_to_triangulate)
    C_new = C_new[:, tracks_to_preserve_2]

    # update pts_prev_indices and n_pts_fix in ba_params
    indices_left_after_error_check = np.arange(len(tracks_to_preserve_1))[tracks_to_preserve_1]
    indices_left_after_baseline_check = np.arange(len(tracks_to_preserve_2))[tracks_to_preserve_2]
    final_indices_left = indices_left_after_error_check[indices_left_after_baseline_check]
    n_pts_fix_new = np.sum(1 * (final_indices_left < p.n_pts_fix))

    # triangulate new points with the observations left
    from .feature_tracks.ft_triangulate import init_pts3d

    pts3d_new = init_pts3d(C_new, p.cameras, p.cam_model, p.pairs_to_triangulate, verbose=verbose)
    if n_pts_fix_new > 0:
        pts3d_new[:n_pts_fix_new, :] = p.pts3d[final_indices_left[final_indices_left < p.n_pts_fix], :]

    from bundle_adjust.ba_params import BundleAdjustmentParameters

    args = [C_new, pts3d_new, p.cameras, p.cam_model, p.pairs_to_triangulate, p.camera_centers]
    d = {
        "n_cam_fix": p.n_cam_fix,
        "n_pts_fix": n_pts_fix_new,
        "reduce": False,
        "verbose": verbose,
        "correction_params": p.cam_params_to_optimize,
        "ref_cam_weight": p.ref_cam_weight,
    }
    new_p = BundleAdjustmentParameters(*args, d)
    new_p.pts_prev_indices = p.pts_prev_indices[final_indices_left]

    return new_p


def compute_obs_to_remove(err, p, predef_thr=None):
    """
    Identify outlier feature track observations based on their reprojection error
    For each camera, a reprojection error threshold T is automatically set
    all feature track observations with reprojection error larger than T are discarded

    Args:
        err: N-valued vector containing the reprojection error of each of the N track observations
        p: the bundle adjustment parameters employed to compute err

    Returns:
        C_new: correspondence matrix after removing all 2d observations detected as outliers
        cam_thr: the reprojection error threshold that was set for each camera
        n_detected_outliers: integer with the total amount of outlier observations that were found
    """

    # compute the reprojection error threshold for each camera
    min_thr = 1.0
    n_obs_in = err.shape[0]
    cam_thr = []
    for cam_idx in range(p.n_cam):
        if predef_thr is None:
            indices_obs = np.arange(n_obs_in)[p.cam_ind == cam_idx]
            elbow_value, success = get_elbow_value(err[indices_obs], verbose=False)
            thr = max(elbow_value, min_thr) if success else np.max(err[indices_obs])
            cam_thr.append(np.round(thr, 2))
        else:
            cam_thr.append(np.round(float(predef_thr), 2))

    obs_to_rm_pts_idx, obs_to_rm_cam_idx = [], []
    for cam_idx, thr in enumerate(cam_thr):
        indices_obs = np.arange(n_obs_in)[p.cam_ind == cam_idx]
        indices_obs_to_delete = np.arange(n_obs_in)[indices_obs[err[indices_obs] > thr]]
        if len(indices_obs_to_delete) > 0:
            obs_to_rm_pts_idx.extend(p.pts_ind[indices_obs_to_delete].tolist())
            obs_to_rm_cam_idx.extend(p.cam_ind[indices_obs_to_delete].tolist())

    # remove outlier 2d observations from the correspondence matrix of the bundle adjustment parameters
    C_new = p.C.copy()
    if len(obs_to_rm_cam_idx) > 0:
        C_new[np.array(obs_to_rm_cam_idx) * 2, np.array(obs_to_rm_pts_idx)] = np.nan
        C_new[np.array(obs_to_rm_cam_idx) * 2 + 1, np.array(obs_to_rm_pts_idx)] = np.nan
    n_detected_outliers = np.sum(~np.isnan(p.C[::2]).ravel()) - np.sum(~np.isnan(C_new[::2]).ravel())

    return C_new, cam_thr, n_detected_outliers


def rm_outliers(err, p, predef_thr=None, verbose=False):
    """
    Remove outlier feature track observations based on their reprojection error

    Args:
        err: N-valued vector containing the reprojection error of each of the N track observations
        p: the bundle adjustment parameters employed to compute err

    Returns:
        new_p: updated bundle adjustments parameters object
    """

    C_new, cam_thr, n_detected_outliers = compute_obs_to_remove(err, p, predef_thr=predef_thr)

    if n_detected_outliers > 0:
        new_p = reset_ba_params_after_outlier_removal(C_new, p, verbose=verbose)
    else:
        new_p = p

    if verbose:
        n_obs_in, n_obs_rm = len(p.cam_ind), n_detected_outliers
        n_tracks_in, n_tracks_rm = p.C.shape[1], p.C.shape[1] - new_p.C.shape[1]
        print("Reprojection error threshold per camera: {} px".format(cam_thr))
        args = [n_obs_rm, n_obs_rm / n_obs_in * 100, n_tracks_rm, n_tracks_rm / n_tracks_in * 100]
        print("Deleted {} observations ({:.2f}%) and {} tracks ({:.2f}%)".format(*args))
        # print("     - Obs per cam before: {}".format(np.sum(1 * ~np.isnan(p.C), axis=1)[::2]))
        # print("     - Obs per cam after:  {}\n".format(np.sum(1 * ~np.isnan(C_new), axis=1)[::2]))

    return new_p
