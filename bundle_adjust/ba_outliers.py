"""
Bundle Adjustment for 3D Reconstruction from Multi-Date Satellite Images
This script implements a series of functions dedicated to the suppression of outliers according to reprojection error
by Roger Mari <roger.mari@ens-paris-saclay.fr>
"""

import numpy as np
import matplotlib.pyplot as plt


def get_elbow_value(err, verbose=False):
    """
    Plot a function that is expected to follow a L-shape and compute elbow value
    We compute the elbow value as the point furthest away between the segment going from min to max values
    Source: https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    Args:
        err: input vector of values
        verbose (optional): boolean, a plot will be displayed if True
    Returns:
        elbow_value: scalar with the elbow value of the function
        success: boolean used to determine whether to trust the result or no
                 success is False when elbow_value falls below the 80-th percentile, i.e. more than 20% of outliers
                 in that case it is extremely likely that the input values simply do not follow an L-shape
    """

    values = np.sort(err).tolist()
    n_pts = len(values)
    all_coord = np.vstack((range(n_pts), values)).T

    # get vector between first and last point - this is the line
    line_vec = all_coord[-1] - all_coord[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))

    # find the distance from each point to the line:
    vec_from_first = all_coord - all_coord[0]
    scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (n_pts, 1)), axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))

    # knee/elbow is the point with max distance value
    elbow_value = values[np.argmax(dist_to_line)]
    #elbow_value = np.percentile(err[err < elbow_value], 99)
    success = False if (elbow_value < np.percentile(err, 80)) else True

    if verbose:
        plt.figure(figsize=(10, 5))
        plt.plot(values)
        plt.axhline(y=elbow_value, color='r', linestyle='-')
        plt.title('Elbow value is {:.3f}'.format(elbow_value))
        plt.show()

    return elbow_value, success


def remove_outliers_from_reprojection_error(err, p, thr=1.0, verbose=False):
    """
    Remove observations from the correspondence matrix C if their reprojection error is larger than a threshold
    Args:
        err: vector of length K with the reprojection error of each 2d observation in C
        p: bundle adjustment parameters object
        thr: threshold with the maximum reprojection error allowed
    Returns:
        new_p: updated bundle adjustments parameters object
    """

    pts3d_new = p.pts3d.copy()
    C_new = p.C.copy()
    n_deleted_obs = 0
    for i in range(len(err)):
        if err[i] > thr:
            n_deleted_obs += 1
            track_where_obs, cam_where_obs = p.pts_ind[i], p.cam_ind[i]
            # count number of obs x track (if the track is formed by only one match, then delete it)
            # otherwise delete only that particular observation
            C_new[2 * cam_where_obs, track_where_obs] = np.nan
            C_new[2 * cam_where_obs + 1, track_where_obs] = np.nan

    # count the updated number of obs per track and keep those tracks with 2 or more observations
    obs_per_track = np.sum(1 * np.invert(np.isnan(C_new)), axis=0)
    tracks_to_preserve_1 = obs_per_track >= 4
    C_new, pts3d_new = C_new[:, tracks_to_preserve_1], pts3d_new[tracks_to_preserve_1, :]

    # remove matches found in pairs with short baseline that were not extended to more images
    from feature_tracks.ft_utils import filter_C_using_pairs_to_triangulate
    tracks_to_preserve_2 = filter_C_using_pairs_to_triangulate(C_new, p.pairs_to_triangulate)
    C_new, pts3d_new = C_new[:, tracks_to_preserve_2], pts3d_new[tracks_to_preserve_2, :]
    n_deleted_tracks = p.C.shape[1] - C_new.shape[1]

    # TODO: Check no camera is left with 0 observations

    # update pts_prev_indices in ba_params
    indices_left_after_error_check = np.arange(len(tracks_to_preserve_1))[tracks_to_preserve_1]
    indices_left_after_baseline_check = np.arange(len(tracks_to_preserve_2))[tracks_to_preserve_2]
    final_indices_left = indices_left_after_error_check[indices_left_after_baseline_check]

    if verbose:
        print('\nRemoval of outliers according to reprojection error completed')
        args = [n_deleted_obs, n_deleted_obs / len(err) * 100,
                n_deleted_tracks, n_deleted_tracks / p.C.shape[1] * 100]
        print('Deleted {} observations ({:.2f}%) and {} tracks ({:.2f}%)'.format(*args))
        print('     - Obs per cam before : {}'.format(np.sum(1 * ~np.isnan(p.C), axis=1)[::2]))
        print('     - Obs per cam after  : {}\n'.format(np.sum(1 * ~np.isnan(C_new), axis=1)[::2]))

    from bundle_adjust.ba_params import BundleAdjustmentParameters
    new_p = BundleAdjustmentParameters(C_new, pts3d_new, p.cameras, p.cam_model, p.pairs_to_triangulate,
                                       reduce=False, verbose=verbose)
    new_p.pts_prev_indices = p.pts_prev_indices[final_indices_left]

    return new_p