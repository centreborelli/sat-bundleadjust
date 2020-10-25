"""
Bundle Adjustment for 3D Reconstruction from Multi-Date Satellite Images
This script implements a series of functions dedicated to the suppression of outliers according to reprojection error
by Roger Mari <roger.mari@ens-paris-saclay.fr>
"""

import numpy as np
import matplotlib.pyplot as plt
import timeit

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
        plt.title('Elbow value is {:.3f}. Success: {}'.format(elbow_value, success))
        plt.show()

    return elbow_value, success


def reset_ba_params_after_outlier_removal(C_new, p, reuse_pts3d=False, verbose=True):

    # count the updated number of obs per track and keep those tracks with 2 or more observations
    obs_per_track = np.sum(1 * np.invert(np.isnan(C_new)), axis=0)
    tracks_to_preserve_1 = obs_per_track >= 4
    C_new, pts3d_new = C_new[:, tracks_to_preserve_1], p.pts3d[tracks_to_preserve_1, :]

    # remove matches found in pairs with short baseline that were not extended to more images
    from feature_tracks.ft_utils import filter_C_using_pairs_to_triangulate
    tracks_to_preserve_2 = filter_C_using_pairs_to_triangulate(C_new, p.pairs_to_triangulate)
    C_new, pts3d_new = C_new[:, tracks_to_preserve_2], pts3d_new[tracks_to_preserve_2, :]

    # TODO: Check no camera is left with 0 observations

    # update pts_prev_indices and n_pts_fix in ba_params
    indices_left_after_error_check = np.arange(len(tracks_to_preserve_1))[tracks_to_preserve_1]
    indices_left_after_baseline_check = np.arange(len(tracks_to_preserve_2))[tracks_to_preserve_2]
    final_indices_left = indices_left_after_error_check[indices_left_after_baseline_check]
    n_pts_fix_new = np.sum(1 * (final_indices_left < p.n_pts_fix))

    if not reuse_pts3d:
        from bundle_adjust.ba_triangulate import init_pts3d
        pts3d_new = init_pts3d(C_new, p.cameras, p.cam_model, p.pairs_to_triangulate, verbose=verbose)
        if n_pts_fix_new > 0:
            pts3d_new[:n_pts_fix_new, :] = p.pts3d[final_indices_left[final_indices_left < p.n_pts_fix], :]

    from bundle_adjust.ba_params import BundleAdjustmentParameters
    new_p = BundleAdjustmentParameters(C_new, pts3d_new, p.cameras, p.cam_model, p.pairs_to_triangulate,
                                       n_cam_fix=p.n_cam_fix, n_pts_fix=n_pts_fix_new, reduce=False, verbose=verbose)
    new_p.pts_prev_indices = p.pts_prev_indices[final_indices_left]

    return new_p


def rm_outliers_based_on_reprojection_error_imagewise(err, p, reuse_pts3d=False, verbose=False):
    """
    Remove observations from the correspondence matrix C
    if their reprojection error is larger than a threshold specific to each camera
    Args:
        err: vector of length K with the reprojection error of each 2d observation in C
        p: bundle adjustment parameters object
    Returns:
        new_p: updated bundle adjustments parameters object
    """

    start = timeit.default_timer()
    n_obs_in = err.shape[0]
    indices_obs_to_delete_pts_idx, indices_obs_to_delete_cam_idx, cam_thr = [], [], []
    for cam_idx in range(p.n_cam):
        indices_obs = np.arange(n_obs_in)[p.cam_ind==cam_idx]
        elbow_value, success = get_elbow_value(err[indices_obs], verbose=True)
        thr = max(elbow_value, 2.0) if success else np.max(err[indices_obs])
        indices_obs_to_delete = np.arange(n_obs_in)[indices_obs[err[indices_obs] > thr]]
        if len(indices_obs_to_delete) > 0:
            indices_obs_to_delete_pts_idx.extend(p.pts_ind[indices_obs_to_delete].tolist())
            indices_obs_to_delete_cam_idx.extend(p.cam_ind[indices_obs_to_delete].tolist())
        cam_thr.append(np.round(thr, 2))

    # delete outlier observations from C
    C_new = p.C.copy()
    if len(indices_obs_to_delete_cam_idx) > 0:
        C_new[np.array(indices_obs_to_delete_cam_idx) * 2, np.array(indices_obs_to_delete_pts_idx)] = np.nan
        C_new[np.array(indices_obs_to_delete_cam_idx) * 2 + 1, np.array(indices_obs_to_delete_pts_idx)] = np.nan
        new_p = reset_ba_params_after_outlier_removal(C_new, p, reuse_pts3d=reuse_pts3d, verbose=verbose)
    else:
        new_p = p

    if verbose:
        n_deleted_obs, n_deleted_tracks = len(indices_obs_to_delete_pts_idx), p.C.shape[1] - C_new.shape[1]
        running_time = timeit.default_timer() - start
        print('Removal of outliers based on reprojection error completed in {:.2f} seconds'.format(running_time))
        print('Reprojection error threshold per camera: {} px'.format(cam_thr))
        args = [n_deleted_obs, n_deleted_obs / n_obs_in * 100,
                n_deleted_tracks, n_deleted_tracks / p.C.shape[1] * 100]
        print('Deleted {} observations ({:.2f}%) and {} tracks ({:.2f}%)'.format(*args))
        print('     - Obs per cam before: {}'.format(np.sum(1 * ~np.isnan(p.C), axis=1)[::2]))
        print('     - Obs per cam after:  {}\n'.format(np.sum(1 * ~np.isnan(C_new), axis=1)[::2]))

    return new_p


def rm_outliers_based_on_reprojection_error_global(err, p, verbose=False, reuse_pts3d=False):
    """
    Remove observations from the correspondence matrix C
    if their reprojection error is larger than a global threshold
    Args:
        err: vector of length K with the reprojection error of each 2d observation in C
        p: bundle adjustment parameters object
    Returns:
        new_p: updated bundle adjustments parameters object
    """

    start = timeit.default_timer()
    elbow_value, success = get_elbow_value(err, verbose=verbose)
    thr = max(elbow_value, 2.0) if success else np.max(err)

    where_obs_to_delete = err > thr
    indices_obs_to_delete_pts_idx = p.pts_ind[where_obs_to_delete].tolist()
    indices_obs_to_delete_cam_idx = p.cam_ind[where_obs_to_delete].tolist()

    # delete outlier observations from C
    C_new = p.C.copy()
    if len(indices_obs_to_delete_cam_idx) > 0:
        C_new[np.array(indices_obs_to_delete_cam_idx) * 2, np.array(indices_obs_to_delete_pts_idx)] = np.nan
        C_new[np.array(indices_obs_to_delete_cam_idx) * 2 + 1, np.array(indices_obs_to_delete_pts_idx)] = np.nan
        new_p = reset_ba_params_after_outlier_removal(C_new, p, reuse_pts3d=reuse_pts3d, verbose=verbose)
    else:
        new_p = p

    if verbose:
        n_deleted_obs, n_deleted_tracks = len(indices_obs_to_delete_pts_idx), p.C.shape[1] - C_new.shape[1]
        running_time = timeit.default_timer() - start
        print('Removal of outliers based on reprojection error completed in {:.2f} seconds'.format(running_time))
        print('Reprojection error threshold: {} px'.format(thr))
        args = [n_deleted_obs, n_deleted_obs / len(err) * 100,
                n_deleted_tracks, n_deleted_tracks / p.C.shape[1] * 100]
        print('Deleted {} observations ({:.2f}%) and {} tracks ({:.2f}%)'.format(*args))
        print('     - Obs per cam before: {}'.format(np.sum(1 * ~np.isnan(p.C), axis=1)[::2]))
        print('     - Obs per cam after:  {}\n'.format(np.sum(1 * ~np.isnan(C_new), axis=1)[::2]))

    return new_p