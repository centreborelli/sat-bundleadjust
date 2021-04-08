"""
A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery
author: Roger Mari <roger.mari@ens-paris-saclay.fr>
year: 2021

This script implements a regularized weighted least squares algorithm
which is used to fit a new RPC model from a grid of 2d-3d point correspondences
"""

import matplotlib.pyplot as plt
import numpy as np
import rpcm

from bundle_adjust import ba_core, cam_utils, geo_utils, ba_rotate


def poly_vect(x, y, z):
    """
    Returns evaluated polynomial vector without the first constant term equal to 1,
    using the order convention defined in rpc_model.apply_poly
    """
    return np.array(
        [
            y,
            x,
            z,
            y * x,
            y * z,
            x * z,
            y * y,
            x * x,
            z * z,
            x * y * z,
            y * y * y,
            y * x * x,
            y * z * z,
            y * y * x,
            x * x * x,
            x * z * z,
            y * y * z,
            x * x * z,
            z * z * z,
        ]
    )


def normalize_target(rpc, target):
    """
    Normalize in image space
    """
    norm_cols = (target[:, 0] - rpc.col_offset) / rpc.col_scale
    norm_rows = (target[:, 1] - rpc.row_offset) / rpc.row_scale
    target_norm = np.vstack((norm_cols, norm_rows)).T
    return target_norm


def normalize_input_locs(rpc, input_locs):
    """
    Normalize in world space
    """
    norm_lons = (input_locs[:, 0] - rpc.lon_offset) / rpc.lon_scale
    norm_lats = (input_locs[:, 1] - rpc.lat_offset) / rpc.lat_scale
    norm_alts = (input_locs[:, 2] - rpc.alt_offset) / rpc.alt_scale
    input_locs_norm = np.vstack((norm_lons, norm_lats, norm_alts)).T
    return input_locs_norm


def update_rpc(rpc, x):
    """
    Update rpc coefficients
    """
    rpc.row_num, rpc.row_den = x[:20], x[20:40]
    rpc.col_num, rpc.col_den = x[40:60], x[60:]
    return rpc


def calculate_RMSE_row_col(rpc, input_locs, target):
    """
    Calculate MSE & RMSE in image domain
    """
    col_pred, row_pred = rpc.projection(lon=input_locs[:, 0], lat=input_locs[:, 1], alt=input_locs[:, 2])
    MSE_col, MSE_row = np.mean((np.hstack([col_pred.reshape(-1, 1), row_pred.reshape(-1, 1)]) - target) ** 2, axis=0)
    MSE_row_col = np.mean([MSE_col, MSE_row])  # the number of data is equal in MSE_col and MSE_row
    RMSE_row_col = np.sqrt(MSE_row_col)
    return RMSE_row_col


def weighted_lsq(target, input_locs, h=1e-3, tol=1e-2, max_iter=20):
    """
    Regularized iterative weighted least squares for calibrating a RPC model
    Warning: this code is to be employed with the rpc_model from rpcm

    Args:
        input_locs: Nx3 array containing the lon-lat-alt coordinates of N 3d points
        target: Nx2 array containing the column-row image coordinates associated to the N 3d points
        h: regularization parameter
        tol: tolerance criterion on improvment of RMSE over iterations
        max_iter: maximum number of iterations allowed

    Returns:
        rpc_to_calibrate: RPC model encoding the mapping from 3d to 2d coordinates
    """

    rpc_to_calibrate = initialize_rpc(target, input_locs)

    reg_matrix = (h ** 2) * np.eye(39)  # regularization matrix
    target_norm = normalize_target(rpc_to_calibrate, target)  # col, row
    input_locs_norm = normalize_input_locs(rpc_to_calibrate, input_locs)  # lon, lat, alt

    # define C, R and M
    C, R = target_norm[:, 0][:, np.newaxis], target_norm[:, 1][:, np.newaxis]
    lon, lat, alt = input_locs_norm[:, 0], input_locs_norm[:, 1], input_locs_norm[:, 2]
    MC = np.hstack(
        [
            np.ones((lon.shape[0], 1)),
            poly_vect(x=lat, y=lon, z=alt).T,
            -C * poly_vect(x=lat, y=lon, z=alt).T,
        ]
    )
    MR = np.hstack(
        [
            np.ones((lon.shape[0], 1)),
            poly_vect(x=lat, y=lon, z=alt).T,
            -R * poly_vect(x=lat, y=lon, z=alt).T,
        ]
    )

    # calculate direct solution
    JR = np.linalg.inv(MR.T @ MR) @ (MR.T @ R)
    JC = np.linalg.inv(MC.T @ MC) @ (MC.T @ C)

    # update rpc and get error
    coefs = np.vstack([JR[:20], 1, JR[20:], JC[:20], 1, JC[20:]]).reshape(-1)
    rpc_to_calibrate = update_rpc(rpc_to_calibrate, coefs)
    RMSE_row_col = calculate_RMSE_row_col(rpc_to_calibrate, input_locs, target)

    for n_iter in range(1, max_iter + 1):
        WR2 = np.diagflat(1 / ((MR[:, :20] @ coefs[20:40]) ** 2))  # diagonal matrix with 1 / denom ** 2
        JR_iter = np.linalg.inv((MR.T @ WR2 @ MR) + reg_matrix) @ (MR.T @ WR2 @ R)
        WC2 = np.diagflat(1 / ((MC[:, :20] @ coefs[60:80]) ** 2))  # diagonal matrix with 1 / denom ** 2
        JC_iter = np.linalg.inv((MC.T @ WC2 @ MC) + reg_matrix) @ (MC.T @ WC2 @ C)

        # update rpc and get error
        coefs = np.vstack([JR_iter[:20], 1, JR_iter[20:], JC_iter[:20], 1, JC_iter[20:]]).reshape(-1)
        rpc_to_calibrate = update_rpc(rpc_to_calibrate, coefs)
        RMSE_row_col_prev = RMSE_row_col
        RMSE_row_col = calculate_RMSE_row_col(rpc_to_calibrate, input_locs, target)

        # check convergence
        if np.abs(RMSE_row_col_prev - RMSE_row_col) < tol:
            break

    return rpc_to_calibrate


def scaling_params(vect):
    """
    Returns scale, offset based on vect min and max values
    """
    min_vect = min(vect)
    max_vect = max(vect)
    scale = (max_vect - min_vect) / 2
    offset = min_vect + scale
    return scale, offset


def initialize_rpc(target, input_locs):
    """
    Creates an empty rpc instance
    """
    d = {}
    listkeys = [
        "LINE_OFF",
        "SAMP_OFF",
        "LAT_OFF",
        "LONG_OFF",
        "HEIGHT_OFF",
        "LINE_SCALE",
        "SAMP_SCALE",
        "LAT_SCALE",
        "LONG_SCALE",
        "HEIGHT_SCALE",
        "LINE_NUM_COEFF",
        "LINE_DEN_COEFF",
        "SAMP_NUM_COEFF",
        "SAMP_DEN_COEFF",
    ]
    for key in listkeys:
        d[key] = "0"

    rpc_init = rpcm.RPCModel(d)
    rpc_init.row_scale, rpc_init.row_offset = scaling_params(target[:, 1])
    rpc_init.col_scale, rpc_init.col_offset = scaling_params(target[:, 0])
    rpc_init.lat_scale, rpc_init.lat_offset = scaling_params(input_locs[:, 1])
    rpc_init.lon_scale, rpc_init.lon_offset = scaling_params(input_locs[:, 0])
    rpc_init.alt_scale, rpc_init.alt_offset = scaling_params(input_locs[:, 2])

    return rpc_init


def fit_rpc_from_projection_matrix(P, input_ecef):
    """
    Fit a new RPC model from a set of 2d-3d correspondences
    The projection mapping is given by a 3x4 projection matrix P
    Args:
        P: 3x4 array, the projection matrix that will be copied by the output RPC
        input_ecef: Nx3 array of N 3d points in ECEF coordinates
                    these points are located in the 3d space area where the output RPC model will be fitted
    Returns:
        rpc_calib: output RPC model
        err: a vector of K values with the reprojection error of each of the K points used to fit the RPC
    """

    input_locs = define_grid3d_from_cloud(input_ecef)
    x, y, z = geo_utils.latlon_to_ecef_custom(input_locs[:, 1], input_locs[:, 0], input_locs[:, 2])
    target = cam_utils.apply_projection_matrix(P, np.vstack([x, y, z]).T)

    rpc_calib = weighted_lsq(target, input_locs)
    rmse_err = check_errors(rpc_calib, input_locs, target)
    return rpc_calib, rmse_err


def fit_Rt_corrected_rpc(Rt_vec, original_rpc, offset, pts3d_ba, n_samples=10):
    """
    Fit a new RPC model from a set of 2d-3d correspondences
    The corrected projection mapping is given by: x = P( R(X - T - C) + C
    where x is a point 2d, X is a point 3d, R is a 3d rotation matrix,
    T is a 3d translation vector, C is the camera center
    and P is the projection function of another RPC model

    Args:
        Rt_vec: 1x9 array with the following structure [alpha, T, C]
                alpha = the 3 Euler angles corresponding to the rotation R
                T = the 3 values of the translation T
                C = the 3 values of the camera center in the object space
        original_rpc: the RPC model with projection function P
        offset: image crop boundaries (0,0, width, height) if we are working with the entire image
        pts3d_ba: Nx3 array of N 3d points in ECEF coordinates
                  these points are located in the 3d space area where the output RPC model will be fitted
        n_samples (optional): integer, the number of samples per dimension of the 3d grid
                              that will be used to fit the RPC model

    Returns:
        rpc_calib: output RPC model
        err: a vector of K values with the reprojection error of each of the K points used to fit the RPC
    """

    def undo_adjust_pts3d(pts3d_adj, Rt_vec):
        # inverse of ba_core.adjust_pts3d (only one can camera)
        # pts3d_adj is in ECEF coordinates
        pts3d = pts3d_adj - Rt_vec[:, 6:9]  # subtract rotation center
        roll, pitch, yaw = Rt_vec[:, :3].ravel()
        inverse_R = np.linalg.inv(ba_rotate.euler_angles_to_R(roll, pitch, yaw))
        pts3d = inverse_R @ pts3d.T
        pts3d = pts3d.T
        pts3d += Rt_vec[:, 6:9]  # add rotation center
        pts3d += Rt_vec[:, 3:6]  # invert translation
        return pts3d

    # define the minimum and maximum altitudes we want to work with
    pts3d_init = undo_adjust_pts3d(pts3d_ba, Rt_vec)
    _, _, alts = geo_utils.ecef_to_latlon_custom(pts3d_init[:, 0],  pts3d_init[:, 1],  pts3d_init[:, 2])
    new_offset = np.median(alts)
    min_alt = -1. * original_rpc.alt_scale + new_offset
    max_alt = +1. * original_rpc.alt_scale + new_offset

    # define a grid covering the input image crop and localize it in the 3d space using the original RPC model
    x, y, w, h = offset["col0"], offset["row0"], offset["width"], offset["height"]
    col_range, lin_range, alt_range = [x, x + w, n_samples], [y, y + h, n_samples], [min_alt, max_alt, n_samples]
    cols, lins, alts = cam_utils.generate_point_mesh(col_range, lin_range, alt_range)
    target = np.vstack([cols, lins]).T
    lons, lats = original_rpc.localization(cols, lins, alts)
    x, y, z = geo_utils.latlon_to_ecef_custom(lats, lons, alts)
    input_ecef_before_correction = np.vstack([x, y, z]).T

    # apply the corrective functions in inverted order to use the correct 3d point coordinates
    input_ecef_after_correction = undo_adjust_pts3d(input_ecef_before_correction, Rt_vec)
    x, y, z = input_ecef_after_correction[:, 0], input_ecef_after_correction[:, 1], input_ecef_after_correction[:, 2]
    lats, lons, alts = geo_utils.ecef_to_latlon_custom(x, y, z)
    input_locs = np.vstack([lons, lats, alts]).T

    rpc_calib = weighted_lsq(target, input_locs)
    err = check_errors(rpc_calib, input_locs, target)
    return rpc_calib, err


def check_errors(rpc_calib, input_locs, target, plot=False):
    """
    Computes the reprojection error obtained using the calibrated RPC model
    """
    lat, lon, alt = input_locs[:, 1], input_locs[:, 0], input_locs[:, 2]
    col_pred, row_pred = rpc_calib.projection(lon, lat, alt)
    err = np.linalg.norm(np.hstack([col_pred.reshape(-1, 1), row_pred.reshape(-1, 1)]) - target, axis=1)
    if plot:
        plt.figure()
        plt.hist(err, bins=30)
        plt.show()
    return err


def define_grid3d_from_cloud(input_ecef, n_samples=10, margin=500):
    """
    Takes a point cloud and defines a regular grid of 3d points
    the output grid is located inside the bounding box that contains the input point cloud

    Args:
        input_ecef: Nx3 array with the ECEF coordinates of N 3d points
        n_samples: the number of samples in each dimension of the grid
                   the output grid will have shape n_samples x n_samples x n_samples
        margin: in meters, a certain margin to add to the bounding box limits where the grid is defined

    Returns:
        input_locs: Nx3 array with the lon-lat-alt coordinates of the K 3d points of the regular grid
                    K = n_samples * n_samples * n_samples
    """
    x, y, z = input_ecef[:, 0], input_ecef[:, 1], input_ecef[:, 2]
    x_grid_coords = np.linspace(np.percentile(x, 5) - margin, np.percentile(x, 95) + margin, n_samples)
    y_grid_coords = np.linspace(np.percentile(y, 5) - margin, np.percentile(y, 95) + margin, n_samples)
    z_grid_coords = np.linspace(np.percentile(z, 5) - margin, np.percentile(z, 95) + margin, n_samples)
    x_grid, y_grid, z_grid = np.meshgrid(x_grid_coords, y_grid_coords, z_grid_coords)
    samples = np.vstack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel())).T
    lat, lon, alt = geo_utils.ecef_to_latlon_custom(samples[:, 0], samples[:, 1], samples[:, 2])
    input_locs = np.vstack((lon, lat, alt)).T  # lon, lat, alt
    return input_locs


def sample_direct(x_min, x_max, y_min, y_max, z_mean, grid_size=(15, 15, 15)):
    """
    Sample regularly spaced points over pixels of the image
    """
    x_grid_coords = np.linspace(x_min, x_max, grid_size[0])
    y_grid_coords = np.linspace(y_min, y_max, grid_size[1])
    z_grid_coords = np.linspace(0, 1.5 * z_mean, grid_size[2])
    x_grid, y_grid, z_grid = np.meshgrid(x_grid_coords, y_grid_coords, z_grid_coords)
    samples = np.zeros((x_grid.size, 3), dtype=np.float32)
    samples[:, 0] = x_grid.ravel()
    samples[:, 1] = y_grid.ravel()
    samples[:, 2] = z_grid.ravel()
    return samples


def localize_target_to_input(rpc, samples):
    """
    Applies localization function over samples
    Returns : (lon, lat, alt)
    """
    input_locations = np.zeros_like(samples)
    input_locations[:, 2] = samples[:, 2]  # copy altitude
    for i in range(samples.shape[0]):
        input_locations[i, 0:2] = rpc.localization(*tuple(samples[i]))  # col, row, alt
    return input_locations  # lon, lat, alt
