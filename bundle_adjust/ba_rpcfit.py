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

from bundle_adjust import ba_core, cam_utils, geo_utils


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


def fit_rpc_from_projection_matrix(P, original_rpc, crop_offset, pts3d_ba, n_samples=10):
    """
    Fit a new RPC model from a set of 2d-3d correspondences
    The corrected projection mapping is given by a 3x4 projection matrix P

    Args:
        P: 3x4 array, the projection matrix that will be copied by the output RPC
        original_rpc: the RPC model with projection function P
        crop_offset: image crop boundaries (0,0, width, height) if we are working with the entire image
        pts3d_ba: Nx3 array of N 3d points in ECEF coordinates
                  these points are located in the 3d space area where the output RPC model will be fitted
        n_samples (optional): integer, the number of samples per dimension of the 3d grid
                              that will be used to fit the RPC model

    Returns:
        rpc_calib: output RPC model
        err: a vector of K values with the reprojection error of each of the K points used to fit the RPC
    """

    # define the altitude range where the RPC will be fitted
    _, _, alts = geo_utils.ecef_to_latlon_custom(pts3d_ba[:, 0], pts3d_ba[:, 1], pts3d_ba[:, 2])
    alt_offset = np.median(alts)
    alt_scale = max(8000, original_rpc.alt_scale)
    min_alt = -1.0 * alt_scale + alt_offset
    max_alt = +1.0 * alt_scale + alt_offset
    alt_range = [min_alt, max_alt, n_samples]

    # define a shapely polygon with the image boundaries
    x0, y0, w, h = crop_offset["col0"], crop_offset["row0"], crop_offset["width"], crop_offset["height"]
    image_corners = np.array([[x0, y0], [x0, y0 + h], [x0 + w, y0 + h], [x0 + w, y0]])
    image_boundary = geo_utils.geojson_to_shapely_polygon(geo_utils.geojson_polygon(image_corners))

    image_fully_covered_by_3d_grid = False
    margin = 100  # margin in image space
    while not image_fully_covered_by_3d_grid:

        # define a grid in the image space covering the input image crop + a certain margin
        col_range = [x0 - margin, x0 + w + margin, n_samples]
        row_range = [y0 - margin, y0 + h + margin, n_samples]

        # localize the 2d grid at the altitude range to obtain the 2d-3d correspondences to fit the RPC
        cols, lins, alts = cam_utils.generate_point_mesh(col_range, row_range, alt_range)
        lons, lats = original_rpc.localization(cols, lins, alts)
        x, y, z = geo_utils.latlon_to_ecef_custom(lats, lons, alts)
        target = cam_utils.apply_projection_matrix(P, np.vstack([x, y, z]).T)
        input_locs = np.vstack([lons, lats, alts]).T

        # check if the entire image is covered by the 2d-3d correspondences
        target_convex_hull = geo_utils.geojson_to_shapely_polygon(geo_utils.geojson_polygon_convex_hull(target))
        intersection = image_boundary.intersection(target_convex_hull)
        image_fully_covered_by_3d_grid = (intersection.area / image_boundary.area) == 1
        if not image_fully_covered_by_3d_grid:
            margin += 500
        if margin > 6000:
            image_fully_covered_by_3d_grid = True

    rpc_calib = weighted_lsq(target, input_locs)
    rmse_err = check_errors(rpc_calib, input_locs, target)
    return rpc_calib, rmse_err, margin


def fit_Rt_corrected_rpc(Rt_vec, original_rpc, crop_offset, pts3d_ba, n_samples=10):
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
        crop_offset: image crop boundaries (0,0, width, height) if we are working with the entire image
        pts3d_ba: Nx3 array of N 3d points in ECEF coordinates
                  these points are located in the 3d space area where the output RPC model will be fitted
        n_samples (optional): integer, the number of samples per dimension of the 3d grid
                              that will be used to fit the RPC model

    Returns:
        rpc_calib: output RPC model
        err: a vector of K values with the reprojection error of each of the K points used to fit the RPC
    """

    # define the altitude range where the RPC will be fitted
    _, _, alts = geo_utils.ecef_to_latlon_custom(pts3d_ba[:, 0], pts3d_ba[:, 1], pts3d_ba[:, 2])
    alt_offset = np.median(alts)
    alt_scale = max(8000, original_rpc.alt_scale)
    min_alt = -1.0 * alt_scale + alt_offset
    max_alt = +1.0 * alt_scale + alt_offset
    alt_range = [min_alt, max_alt, n_samples]

    # define a shapely polygon with the image boundaries
    x0, y0, w, h = crop_offset["col0"], crop_offset["row0"], crop_offset["width"], crop_offset["height"]
    image_corners = np.array([[x0, y0], [x0, y0 + h], [x0 + w, y0 + h], [x0 + w, y0]])
    image_boundary = geo_utils.geojson_to_shapely_polygon(geo_utils.geojson_polygon(image_corners))

    image_fully_covered_by_3d_grid = False
    margin = 100  # margin in image space
    while not image_fully_covered_by_3d_grid:

        # define a grid in the image space covering the input image crop + a certain margin
        col_range = [x0 - margin, x0 + w + margin, n_samples]
        row_range = [y0 - margin, y0 + h + margin, n_samples]

        # localize the 2d grid at the altitude range to obtain the 2d-3d correspondences to fit the RPC
        cols, lins, alts = cam_utils.generate_point_mesh(col_range, row_range, alt_range)
        lons, lats = original_rpc.localization(cols, lins, alts)
        x, y, z = geo_utils.latlon_to_ecef_custom(lats, lons, alts)
        pts3d_adj = ba_core.adjust_pts3d(np.vstack([x, y, z]).T, Rt_vec)
        target = cam_utils.apply_rpc_projection(original_rpc, pts3d_adj)
        input_locs = np.vstack([lons, lats, alts]).T

        # check if the entire image is covered by the 2d-3d correspondences
        target_convex_hull = geo_utils.geojson_to_shapely_polygon(geo_utils.geojson_polygon_convex_hull(target))
        intersection = image_boundary.intersection(target_convex_hull)
        image_fully_covered_by_3d_grid = (intersection.area / image_boundary.area) == 1
        if not image_fully_covered_by_3d_grid:
            margin += 500
        if margin > 6000:
            image_fully_covered_by_3d_grid = True

    rpc_calib = weighted_lsq(target, input_locs)
    rmse_err = check_errors(rpc_calib, input_locs, target)
    return rpc_calib, rmse_err, margin


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
