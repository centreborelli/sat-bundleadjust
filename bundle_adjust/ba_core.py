"""
A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery
author: Roger Mari <roger.mari@ens-paris-saclay.fr>
year: 2021

This script implements the most important functions for the resolution of a bundle adjustment optimization
Inspired by https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from bundle_adjust.loader import flush_print


def rotate_rodrigues(pts, axis_angle):
    """
    Rotates a set of 3d points using axis-angle rotation vectors by means of the Rodrigues formula

    Args:
        pts: Nx3 array with N (x,y,z) ECEF coordinates to rotate
        axis_angle: Nx3 array with the axis_angle vectors that will be used to rotate each point

    Returns:
        ptsR: Nx3 array witht the rotated 3d points
    """
    theta = np.linalg.norm(axis_angle, axis=1)[:, np.newaxis]
    v = axis_angle / theta
    dot = np.sum(pts * v, axis=1)[:, np.newaxis]
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    ptsR = cos_theta * pts + sin_theta * np.cross(v, pts) + dot * (1 - cos_theta) * v
    return ptsR


def rotate_euler(pts, euler_angles):
    """
    Rotates a set of 3d points using the Euler angles representation

    Args:
        pts: Nx3 array with N (x,y,z) ECEF coordinates to rotate
        euler_angles: Nx3 array with the euler angles vectors that will be used to rotate each point

    Returns:
        ptsR: Nx3 array with the rotated 3d points in ECEF coordinates
    """
    cosx, sinx = np.cos(euler_angles[:, 0]), np.sin(euler_angles[:, 0])
    cosy, siny = np.cos(euler_angles[:, 1]), np.sin(euler_angles[:, 1])
    cosz, sinz = np.cos(euler_angles[:, 2]), np.sin(euler_angles[:, 2])
    # rotate along x-axis
    ptsR = np.vstack((pts[:, 0], cosx * pts[:, 1] - sinx * pts[:, 2], sinx * pts[:, 1] + cosx * pts[:, 2])).T
    # rotate along y-axis
    ptsR = np.vstack((cosy * ptsR[:, 0] + siny * ptsR[:, 2], ptsR[:, 1], -siny * ptsR[:, 0] + cosy * ptsR[:, 2])).T
    # rotate along z-axis
    ptsR = np.vstack((cosz * ptsR[:, 0] - sinz * ptsR[:, 1], sinz * ptsR[:, 0] + cosz * ptsR[:, 1], ptsR[:, 2])).T
    return ptsR


def project_affine(pts3d, cam_params, pts_ind, cam_ind):
    """
    Projects a set ot 3d points using the parameters of an affine projection matrix

    Args:
        pts3d: Nx3 array with N (x,y,z) ECEF coordinates to project
        cam_params: Mx8 array with the the parameters of the M cameras to be used for the projection of each point
        pts_ind: 1xK vector containing the index of the 3d point causing the k-th 2d observation
        cam_ind: 1xK vector containing the index of the camera where the k-th 2d observation is seen

    Returns:
        pts_proj: nx2 array with the 2d (col, row) coordinates of each projection
    """
    cam_params_ = cam_params[cam_ind]
    # apply extrinsics
    pts_proj = rotate_euler(pts3d[pts_ind], cam_params_[:, :3])
    pts_proj = pts_proj[:, :2]
    pts_proj += cam_params_[:, 3:5]
    # apply intrinsics
    fx, fy, skew = cam_params_[:, 5], cam_params_[:, 6], cam_params_[:, 7]
    pts_proj[:, 0] = fx * pts_proj[:, 0] + skew * pts_proj[:, 1]
    pts_proj[:, 1] = fy * pts_proj[:, 1]
    return pts_proj


def project_perspective(pts3d, cam_params, pts_ind, cam_ind):
    """
    Projects a set ot 3d points using the parameters of a perspective projection matrix

    Args:
        pts3d: Nx3 array with N (x,y,z) ECEF coordinates to project
        cam_params: Mx11 array with the the parameters of the M cameras to be used for the projection of each point
        pts_ind: 1xK vector containing the index of the 3d point causing the k-th 2d observation
        cam_ind: 1xK vector containing the index of the camera where the k-th 2d observation is seen

    Returns:
        pts_proj: nx2 array with the 2d (col, row) coordinates of each projection
    """
    cam_params_ = cam_params[cam_ind]
    # apply extrinsics
    pts_proj = rotate_euler(pts3d[pts_ind], cam_params_[:, :3])
    pts_proj += cam_params_[:, 3:6]
    # apply intrinsics
    fx, fy, skew = cam_params_[:, 6], cam_params_[:, 7], cam_params_[:, 8]
    cx, cy = cam_params_[:, 9], cam_params_[:, 10]
    pts_proj[:, 0] = fx * pts_proj[:, 0] + skew * pts_proj[:, 1] + cx * pts_proj[:, 2]
    pts_proj[:, 1] = fy * pts_proj[:, 1] + cy * pts_proj[:, 2]
    pts_proj = pts_proj[:, :2] / pts_proj[:, 2, np.newaxis]
    return pts_proj


def adjust_pts3d(pts3d, Rt_vec):
    """
    Corrects the object coordinates of a set of tie points
    The correction mapping is given by: X' = R(X - T - C) + C)
    Used by project_rpc

    Args:
        pts3d: Nx3 array with N (x,y,z) ECEF coordinates
        Rt_vec: 2d array with 9 columns with the following structure [alpha, T, C]
                alpha = the 3 Euler angles corresponding to the rotation R
                T = the 3 values of the translation T
                C = the 3 values of the camera center in the object space

    Returns:
        pts3d_adj: Nx3 array with N (x,y,z) ECEF coordinates after the correction mapping
    """
    pts3d_adj = pts3d - Rt_vec[:, 3:6]  # apply translation
    pts3d_adj -= Rt_vec[:, 6:9]  # subtract rotation center
    pts3d_adj = rotate_euler(pts3d_adj, Rt_vec[:, :3])  # rotate
    pts3d_adj += Rt_vec[:, 6:9]  # add rotation center
    return pts3d_adj


def project_rpc(pts3d, rpcs, cam_params, pts_ind, cam_ind):
    """
    Projects a set ot 3d points using an original rpc and a prior corrective rotation

    Args:
        pts3d: Nx3 array with N (x,y,z) ECEF coordinates to project
        rpcs: list of M rpcm rpc models to be used for the projection of each point
        cam_params: Mx3 array with the euler angles of the M corrective matrices associated to each rpc
        pts_ind: 1xK vector containing the index of the 3d point causing the k-th 2d observation
        cam_ind: 1xK vector containing the index of the camera where the k-th 2d observation is seen

    Returns:
        pts_proj: nx2 array with the 2d (col, row) coordinates of each projection
    """
    from bundle_adjust.cam_utils import apply_rpc_projection

    pts3d_adj = adjust_pts3d(pts3d[pts_ind], cam_params[cam_ind])
    pts_proj = np.zeros((pts_ind.shape[0], 2), dtype=np.float32)
    for c_idx in np.unique(cam_ind).tolist():
        where_c_idx = cam_ind == c_idx
        pts_proj[where_c_idx] = apply_rpc_projection(rpcs[c_idx], pts3d_adj[where_c_idx])
    return pts_proj


def fun(v, p):
    """
    Compute bundle adjustment residuals

    Args:
        v: initial guess on the variables to optimize
        p: bundle adjustment parameters object with everything that is needed

    Returns:
        residuals: 1x2K vector containing the residuals (x'-x, y'-y) of each reprojected observation
                   where K is the total number of feature track observations
    """

    # project 3d points using the current camera parameters
    pts3d, cam_params = p.get_vars_ready_for_fun(v)
    if p.cam_model == "perspective":
        pts_proj = project_perspective(pts3d, cam_params, p.pts_ind, p.cam_ind)
    elif p.cam_model == "affine":
        pts_proj = project_affine(pts3d, cam_params, p.pts_ind, p.cam_ind)
    else:
        pts_proj = project_rpc(pts3d, p.cameras, cam_params, p.pts_ind, p.cam_ind)

    # compute reprojection residuals
    obs_weights = np.repeat(p.pts2d_w, 2, axis=0)
    residuals = obs_weights * (pts_proj - p.pts2d).ravel()

    return residuals


def build_jacobian_sparsity(p):
    """
    Builds the sparse matrix employed to compute the Jacobian of the bundle adjustment problem

    Args:
        p: bundle adjustment parameters object with everything that is needed

    Returns:
        A: output sparse matrix
    """
    from scipy.sparse import lil_matrix

    # compute shape of sparse matrix
    n_params = p.n_params
    m = p.pts_ind.size * 2
    n_params_K = 3 if p.cam_model == "affine" else 5
    common_K = "K" in p.cam_params_to_optimize and "COMMON_K" in p.cam_params_to_optimize
    if common_K:
        n_params -= n_params_K
    n = common_K * n_params_K + p.n_cam * n_params + p.n_pts * 3
    A = lil_matrix((m, n), dtype=int)

    # fill matrix
    i = np.arange(p.pts_ind.size)
    for s in range(n_params):
        A[2 * i, common_K * n_params_K + p.cam_ind * n_params + s] = 1
        A[2 * i + 1, common_K * n_params_K + p.cam_ind * n_params + s] = 1
    for s in range(3):
        A[2 * i, common_K * n_params_K + p.n_cam * n_params + p.pts_ind * 3 + s] = 1
        A[2 * i + 1, common_K * n_params_K + p.n_cam * n_params + p.pts_ind * 3 + s] = 1
    if common_K:
        A[:, :n_params_K] = np.ones((m, n_params_K))

    return A


def init_optimization_config(config=None):
    """
    Initializes the configuration of the bundle adjustment optimization algorithm

    Args:
        config: dict possibly containing values that we want to be different from default
                the default configuration is used for all parameters not specified in config

    Returns:
        output_config: dict where keys identify the parameters and values their assigned value
    """
    keys = ["loss", "ftol", "xtol", "f_scale", "max_iter", "verbose"]
    default_values = ["linear", 1e-4, 1e-10, 1.0, 300, 1]
    output_config = {}
    if config is not None:
        for v, k in zip(default_values, keys):
            output_config[k] = config[k] if k in config.keys() else v
    else:
        output_config = dict(zip(keys, default_values))
    return output_config


def run_ba_optimization(p, ls_params=None, verbose=False, plots=True):
    """
    Solves the bundle adjustment optimization problem

    Args:
        p: bundle adjustment parameters object with everything that is needed
        ls_params (optional): dictionary specifying a particular configuration for the optimization algorithm
        verbose (optional): boolean, set to True to check all sorts of informationa about the process
        plots (optional): boolean, set to True to see histograms of reprojection error

    Returns:
        vars_init: the vector with the initial variables input to the solver
        vars_ba: the vector with the final variables optimized by the solver
        err_init: vector with the initial reprojection error of each 2d feature track observation
        err_ba: vector with the final reprojection error of each 2d feature track observation
        err_init_per_cam: the average initial reprojection error associated to each camera
        err_ba_per_cam: the average final reprojection error associated to each camera
        iterations: number of iterations that were needed to converge
    """

    import time

    from scipy.optimize import least_squares

    ls_params = init_optimization_config(ls_params)
    if verbose:
        print("\nRunning bundle adjustment...")
        from bundle_adjust.loader import display_dict

        display_dict(ls_params)

    # compute cost at initial variable values and define jacobian sparsity matrix
    vars_init = p.params_opt.copy()
    residuals_init = fun(vars_init, p)
    A = build_jacobian_sparsity(p)
    if verbose:
        flush_print("Shape of Jacobian sparsity: {}x{}".format(*A.shape))

    # run bundle adjustment
    t0 = time.time()
    res = least_squares(
        fun,
        vars_init,
        jac_sparsity=A,
        verbose=ls_params["verbose"],
        x_scale="jac",
        method="trf",
        ftol=ls_params["ftol"],
        xtol=ls_params["xtol"],
        loss=ls_params["loss"],
        f_scale=ls_params["f_scale"],
        max_nfev=ls_params["max_iter"],
        args=(p,),
    )
    if verbose:
        flush_print("Optimization took {:.2f} seconds\n".format(time.time() - t0))

    # check error and plot residuals before and after the optimization
    iterations = res.nfev
    residuals_ba, vars_ba = res.fun, res.x
    err_init = compute_reprojection_error(residuals_init, p.pts2d_w)
    err_ba = compute_reprojection_error(residuals_ba, p.pts2d_w)
    err_init_per_cam, err_ba_per_cam = [], []
    if verbose:
        to_print = [np.mean(err_init), np.median(err_init)]
        flush_print("Reprojection error before BA (mean / median): {:.2f} / {:.2f}".format(*to_print))
        to_print = [np.mean(err_ba), np.median(err_ba)]
        flush_print("Reprojection error after  BA (mean / median): {:.2f} / {:.2f}\n".format(*to_print))

        for cam_idx in range(int(p.C.shape[0] / 2)):
            err_init_per_cam.append(np.mean(err_init[p.cam_ind == cam_idx]))
            err_ba_per_cam.append(np.mean(err_ba[p.cam_ind == cam_idx]))
            n_obs = np.sum(1 * ~np.isnan(p.C[2 * cam_idx, :]))
            to_print = [cam_idx, n_obs, err_init_per_cam[-1], err_ba_per_cam[-1]]
            flush_print("    - cam {:3} - {:5} obs - (mean before / mean after): {:.2f} / {:.2f}".format(*to_print))
        print("\n")

    if plots:
        _, f = plt.subplots(1, 3, figsize=(15, 3))
        f[0].plot(residuals_init)
        f[0].plot(residuals_ba)
        f[0].title.set_text("Residuals before and after BA")
        f[1].hist(err_init, bins=40)
        f[1].title.set_text("Reprojection error before BA")
        f[2].hist(err_ba, bins=40, range=(err_init.min(), err_init.max()))
        f[2].title.set_text("Reprojection error after BA")
        plt.show()

    return vars_init, vars_ba, err_init, err_ba, iterations


def compute_reprojection_error(residuals, pts2d_w=None):
    """
    Computes the reprojection error from the bundle adjustment residuals

    Args:
        residuals: 1x2N vector containing the residual for each coordinate of the N 2d feature track observations
        pts2d_w (optional): 1xN vector with the weight given to each feature track observation

    Returns:
        err: 1xN vector with the reprojection error of each observation, computed as the L2 norm of the residuals
    """
    n_pts = int(residuals.size / 2)
    obs_weights = np.ones(residuals.size, dtype=np.float32) if pts2d_w is None else np.repeat(pts2d_w, 2, axis=0)
    err = np.linalg.norm(abs(residuals / obs_weights).reshape(n_pts, 2), axis=1)
    return err


def compute_mean_reprojection_error_per_track(err, pts_ind, cam_ind):
    """
    Computes efficiently the average reprojection error of each track used for bundle adjustment

    Args:
        err: 1xN vector with the reprojection error of each 2d observation
        pts_ind: 1xN vector with the track index of each 2d observation
        cam_ind: 1xN vector with the camera where each 2d observation is seen

    Returns:
        track_err: 1xK vector with the average reprojection error of each track, K is the number of tracks
    """
    n_cam, n_pts = cam_ind.max() + 1, pts_ind.max() + 1
    C_reproj = np.zeros((n_cam, n_pts))
    C_reproj[:] = np.nan
    for i, e in enumerate(err):
        C_reproj[cam_ind[i], pts_ind[i]] = e
    track_err = np.nanmean(C_reproj, axis=0).astype(np.float32)
    return track_err


#--- functions that generate output illustrations ---


def save_histogram_of_errors(img_path, err_init, err_ba, plot=False):
    """
    Writes a png image with the histogram of errors before and after run_ba_optimization

    Args:
        img_path: string, filename of the png image that will be written on the disk
        err_init: vector with the reprojection error of each 2d observation before bundle adjustment
        err_ba: vector with the reprojection error of each 2d observation after bundle adjustment
        plot (optional): plot a matplotlib figure instead of saving output image
    """
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 2, 1)
    plt.hist(err_init, bins=40)
    plt.title("Before BA")
    plt.ylabel("Number of tie point observations")
    plt.xlabel("Reprojection error (pixel units)")

    plt.subplot(1, 2, 2)
    plt.hist(err_ba, bins=40, range=(err_init.min(), err_init.max()))
    plt.title("After BA")
    plt.ylabel("Number of tie point observations")
    plt.xlabel("Reprojection error (pixel units)")
    if plot:
        plt.show()
    else:
        plt.savefig(img_path, bbox_inches="tight")


def save_heatmap_of_reprojection_error(img_path, p, err, input_ims_footprints_lonlat,
                                       aoi_lonlat_roi=None, plot=False, smooth=20, global_transform=None):
    """
    Writes a georeferenced raster with the reprojection errors of a set of tie points interpolated across an AOI

    Args:
        img_path: string, filename of the tif image that will be written on the disk
        p: bundle adjustment parameters object
        err: vector with the reprojection error of each 2d tie point observation
        aoi_lonlat_ims: geojson polygon in lon lat coordinates with the silhouette of the input images
        aoi_lonlat_roi: geojson polygon in lon lat coordinates with the silhouette of the area of interest
        plot (optional): plot a matplotlib figure instead of saving output image
        smooth (optional): sigma for gaussian filtering, set to 0 to visualize raw interpolation
    """
    from scipy.ndimage import gaussian_filter
    from bundle_adjust import geo_utils
    from bundle_adjust import loader
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    tif = True if os.path.splitext(img_path)[1] == ".tif" else False
    aoi_lonlat_union_ims = geo_utils.combine_lonlat_geojson_borders(input_ims_footprints_lonlat)

    # extract the utm bbox that contains the area of interest and set a reasonable resolution
    max_size = 1000  # maximum size allowed per raster dimension (in pixels)
    utm_bbx = geo_utils.utm_bbox_from_aoi_lonlat(aoi_lonlat_union_ims)
    height, width = geo_utils.utm_bbox_shape(utm_bbx, 1.0)
    resolution = float(max(height, width)) / max_size  # raster resolution (in meters)

    # compute the average reprojection error per tie point before and after BA
    track_err = compute_mean_reprojection_error_per_track(err, p.pts_ind, p.cam_ind)

    # convert the tie points object coordinates to the UTM system
    pts3d_ecef = p.pts3d_ba.copy()
    if global_transform is not None:
        pts3d_ecef -= global_transform
    lats, lons, alts = geo_utils.ecef_to_latlon_custom(pts3d_ecef[:, 0], pts3d_ecef[:, 1], pts3d_ecef[:, 2])
    pts2d_utm = np.vstack([geo_utils.utm_from_lonlat(lons, lats)]).T

    # discretize the utm bbox and get the relative position of the tie points on it
    pts2d_ = geo_utils.compute_relative_utm_coords_inside_utm_bbx(pts2d_utm, utm_bbx, resolution)

    # keep only those points and their error inside the utm bbx limits
    cols, rows = pts2d_.T
    height, width = geo_utils.utm_bbox_shape(utm_bbx, resolution)
    valid_pts = np.logical_and(cols < width, cols >= 0) & np.logical_and(rows < height, rows >= 0)
    pts2d, track_err = pts2d_[valid_pts], track_err[valid_pts]

    # interpolate the reprojection error across the utm bbx
    all_cols, all_rows = np.meshgrid(np.arange(width), np.arange(height))
    pts2d_i = np.vstack([all_cols.ravel(), all_rows.ravel()]).T
    track_err_interp = idw_interpolation(pts2d, track_err, pts2d_i).reshape((height, width))
    track_err_interp = track_err_interp.reshape((height, width))

    # smooth the interpolation result to improve visualization
    track_err_interp = gaussian_filter(track_err_interp, sigma=smooth)

    # apply mask of image footprints
    mask = np.ones((height, width)).astype(bool)
    for i, aoi_lonlat in enumerate(input_ims_footprints_lonlat):
        aoi_utm = geo_utils.utm_geojson_from_lonlat_geojson(aoi_lonlat)
        pts2d_utm = np.array(aoi_utm["coordinates"][0])
        aoi_pts2d = geo_utils.compute_relative_utm_coords_inside_utm_bbx(pts2d_utm, utm_bbx, resolution)
        tmp = geo_utils.geojson_to_shapely_polygon(geo_utils.geojson_polygon(aoi_pts2d))
        mask &= ~loader.mask_from_shapely_polygons([tmp], (height, width)).astype(bool)
    track_err_interp[mask.astype(bool)] = np.nan

    # compute borders of the previous mask
    utm_geojson_list = [geo_utils.utm_geojson_from_lonlat_geojson(x) for x in input_ims_footprints_lonlat]
    from shapely.ops import cascaded_union
    geoms = [geo_utils.geojson_to_shapely_polygon(g) for g in utm_geojson_list]
    union_shapely = cascaded_union([geom if geom.is_valid else geom.buffer(0) for geom in geoms])
    borders = []
    polys = union_shapely if union_shapely.geom_type == "MultiPolygon" else [union_shapely]
    for x in polys:
        tmp = np.array(geo_utils.geojson_from_shapely_polygon(x)["coordinates"][0])
        tmp = geo_utils.compute_relative_utm_coords_inside_utm_bbx(tmp, utm_bbx, resolution)
        tmp = geo_utils.geojson_to_shapely_polygon(geo_utils.geojson_polygon(tmp))
        borders.append(tmp)

    # prepare plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.invert_yaxis()
    ax.axis("equal")
    ax.axis("off")
    vmin, vmax = 0.0, 2.0
    im = plt.imshow(track_err_interp, vmin=vmin, vmax=vmax)
    for x in borders:
        plt.plot(*x.exterior.xy, color="black")
    plt.scatter(pts2d[:, 0], pts2d[:, 1], 30, track_err, edgecolors="k", vmin=vmin, vmax=vmax)
    # draw aoi if available
    if aoi_lonlat_roi is not None:
        roi_utm = geo_utils.utm_geojson_from_lonlat_geojson(aoi_lonlat_roi)
        tmp = np.array(roi_utm["coordinates"][0])
        tmp = geo_utils.compute_relative_utm_coords_inside_utm_bbx(tmp, utm_bbx, resolution)
        tmp = geo_utils.geojson_to_shapely_polygon(geo_utils.geojson_polygon(tmp))
        plt.plot(*tmp.exterior.xy, color="red", linewidth=3.0)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im, cax=cax)
    n_ticks = 9
    ticks = np.linspace(vmin, vmax, n_ticks)
    cbar.set_ticks(ticks)
    tick_labels = ["{:.2f}".format(vmin + t * (vmax - vmin)) for t in np.linspace(0, 1, n_ticks)]
    tick_labels[-1] = ">=" + tick_labels[-1]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label("Reprojection error across AOI (pixel units)", rotation=270, labelpad=25)
    if plot:
        plt.show()
    else:
        if tif:
            # save georeferenced tif
            utm_zs = geo_utils.zonestring_from_lonlat(*aoi_lonlat_union_ims["center"])
            epsg = geo_utils.epsg_code_from_utm_zone(utm_zs)
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            loader.write_georeferenced_raster_utm_bbox(img_path, track_err_interp, utm_bbx, epsg, resolution)
        else:
            # save png
            plt.savefig(img_path, bbox_inches="tight")


def idw_interpolation(pts2d, z, pts2d_query, N=8):
    """
    Interpolates each query point pts2d_query from the N nearest known data points in pts2d
    each neighbor contribution follows inverse distance weighting IDW (closest points are given larger weights)
    inspired by https://stackoverflow.com/questions/3104781/inverse-distance-weighted-idw-interpolation-with-python

    Example: given a query point q and N=3, finds the 3 data points nearest q at distances d1 d2 d3
             and returns the IDW average of the known values z1 z2 z3 at distances d1 d3 d3
             z(q) = (z1/d1 + z2/d2 + z3/d3) / (1/d1 + 1/d2 + 1/d3)

    Args:
        pts2d: Kx2 array, contains K 2d points whose value z is known
        z: Kx1 array, the known value of each point in pts2d
        pts2d_query: Qx2 array, contains Q 2d points that we want to interpolate
        N (optional): integer, nearest neighbours that will be employed to interpolate

    Returns:
        z_query: Qx1 array, contans the interpolated value of each input query point
    """
    from scipy.spatial import cKDTree as KDTree

    # build a KDTree using scipy, to find nearest neighbours quickly
    tree = KDTree(pts2d)

    # find the N nearest neighbours of each query point
    nn_distances, nn_indices = tree.query(pts2d_query, k=N)

    if N == 1:
        # particular case 1:
        # only one nearest neighbour to use, which is given all the weight
        z_query = z[nn_indices]
    else:
        # general case
        # interpolate by weighting the N nearest known points by 1/dist
        w = 1.0 / nn_distances
        w /= np.tile(np.sum(w, axis=1), (N, 1)).T
        z_query = np.sum(w * z[nn_indices], axis=1)

        # particular case 2:
        # the query point falls on a known point, which is given all the weight
        known_query_indices = np.where(nn_distances[:, 0] < 1e-10)[0]
        z_query[known_query_indices] = z[nn_indices[known_query_indices, 0]]
    return z_query
