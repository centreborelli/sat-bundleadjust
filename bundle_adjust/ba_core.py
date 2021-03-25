"""
Bundle Adjustment for 3D Reconstruction from Multi-Date Satellite Images
This script implements the most important functions for the resolution of a bundle adjustment optimization
It is highly based on the tutorial in https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
by Roger Mari <roger.mari@ens-paris-saclay.fr>
"""

import matplotlib.pyplot as plt
import numpy as np

from bundle_adjust.loader import flush_print


def rotate_euler(pts, euler_angles):
    """
    Rotates 3d points using the Euler angles representation
    Args:
        pts: Nx3 array with N (x,y,z) ECEF coordinates to rotate
        euler_angles: Nx3 array with the euler angles vectors that will be used to rotate each point
    Returns:
        ptsR: Nx3 array with the rotated 3d points
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
    pts_proj = rotate_euler(pts3d[pts_ind], cam_params_[:, :3])
    pts_proj = pts_proj[:, :2]
    fx, fy, skew = cam_params_[:, 5], cam_params_[:, 6], cam_params_[:, 7]
    pts_proj[:, 0] = fx * pts_proj[:, 0] + skew * pts_proj[:, 1]
    pts_proj[:, 1] = fy * pts_proj[:, 1]
    pts_proj += cam_params_[:, 3:5]
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
    pts_proj = rotate_euler(pts3d[pts_ind], cam_params_[:, :3])
    pts_proj += cam_params_[:, 3:6]
    fx, fy, skew = cam_params_[:, 6], cam_params_[:, 7], cam_params_[:, 8]
    cx, cy = cam_params_[:, 9], cam_params_[:, 10]
    pts_proj[:, 0] = fx * pts_proj[:, 0] + skew * pts_proj[:, 1] + cx * pts_proj[:, 2]
    pts_proj[:, 1] = fy * pts_proj[:, 1] + cy * pts_proj[:, 2]
    pts_proj = pts_proj[:, :2] / pts_proj[:, 2, np.newaxis]
    return pts_proj


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
    from bundle_adjust.camera_utils import apply_rpc_projection

    cam_params_ = cam_params[cam_ind]
    pts_3d_adj = pts3d[pts_ind] - cam_params_[:, 3:6]  # apply translation
    pts_3d_adj -= cam_params_[:, 6:9]  # subtract rotation center
    pts_3d_adj = rotate_euler(pts_3d_adj, cam_params_[:, :3])  # rotate
    pts_3d_adj += cam_params_[:, 6:9]  # add rotation center
    pts_proj = np.zeros((pts_ind.shape[0], 2), dtype=np.float32)
    for c_idx in np.unique(cam_ind).tolist():
        where_c_idx = cam_ind == c_idx
        pts_proj[where_c_idx] = apply_rpc_projection(rpcs[c_idx], pts_3d_adj[where_c_idx])
    return pts_proj


def fun(v, p):
    """
    Compute bundle adjustment residuals
    Args:
        params_opt: initial guess on the variables to optimize
        p: BA_Parameters object with everything that is needed
    Returns:
        residuals: 1x2K vector containing the residuals (x'-x, y'-y) of each reprojected observation
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
        p: BA_Parameters object with everything that is needed
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
        p: BA_Parameters object with everything that is needed
        ls_params (optional): dictionary specifying a particular configuration for the least squares optimization
        verbose (optional): boolean specifying whether if plots and other information is to be displayed
    Returns:
        vars_init: the vector with the initial variables input to the solver
        vars_ba: the vector with the final variables optimized by the solver
        err_init: vector with the initial reprojection error of each 2d feature track observation
        err_ba: vector with the final reprojection error of each 2d feature track observation
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

    return vars_init, vars_ba, [err_init, err_ba, err_init_per_cam, err_ba_per_cam], iterations


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
