"""
Bundle Adjustment for 3D Reconstruction from Multi-Date Satellite Images
This script implements the most important functions for the resolution of a bundle adjustment optimization
It is highly based on the tutorial in https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
by Roger Mari <roger.mari@ens-paris-saclay.fr>
"""

import numpy as np
import matplotlib.pyplot as plt


def rotate_rodrigues(pts, axis_angle):
    """
    Rotates 3d points using axis-angle rotation vectors by means of the Rodrigues formula
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
    pts_3d_adj = rotate_euler(pts3d[pts_ind], cam_params_[:, :3])
    pts_proj = np.zeros((pts_ind.shape[0], 2), dtype=np.float32)
    for c_idx in np.unique(cam_ind).tolist():
        where_c_idx = cam_ind == c_idx
        pts_proj[where_c_idx] = apply_rpc_projection(rpcs[c_idx], pts_3d_adj[where_c_idx])
    return pts_proj


def fun(vars, p):
    """
    Compute bundle adjustment residuals
    Args:
        params_opt: initial guess on the variables to optimize
        p: BA_Parameters object with everything that is needed
    Returns:
        residuals: 1x2K vector containing the residuals (x'-x, y'-y) of each reprojected observation
    """

    # project 3d points using the current camera parameters
    pts3d, cam_params = p.get_vars_ready_for_fun(vars)
    if p.cam_model == 'perspective':
        pts_proj = project_perspective(pts3d, cam_params, p.pts_ind, p.cam_ind)
    elif p.cam_model == 'affine':
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
    n_params_K = 3 if p.cam_model == 'affine' else 5
    common_K = 'K' in p.cam_params_to_optimize and 'COMMON_K' in p.cam_params_to_optimize
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

    from scipy.optimize import least_squares
    import time
    ls_params = {'loss': 'linear', 'ftol': 1e-8, 'xtol': 1e-8, 'f_scale': 1.0} if ls_params is None else ls_params
    if verbose:
        print('\nRunning bundle adjustment...')
        print('     - loss:    {}'.format(ls_params['loss']))
        print('     - ftol:    {}'.format(ls_params['ftol']))
        print('     - xtol:    {}'.format(ls_params['xtol']))
        print('     - f_scale: {}\n'.format(ls_params['f_scale']), flush=True)

    # compute cost at initial variable values and define jacobian sparsity matrix
    vars_init = p.params_opt.copy()
    residuals_init = fun(vars_init, p)
    A = build_jacobian_sparsity(p)
    if verbose:
        print('Shape of Jacobian sparsity: {}x{}'.format(*A.shape), flush=True)

    # run bundle adjustment
    t0 = time.time()
    res = least_squares(fun, vars_init, jac_sparsity=A,
                        verbose=1, x_scale='jac', method='trf',
                        ftol=ls_params['ftol'], xtol=ls_params['xtol'],
                        loss=ls_params['loss'], f_scale=ls_params['f_scale'], args=(p,))
    if verbose:
        print("Optimization took {:.2f} seconds\n".format(time.time() - t0), flush=True)

    # check error and plot residuals before and after the optimization
    residuals_ba, vars_ba = res.fun, res.x
    err_init = compute_reprojection_error(residuals_init, p.pts2d_w)
    err_ba = compute_reprojection_error(residuals_ba, p.pts2d_w)
    if verbose:
        args = [np.mean(err_init), np.median(err_init)]
        print('Reprojection error before BA (mean / median): {:.2f} / {:.2f}'.format(*args))
        args = [np.mean(err_ba), np.median(err_ba)]
        print('Reprojection error after  BA (mean / median): {:.2f} / {:.2f}\n'.format(*args), flush=True)
    if plots:
        _, f = plt.subplots(1, 3, figsize=(15, 3))
        f[0].plot(residuals_init)
        f[0].plot(residuals_ba)
        f[0].title.set_text('Residuals before and after BA')
        f[1].hist(err_init, bins=40)
        f[1].title.set_text('Reprojection error before BA')
        f[2].hist(err_ba, bins=40, range=(err_init.min(), err_init.max()))
        f[2].title.set_text('Reprojection error after BA')
        plt.show()

    return vars_init, vars_ba, err_init, err_ba


def compute_reprojection_error(residuals, pts2d_w=None):
    """
    Computes the reprojection error from the bundle adjustment residuals
    Args:
        residuals: 1x2N vector containing the residual for each coordinate of the N 2d feature track observations
        pts2d_w (optional): 1xN vector with the weight given to each feature track observation
    Returns:
        err: 1xN vector with the reprojection error of each observation, computed as the L2 norm of the residuals
    """
    n_pts = int(residuals.size/2)
    obs_weights = np.ones(residuals.size, dtype=np.float32) if pts2d_w is None else np.repeat(pts2d_w, 2, axis=0)
    err = np.linalg.norm(abs(residuals/obs_weights).reshape(n_pts, 2), axis=1)
    return err
