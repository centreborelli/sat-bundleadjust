"""
Bundle Adjustment for 3D Reconstruction from Multi-Date Satellite Images
This script implements all functions necessary to define all variables involved in the bundle adjustment
by Roger Mari <roger.mari@ens-paris-saclay.fr>
"""

import numpy as np
from bundle_adjust import ba_triangulate
from bundle_adjust import ba_rotate
from bundle_adjust import camera_utils
from bundle_adjust import rpc_fit


class Error(Exception):
    pass


def check_valid_cam_model(cam_model):
    """
    Args:
        cam_model: string stating the camera model
    Returns:
        raises an error if cam_model is not valid
    """
    if cam_model not in ['rpc', 'affine', 'perspective']:
        raise Error('cam_model is not valid')


def load_cam_params_from_camera(camera, camera_center, cam_model):
    """
    Args:
        camera: either an 3x4 perspective or affine projection matrix, or an rpc
        cam_model: string stating the camera model
    Returns:
        cam_params: vector containing the camera parameters associated to the input camera, in the form used to BA
    """
    check_valid_cam_model(cam_model)
    if cam_model == 'affine':
        K, R, vecT = camera_utils.decompose_affine_camera(camera)
        vecR = np.array(ba_rotate.euler_angles_from_R(R))
        fx, fy, skew = K[0,0], K[1,1], K[0,1]
        cam_params = np.hstack((vecR.ravel(),vecT.ravel(),fx,fy,skew))
    elif cam_model == 'perspective':
        K, R, vecT, _ = camera_utils.decompose_perspective_camera(camera)
        K = K/K[2,2]
        vecR = np.array(ba_rotate.euler_angles_from_R(R))
        fx, fy, skew, cx, cy = K[0,0], K[1,1], K[0,1], K[0,2], K[1,2]
        cam_params = np.hstack((vecR.ravel(),vecT.ravel(),fx,fy,skew,cx,cy))
    else:
        cam_params = np.zeros(6, dtype=np.float32)
        cam_params = np.hstack([cam_params, camera_center])
    return cam_params


def load_camera_from_cam_params(cam_params, cam_model):
    """
    Args:
        cam_params: vector containing the camera parameters associated to the input camera, in the form used to BA
        cam_model: string stating the camera model
    Returns:
        camera: either an 3x4 perspective or affine projection matrix, or the 3X3 corrective rotation for an rpc
    """
    check_valid_cam_model(cam_model)
    if cam_model == 'affine':
        vecR, vecT, fx, fy, skew = cam_params[0:3], cam_params[3:5], cam_params[5], cam_params[6], cam_params[7]
        K = np.array([[fx, skew], [0., fy]])
        R = ba_rotate.euler_angles_to_R(*vecR.tolist())
        P = np.vstack((np.hstack((K @ R[:2, :], np.array([vecT]).T)), np.array([[0, 0, 0, 1]])))
        camera = P / P[2, 3]
    elif cam_model == 'perspective':
        vecR, vecT = cam_params[0:3], cam_params[3:6]
        fx, fy, skew, cx, cy = cam_params[6], cam_params[7], cam_params[8], cam_params[9], cam_params[10]
        K = np.array([[fx, skew, cx], [0., fy, cy], [0., 0., 1.]])
        R = ba_rotate.euler_angles_to_R(*vecR.tolist())
        P = K @ np.hstack((R, vecT.reshape((3, 1))))
        camera = P / P[2, 3]
    else:
        camera = cam_params.reshape((1,9))
    return camera


class BundleAdjustmentParameters:
    def __init__(self, C, pts3d, cameras, cam_model, pairs_to_triangulate, camera_centers,
                 n_cam_fix=0, n_pts_fix=0, reduce=True, verbose=False, cam_params_to_optimize=['R']):
        """
        Args:
            C: ndarray representing a correspondence matrix containing a set of feature tracks
            pts3d: Nx3 array with the initial ECEF xyz coordinates of the 3d points observed in C
            cameras: either a list of M initial 3x4 projection matrices or a list of M initial rpc models
            pairs_to_triangulate: list of pairs suitable for triangulation
            camera_centers: list of 1x3 arrays with the projective camera centers
            n_cam_fix (optional): number of cameras to freeze (will not be optimized)
            n_pts_fix (optional): number of 3d points to freeze (will not be optimized)
            reduce (optional): if True, only the points and cameras to update will be used
            cam_params_to_optimize (optional): a list with the parameters to optimize from each camera model
                                               accepted values: 'R' (rotation), 'T' (translation), 'K' (calibration)
                                               or 'COMMON_K' to fix a common K in all cams when 'K' is in the list
        """

        # other ba parameters
        if verbose:
            print('\nDefining bundle adjustment parameters...')
            print('     - cam_params_to_optimize: {}\n'.format(cam_params_to_optimize))

        # (1) set dummy variables
        check_valid_cam_model(cam_model)
        for v in cam_params_to_optimize:
            if v not in ['R', 'T', 'K', 'COMMON_K']:
                raise Error('{} is not a valid camera parameter to optimize'.format(v))
        self.cam_params_to_optimize = cam_params_to_optimize
        self.cam_model = cam_model
        self.C = C.copy()
        self.pts3d = pts3d.copy()
        self.cameras = cameras.copy()
        self.camera_centers = camera_centers.copy()
        self.pairs_to_triangulate = pairs_to_triangulate.copy()
        self.n_cam = int(C.shape[0] / 2)
        self.n_pts = int(C.shape[1])
        self.n_cam_fix = n_cam_fix
        self.n_cam_opt = self.n_cam - self.n_cam_fix
        self.cam_prev_indices = np.arange(self.n_cam)
        self.n_pts_fix = n_pts_fix
        self.n_pts_opt = self.n_pts - self.n_pts_fix
        self.pts_prev_indices = np.arange(self.n_pts)
        if reduce:
            self.reduce(C, pts3d, cameras, pairs_to_triangulate, camera_centers)
            if verbose:
                print('C.shape before reduce', C.shape)
                print('C.shape after reduce', self.C.shape)

        # (2) define camera parameters as needed in the bundle adjustment procedure
        self.cam_params = np.array([load_cam_params_from_camera(c, oC, self.cam_model)
                                    for c, oC in zip(self.cameras, self.camera_centers)])

        # (3) define camera_ind, points_ind, points_2d as needed in bundle adjustment
        pts_ind, cam_ind, pts2d = [], [], []
        true_where_obs = np.invert(np.isnan(self.C[::2, :]))
        cam_indices = np.arange(self.n_cam)
        for i in range(self.n_pts):
            #print(true_where_obs.shape)
            #print(cam_indices.shape)
            cam_indices_where_obs = cam_indices[true_where_obs[:, i]]
            for j in cam_indices_where_obs:
                pts_ind.append(i)
                cam_ind.append(j)
                pts2d.append(self.C[(j * 2):(j * 2 + 2), i])
        self.pts_ind, self.cam_ind, self.pts2d = np.array(pts_ind), np.array(cam_ind), np.vstack(pts2d)
        self.n_obs = self.pts2d.shape[0]

        # (4) define the vector of parameters/variables to optimize, i.e. params_opt
        self.n_params = 0
        if 'R' in self.cam_params_to_optimize:
            self.n_params += 3
            cam_params_opt = self.cam_params[:, :3]
            if 'T' in self.cam_params_to_optimize:
                n_params_T = 2 if self.cam_model == 'affine' else 3
                self.n_params += n_params_T
                cam_params_opt = np.hstack((cam_params_opt, self.cam_params[:, 3:3 + n_params_T]))
                if 'K' in self.cam_params_to_optimize:
                    n_params_K = 3 if self.cam_model == 'affine' else 5
                    self.n_params += n_params_K
                    cam_params_opt = np.hstack((cam_params_opt, self.cam_params[:, 3:3 + n_params_K]))
        else:
            cam_params_opt = []
        # if K is to be optimized and shared among all cameras, extract it and put its values at the beginning
        if 'K' in self.cam_params_to_optimize and 'COMMON_K' in self.cam_params_to_optimize:
            n_params_K = 3 if self.cam_model == 'affine' else 5
            K = cam_params_opt[0, -n_params_K:]
            cam_params_opt = np.hstack([cam_params_opt[cam_idx, :-n_params_K] for cam_idx in range(self.n_cam_opt)])
            cam_params_opt = np.hstack((K, cam_params_opt))
        self.params_opt = np.hstack((cam_params_opt.ravel(), self.pts3d.ravel()))
        self.pts2d_w = np.ones(self.pts2d.shape[0])

        if verbose:
            print('{} 3d points, {} fixed and {} to be optimized'.format(self.n_pts, self.n_pts_fix, self.n_pts_opt))
            print('{} cameras, {} fixed and {} to be optimized'.format(self.n_cam, self.n_cam_fix, self.n_cam_opt))
            print('{} parameters to optimize per camera\n'.format(self.n_params))


    def reduce(self, C, pts3d, cameras, pairs_to_triangulate, camera_centers):
        """
        Reduce the number of parameters, if possible, by selecting only feature tracks with observations located
        in the cameras to optimize. This may be useful to save computational cost when some cameras are fixed.
        Args:
            C: ndarray representing a correspondence matrix containing a set of feature tracks
            cameras: either a list of M 3x4 projection matrices or a list of rpc models
            pairs_to_triangulate: list of pairs suitable for triangulation
        """

        # select only those feature tracks containing observations in the cameras to optimize
        # (i.e. columns of C with values different from nan in the rows of the cams to be optimized)
        cols_where_obs = np.sum(1 * ~np.isnan(C[::2, :])[-self.n_cam_opt:], axis=0)
        cols_where_obs = cols_where_obs.astype(bool)
        self.C = C[:, cols_where_obs].copy()
        self.pts_prev_indices = np.arange(self.n_pts, dtype=int)[cols_where_obs]
        self.n_pts_fix -= np.sum(1 * ~cols_where_obs[:self.n_pts_fix])
        self.n_pts_opt -= np.sum(1 * ~cols_where_obs[-self.n_pts_opt:])
        self.pts3d = pts3d[self.pts_prev_indices, :].copy()

        # remove possible cameras containing 0 observations after the previous process
        obs_per_cam = np.sum(1 * ~(np.isnan(self.C[::2, :])), axis=1)
        cams_to_keep = obs_per_cam > 0
        self.C = self.C[np.repeat(cams_to_keep, 2), :]
        self.n_cam = int(self.C.shape[0] / 2)
        self.n_pts = int(self.C.shape[1])
        self.cam_prev_indices = np.arange(self.n_cam, dtype=int)[cams_to_keep]
        self.n_cam_fix -= np.sum(1 * ~cams_to_keep[:self.n_cam_fix])
        self.n_cam_opt -= np.sum(1 * ~cams_to_keep[-self.n_cam_opt:])
        self.cameras = [cameras[idx] for idx in self.cam_prev_indices]
        self.camera_centers = [camera_centers[idx] for idx in self.cam_prev_indices]

        # update pairs to triangulate with the new camera indices
        new_cam_idx_from_old_cam_idx = np.array([-1] * len(cams_to_keep))
        new_cam_idx_from_old_cam_idx[cams_to_keep] = np.arange(np.sum(cams_to_keep))
        self.pairs_to_triangulate = []
        for [idx_r, idx_l] in pairs_to_triangulate:
            if cams_to_keep[idx_r] and cams_to_keep[idx_l]:
                new_idx_r = new_cam_idx_from_old_cam_idx[idx_r]
                new_idx_l = new_cam_idx_from_old_cam_idx[idx_l]
                self.pairs_to_triangulate.append((new_idx_r, new_idx_l))


    def get_vars_ready_for_fun(self, v):
        """
        Given the vector of variables involved in the bundle adjustment optimization problem,
        use the current BA_Parameters instance to set it ready for function ba_core.fun
        Args:
            vars: variables to optimize or optimized
        Returns:
            pts3d: Nx3 array with the (x,y,z) ECEF coordinates of N 3d points
            cam_params: MxP array with the the P parameters of the M cameras used for the projection of the 3d points
        """

        # handle K (1st part)
        n_params = self.n_params
        if 'K' in self.cam_params_to_optimize and 'COMMON_K' in self.cam_params_to_optimize:
            # vars is organized as: [ K + params cam 1 + ... + params cam N + pt 3D 1 + ... + pt 3D N ]
            n_params_K = 3 if cam_model == 'affine' else 5
            K = v[:n_params_K]
            v = v[n_params_K:]
            n_params -= params_in_K
        else:
            # vars is organized as: [ params cam 1 + ... + params cam N + pt 3D 1 + ... + pt 3D N ]
            K = np.array([])

        # get 3d points
        pts3d = v[self.n_cam * n_params:].reshape((self.n_pts, 3)).copy()
        if self.n_pts_fix > 0:
            # fixed pts are at first rows if any
            pts3d[:self.n_pts_fix, :] = self.pts3d[:self.n_pts_fix, :]

        # get camera params
        cam_params_opt = v[:self.n_cam * n_params].reshape((self.n_cam, n_params))
        if self.n_cam_fix > 0:
            # fixed cams are at first rows if any
            cam_params_opt[:self.n_cam_fix, :] = self.cam_params[:self.n_cam_fix, :n_params]
        # add fixed camera params
        cam_params = np.hstack((cam_params_opt, self.cam_params[:, n_params:]))

        # handle K (2nd part)
        if 'K' in self.cam_params_to_optimize and 'COMMON_K' in self.cam_params_to_optimize:
            cam_params[:, -n_params_K:] = np.repeat(np.array([K]), cam_params.shape[0], axis=0)

        return pts3d, cam_params


    def reconstruct_vars(self, v, pts3d, cameras):
        """
        Given the vector of variables involved in the bundle adjustment optimization problem,
        reconstruct the corresponding 3d points and camera models
        Args:
            vars: variables to optimize or optimized
        Returns:
            pts3d: Nx3 array with the (x,y,z) ECEF coordinates of N 3d points
            cameras_ba: list containing the M cameras reconstructed from the vector of variables
        """

        self.pts3d_ba, cam_params = self.get_vars_ready_for_fun(v)
        self.cameras_ba = [load_camera_from_cam_params(cam_params[i, :], self.cam_model) for i in range(self.n_cam)]

        self.estimated_params = []
        for i in range(cam_params.shape[0]):
            current_cam_estimated_params = {}
            if 'R' in self.cam_params_to_optimize:
                current_cam_estimated_params['R'] = cam_params[i, :3]
            if 'T' in self.cam_params_to_optimize:
                current_cam_estimated_params['T'] = cam_params[i, 3:6]
            if self.cam_model == 'rpc':
                current_cam_estimated_params['C'] = cam_params[i, 6:9]
            self.estimated_params.append(current_cam_estimated_params)

        if self.cam_model == 'rpc':
            for cam_idx in np.arange(self.n_cam_fix):
                self.cameras_ba[cam_idx] = cameras[self.cam_prev_indices[cam_idx]]
            for cam_idx in np.arange(self.n_cam_fix, self.n_cam_fix + self.n_cam_opt):
                args = [self.cameras_ba[cam_idx], self.cameras[self.cam_prev_indices[cam_idx]], self.pts3d_ba]
                self.cameras_ba[cam_idx], err = rpc_fit.fit_Rt_corrected_rpc(*args)
                to_print = [cam_idx, 1e4*err.max(), 1e4*err.mean()]
                print('cam {:2} - RPC fit error per obs [1e-4 px] (max / avg): {:.2f} / {:.2f}'.format(*to_print))

        print('\n')
        corrected_pts3d, corrected_cameras = pts3d.copy(), cameras.copy()
        for ba_idx, prev_idx in enumerate(self.pts_prev_indices):
            corrected_pts3d[prev_idx] = self.pts3d_ba[ba_idx]
        for ba_idx, prev_idx in enumerate(self.cam_prev_indices):
            corrected_cameras[prev_idx] = self.cameras_ba[ba_idx]

        return corrected_pts3d, corrected_cameras