"""
* Bundle Adjustment (BA) for 3D Reconstruction from Multi-Date Satellite Images
* Based on https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
* by Roger Mari <mari@cmla.ens-cachan.fr>
"""
import numpy as np
import matplotlib.pyplot as plt
import srtm4
from scipy import linalg
from scipy.sparse import lil_matrix

from bundle_adjust import ba_utils
from bundle_adjust.ba_triangulation import initialize_3d_points
from bundle_adjust import ba_rotations as ba_R

def project(pts, cam_params, cam_model, K):
    """
    Convert 3D points to 2D by projecting onto images
    """

    pts_proj = ba_R.rotate_euler(pts, cam_params[:, :3])
    
    if K.shape[0] > 0:
        if cam_model == 'Affine':
            pts_proj = pts_proj[:,:2]
            fx, fy, skew = K[0], K[1], K[2]
            pts_proj[:,0] = fx * pts_proj[:,0] + skew * pts_proj[:,1]
            pts_proj[:,1] = fy * pts_proj[:,1]
            pts_proj += cam_params[:, 3:5]
        else:
            pts_proj += cam_params[:, 3:6]
            fx, fy, skew, cx, cy = K[0], K[1], K[2], K[3], K[4]
            pts_proj[:,0] = fx * pts_proj[:,0] + skew * pts_proj[:,1] + cx * pts_proj[:,2]
            pts_proj[:,1] = fy * pts_proj[:,1] + cy * pts_proj[:,2]
            pts_proj = pts_proj[:, :2] / pts_proj[:, 2, np.newaxis] # set scale = 1  
    else:
        if cam_model == 'Affine':
            pts_proj = pts_proj[:,:2]
            fx, fy, skew = cam_params[:, 5], cam_params[:, 6], cam_params[:, 7]
            pts_proj[:,0] = fx * pts_proj[:,0] + skew * pts_proj[:,1]
            pts_proj[:,1] = fy * pts_proj[:,1]
            pts_proj += cam_params[:, 3:5]
        else:
            pts_proj += cam_params[:, 3:6]
            fx, fy, skew, cx, cy = cam_params[:, 6], cam_params[:, 7], cam_params[:, 8], cam_params[:, 9], cam_params[:, 10]
            pts_proj[:,0] = fx * pts_proj[:,0] + skew * pts_proj[:,1] + cx * pts_proj[:,2]
            pts_proj[:,1] = fy * pts_proj[:,1] + cy * pts_proj[:,2]
            pts_proj = pts_proj[:, :2] / pts_proj[:, 2, np.newaxis] # set scale = 1  
    return pts_proj


def fun(params, cam_ind, pts_ind, pts_2d, cam_params, pts_3d, ba_params, pts_2d_w, fix_1st_cam=False):
    """
    Compute Bundle Adjustment residuals.
    'params' contains those parameters to be optimized (3D points + camera paramters)
    """
    n_cam_opt, n_cam_fix, n_cam = ba_params['n_cam_opt'], ba_params['n_cam_fix'], ba_params['n_cam']
    cam_model, n_params = ba_params['cam_model'], ba_params['n_params']
    n_pts_opt, n_pts_fix, n_pts = ba_params['n_pts_opt'], ba_params['n_pts_fix'], ba_params['n_pts']

    if ba_params['opt_K'] and ba_params['fix_K']:
        # params is organized as: [ K + params cam 1 + ... + params cam N + pt 3D 1 + ... + pt 3D N ]
        params_in_K = 3 if cam_model == 'Affine' else 5
        K_tmp = params[:params_in_K]
        K = cam_params[0, -params_in_K:]
        K[0] = K_tmp[0] # per optimitzar fx
        params = params[params_in_K:]
        n_params -= params_in_K
    else:
        # params is organized as: [ params cam 1 + ... + params cam N + pt 3D 1 + ... + pt 3D N ]
        K = np.array([])

    # get 3d points to optimize 
    pts_3d_opt = np.vstack((pts_3d[:n_pts_fix,:], #fixed pts are at the first rows if any
                            params[n_cam * n_params:].reshape((n_pts, 3))))
        
    # get camera params to optimize
    cam_params_opt = np.vstack((cam_params[:n_cam_fix,:n_params], #fixed cameras are at the first rows if any
                                params[n_cam_fix * n_params : n_cam * n_params].reshape((n_cam_opt, n_params))))
    
    # add fixed camera params
    cam_params_ba = np.hstack((cam_params_opt,cam_params[:,n_params:]))      
    
    # project 3d points using the current camera params   
    points_proj = project(pts_3d_opt[pts_ind], cam_params_ba[cam_ind], cam_model, K)
    
    # compute reprojection errors
    weights = np.repeat(pts_2d_w,2, axis=0)
    err = weights*(points_proj - pts_2d).ravel()   
    
    return err


def bundle_adjustment_sparsity(cam_ind, pts_ind, ba_params):
    '''
    Builds the sparse matrix employed to compute the Jacobian of the bundle adjustment
    '''
    
    n_cam, n_pts, n_params = ba_params['n_cam'], ba_params['n_pts'], ba_params['n_params']   
    m = pts_ind.size * 2
    params_in_K = 3 if ba_params['cam_model'] == 'Affine' else 5
    
    common_K = ba_params['opt_K'] and ba_params['fix_K']
    if common_K:
        n_params -= params_in_K
    
    n = common_K * params_in_K + n_cam * n_params + n_pts * 3
    A = lil_matrix((m, n), dtype=int)
    print('Shape of matrix A: {}x{}'.format(m,n))
    
    i = np.arange(pts_ind.size)
    for s in range(n_params):
        A[2 * i, common_K * params_in_K + cam_ind * n_params + s] = 1
        A[2 * i + 1, common_K * params_in_K + cam_ind * n_params + s] = 1
        
    for s in range(3):
        A[2 * i, common_K * params_in_K + n_cam * n_params + pts_ind * 3 + s] = 1
        A[2 * i + 1, common_K * params_in_K + n_cam * n_params + pts_ind * 3 + s] = 1
    
    if common_K:
        A[:, :params_in_K] = np.ones((m, params_in_K))
            
    return A

def ba_cam_params_to_P(cam_params, cam_model):
    '''
    Recover the 3x4 projection matrix P from the camera parameters format used by the bundle adjustment
    '''
    
    if cam_model == 'Affine':
        vecR, vecT, fx, fy, skew = cam_params[0:3], cam_params[3:5], cam_params[5], cam_params[6], cam_params[7]
        K = np.array([[fx, skew], [0., fy]])
        R = ba_R.euler_angles_to_R(vecR)
        P = np.vstack( (np.hstack((K @ R[:2,:], np.array([vecT]).T)), np.array([[0,0,0,1]])) )
    else:
        vecR, vecT = cam_params[0:3], cam_params[3:6]
        fx, fy, skew, cx, cy = cam_params[6], cam_params[7], cam_params[8], cam_params[9], cam_params[10]
        K = np.array([[fx, skew, cx], [0., fy, cy], [0., 0., 1.]])
        R = euler_angles_to_R(vecR)
        P = K @ np.hstack((R, vecT.reshape((3,1))))
    return P/P[2,3]

def ba_cam_params_from_P(P, cam_model):
    '''
    Convert the 3x4 projection matrix P to the camera parameters format used by the bundle adjustment
    '''
    if cam_model == 'Affine':
        K, R, vecT = decompose_affine_camera(P)
        vecR = ba_R.euler_angles_from_R(R)
        u, s, vh = np.linalg.svd(R, full_matrices=False)
        fx, fy, skew = K[0,0], K[1,1], K[0,1]
        cam_params = np.hstack((vecR.ravel(),vecT.ravel(),fx,fy,skew))
    else:
        K, R, vecT, _ = decompose_perspective_camera(P)
        K = K/K[2,2]
        vecR = ba_R.euler_angles_from_R(R)
        fx, fy, skew, cx, cy = K[0,0], K[1,1], K[0,1], K[0,2], K[1,2]
        cam_params = np.hstack((vecR.ravel(),vecT.ravel(),fx,fy,skew,cx,cy))
    return cam_params

def get_elbow_value(init_e, percentile_value=95, verbose=False):

    sort_indices = np.argsort(init_e)
    
    values = np.sort(init_e).tolist()
    nPoints = len(values)
    allCoord = np.vstack((range(nPoints), values)).T

    # get the first point
    firstPoint = allCoord[0]
    # get vector between first and last point - this is the line
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))

    # find the distance from each point to the line:
    # vector between all points and first point
    vecFromFirst = allCoord - firstPoint

    scalarProduct = np.sum(vecFromFirst * np.tile(lineVecNorm, (nPoints, 1)), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel

    # distance to line is the norm of vecToLine
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))

    # knee/elbow is the point with max distance value
    elbow_value = values[np.argmax(distToLine)]

    elbow_value = np.percentile(init_e[init_e < elbow_value], percentile_value)
    
    success = True
    if elbow_value < np.percentile(init_e, 80):
        success = False
    
    if verbose:
        print('Elbow value is {}'.format(elbow_value))
        fig = plt.figure(figsize=(10,5))
        plt.plot(np.sort(init_e))
        plt.axhline(y=elbow_value, color='r', linestyle='-')
        plt.show()
    
    return elbow_value, success


def remove_outlier_obs(reprojection_err, pts_ind, cam_ind, C, pairs_to_triangulate, thr=1.0):
    '''
    Given the reprojection error associated each feature observation involved in the bundle adjustment process 
    (i.e. ba_output_err), those observations with error larger than 'outlier_thr' are removed
    The correspondence matrix C is updated with the remaining observations
    '''
    
    C_new = C.copy()
    n_old_tracks = C_new.shape[1]
    n_img = int(C_new.shape[0]/2)
    cont = 0
    for i in range(len(reprojection_err)):
        if reprojection_err[i] > thr:
            cont += 1
            track_where_obs, cam_where_obs = pts_ind[i], cam_ind[i]
            # count number of obs x track (if the track is formed by only one match, then delete it)
            # otherwise delete only that particular observation
            C_new[2*cam_where_obs, track_where_obs] = np.nan
            C_new[2*cam_where_obs+1, track_where_obs] = np.nan
    
    # count the updated number of obs per track and keep those tracks with 2 or more observations 
    obs_per_track = np.sum(1*np.invert(np.isnan(C_new)), axis=0)
    columns_to_preserve = obs_per_track >=4
    
    # remove matches found in pairs with short baseline that were not extended to more images
    # since these columns of C will not be triangulated
    for i in range(C.shape[1]):
        if columns_to_preserve[i]:
            im_ind = [k for k, j in enumerate(range(n_img)) if not np.isnan(C_new[j*2,i])]
            all_pairs = [(im_i, im_j) for im_i in im_ind for im_j in im_ind if im_i != im_j and im_i<im_j]
            good_pairs = [pair for pair in all_pairs if pair in pairs_to_triangulate]
            if len(good_pairs) == 0:
                cont += len(all_pairs)
            columns_to_preserve[i] = len(good_pairs) > 0

    C_new = C_new[:, columns_to_preserve]
    n_tracks_del = n_old_tracks - C_new.shape[1]
    
    print('Observations per cam before outlier removal', np.sum(1*~np.isnan(C), axis=1)[::2])
    print('Observations per cam after outlier removal', np.sum(1*~np.isnan(C_new), axis=1)[::2])
    
    print('Deleted {} observations ({:.2f}%) and {} tracks ({:.2f}%)\n' \
          .format(cont, cont/pts_ind.shape[0]*100, n_tracks_del, n_tracks_del/n_old_tracks*100))
    
    
    track_indices_to_preserve = np.arange(C.shape[1])[columns_to_preserve]
    
    return C_new, track_indices_to_preserve


def get_ba_output(params, ba_params, cam_params, pts_3d):
    '''
    Recover the Bundle Adjustment output from 'optimized_params'.
    Output: pts_3d_ba     - (Nx3) array with the optimized N 3D point locations
            cam_params_ba - optimized camera parameters in the bundle adjustment format
            P_ba          - list with the optimized 3x4 projection matrices
    '''
    
    # remember the structure of 'params': camera parameters + 3d points 
    n_cam_opt, n_cam_fix, n_cam = ba_params['n_cam_opt'], ba_params['n_cam_fix'], ba_params['n_cam'] 
    n_pts, n_params = ba_params['n_pts'], ba_params['n_params']
    params_in_K = 3 if ba_params['cam_model'] == 'Affine' else 5
    
    if ba_params['opt_K'] and ba_params['fix_K']:
        K = optimized_params[:params_in_K]
        optimized_params = optimized_params[params_in_K:]
        n_params -= params_in_K
    
    # get 3d points
    if ba_params['opt_X']:
        pts_3d_ba = params[n_cam * n_params:].reshape((n_pts, 3))
    else:
        pts_3d_ba = pts_3d
    
    # get camera params to optimize
    cam_params_opt = np.vstack((cam_params[:n_cam_fix,:n_params], #fixed cameras are at the first rows if any
                                params[n_cam_fix * n_params : n_cam * n_params].reshape((n_cam_opt, n_params))))
    
    # add fixed camera params
    cam_params_ba = np.hstack((cam_params_opt,cam_params[:,n_params:]))     
    
    if ba_params['opt_K'] and ba_params['fix_K']:
        cam_params_ba[:, -params_in_K:] = np.repeat(np.array([K]), cam_params_ba.shape[0], axis=0)
    
    P_ba = ba_params['input_P'].copy()
    for (idx, it) in zip(ba_params['prev_cam_indices'], range(cam_params_ba.shape[0])):
        P_ba[idx] = ba_cam_params_to_P(cam_params_ba[it,:], ba_params['cam_model'])
    
    return pts_3d_ba, cam_params_ba, P_ba


def get_ba_error(ba_residuals, pts_2d_w=None):
    

    if pts_2d_w is None:
        pts_2d_w = np.ones(int(len(ba_residuals)/2))
        
    des_norm = np.repeat(pts_2d_w,2, axis=0)
    
    error_per_obs = np.add.reduceat(abs(ba_residuals.astype(float)/des_norm), np.arange(0, len(ba_residuals), 2))
    mean_error = np.mean(error_per_obs)
    median_error = np.median(error_per_obs)
    
    return error_per_obs, mean_error, median_error
    

def check_ba_error(error_before, error_after, pts_2d_w, display_plots=True):

    if pts_2d_w is None:
        pts_2d_w = np.ones(pts_2d_w.shape[0])

    init_e, init_e_mean, init_e_median = get_ba_error(error_before, pts_2d_w)
    ba_e, ba_e_mean, ba_e_median = get_ba_error(error_after, pts_2d_w)
    
    if display_plots:
        _,f = plt.subplots(1, 2, figsize=(10,3))
        f[0].hist(init_e, bins=40);
        f[1].hist(ba_e, bins=40); 
    
    print('Error before BA (mean / median): {:.2f} / {:.2f}'.format(init_e_mean, init_e_median))
    print('Error after  BA (mean / median): {:.2f} / {:.2f}\n'.format(ba_e_mean, ba_e_median))
    return ba_e, init_e

def ba_cam_params_to_P(cam_params, cam_model):
    '''
    Recover the 3x4 projection matrix P from the camera parameters format used by the bundle adjustment
    '''
    
    if cam_model == 'Affine':
        vecR, vecT, fx, fy, skew = cam_params[0:3], cam_params[3:5], cam_params[5], cam_params[6], cam_params[7]
        K = np.array([[fx, skew], [0., fy]])
        R = ba_R.euler_angles_to_R(vecR)
        P = np.vstack( (np.hstack((K @ R[:2,:], np.array([vecT]).T)), np.array([[0,0,0,1]])) )
    else:
        vecR, vecT = cam_params[0:3], cam_params[3:6]
        fx, fy, skew, cx, cy = cam_params[6], cam_params[7], cam_params[8], cam_params[9], cam_params[10]
        K = np.array([[fx, skew, cx], [0., fy, cy], [0., 0., 1.]])
        R = ba_R.euler_angles_to_R(vecR)
        P = K @ np.hstack((R, vecT.reshape((3,1))))
    return P/P[2,3]

def ba_cam_params_from_P(P, cam_model):
    '''
    Convert the 3x4 projection matrix P to the camera parameters format used by the bundle adjustment
    '''
    if cam_model == 'Affine':
        K, R, vecT = decompose_affine_camera(P)
        vecR = ba_R.euler_angles_from_R(R)
        u, s, vh = np.linalg.svd(R, full_matrices=False)
        fx, fy, skew = K[0,0], K[1,1], K[0,1]
        cam_params = np.hstack((vecR.ravel(),vecT.ravel(),fx,fy,skew))
    else:
        K, R, vecT, _ = decompose_perspective_camera(P)
        K = K/K[2,2]
        vecR = ba_R.euler_angles_from_R(R)
        fx, fy, skew, cx, cy = K[0,0], K[1,1], K[0,1], K[0,2], K[1,2]
        cam_params = np.hstack((vecR.ravel(),vecT.ravel(),fx,fy,skew,cx,cy))
    return cam_params


def set_ba_params(P, C, cam_model, n_cam_fix, n_pts_fix, pairs_to_triangulate, pts_3d=None, reduce=True, verbose=True):
    '''
    Given a set of input feature tracks (correspondence matrix C) and a set of initial projection matrices (P),
    define the input parameters needed by Bundle Adjustment
    '''
    
    n_cam_init = int(C.shape[0]/2)
    n_cam_opt = n_cam_init - n_cam_fix
    
    n_pts_init = int(C.shape[1])
    n_pts_opt = n_pts_init - n_pts_fix
    
    if reduce:
        
        # pick only the points that have to be updated 
        # (i.e. list the columns of C with values different from nan in the rows of the cams to be optimized)
        true_where_new_track = np.sum(~np.isnan(C[np.arange(0, C.shape[0], 2), :])[-n_cam_opt:]*1,axis=0).astype(bool)
        C_new = C[:, true_where_new_track]
        prev_pts_indices = np.arange(len(true_where_new_track))[true_where_new_track]
        n_pts_fix -= np.sum(np.invert(true_where_new_track[:n_pts_fix])*1)
        n_pts_opt -= np.sum(np.invert(true_where_new_track[-n_pts_opt:])*1)
        
        # remove cameras that dont need to be adjusted
        obs_per_cam = np.sum(1*~(np.isnan(C_new[np.arange(0, C_new.shape[0], 2), :])), axis=1)
        cams_to_keep = obs_per_cam > 0
        C_new = C_new[np.repeat(cams_to_keep,2),:]
        negative_else_new_idx = np.array([-1] * len(cams_to_keep)) 
        negative_else_new_idx[cams_to_keep] = np.arange(np.sum(cams_to_keep))
        prev_cam_indices = np.arange(len(cams_to_keep))[cams_to_keep]
        P_new = [P[idx] for idx in prev_cam_indices]
        n_cam_fix -= np.sum(np.invert(cams_to_keep[:n_cam_fix])*1)
        n_cam_opt -= np.sum(np.invert(cams_to_keep[-n_cam_opt:])*1)    
    
        #print('C shape:', C.shape)
        #print('C_new shape:', C_new.shape)
    
        # update pairs_to_triangulate with the new indices
        pairs_to_triangulate_new = []
        for [idx_r, idx_l] in pairs_to_triangulate:
            new_idx_r, new_idx_l = negative_else_new_idx[idx_r], negative_else_new_idx[idx_l]
            if new_idx_r >= 0 and new_idx_l >= 0:
                pairs_to_triangulate_new.append((new_idx_r, new_idx_l))
    else:
        P_new = P.copy()
        C_new = C.copy()
        pairs_to_triangulate_new = pairs_to_triangulate.copy()
        prev_pts_indices = np.arange(int(C.shape[1]))
        prev_cam_indices = np.arange(int(C.shape[0]/2))
    
    
    n_cam = len(P_new)
    n_pts = C_new.shape[1]
    
    # other ba parameters
    ba_params = {
    'cam_model'        : cam_model,
    'n_cam_fix'        : n_cam_fix,
    'n_cam_opt'        : n_cam_opt,
    'n_cam'            : n_cam_fix + n_cam_opt,
    'n_pts_fix'        : n_pts_fix,
    'n_pts_opt'        : n_pts_opt,
    'n_pts'            : n_pts_fix + n_pts_opt,
    'n_params'         : 0,
    'opt_X'            : True,
    'opt_R'            : True,
    'opt_T'            : False,
    'opt_K'            : False,
    'fix_K'            : False,
    'prev_pts_indices' : prev_pts_indices,
    'prev_cam_indices' : prev_cam_indices,
    'input_P'          : np.array(P),
    'input_3d_pts'     : pts_3d
    }
    
    # (1) init 3d points 
    if pts_3d is None:
        pts_3d = initialize_3d_points(P_new, C_new, pairs_to_triangulate_new, cam_model)
    else:
        pts_3d = pts_3d[prev_pts_indices,:].copy()
    
    # (2) define camera_params as needed in bundle adjustment
    n_cam_params = 8 if cam_model == 'Affine' else 11
    cam_params = np.zeros((n_cam,n_cam_params))
    for i in range(n_cam):
        cam_params[i, :] = ba_cam_params_from_P(P_new[i], cam_model)

    # (3) define camera_ind, points_ind, points_2d as needed in bundle adjustment
    point_ind, camera_ind, points_2d = [], [], []
    true_where_track = np.invert(np.isnan(C_new[np.arange(0, C_new.shape[0], 2), :]))
    cam_indices = np.arange(n_cam) 
    for i in range(n_pts):
        im_ind = cam_indices[true_where_track[:,i]]
        for j in im_ind:
            point_ind.append(i)
            camera_ind.append(j)
            points_2d.append(C_new[(j*2):(j*2+2),i])
    pts_ind, cam_ind = np.array(point_ind), np.array(camera_ind)
    pts_2d = np.vstack(points_2d)
    
    # Apparently the sparsity matrix A can't have a fixed shape (scipy least squares behaves wierd when A has colums of 0s)
    # Consequently, the number of camera parameters to be optimized is set in an incremental way so that A is well-posed
    # In other words: translations can only be refined along with rotations and not on their own
    n_params = 0
    if ba_params['opt_R']:
        n_params += 3
        cam_params_opt = cam_params[:,:3]
        if ba_params['opt_T']:
            if cam_model == 'Affine': 
                n_params += 2
                cam_params_opt = np.hstack((cam_params_opt, cam_params[:,3:5]))  # REVIEW !!!
            else:
                n_params += 3
                cam_params_opt = np.hstack((cam_params_opt, cam_params[:,3:6]))  # REVIEW !!!
            if ba_params['opt_K']:
                if cam_model == 'Affine':
                    n_params += 3
                    cam_params_opt = np.hstack((cam_params_opt, cam_params[:,6:]))   # REVIEW !!!
                else:
                    n_params += 5
                    cam_params_opt = np.hstack((cam_params_opt, cam_params[:,6:]))   # REVIEW !!!
    else:
        cam_params_opt = []
    ba_params['n_params'] = n_params     
    
    if verbose:
        print('{} 3d points in total, {} fixed and {} to be optimized'.format(n_pts, n_pts_fix, n_pts_opt))
        print('{} cameras in total, {} fixed and {} to be optimized'.format(n_cam, n_cam_fix, n_cam_opt))
        print('{} parameters per camera'.format(n_params))
    
    if ba_params['opt_K'] and ba_params['fix_K']:
        params_in_K = 3 if cam_model == 'Affine' else 5
        K = cam_params_opt[0,-params_in_K:]
        cam_params_opt2 = np.hstack([cam_params_opt[cam_id, :-params_in_K] for cam_id in range(n_cam_opt)])
        cam_params_opt = np.hstack((K, cam_params_opt2))

    pts_3d_opt = pts_3d.copy()
    params_opt = np.hstack((cam_params_opt.ravel(), pts_3d_opt.ravel()))
        
    return params_opt, cam_params, pts_3d, pts_2d, cam_ind, pts_ind, ba_params, C_new

def decompose_perspective_camera(P):
    """
    Decomposition of the perspective camera matrix as P = KR[I|-C] (Hartley and Zissermann 6.2.4)  
    Let  P = [M|T]. Compute internal and rotation as [K,R] = rq(M). Fix the sign so that diag(K) is positive.
    Camera center is computed with the formula C = -M^-1 T
    """
    
    # rq decomposition of M gives rotation and calibration
    M, T = P[:,:-1], P[:,-1]
    K, R = linalg.rq(M)
    # fix sign of the scale params
    R = np.diag(np.sign(np.diag(K))).dot(R)
    K = K.dot(np.diag(np.sign(np.diag(K))))  
    # optical center
    oC = -((np.linalg.inv(M)).dot(T))
    # translation vector of the camera 
    vecT = (R @ - oC[:, np.newaxis]).T[0]
    
    # fix sign of the scale params
    R = np.diag(np.sign(np.diag(K))).dot(R)
    K = K.dot(np.diag(np.sign(np.diag(K))))  
    
    #Preconstructed = K @ R @ np.hstack((np.eye(3), - oC[:, np.newaxis]))
    #print(np.allclose(P, Preconstructed))
    
    return K, R, vecT, oC


def decompose_affine_camera(P):
    """
    Decomposition of the affine camera matrix
    """
    
    M, vecT = P[:2,:3], np.array([P[:2,-1]])
    MMt = M @ M.T
    a, b, c, d = MMt[0,0], MMt[0,1], MMt[1,0], MMt[1,1]
    fy = np.sqrt(d)
    s = c/fy
    fx = np.sqrt(a - s**2)
    K = np.array([[fx, s],[0, fy]])
    # check that the solution of K is valid:
    #print(np.allclose(np.identity(2), np.linalg.inv(K) @ MMt @ np.linalg.inv(K.T)  ))
    R = np.linalg.inv(K) @ M
    r1 = np.array([R[0, :]]).T
    r2 = np.array([R[1, :]]).T
    r3 = np.cross(r1, r2, axis=0)
    R = np.vstack((r1.T, r2.T, r3.T))
   
    #Preconstructed = np.vstack( (np.hstack((K @ R[:2,:], vecT.T)), np.array([[0,0,0,1]])) )
    #print(np.allclose(P, Preconstructed))  
    
    return K, R, vecT


def rpc_affine_approx_for_bundle_adjustment(rpc, p):
    """
    Compute the first order Taylor approximation of an RPC projection function.

    Args:
        rpc: instance of the rpc_model.RPCModel class
        p: x, y, z coordinates in ecef system

    Return:
        array of shape (3, 4) representing the affine camera matrix equal to the
        first order Taylor approximation of the RPC projection function at point p.
    """ 
    import ad
    
    p = ad.adnumber(p)
    lat, lon, alt = ba_utils.ecef_to_latlon_custom_ad(*p)
    q = rpc.projection(lon, lat, alt)
    J = ad.jacobian(q, p)
    
    A = np.zeros((3, 4))
    A[:2, :3] = J
    A[:2, 3] = np.array(q) - np.dot(J, p)
    A[2, 3] = 1
    return A

def get_perspective_cam_from_rpc(rpc, crop):
    from bundle_adjust.rpc_utils import approx_rpc_as_proj_matrix
    # approximate current rpc as a perspective 3x4 matrix
    x, y, w, h = crop['col0'], crop['row0'], crop['crop'].shape[1], crop['crop'].shape[0]
    P_img = approx_rpc_as_proj_matrix(rpc, [x,x+w,10], [y,y+h,10], [rpc.alt_offset - 100, rpc.alt_offset + 100, 10])
    #express P in terms of crop coord by applying the translation x0, y0 (i.e.top-left corner of the crop)
    T_crop = np.array([[1., 0., -crop['col0']], [0., 1., -crop['row0']], [0., 0., 1.]])
    current_P = T_crop @ P_img
    return current_P/current_P[2,3]

def get_affine_cam_from_rpc(rpc, crop, lon, lat, alt):
    p_x, p_y, p_z = ba_utils.latlon_to_ecef_custom(lat, lon, alt)
    p_geocentric = [p_x, p_y, p_z]
    P_img = rpc_affine_approx_for_bundle_adjustment(rpc, p_geocentric)
    T_crop = np.array([[1., 0., -crop['col0']], [0., 1., -crop['row0']], [0., 0., 1.]])
    current_P = T_crop @ P_img
    return current_P/current_P[2,3]

def approximate_rpcs_as_proj_matrices(myrpcs_new, mycrops_new, aoi, cam_model='Perspective'):
    
    print('Approximating RPCs as {} projection matrices'.format(cam_model))
    n_ims, myprojmats_new, err_indices = len(myrpcs_new), [], []

    if cam_model =='Affine':
        
        lon, lat = aoi['center'][0], aoi['center'][1]
        alt = srtm4.srtm4(lon, lat)
        for im_idx, rpc, crop in zip(np.arange(n_ims), myrpcs_new, mycrops_new):
            
            myprojmats_new.append(get_affine_cam_from_rpc(rpc, crop, lon, lat, alt))
            print('\r{} projection matrices / {} ({} err)'.format(im_idx+1, n_ims, len(err_indices)),end='\r')
    else:
        
        # Perspective model
        for im_idx, rpc, crop in zip(np.arange(n_ims), myrpcs_new, mycrops_new):
           
            myprojmats_new.append(get_perspective_cam_from_rpc(rpc, crop))
            print('\r{} projection matrices / {} ({} err)'.format(im_idx+1, n_ims, len(err_indices)),end='\r') 

        print('\nDone!\n')
    return myprojmats_new

