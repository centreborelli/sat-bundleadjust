"""
* Bundle Adjustment (BA) for 3D Reconstruction from Multi-Date Satellite Images
* Based on https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
* by Roger Mari <mari@cmla.ens-cachan.fr>
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.sparse import lil_matrix
import rectification
import triangulation
import stereo
import utils
import cv2
import re
import math

def rotate_rodrigues(pts, vecR, proper_R_axis):
    """
    Rotate points by given rotation vectors using Rodrigues' formula
    """
    theta = np.linalg.norm(vecR, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = vecR/ theta
        v = np.nan_to_num(v)
    v = vecR / theta
    dot = np.sum(pts * v, axis=1)[:, np.newaxis]
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    
    p = np.where(proper_R_axis == 99)[0] #indices where proper rotations
    q = np.where(proper_R_axis != 99)[0] #indices where improper rotations
       
    rotated_pts = np.zeros(pts.shape)
    
    # Rodriguez formula for a proper rotation 
    rotated_pts[p] = cos_theta[p] * pts[p] + sin_theta[p] * np.cross(v[p], pts[p]) + dot[p] * (1 - cos_theta[p]) * v[p] 
    # Rodriguez formula for an improper rotation 
    rotated_pts[q] = cos_theta[q] * pts[q] + sin_theta[q] * np.cross(v[q], pts[q]) - dot[q] * (1 + cos_theta[q]) * v[q]
    
    return rotated_pts

def rotate_euler(pts, vecR):
    """
    Rotate points by using Euler angles
    """
    # R = R(z)R(y)R(x)
    cosx, sinx = np.cos(vecR[:,0]), np.sin(vecR[:,0]) 
    cosy, siny = np.cos(vecR[:,1]), np.sin(vecR[:,1])
    cosz, sinz = np.cos(vecR[:,2]), np.sin(vecR[:,2])
    
    # rotate along x-axis
    pts_Rx = np.vstack((pts[:,0], cosx*pts[:,1]-sinx*pts[:,2], sinx*pts[:,1] + cosx*pts[:,2])).T
    
    # rotate along y-axis
    pts_Ryx = np.vstack((cosy*pts_Rx[:,0] + siny*pts_Rx[:,2], pts_Rx[:,1], -siny*pts_Rx[:,0] + cosy*pts_Rx[:,2])).T
    
    # rotate along z-axis
    pts_Rzyx = np.vstack((cosz*pts_Ryx[:,0] - sinz*pts_Ryx[:,1], sinz*pts_Ryx[:,0] + cosz*pts_Ryx[:,1], pts_Ryx[:,2])).T
    
    return pts_Rzyx

def project(pts, cam_params, proper_R_axis, cam_model, R_params, K):
    """
    Convert 3D points to 2D by projecting onto images
    """
    if R_params == 'Rodrigues':
        pts_proj = rotate_rodrigues(pts, cam_params[:, :3], proper_R_axis)
    else:
        pts_proj = rotate_euler(pts, cam_params[:, :3])
    
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

def fun(params, cam_ind, pts_ind, pts_2d, proper_rotations, cam_params, pts_3d, ba_params, pts_2d_w, fix_1st_cam=False):
    """
    Compute Bundle Adjustment residuals.
    'params' contains those parameters to be optimized (3D points + camera paramters)
    """
    n_cam, n_pts, n_params = ba_params['n_cam'], ba_params['n_pts'], ba_params['n_params']
    R_params, cam_model = ba_params['R_params'], ba_params['cam_model']
    
    # to fix the first camera
    if fix_1st_cam:
        n_cam -= 1
        params = params[n_params:]
    
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
    
    # get 3d points
    if ba_params['opt_X']:
        pts_3d_ba = params[n_cam * n_params:].reshape((n_pts, 3))
    else:
        pts_3d_ba = pts_3d

    #get camera matrices
    cam_params_opt = params[:n_cam * n_params].reshape((n_cam, n_params))

    if fix_1st_cam:
        cam_params_opt = np.vstack((cam_params[0,:n_params], params[:n_cam * n_params].reshape((n_cam, n_params))))

    if n_params > 0:
        cam_params_ba = np.hstack((cam_params_opt,cam_params[:,n_params:]))
    else:
        cam_params_ba = cam_params       

    # project 3d points using the current camera params
    points_proj = project(pts_3d_ba[pts_ind], cam_params_ba[cam_ind], proper_rotations[cam_ind], cam_model, R_params, K)
    
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
    
    n = common_K * params_in_K + n_cam * n_params + ba_params['opt_X'] * n_pts * 3
    A = lil_matrix((m, n), dtype=int)
    print('Shape of matrix A: {}x{}'.format(m,n))
        
    i = np.arange(pts_ind.size)
    for s in range(n_params):
        A[2 * i, common_K * params_in_K + cam_ind * n_params + s] = 1
        A[2 * i + 1, common_K * params_in_K + cam_ind * n_params + s] = 1
        
    if ba_params['opt_X']:
        for s in range(3):
            A[2 * i, common_K * params_in_K + n_cam * n_params + pts_ind * 3 + s] = 1
            A[2 * i + 1, common_K * params_in_K + n_cam * n_params + pts_ind * 3 + s] = 1
    
    if common_K:
        A[:, :params_in_K] = np.ones((m, params_in_K))
            
    return A

def decompose_projection_matrix(P):
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
    # center
    C = -((np.linalg.inv(M)).dot(T))
    # translation vector of the camera 
    vecT = (R @ - C[:, np.newaxis]).T[0]
    
    # fix sign of the scale params
    R = np.diag(np.sign(np.diag(K))).dot(R)
    K = K.dot(np.diag(np.sign(np.diag(K))))  
    
    #Preconstructed = K @ R @ np.hstack((np.eye(3), - C[:, np.newaxis]))
    #print(np.allclose(P, Preconstructed))
    
    return K, R, vecT, C

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
   
    #Preconstructed = np.vstack( (np.hstack((K @ R, vecT)), np.array([[0,0,0,1]])) )
    #print(np.allclose(P, Preconstructed))  
    
    return K, R, vecT

def euler_angles_from_R(R) :
    """
    Convert a 3x3 rotation matrix R to the Euler angles representation
    Source: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    """
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
 
    if not singular:
        xa, ya, za = np.arctan2(R[2,1] , R[2,2]), np.arctan2(-R[2,0], sy), np.arctan2(R[1,0], R[0,0])
    else:
        xa, ya, za = np.arctan2(-R[1,2], R[1,1]), np.arctan2(-R[2,0], sy), 0
 
    return np.array([xa, ya, za])

def euler_angles_to_R(vecR):
    """
    Recover the 3x3 rotation matrix R from the Euler angles representation
    Source: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    """
    
    R_x = np.array([[1, 0,                0              ],
                    [0, np.cos(vecR[0]), -np.sin(vecR[0])],
                    [0, np.sin(vecR[0]),  np.cos(vecR[0])]])
         
    R_y = np.array([[ np.cos(vecR[1]), 0, np.sin(vecR[1])],
                    [ 0,               1, 0              ],
                    [-np.sin(vecR[1]), 0, np.cos(vecR[1])]])
                 
    R_z = np.array([[np.cos(vecR[2]), -np.sin(vecR[2]), 0],
                    [np.sin(vecR[2]),  np.cos(vecR[2]), 0],
                    [0,                0,               1]])
                     
    R = R_z @ R_y @ R_x
    
    return R

def axis_angle_from_R(R):
    """
    Convert a 3x3 rotation matrix R to the axis-angle representation
    Source: https://github.com/Wallacoloo/printipi/blob/master/util/rotation_matrix.py
    """
    # Axis
    axis = np.array([ R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
    # Angle
    r, t = np.hypot(axis[0], np.hypot(axis[1], axis[2])), R[0,0] + R[1,1] + R[2,2]
    theta = np.arctan2(r, t-1)
    # Normalise axis
    axis = axis / r
    return axis, theta

def axis_angle_to_R(axis, angle):
    """
    Recover the 3x3 rotation matrix R from the axis-angle representation
    Source: https://github.com/Wallacoloo/printipi/blob/master/util/rotation_matrix.py
    """
    # Trig factors
    ca, sa = np.cos(angle), np.sin(angle)
    C = 1 - ca
    # Depack the axis
    x, y, z = axis
    # Multiplications (to remove duplicate calculations)
    xs, ys, zs, xC, yC, zC = x*sa, y*sa, z*sa, x*C, y*C, z*C
    xyC, yzC, zxC = x*yC, y*zC, z*xC
    # Update the rotation matrix
    R = np.array([[x*xC + ca, xyC - zs, zxC + ys], [xyC + zs, y*yC + ca, yzC - xs], [zxC - ys, yzC + xs, z*zC + ca]])
    return R

def get_proper_R(R):
    """
    Get proper rotation matrix (i.e. det(R) = 1) from improper rotation matrix (i.e. det(R) = -1)
    Notes on proper and improper rotations: https://pdfs.semanticscholar.org/9934/061eedc830fab32edd97cd677b95f21248e1.pdf
    """
    ind_of_reflection_axis = 99   # arbitrary value to indicate that R is a proper rotation matrix
    if (np.linalg.det(R) - (-1.0)) < 1e-5:
        L, V = np.linalg.eig(R)
        ind_of_reflection_axis = np.where(np.abs(L + 1.0) < 1e-5)[0][0]
        D = np.array([1,1,1])
        D[ind_of_reflection_axis] *= -1
        D = np.diag(D)
        R = np.real(V @ D @ np.diag(L) @ np.linalg.inv(V))
    return R, ind_of_reflection_axis

def ba_cam_params_to_P(cam_params, proper_R_axis, cam_model, R_params='Euler'):
    '''
    Recover the 3x4 projection matrix P from the camera parameters format used by the bundle adjustment
    '''
    proper_R_axis = int(proper_R_axis)
    
    if cam_model == 'Affine':
        vecR, vecT, fx, fy, skew = cam_params[0:3], cam_params[3:5], cam_params[5], cam_params[6], cam_params[7]
        K = np.array([[fx, skew], [0., fy]])
        if R_params == 'Rodrigues':
            angle = np.linalg.norm(vecR)
            axis = vecR / angle
            R = axis_angle_to_R(axis, angle)
            if proper_R_axis < 99:
                L, V = np.linalg.eig(R)
                D = np.array([1,1,1])
                D[proper_R_axis] *= -1
                D = np.diag(D)
                R = np.real(V @ D @ np.diag(L) @ np.linalg.inv(V))
        else:
            R = euler_angles_to_R(vecR)
        P = np.vstack( (np.hstack((K @ R[:2,:], np.array([vecT]).T)), np.array([[0,0,0,1]])) )
    else:
        vecR, vecT = cam_params[0:3], cam_params[3:6]
        fx, fy, skew, cx, cy = cam_params[6], cam_params[7], cam_params[8], cam_params[9], cam_params[10]
        K = np.array([[fx, skew, cx], [0., fy, cy], [0., 0., 1.]])
        if R_params == 'Rodrigues':
            angle = np.linalg.norm(vecR)
            axis = vecR / angle
            R = axis_angle_to_R(axis, angle)
            if proper_R_axis < 99:
                L, V = np.linalg.eig(R)
                D = np.array([1,1,1])
                D[proper_R_axis] *= -1
                D = np.diag(D)
                R = np.real(V @ D @ np.diag(L) @ np.linalg.inv(V))
        else:
            R = euler_angles_to_R(vecR)
        P = K @ np.hstack((R, vecT.reshape((3,1))))
    return P/P[2,3]

def ba_cam_params_from_P(P, cam_model, R_params='Euler'):
    '''
    Convert the 3x4 projection matrix P to the camera parameters format used by the bundle adjustment
    '''
    if cam_model == 'Affine':
        K, R, vecT = decompose_affine_camera(P)
        properR, proper_R_axis = get_proper_R(R)
        if R_params == 'Rodrigues':
            axis, theta = axis_angle_from_R(properR)
            vecR = axis*theta
        else:
            vecR = euler_angles_from_R(R)
        u, s, vh = np.linalg.svd(R, full_matrices=False)
        fx, fy, skew = K[0,0], K[1,1], K[0,1]
        cam_params = np.hstack((vecR.ravel(),vecT.ravel(),fx,fy,skew))
    else:
        K, R, vecT, _ = decompose_projection_matrix(P)
        K = K/K[2,2]
        properR, proper_R_axis = get_proper_R(R)
        if R_params == 'Rodrigues':
            axis, theta = axis_angle_from_R(properR)
            vecR = axis*theta
        else:
            vecR = euler_angles_from_R(R)
        fx, fy, skew, cx, cy = K[0,0], K[1,1], K[0,1], K[0,2], K[1,2]
        cam_params = np.hstack((vecR.ravel(),vecT.ravel(),fx,fy,skew,cx,cy))
    return cam_params, proper_R_axis

def find_SIFT_kp(im):
    '''
    Detect SIFT keypoints in an input image
    '''
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(im,None)
    kpts = np.array([kp[idx].pt for idx in range(len(kp))])
    return kpts, des

def match_pair(kp1, kp2, des1, des2, dist_thresold=0.6):
    '''
    Match SIFT keypoints from an stereo pair
    '''

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)

    #FLANN_INDEX_KDTREE = 0
    #index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    #search_params = dict(checks=50)
    #flann = cv2.FlannBasedMatcher(index_params, search_params)
    #matches = flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)
    
    # Apply ratio test as in Lowe's paper
    pts1, pts2, filt_matches, all_matches = [], [], [], []
    for m,n in matches:
        all_matches.append(m)
        if m.distance < dist_thresold*n.distance:
            pts2.append(kp2[m.trainIdx])
            pts1.append(kp1[m.queryIdx])
            filt_matches.append(m)
            
    # Geometric filtering using the Fundamental matrix
    pts1, pts2 = np.array(pts1), np.array(pts2)
    if pts1.shape[0] > 0 and pts2.shape[0] > 0:
        F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
        idx_of_matched_kp1 = []
        idx_of_matched_kp2 = []

        # We select only inlier points
        if mask is None:
            # Special case: no matches after geometric filtering
            pts1, pts2, filt_matches = None, None, []
        else:
            mask_bool = mask.ravel()==1
            pts1, pts2 = pts1[mask_bool], pts2[mask_bool] 
            filt_matches_g = [filt_matches[i] for i, x in enumerate(mask_bool) if x]
            for i, x in enumerate(mask_bool):
                if x:
                    idx_of_matched_kp1.append(filt_matches[i].queryIdx)
                    idx_of_matched_kp2.append(filt_matches[i].trainIdx)
    else:
        # no matches were after ratio test
        pts1, pts2, filt_matches = None, None, []
    
    return pts1, pts2, kp1, kp2, filt_matches_g, all_matches, idx_of_matched_kp1, idx_of_matched_kp2
  
def linear_triangulation_single_pt(pt1, pt2, P1, P2):
    '''
    Linear triangulation of a single stereo correspondence (does the same as triangulate points from OpenCV)
    '''
    x1, y1, x2, y2 = pt1[0], pt1[1], pt2[0], pt2[1]
    A = np.array([x1*P1[2,:]-P1[0,:], y1*P1[2,:]-P1[1,:], x2*P2[2,:]-P2[0,:], y2*P2[2,:]-P2[1,:]])
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    pt_3d = vh.T[:3,-1] / vh.T[-1,-1]
    #print(np.allclose(A, u @ np.diag(s) @ vh))  # to check that svd is applied properly
    #pt_3d_opencv = cv2.triangulatePoints(P1,P2,pt1,pt2)[:3,0]/cv2.triangulatePoints(P1,P2,pt1,pt2)[-1,0]
    #print(np.allclose(pt_3d, pt_3d_opencv)) # to check that linear triangulation works properly
    return pt_3d 
   
def triangulate_list_of_matches(pts1, pts2, P1, P2):
    '''
    Linear triangulation of multiple stereo correspondences
    '''
    X = cv2.triangulatePoints(P1,P2,pts1,pts2)
    X = X[:3,:] / X[-1,:]
    return X.T

def dist_between_proj_rays(pt1, pt2, P1, P2):
    '''
    Input: two 2D correspondences (pt1, pt2) and two projection matrices (P1, P2)
    Output: the distance between the projection rays that go from the optical center of the cameras to the points pt1, pt2
    
    If the camera calibration and correspondence are both good, the rays should intersect and the distance should be 0
    This is why this distance gives an idea about the triangulation error
    
    Inspired by https://math.stackexchange.com/questions/2213165/find-shortest-distance-between-lines-in-3d
    '''
    K, R, vecT, C1 = decompose_projection_matrix(P1)
    dir_vec_ray1 = np.linalg.inv(K @ R) @ np.expand_dims(np.hstack((pt1, np.ones(1))), axis=1)
    K, R, vecT, C2 = decompose_projection_matrix(P2)
    dir_vec_ray2 = np.linalg.inv(K @ R) @ np.expand_dims(np.hstack((pt2, np.ones(1))), axis=1)
    n = np.cross(dir_vec_ray1, dir_vec_ray2, axis=0)
    d = np.dot((C2 - C1), n) / np.linalg.norm(n)
    return abs(d[0])

def corresp_matrix_from_tracks(feature_tracks, r):
    '''
    Build a correspondence matrix C from a set of input feature tracks
    
    C = x11 ... x1n
        y11 ... y1n
        x21 ... x2n
        y21 ... y2n
        ... ... ...
        xm1 ... xmn
        ym1 ... ymn
 
        where (x11, y11) is the observation of feature track 1 in camera 1
              (xm1, ym1) is the observation of feature track 1 in camera m
              (x1n, y1n) is the observation of feature track n in camera 1
              (xmn, ymn) is the observation of feature track n in camera m
              
    Consequently, the shape of C is  (2*number of cameras) x number of feature tracks
    '''
    
    N = feature_tracks.shape[0]
    M = r.shape[1]
    C = np.zeros((2*M,N))
    C[:] = np.nan
    for i in range(N):
        im_ind = [k for k, j in enumerate(range(M)) if r[i,j]!=0]
        for ind in im_ind:
            C[ind*2:(ind*2)+2,i] = feature_tracks[i,:,ind]        
    return C  


def cloud_from_pair(i, j, P1, P2, cam_model, myimages, mycrops, aoi):
    '''
    Input:   i,j         - index of the stereo pair of views to be picked from 'myimages'
             P1, P2      - RPCs approximated as projection matrices and corrected via Bundle Adjustment
             cam_model   - camera model used to approximate the RPCs, 'Affine' or 'Perspective'
             myimages    - filenames of the satellite images to be used
             aoi         - area of interest to be reconstructed
             mycrops     - slices resulting from cropping the area of interest in each input satellite images
             
    Output:  dense_cloud - (Nx3) array containing N points 3D definining the reconstructed surface of the aoi
    '''
    # affine rectification and disparity computation
    rect1, rect2, S1, S2, dmin, dmax, PA, PB = rectification.rectify_aoi(myimages[i], myimages[j], aoi)
    LRS, _, _ =  stereo.compute_disparity_map(rect1,rect2,dmin-50,dmax+50,cost='census', lam=10)
    
    # matched coordinates in im1 and im2 after rectification and disparity computation
    x_im1, y_im1 = np.meshgrid(np.arange(0,LRS.shape[1]),np.arange(0,LRS.shape[0]))
    x_im2, y_im2 = x_im1 + LRS, y_im1 
    
    # matched coordinates in affine rectified im1 and im2
    pts_im1 = np.array([x_im1.flatten(), y_im1.flatten()]).T
    pts_im2 = np.array([x_im2.flatten(), y_im2.flatten()]).T

    # remove coordinates where disparity is not valid
    pts_im1_filt = pts_im1[abs(pts_im2[:,0])!=np.inf,:]
    pts_im2_filt = pts_im2[abs(pts_im2[:,0])!=np.inf,:]

    # matched coordinates in original im1 and im2 crops
    pts_im1_org = utils.points_apply_homography(np.linalg.inv(S1), pts_im1_filt)
    pts_im2_org = utils.points_apply_homography(np.linalg.inv(S2), pts_im2_filt)
    pts_im1_org[:,0] -= mycrops[i]['x0']
    pts_im1_org[:,1] -= mycrops[i]['y0']
    pts_im2_org[:,0] -= mycrops[j]['x0']
    pts_im2_org[:,1] -= mycrops[j]['y0']
    
    #build point cloud 
    if cam_model == 'Affine':
        x1, y1, x2, y2 = pts_im1_org[:,0], pts_im1_org[:,1], pts_im2_org[:,0], pts_im2_org[:,1]
        lon, lat, h, _ = triangulation.triangulation_affine(P1, P2, x1, y1, x2, y2)
        dense_cloud = np.vstack([lon, lat, h]).T
    else:
        dense_cloud = triangulate_list_of_matches(pts_im1_org.T, pts_im2_org.T, P1, P2)
    
    return dense_cloud

def initialize_3d_points(P_crop, C, cam_model, var_filt=True, var_hist=True):
    '''
    Initialize the 3D point corresponding to each feature track.
    How? Pick the average value of all possible triangulated points within each track.
    Additionally, compute the sum of variances in each dimension for each set of candidates.
    If the total variance is larger than a certain threshold, then remove that 3d point due to low reliability.
    '''
    n_pts, n_cam = C.shape[1], int(C.shape[0]/2) 
    pts_3d = np.zeros((n_pts,3))
    variance = np.zeros((n_pts,1))
    for i in range(n_pts):
        im_ind = [k for k, j in enumerate(range(n_cam)) if not np.isnan(C[j*2,i])]
        all_pairs = [(im_i, im_j) for im_i in im_ind for im_j in im_ind if im_i != im_j and im_i<im_j]
        current_track_candidates = []
        for current_pair in all_pairs:
            q, p = current_pair[0], current_pair[1]
            pt1, pt2 = C[(q*2):(q*2+2),i], C[(p*2):(p*2+2),i]
            P1, P2 = P_crop[q], P_crop[p]
            if cam_model == 'Affine':
                lon, lat, h, _ = triangulation.triangulation_affine(P1,P2, np.array([pt1[0]]), np.array([pt1[1]]), 
                                                                    np.array([pt2[0]]), np.array([pt2[1]]) )
                candidate_from_pair = np.hstack([lon, lat, h])
            else:
                candidate_from_pair = linear_triangulation_single_pt(pt1,pt2,P1,P2)
                #candidate_from_pair = cv2.triangulatePoints(P1, P2, pt1, pt2)[:,0]
            current_track_candidates.append(candidate_from_pair)
        if len(current_track_candidates) > 1:
            variance[i,:] = sum(np.var(np.array(current_track_candidates), axis=0))
        else:
            variance[i,:] = -1.0
        pts_3d[i,:] = np.mean(np.array(current_track_candidates), axis=0)
    
    var_thr = np.percentile(variance, 95)
    if var_filt:
        old_n_pts = n_pts
        if var_hist:
            # visualize variance of point candidates
            plt.hist(variance[variance[:,0]>0], bins=50, range=(0,2000))
            plt.axvline(x=var_thr, color='r', linestyle='--', linewidth=3.0) 
        # filter points according to variance
        idx_to_preserve = variance[:,0]<var_thr
        pts_3d = pts_3d[idx_to_preserve]
        C = C[:,idx_to_preserve]
        n_pts = C.shape[1]
        n_pts_gone = old_n_pts-n_pts
        print('Variance filtering removed {} points ({:.2f}%)\n'.format(n_pts_gone, n_pts_gone/old_n_pts * 100))
        
    return pts_3d, C


def remove_outlier_obs(reprojection_err, pts_ind, cam_ind, C, outlier_thr=1.0):
    '''
    Given the reprojection error associated each feature observation involved in the bundle adjustment process 
    (i.e. ba_output_err), those observations with error larger than 'outlier_thr' are removed
    The correspondence matrix C is updated with the remaining observations
    '''
    cont = 0
    for i in range(len(reprojection_err)):
        if reprojection_err[i] > outlier_thr:
            cont += 1
            track_where_obs, cam_where_obs = pts_ind[i], cam_ind[i]
            # count number of obs x track (if the track is formed by only one match, then delete it)
            # otherwise delete only that particular observation
            C[2*cam_where_obs, track_where_obs] = np.nan
            C[2*cam_where_obs+1, track_where_obs] = np.nan
    
    # count the updated number of obs per track and keep those tracks with 2 or more observations 
    obs_per_track = np.sum(1*np.invert(np.isnan(C)), axis=0)
    n_old_tracks = C.shape[1]
    C = C[:, obs_per_track >=4]
    n_tracks_del = n_old_tracks - C.shape[1]

    print('Deleted {} observations ({:.2f}%) and {} tracks ({:.2f}%)\n' \
          .format(cont, cont/pts_ind.shape[0]*100, n_tracks_del, n_tracks_del/n_old_tracks*100))  
    return C


def set_ba_params(P, C, cam_model, R_params='Euler', opt_X=True,opt_R=True,opt_T=False,opt_K=False,fix_K=True):
    '''
    Given a set of input feature tracks (correspondence matrix C) and a set of initial projection matrices (P),
    define the input parameters needed by Bundle Adjustment
    '''
    n_pts, n_cam = C.shape[1], int(C.shape[0]/2) 
   
    # define camera_params as needed in bundle adjustment
    n_cam_params = 8 if cam_model == 'Affine' else 11
    cam_params, proper_R_axis = np.zeros((n_cam,n_cam_params)), np.zeros((n_cam))
    for i in range(n_cam):
        cam_params[i, :], proper_R_axis[i] = ba_cam_params_from_P(P[i], cam_model, R_params)

    # define camera_ind, points_ind, points_2d as needed in bundle adjustment
    point_ind, camera_ind, points_2d = [], [], []
    for i in range(n_pts):
        im_ind = [k for k, j in enumerate(range(n_cam)) if not np.isnan(C[j*2,i])]
        for j in im_ind:
            point_ind.append(i)
            camera_ind.append(j)
            points_2d.append(C[(j*2):(j*2+2),i])
    pts_ind, cam_ind, pts_2d = np.array(point_ind), np.array(camera_ind), np.vstack(points_2d)
    
    # other ba parameters
    ba_params = {
    'cam_model' : cam_model,
    'R_params'  : R_params, # use Euler or Rodrigues here
    'n_cam'     : n_cam,
    'n_pts'     : n_pts,
    'n_params'  : 0,
    'opt_X'     : opt_X,
    'opt_R'     : opt_R,
    'opt_T'     : opt_T,
    'opt_K'     : opt_K,
    'fix_K'     : fix_K
    } 
    
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
    
    return cam_params, cam_params_opt, proper_R_axis, pts_2d, cam_ind, pts_ind, ba_params

def get_ba_output(optimized_params, ba_params, proper_R_axis, cam_params, pts_3d):
    '''
    Recover the Bundle Adjustment output from 'optimized_params'.
    Output: pts_3d_ba     - (Nx3) array with the optimized N 3D point locations
            cam_params_ba - optimized camera parameters in the bundle adjustment format
            P_ba          - list with the optimized 3x4 projection matrices
    '''
    n_cam, n_pts, n_params = ba_params['n_cam'], ba_params['n_pts'], ba_params['n_params']
    params_in_K = 3 if ba_params['cam_model'] == 'Affine' else 5

    if ba_params['opt_K'] and ba_params['fix_K']:
        K = optimized_params[:params_in_K]
        optimized_params = optimized_params[params_in_K:]
        n_params -= params_in_K
    
    # get 3d points
    if ba_params['opt_X']:
        pts_3d_ba = optimized_params[n_cam * n_params:].reshape((n_pts, 3))
    else:
        pts_3d_ba = pts_3d
    
    # get camera matrices
    cam_params_opt = optimized_params[:n_cam * n_params].reshape((n_cam, n_params))
    if n_params > 0:
        cam_params_ba = np.hstack((cam_params_opt,cam_params[:,n_params:]))
    else:
        cam_params_ba = cam_params
    
    if ba_params['opt_K'] and ba_params['fix_K']:
        cam_params_ba[:, -params_in_K:] = np.repeat(np.array([K]), cam_params_ba.shape[0], axis=0)
    
    P_ba = []
    for i in range(n_cam):
        P_ba.append(ba_cam_params_to_P(cam_params_ba[i,:], proper_R_axis[i], ba_params['cam_model'], ba_params['R_params']))
        
    return pts_3d_ba, cam_params_ba, P_ba

def get_predefined_pairs(fname):
    '''
    For the IARPA experiments in 'Bundle Adjustment for 3D Reconstruction from Multi-Date Satellite Images' (april 2019)
    
    reads pairs from 'iarpa_oracle_pairs.txt' and 'iarpa_sift_pairs.txt'
    '''
    
    pairs = []
    with open(fname) as f:
        for i in range(50):
            current_str = f.readline()
            a = [int(s) for s in current_str.split() if s.isdigit()]
            p, q = a[0]-1, a[1]-1
            pairs.append((p,q))
    return pairs

def read_point_cloud_ply(filename):
    '''
    to read a point cloud from a ply file
    the header of the file is expected to be as in the e.g., with vertices coords starting the line after end_header
    
    ply
    format ascii 1.0
    element vertex 541636
    property float x
    property float y
    property float z
    end_header
    
    '''
    
    with open(filename, 'r') as f_in:
        lines = f_in.readlines()
        content = [x.strip() for x in lines]
        n_pts = len(content)-7
        pt_cloud = np.zeros((n_pts,3))
        for i in range(n_pts):
            coords = re.findall(r"[-+]?\d*\.\d+|\d+", content[i+7])
            pt_cloud[i,:] = np.array([float(coords[0]),float(coords[1]),float(coords[2])])
    return pt_cloud

def read_point_cloud_txt(filename):
    '''
    to read a point cloud from a txt file
    where each line has 3 floats representing the x y z coordinates of a 3D point
    '''
    
    with open(filename, 'r') as f_in:
        lines = f_in.readlines()
        content = [x.strip() for x in lines]
        n_pts = len(content)
        pt_cloud = np.zeros((n_pts,3))
        for i in range(n_pts):
            coords = re.findall(r"[-+]?\d*\.\d+|\d+", content[i])
            pt_cloud[i,:] = np.array([float(coords[0]),float(coords[1]),float(coords[2])])
    return pt_cloud

def write_point_cloud_ply(filename, point_cloud, color=np.array([255,255,255])):
    with open(filename, 'w') as f_out:
        n_points = point_cloud.shape[0]
        # write output ply file with the point cloud
        f_out.write('ply\n')
        f_out.write('format ascii 1.0\n')
        f_out.write('element vertex {}\n'.format(n_points))
        f_out.write('property float x\nproperty float y\nproperty float z\n')
        if not (color[0] == 255 and color[1] == 255 and color[2] == 255):
            f_out.write('property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n')
            f_out.write('element face 0\nproperty list uchar int vertex_indices\n')
        f_out.write('end_header\n')
        # write 3d points
        for i in range(n_points):
            p_3d = point_cloud[i,:]
            f_out.write('{} {} {}'.format(p_3d[0], p_3d[1], p_3d[2]))
            if not (color[0] == 255 and color[1] == 255 and color[2] == 255):
                f_out.write(' {} {} {} 255'.format(color[0], color[1], color[2]))
            f_out.write('\n')

def write_ply_cam(input_P, crop, filename, s=100.):
    
    h, w = crop['crop'].shape
    with open(filename, 'w') as f_out:
        f_out.write('ply\n')
        f_out.write('format ascii 1.0\n')
        f_out.write('element vertex 5\n')
        f_out.write('property float x\n')
        f_out.write('property float y\n')
        f_out.write('property float z\n')
        f_out.write('element edge 8\n')
        f_out.write('property int vertex1\n')
        f_out.write('property int vertex2\n')
        f_out.write('end_header\n')

        K, R, t, oC = decompose_projection_matrix(input_P)
        KRinv = np.linalg.inv(K @ R)
             
        p4 = oC + KRinv @ np.array([-w/2 * s, -h/2 * s,1]).T
        p3 = oC + KRinv @ np.array([ w/2 * s, -h/2 * s,1]).T
        p2 = oC + KRinv @ np.array([ w/2 * s,  h/2 * s,1]).T
        p1 = oC + KRinv @ np.array([-w/2 * s,  h/2 * s,1]).T
        p5 = oC
        
        # write 3d points
        f_out.write('{} {} {}\n'.format(p1[0] - 2611000, p1[1] - 4322000, p1[2] - 3506000))
        f_out.write('{} {} {}\n'.format(p2[0] - 2611000, p2[1] - 4322000, p2[2] - 3506000))
        f_out.write('{} {} {}\n'.format(p3[0] - 2611000, p3[1] - 4322000, p3[2] - 3506000))
        f_out.write('{} {} {}\n'.format(p4[0] - 2611000, p4[1] - 4322000, p4[2] - 3506000))
        f_out.write('{} {} {}\n'.format(p5[0] - 2611000, p5[1] - 4322000, p5[2] - 3506000))

        # write edges
        f_out.write('0 1\n1 2\n2 3\n3 0\n0 4\n1 4\n2 4\n3 4')       

import rasterio

def save_rpc(rpc, filename):
    with open(filename, 'w') as f_out:
        fout.write('LINE_OFF: {}\nSAMPLE_OFF: {}\n'.format(rpc.linOff, rpc.colOff))
        fout.write('LAT_OFF: {}\nLONG_OFF: {}\nHEIGHT_OFF{}\n'.format(rpc.latOff, rpc.lonOff, rpc.altOff))
        fout.write('LINE_SCALE: {}\nSAMPLE_SCALE: {}\n'.format(rpc.linScale, rpc.colScale))
        fout.write('LAT_SCALE: {}\nLONG_SCALE: {}\nHEIGHT_SCALE{}\n'.format(rpc.latScale, rpc.lonScale, rpc.altScale))
        for n in range(1,21):
            fout.write('LINE_NUM_COEFF_{}: {}\n'.format(n, rpc.inverseLinNum[n]))
        for n in range(1,21):
            fout.write('LINE_DEN_COEFF_{}: {}\n'.format(n, rpc.inverseLinDen[n]))
        for n in range(1,21):
            fout.write('SAMP_NUM_COEFF_{}: {}\n'.format(n, rpc.inverseColNum[n]))
        for n in range(1,21):
            fout.write('SAMP_DEN_COEFF_{}: {}\n'.format(n, rpc.inverseColDen[n]))

def save_geotiff(filename, input_im, epsg_code, x, y, r=0.5):
    # (x,y) - geographic coordinates of the top left pixel
    # r - pixel resolution in meters
    # code epsg (e.g. buenos aires is in the UTM zone 21 south - epsg: 32721)
    h, w = input_im.shape
    profile = {'driver': 'GTiff',
               'count':  1,
               'width':  w,
               'height': h,
               'dtype': rasterio.dtypes.float64,
               'crs': 'epsg:32614',  # UTM zone 14 north
               'transform': rasterio.transform.from_origin(x - r / 2, y + r / 2, r, r)}
    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(np.asarray([input_im]))
        
def latlon_to_ecef_custom(lat, lon, alt):
    '''
    to convert from geodetic (lat, lon, alt) to geocentric coordinates (x, y, z)
    '''
    rad_lat = lat * (np.pi / 180.0)
    rad_lon = lon * (np.pi / 180.0)

    a = 6378137.0
    finv = 298.257223563
    f = 1 / finv
    e2 = 1 - (1 - f) * (1 - f)
    v = a / np.sqrt(1 - e2 * np.sin(rad_lat) * np.sin(rad_lat))

    x = (v + alt) * np.cos(rad_lat) * np.cos(rad_lon)
    y = (v + alt) * np.cos(rad_lat) * np.sin(rad_lon)
    z = (v * (1 - e2) + alt) * np.sin(rad_lat)

    return x, y, z

def ecef_to_latlon_custom(x, y, z):
    '''
    to convert from geocentric coordinates (x, y, z) to geodetic (lat, lon, alt)
    '''
    a = 6378137.0
    e = 8.1819190842622e-2

    asq = a ** 2
    esq = e ** 2

    b   = np.sqrt(asq * (1 - esq))
    bsq = b ** 2

    ep  = np.sqrt((asq - bsq)/bsq)
    p   = np.sqrt( (x ** 2) + (y ** 2) )
    th  = np.arctan2(a * z, b * p)

    lon = np.arctan2(y, x)
    lat = np.arctan2( (z + (ep ** 2) * b * (np.sin(th) ** 3) ), (p - esq * a * (np.cos(th) ** 3) ) )
    N = a / ( np.sqrt(1 - esq * (np.sin(lat) ** 2) ) )
    alt = p / np.cos(lat) - N 

    lon = lon * 180 / np.pi
    lat = lat * 180 / np.pi

    return lat, lon, alt

def ecef_to_latlon_custom_ad(x, y, z):
    # the 'ad' package is unable to differentiate numpy trigonometry functions (sin, tan, etc.)
    # also, 'ad.admath' can't handle lists/arrays, so x, y, z are expected to be floats here
    from ad import admath as math
    a = 6378137.0
    e = 8.1819190842622e-2

    asq = a ** 2
    esq = e ** 2

    b   = math.sqrt(asq * (1 - esq))
    bsq = b ** 2

    ep  = math.sqrt((asq - bsq)/bsq)
    p   = math.sqrt( (x ** 2) + (y ** 2) )
    th  = math.atan2(a * z, b * p)

    lon = math.atan2(y, x)
    lat = math.atan2( (z + (ep ** 2) * b * (math.sin(th) ** 3) ), (p - esq * a * (math.cos(th) ** 3) ) )
    N = a / ( math.sqrt(1 - esq * (math.sin(lat) ** 2) ) )
    alt = p / math.cos(lat) - N 

    lon = lon * 180 / math.pi
    lat = lat * 180 / math.pi

    return lat, lon, alt

def plot_connectivity_graph(C, thr_matches, save_pgf=False):
    # (1) Build connectivity matrix A, where position (i,j) contains the number of matches between images i and j
    n_cam = int(C.shape[0]/2)
    A, n_correspondences_filt, tmp_pairs = np.zeros((n_cam,n_cam)), [], []
    for im1 in range(n_cam):
        for im2 in range(im1+1,n_cam):
            obs_im1 = 1*np.invert(np.isnan(C[2*im1,:]))
            obs_im2 = 1*np.invert(np.isnan(C[2*im2,:]))
            n_matches = np.sum(np.sum(np.vstack((obs_im1, obs_im2)), axis=0) == 2)
            n_correspondences_filt.append(n_matches)
            tmp_pairs.append((im1,im2))
            A[im1,im2] = n_matches
            A[im2,im1] = n_matches
 
    # (2) Filter graph edges according to the threshold on the number of matches
    pairs_to_draw = []
    for i in range(len(tmp_pairs)):
        if n_correspondences_filt[i] > thr_matches:
            pairs_to_draw.append(tmp_pairs[i])

    # (3) Draw the graph and save it as a .pgf image
    import networkx as nx
    
    if save_pgf:
        fig_width_pt = 229.5 # CVPR
        inches_per_pt = 1.0/72.27               # Convert pt to inches
        golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt  # width in inches
        fig_height =fig_width*golden_mean       # height in inches
        fig_size = [fig_width,fig_height]
        params = {'backend': 'pgf', 'axes.labelsize': 8, 'font.size': 8, 'legend.fontsize': 8,
                  'xtick.labelsize': 7, 'ytick.labelsize': 8, 'text.usetex': True, 'figure.figsize': fig_size}
        plt.rcParams.update(params)
    
    fig = plt.gcf()
    fig.set_size_inches(15, 15)

    # create networkx graph
    G=nx.Graph()

    # add edges
    for edge in pairs_to_draw:
        G.add_edge(edge[0], edge[1])
    
    # draw all nodes in a circle
    G_pos = nx.circular_layout(G)
    
    # draw nodes
    nx.draw_networkx_nodes(G, G_pos, node_size=600, node_color='#FFFFFF', edgecolors='#000000')
    
    # paint subgroup of nodes
    nx.draw_networkx_nodes(G, G_pos, nodelist=[41,42, 43, 44, 45], node_size=600, node_color='#FF6161',edgecolors='#000000')
    
    # draw edges and labels
    nx.draw_networkx_edges(G, G_pos)
    nx.draw_networkx_labels(G, G_pos, font_size=12, font_family='sans-serif')
    
    # get list of connected components (to see if there is any disconnected subgroup)
    G_cc = list(nx.connected_component_subgraphs(G))
    if len(G_cc) > 1:
        print('Attention! Graph G contains {} connected components'.format(len(G_cc)))
    
    # show graph and save it as .pgf
    plt.axis('off')
    if save_pgf:
        plt.savefig('graph.pgf', pad_inches=0, bbox_inches='tight', dpi=200)
    plt.show()

    return A