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
import os
from PIL import Image
import srtm4
import rpcm
from shapely.geometry import Polygon, mapping, shape
import geojson

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

def project(pts, cam_params, cam_model, K):
    """
    Convert 3D points to 2D by projecting onto images
    """

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


def fun(params, cam_ind, pts_ind, pts_2d, cam_params, pts_3d, ba_params, pts_2d_w, fix_1st_cam=False):
    """
    Compute Bundle Adjustment residuals.
    'params' contains those parameters to be optimized (3D points + camera paramters)
    """
    n_cam_opt, n_cam_fix, n_cam = ba_params['n_cam_opt'], ba_params['n_cam_fix'], ba_params['n_cam']
    cam_model, n_params = ba_params['cam_model'], ba_params['n_params']
    n_pts = ba_params['n_pts']

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

    # get 3d points to optimize (or not)
    if ba_params['opt_X']:
        pts_3d_ba = params[n_cam * n_params:].reshape((n_pts, 3))
    else:
        pts_3d_ba = pts_3d

    # get camera params to optimize
    cam_params_opt = np.vstack((cam_params[:n_cam_fix,:n_params], #fixed cameras are at the first rows if any
                                params[n_cam_fix * n_params : n_cam * n_params].reshape((n_cam_opt, n_params))))
    
    # add fixed camera params
    cam_params_ba = np.hstack((cam_params_opt,cam_params[:,n_params:]))      
    
    # project 3d points using the current camera params
    points_proj = project(pts_3d_ba[pts_ind], cam_params_ba[cam_ind], cam_model, K)
    
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
   
    #Preconstructed = np.vstack( (np.hstack((K @ R[:2,:], vecT.T)), np.array([[0,0,0,1]])) )
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

def ba_cam_params_to_P(cam_params, cam_model):
    '''
    Recover the 3x4 projection matrix P from the camera parameters format used by the bundle adjustment
    '''
    
    if cam_model == 'Affine':
        vecR, vecT, fx, fy, skew = cam_params[0:3], cam_params[3:5], cam_params[5], cam_params[6], cam_params[7]
        K = np.array([[fx, skew], [0., fy]])
        R = euler_angles_to_R(vecR)
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
        vecR = euler_angles_from_R(R)
        u, s, vh = np.linalg.svd(R, full_matrices=False)
        fx, fy, skew = K[0,0], K[1,1], K[0,1]
        cam_params = np.hstack((vecR.ravel(),vecT.ravel(),fx,fy,skew))
    else:
        K, R, vecT, _ = decompose_projection_matrix(P)
        K = K/K[2,2]
        vecR = euler_angles_from_R(R)
        fx, fy, skew, cx, cy = K[0,0], K[1,1], K[0,1], K[0,2], K[1,2]
        cam_params = np.hstack((vecR.ravel(),vecT.ravel(),fx,fy,skew,cx,cy))
    return cam_params

def find_SIFT_kp(im, enforce_large_size = False, min_kp_size = 3., max_kp = np.inf):
    '''
    Detect SIFT keypoints in an input image
    Requirement: pip3 install opencv-contrib-python==3.4.0.12
    '''
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(im,None)
    
    # reduce number of keypoints if there are more than allowed
    if len(kp) > max_kp:
        prev_idx = np.arange(len(kp))
        new_idx = np.random.choice(prev_idx,max_kp_per_im,replace=False)
        kp, des = kp[new_idx], des[new_idx]
    
    # pick only keypoints from the first scale (satellite imagery)
    if enforce_large_size:
        kp, des = np.array(kp), np.array(des) 
        large_size_indices = np.array([current_kp.size > min_kp_size for current_kp in kp])
        kp, des = kp[large_size_indices].tolist(), des[large_size_indices].tolist()
    
    pts = np.array([kp[idx].pt for idx in range(len(kp))])
    
    return kp, des, pts

def match_pair(kp1, kp2, des1, des2, dist_thresold=0.6):
    '''
    Match SIFT keypoints from an stereo pair
    '''
    
    # bruteforce matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)
    
    # FLANN parameters
    # from https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
    #FLANN_INDEX_KDTREE = 1
    #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #search_params = dict(checks=50)   # or pass empty dictionary
    #flann = cv2.FlannBasedMatcher(index_params,search_params)
    #matches = flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)
    
    
    # Apply ratio test as in Lowe's paper
    pts1, pts2, idx_matched_kp1, idx_matched_kp2 = [], [], [], []
    for m,n in matches:
        if m.distance < dist_thresold*n.distance:
            pts2.append(kp2[m.trainIdx])
            pts1.append(kp1[m.queryIdx])
            idx_matched_kp1.append(m.queryIdx)
            idx_matched_kp2.append(m.trainIdx)
            
    # Geometric filtering using the Fundamental matrix
    pts1, pts2 = np.array(pts1), np.array(pts2)
    idx_matched_kp1, idx_matched_kp2 = np.array(idx_matched_kp1), np.array(idx_matched_kp2)
    if pts1.shape[0] > 0 and pts2.shape[0] > 0:
        F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

        # We select only inlier points
        if mask is None:
            # no matches after geometric filtering
            idx_matched_kp1, idx_matched_kp2 = None, None
        else:
            mask_bool = mask.ravel()==1
            idx_matched_kp1, idx_matched_kp2 = idx_matched_kp1[mask_bool], idx_matched_kp2[mask_bool]
    else:
        # no matches were after ratio test
        idx_matched_kp1, idx_matched_kp2 = None, None
    
    return idx_matched_kp1, idx_matched_kp2
  
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

def initialize_3d_points(P_crop, C, pairs_to_triangulate, cam_model):
    '''
    Initialize the 3D point corresponding to each feature track.
    How? Pick the average value of all possible triangulated points within each track.
    Additionally, compute the sum of variances in each dimension for each set of candidates.
    If the total variance is larger than a certain threshold, then remove that 3d point due to low reliability.
    '''
    
    n_pts, n_cam = C.shape[1], int(C.shape[0]/2) 
    pts_3d = np.zeros((n_pts,3))
    
    verbose = True if n_pts > 10000 else False
    
    
    true_where_track = np.invert(np.isnan(C[np.arange(0, C.shape[0], 2), :])) #(i,j)=True if j-th point seen in i-th image 
    cam_indices = np.arange(n_cam)
    for track_id in range(n_pts):
        
        if verbose == True:
            print('\rTrack {} / {} done'.format(track_id+1, C.shape[1]), end = '\r')
        
        im_ind = cam_indices[true_where_track[:,track_id]] # cameras where track is observed
        all_pairs = [(im_i, im_j) for im_i in im_ind for im_j in im_ind if im_i != im_j and im_i<im_j]
        good_pairs = [pair for pair in all_pairs if pair in pairs_to_triangulate]
        
        current_track_candidates = []
        for [im1, im2] in good_pairs:
            pt1, pt2, P1, P2 = C[(im1*2):(im1*2+2),track_id], C[(im2*2):(im2*2+2),track_id], P_crop[im1], P_crop[im2]
            if cam_model == 'Affine':
                lon, lat, h, _ = triangulation.triangulation_affine(P1,P2, np.array([pt1[0]]), np.array([pt1[1]]), 
                                                                    np.array([pt2[0]]), np.array([pt2[1]]) )
                candidate_from_pair = np.hstack([lon, lat, h])
            else:
                candidate_from_pair = linear_triangulation_single_pt(pt1,pt2,P1,P2)
                #candidate_from_pair = cv2.triangulatePoints(P1, P2, pt1, pt2)[:,0]
            current_track_candidates.append(candidate_from_pair)
        
        pts_3d[track_id,:] = np.mean(np.array(current_track_candidates), axis=0)
        
    return pts_3d

def remove_outlier_obs(reprojection_err, pts_ind, cam_ind, C, pairs_to_triangulate, thr=1.0):
    '''
    Given the reprojection error associated each feature observation involved in the bundle adjustment process 
    (i.e. ba_output_err), those observations with error larger than 'outlier_thr' are removed
    The correspondence matrix C is updated with the remaining observations
    '''
    
    Cnew = C.copy()
    n_img = int(Cnew.shape[0]/2)
    cont = 0
    for i in range(len(reprojection_err)):
        if reprojection_err[i] > thr:
            cont += 1
            track_where_obs, cam_where_obs = pts_ind[i], cam_ind[i]
            # count number of obs x track (if the track is formed by only one match, then delete it)
            # otherwise delete only that particular observation
            Cnew[2*cam_where_obs, track_where_obs] = np.nan
            Cnew[2*cam_where_obs+1, track_where_obs] = np.nan
    
    # count the updated number of obs per track and keep those tracks with 2 or more observations 
    obs_per_track = np.sum(1*np.invert(np.isnan(Cnew)), axis=0)
    n_old_tracks = Cnew.shape[1]
    Cnew = Cnew[:, obs_per_track >=4]
    
    # remove matches found in pairs with short baseline that were not extended to more images
    # since these columns of C will not be triangulated
    columns_to_preserve = []
    for i in range(Cnew.shape[1]):
        im_ind = [k for k, j in enumerate(range(n_img)) if not np.isnan(Cnew[j*2,i])]
        all_pairs = [(im_i, im_j) for im_i in im_ind for im_j in im_ind if im_i != im_j and im_i<im_j]
        good_pairs = [pair for pair in all_pairs if pair in pairs_to_triangulate]
        if len(good_pairs) == 0:
            cont += len(all_pairs)
        columns_to_preserve.append( len(good_pairs) > 0 )
    Cnew = Cnew[:, columns_to_preserve]
    n_tracks_del = n_old_tracks - Cnew.shape[1]
    
    print('Deleted {} observations ({:.2f}%) and {} tracks ({:.2f}%)\n' \
          .format(cont, cont/pts_ind.shape[0]*100, n_tracks_del, n_tracks_del/n_old_tracks*100))
    
    return Cnew


def set_ba_params(P, C, cam_model, n_cam_fix, n_cam_opt, pairs_to_triangulate):
    '''
    Given a set of input feature tracks (correspondence matrix C) and a set of initial projection matrices (P),
    define the input parameters needed by Bundle Adjustment
    '''
    
    # pick only the points that have to be updated 
    # (i.e. list the columns of C with values different from nan in the rows of the cams to be optimized)
    true_where_track = np.sum(np.invert(np.isnan(C[np.arange(0, C.shape[0], 2), :]))[-n_cam_opt:]*1,axis=0).astype(bool)
    C_new = C[:, true_where_track]
    prev_pts_indices = np.arange(len(true_where_track))[true_where_track]

    # remove cameras that dont need to be adjusted
    obs_per_cam = np.sum(1*np.invert(np.isnan(C_new[np.arange(0, C_new.shape[0], 2), :])), axis=1)
    cams_to_keep = obs_per_cam > 0
    C_new = C_new[np.repeat(cams_to_keep,2),:]
    negative_else_new_idx = np.array([-1] * len(cams_to_keep)) 
    negative_else_new_idx[cams_to_keep] = np.arange(np.sum(cams_to_keep))
    prev_cam_indices = np.arange(len(cams_to_keep))[cams_to_keep]
    P_new = [P[idx] for idx in prev_cam_indices]
    n_cam_fix -= np.sum(np.invert(cams_to_keep[:n_cam_fix])*1)
    n_cam_opt -= np.sum(np.invert(cams_to_keep[-n_cam_opt:])*1)    
    
    # update pairs_to_triangulate with the new indices
    pairs_to_triangulate_new = []
    for [idx_r, idx_l] in pairs_to_triangulate:
        new_idx_r, new_idx_l = negative_else_new_idx[idx_r], negative_else_new_idx[idx_l]
        if new_idx_r >= 0 and new_idx_l >= 0:
            pairs_to_triangulate_new.append((new_idx_r, new_idx_l))
            
    n_cam = len(P_new)
    n_pts = C_new.shape[1]
    
    # other ba parameters
    ba_params = {
    'cam_model'        : cam_model,
    'n_cam_fix'        : n_cam_fix,
    'n_cam_opt'        : n_cam_opt,
    'n_cam'            : n_cam_fix + n_cam_opt,
    'n_pts'            : n_pts,
    'n_params'         : 0,
    'opt_X'            : True,
    'opt_R'            : True,
    'opt_T'            : False,
    'opt_K'            : False,
    'fix_K'            : False,
    'prev_cam_indices' : prev_cam_indices,
    'input_P'          : np.array(P),
    }
    
    #ue +=1
    
    # (1) init 3d points   
    pts_3d = initialize_3d_points(P_new, C_new, pairs_to_triangulate_new, cam_model)
    
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
    pts_ind, cam_ind, pts_2d = np.array(point_ind), np.array(camera_ind), np.vstack(points_2d)
    
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
    
    print('{} cameras in total, {} fixed and {} to be adjusted'.format(n_cam, n_cam_fix, n_cam_opt))
    print('{} parameters per camera and {} 3d points to be optimized'.format(n_params, n_pts))
    
    if ba_params['opt_K'] and ba_params['fix_K']:
        params_in_K = 3 if cam_model == 'Affine' else 5
        K = cam_params_opt[0,-params_in_K:]
        cam_params_opt2 = np.hstack([cam_params_opt[cam_id, :-params_in_K] for cam_id in range(n_cam_opt)])
        cam_params_opt = np.hstack((K, cam_params_opt2))

    pts_3d_opt = pts_3d.copy()
    params_opt = np.hstack((cam_params_opt.ravel(), pts_3d_opt.ravel()))
        
    return params_opt, cam_params, pts_3d, pts_2d, cam_ind, pts_ind, ba_params

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
    
    P_ba = (ba_params['input_P'].copy()).tolist()
    for (idx, it) in zip(ba_params['prev_cam_indices'], range(cam_params_ba.shape[0])):
        P_ba[idx] = ba_cam_params_to_P(cam_params_ba[it,:], ba_params['cam_model'])
    
    return pts_3d_ba, cam_params_ba, P_ba

def get_predefined_pairs(fname, site, order, myimages):
    pairs = []
    with open(fname) as f:
        if order in ['oracle', 'sift']:
            for i in range(50):
                current_str = f.readline()
                a = [int(s) for s in current_str.split() if s.isdigit()]
                p, q = a[0]-1, a[1]-1
                pairs.append((p,q))
        else:
            # reads pairs from the heuristics order
            myimages_fn = [os.path.basename(i) for i in myimages]
            if site == 'IARPA':
                while len(pairs) < 50:
                    current_str = f.readline().split(' ')
                    im1_fn, im2_fn = os.path.basename(current_str[0]), os.path.basename(current_str[1])
                    if im1_fn in myimages_fn and im2_fn in myimages_fn:
                        p, q = myimages_fn.index(im1_fn), myimages_fn.index(im2_fn)
                        pairs.append((p,q))
            else:
                while len(pairs) < 50:
                    current_str = f.readline().split(' ')
                    im1_fn, im2_fn = os.path.basename(current_str[0]+'.tif'), os.path.basename(current_str[1]+'.tif')
                    if im1_fn in myimages_fn and im2_fn in myimages_fn:
                        p, q = myimages_fn.index(im1_fn), myimages_fn.index(im2_fn)
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
    #nx.draw_networkx_nodes(G, G_pos, nodelist=[41,42, 43, 44, 45], node_size=600, node_color='#FF6161',edgecolors='#000000')
    
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

def plot_dsm(fname, vmin=None, vmax=None, color_bar='jet', save_pgf=False):
    
    f_size = (13,10)
    fig = plt.figure()
    params = {'backend': 'pgf',
              'axes.labelsize': 22,
              'ytick.labelleft': False,
              'font.size': 22,
              'legend.fontsize': 22,
              'xtick.labelsize': 22,
              'ytick.labelsize': 22,
              'xtick.top': False,
              'xtick.bottom': False,
              'xtick.labelbottom': False,
              'ytick.left': False,
              'ytick.right': False,   
              'text.usetex': True, # use TeX for text
              'font.family': 'serif',
              'legend.loc': 'upper left',
              'legend.fontsize': 22}
    plt.rcParams.update(params)
    plt.figure(figsize=f_size)
    
    im = np.array(Image.open(fname))
    vmin_in = np.min(im.squeeze()) if vmin is None else vmin
    vmax_in = np.max(im.squeeze()) if vmax is None else vmin
    plt.imshow(im.squeeze(), cmap=color_bar, vmin=vmin, vmax=vmax)
    plt.axis('equal')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=40)
    if save_pgf:
        plt.savefig(os.path.splitext(fname) + '.pgf', pad_inches=0, bbox_inches='tight', dpi=200)
    plt.show()

def get_elbow_value(init_e, percentile_value):

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
    
    return elbow_value


def match_kp_within_intersection_polygon(kp_i, kp_j, des_i, des_j, pair):
       
    east_i, north_i, east_j, north_j = kp_i[:,0], kp_i[:,1], kp_j[:,0], kp_j[:,1]
    
    # get instersection polygon utm coords
    east_poly, north_poly = pair['intersection_poly'].exterior.coords.xy
    east_poly, north_poly = np.array(east_poly), np.array(north_poly)
        
    # get centroid of the intersection polygon in utm coords
    #east_centroid, north_centroid = pair['intersection_poly'].centroid.coords.xy # centroid = baricenter ?
    #east_centroid, north_centroid = np.array(east_centroid), np.array(north_centroid)    
    #centroid_utm = np.array([east_centroid[0], north_centroid[0]])
    
    # use the rectangle containing the intersection polygon as AOI 
    min_east, max_east, min_north, max_north = min(east_poly), max(east_poly), min(north_poly), max(north_poly)
    
    east_ok_i = np.logical_and(east_i > min_east, east_i < max_east)
    north_ok_i = np.logical_and(north_i > min_north, north_i < max_north)
    indices_i_poly_bool, all_indices_i = np.logical_and(east_ok_i, north_ok_i), np.arange(kp_i.shape[0])
    indices_i_poly_int = all_indices_i[indices_i_poly_bool]
    
    if not any(indices_i_poly_bool):
        return [], []
    
    east_ok_j = np.logical_and(east_j > min_east, east_j < max_east)
    north_ok_j = np.logical_and(north_j > min_north, north_j < max_north)
    indices_j_poly_bool, all_indices_j = np.logical_and(east_ok_j, north_ok_j), np.arange(kp_j.shape[0])
    indices_j_poly_int = all_indices_j[indices_j_poly_bool]
    
    if not any(indices_j_poly_bool):
        return [], []
    
    # pick kp in overlap area and the descriptors
    kp_i_poly, des_i_poly = kp_i[indices_i_poly_bool], des_i[indices_i_poly_bool] 
    kp_j_poly, des_j_poly = kp_j[indices_j_poly_bool], des_j[indices_j_poly_bool]   
    
    '''
    import matplotlib.patches as patches
    fig, ax = plt.subplots(figsize=(10,6))
    plt.scatter(kp_i[:,0], kp_i[:,1], c='b')
    plt.scatter(kp_j[:,0], kp_j[:,1], c='r')
    plt.scatter(kp_i[indices_i_poly_int,0], kp_i[indices_i_poly_int,1], c='g')
    plt.scatter(kp_j[indices_j_poly_int,0], kp_j[indices_j_poly_int,1], c='g')
    rect = patches.Rectangle((min_east,min_north),max_east-min_east, max_north-min_north, facecolor='g', alpha=0.4)
    #plt.scatter(east_centroid, north_centroid, s=100, c='g')
    #plt.scatter(east_poly, north_poly, s=100, c='g')
    ax.add_patch(rect)
    plt.show()   
    '''
    
    indices_m_kp_i_poly, indices_m_kp_j_poly = match_pair(kp_i_poly, kp_j_poly, des_i_poly, des_j_poly)
    
    # go back from the filtered indices inside the polygon to the original indices of all the kps in the image
    if indices_m_kp_i_poly is None:
        indices_m_kp_i, indices_m_kp_j = [], []
    else:
        indices_m_kp_i, indices_m_kp_j = indices_i_poly_int[indices_m_kp_i_poly], indices_j_poly_int[indices_m_kp_j_poly]

    return indices_m_kp_i, indices_m_kp_j

def display_ba_error_particular_view(P_before, P_after, pts3d_before, pts3d_after, pts2d, image):

    n_pts = pts3d_before.shape[0]

    # reprojections before bundle adjustment
    proj = P_before @ np.hstack((pts3d_before, np.ones((n_pts,1)))).T
    pts_reproj_before = (proj[:2,:]/proj[-1,:]).T

    # reprojections after bundle adjustment
    proj = P_after @ np.hstack((pts3d_after, np.ones((n_pts,1)))).T
    pts_reproj_after = (proj[:2,:]/proj[-1,:]).T

    err_before = np.sum(abs(pts_reproj_before - pts2d), axis=1)
    err_after = np.sum(abs(pts_reproj_after - pts2d), axis=1)

    print('Mean abs reproj error before BA: {:.4f}'.format(np.mean(err_before)))
    print('Mean abs reproj error after  BA: {:.4f}'.format(np.mean(err_after)))

    # reprojection error histograms for the selected image
    fig = plt.figure(figsize=(10,3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text('Reprojection error before BA')
    ax2.title.set_text('Reprojection error after  BA')
    ax1.hist(err_before, bins=40); 
    ax2.hist(err_after, bins=40);

    plt.show()

    # warning: this is slow...

    fig = plt.figure(figsize=(20,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text('Before BA')
    ax2.title.set_text('After  BA')
    ax1.imshow(image, cmap="gray")
    ax2.imshow(image, cmap="gray")
    for k in range(min(1000,n_pts)):
        # before bundle adjustment
        ax1.plot([pts2d[k,0], pts_reproj_before[k,0] ], [pts2d[k,1], pts_reproj_before[k,1]], 'r-')
        ax1.plot(*pts2d[k], 'yx')
        # after bundle adjustment
        ax2.plot([pts2d[k,0], pts_reproj_after[k,0] ], [pts2d[k,1], pts_reproj_after[k,1]], 'r-')
        ax2.plot(*pts2d[k], 'yx')
    plt.show()

def feature_detection(input_seq, enforce_large_size = False, min_kp_size = 3.):
    # finds SIFT keypoints in a sequence of grayscale images
    # saves the keypoints coordinates, the descriptors, and assigns a unique id to each keypoint that is found
    kp_cont, n_img = 0, len(input_seq)
    features, all_keypoints, all_vertices = [], [], []
    for idx in range(n_img):
        kp, des, pts = find_SIFT_kp(input_seq[idx], enforce_large_size, min_kp_size)
        kp_id = np.arange(kp_cont, kp_cont + pts.shape[0]).tolist()
        features.append({ 'kp': pts, 'des': np.array(des), 'id': np.array(kp_id) })
        all_keypoints.extend(pts.tolist())
        tmp = np.vstack((np.ones(pts.shape[0]).astype(int)*idx, kp_id)).T
        all_vertices.extend( tmp.tolist() )
        print('Found', pts.shape[0], 'keypoints in image', idx)
        kp_cont += pts.shape[0]
        #im_kp=cv2.drawKeypoints(input_seq[idx],kp,outImage=np.array([]))
        #vistools.display_image(im_kp) 
    return features
    
def feature_detection_skysat(input_seq, input_rpcs, footprints):
    features = feature_detection(input_seq, enforce_large_size = True, min_kp_size = 4.)
    for idx, features_current_im in enumerate(features):
        # convert im coords to utm coords
        pts = features[idx]['kp']
        cols, rows, alts = pts[:,0].tolist(), pts[:,1].tolist(), [footprints[idx]['z']] * pts.shape[0]
        lon, lat = input_rpcs[idx].localization(cols, rows, alts)
        east, north = utils.utm_from_lonlat(lon, lat)
        features[idx]['kp_utm'] = np.vstack((east, north)).T
    return features
 
def footprint_from_rpc_file(rpc, w, h):
    z = srtm4.srtm4(rpc.lon_offset, rpc.lat_offset)
    lons, lats = rpc.localization([0, 0, w, w, 0], [0, h, h, 0, 0], [z, z, z, z, z])
    return geojson.Feature(geometry=geojson.Polygon([list(zip(lons, lats))]))

def get_image_footprints(myrpcs, crops):
    footprints = []
    for current_rpc, current_im, iter_cont in zip(myrpcs, crops, range(len(myrpcs))):
        z_footprint = srtm4.srtm4(current_rpc.lon_offset, current_rpc.lat_offset)
        this_footprint = footprint_from_rpc_file(current_rpc, current_im.shape[1], current_im.shape[0])['geometry']
        this_footprint_lon = np.array(this_footprint["coordinates"][0])[:,0]
        this_footprint_lat = np.array(this_footprint["coordinates"][0])[:,1]
        this_footprint_east, this_footprint_north = utils.utm_from_lonlat(this_footprint_lon, this_footprint_lat)
        this_footprint_utm = np.vstack((this_footprint_east, this_footprint_north)).T
        this_footprint["coordinates"] = [this_footprint_utm.tolist()]
        footprints.append({'poly': shape(this_footprint), 'z': z_footprint})
        #print('\r{} / {} done'.format(iter_cont+1, len(P_crop)), end = '\r')
    return footprints

def filter_pairs_to_match_skysat(init_pairs, footprints, projection_matrices):

    # get optical centers and footprints
    optical_centers, n_img = [], len(footprints)
    for current_P in projection_matrices:
        _, _, _, current_optical_center = decompose_projection_matrix(current_P)
        optical_centers.append(current_optical_center)
        
    pairs_to_match, pairs_to_triangulate = [], []
    for (i, j) in init_pairs:
        
            # check if the baseline between both cameras is large enough
            baseline = np.linalg.norm(optical_centers[i] - optical_centers[j])
            baseline_ok = baseline > 150000
            
            # check there is enough overlap between the images (at least 10% w.r.t image 1)
            intersection_polygon = footprints[i]['poly'].intersection(footprints[j]['poly']) 
            overlap_ok = intersection_polygon.area/footprints[i]['poly'].area >= 0.1
            
            if overlap_ok:    
                pairs_to_match.append({'im_i' : i, 'im_j' : j,
                       'footprint_i' : footprints[i], 'footprint_j' : footprints[j],
                       'baseline' : baseline, 'intersection_poly': intersection_polygon})
                
                if baseline_ok:
                    pairs_to_triangulate.append((i,j))
                    
    print('{} / {} pairs to be matched'.format(len(pairs_to_match),int((n_img*(n_img-1))/2)))  
    return pairs_to_match, pairs_to_triangulate

def matching_skysat(pairs_to_match, features):

    all_pairwise_matches = []
    for idx, pair in enumerate(pairs_to_match):
        i, j = pair['im_i'], pair['im_j']  
        kp_i, des_i, kp_i_id = features[i]['kp'], features[i]['des'], features[i]['id']
        kp_j, des_j, kp_j_id = features[j]['kp'], features[j]['des'], features[j]['id']
        kp_i_utm, kp_j_utm = features[i]['kp_utm'], features[j]['kp_utm']
        
        # pick only those keypoints within the intersection area
        indices_m_kp_i, indices_m_kp_j = match_kp_within_intersection_polygon(kp_i_utm, kp_j_utm, des_i, des_j, pair)
        n_matches = 0 if indices_m_kp_i is None else len(indices_m_kp_i)
        print('Pair ({},{}) -> {} matches'.format(i,j,n_matches))

        if indices_m_kp_i is not None:
            matches_i_j = np.vstack((kp_i_id[indices_m_kp_i], kp_j_id[indices_m_kp_j])).T
            all_pairwise_matches.extend(matches_i_j.tolist())

    return all_pairwise_matches

def feature_tracks_from_pairwise_matches(features, pairwise_matches, pairs_to_triangulate):

    # prepreate data to build correspondence matrix
    keypoints_coord, keypoints_im_id, n_cam = [], [], len(features)
    for im_id, features_current_im in enumerate(features):
        pts = features_current_im['kp']
        keypoints_coord.extend(pts.tolist())
        keypoints_im_id.extend(np.ones(pts.shape[0]).astype(int)*im_id)
    
    def find(parents, feature_id):
        p = parents[feature_id]
        return feature_id if not p else find(parents, p)

    def union(parents, feature_i_idx, feature_j_idx):
        p_1, p_2 = find(parents, feature_i_idx), find(parents, feature_j_idx)
        if p_1 != p_2: 
            parents[p_1] = p_2

    parents = [None]*(len(keypoints_im_id))
    for feature_i_idx, feature_j_idx in pairwise_matches:
        union(parents, feature_i_idx, feature_j_idx)

    # handle parents without None
    parents = [find(parents, feature_id) for feature_id, v in enumerate(parents)]
    
    # parents = track_id
    _, parents_indices, parents_counts = np.unique(parents, return_inverse=True, return_counts=True)
    n_tracks = np.sum(1*(parents_counts>1))
    track_parents = np.array(parents)[parents_counts[parents_indices] > 1]
    _, track_idx_from_parent, _ = np.unique(track_parents, return_inverse=True, return_counts=True)
    
    # t_idx, parent_id
    track_indices = np.zeros(len(parents))
    track_indices[:] = np.nan
    track_indices[parents_counts[parents_indices] > 1] = track_idx_from_parent
        
    # build correspondence matrix
    C = np.zeros((2*n_cam, n_tracks))
    C[:] = np.nan
    for (feature_i_id, feature_j_id) in pairwise_matches:
        t_idx, t_idx2 = int(track_indices[feature_i_id]), int(track_indices[feature_j_id])
        im_id_i, im_id_j = keypoints_im_id[feature_i_id], keypoints_im_id[feature_j_id]
        C[(2*im_id_i):(2*im_id_i+2), t_idx] = np.array(keypoints_coord[feature_i_id])
        C[(2*im_id_j):(2*im_id_j+2), t_idx] = np.array(keypoints_coord[feature_j_id])
    
    # remove matches found in pairs with short baseline that were not extended to more images
    # since these columns of C will not be triangulated
    columns_to_preserve = []
    for i in range(C.shape[1]):
        im_ind = [k for k, j in enumerate(range(n_cam)) if not np.isnan(C[j*2,i])]
        all_pairs = [(im_i, im_j) for im_i in im_ind for im_j in im_ind if im_i != im_j and im_i<im_j]
        good_pairs = [pair for pair in all_pairs if pair in pairs_to_triangulate]
        columns_to_preserve.append( len(good_pairs) > 0 )
    C = C[:, columns_to_preserve]
    
    print('Found {} tracks in total'.format(C.shape[1]))
    return C
    
def check_ba_error(error_before, error_after, pts_2d_w):

    des_norm = np.repeat(pts_2d_w,2, axis=0)

    init_e = np.add.reduceat(abs(error_before.astype(float)/des_norm), np.arange(0, len(error_before), 2))
    init_e_mean = np.mean(init_e)
    init_e_median = np.median(init_e)

    ba_e = np.add.reduceat(abs(error_after.astype(float)/des_norm), np.arange(0, len(error_after), 2))
    ba_e_mean = np.mean(ba_e)
    ba_e_median = np.median(ba_e)

    _,f = plt.subplots(1, 2, figsize=(10,3))
    f[0].hist(init_e, bins=40);
    f[1].hist(ba_e, bins=40); 
    
    print('Error before BA (mean / median): {:.2f} / {:.2f}'.format(init_e_mean, init_e_median))
    print('Error after  BA (mean / median): {:.2f} / {:.2f}'.format(ba_e_mean, ba_e_median))
    return ba_e
    
def rescale_P(input_P, alpha):
    alpha = float(alpha)
    return np.array([[alpha, 0., 0.],[0., alpha, 0.],[0., 0., 1.]]) @ input_P

def rescale_RPC(input_rpc, alpha):
    alpha = float(alpha)
    input_rpc.row_offset *= alpha
    input_rpc.col_offset *= alpha
    input_rpc.row_scale *= alpha
    input_rpc.row_scale *= alpha
    return input_rpc

def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = np.arctan2(t3, t4)
    return X, Y, Z

def quaternion_to_R(q0, q1, q2, q3):
    """Convert a quaternion into rotation matrix form.
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    matrix = np.zeros((3,3))
    matrix[0, 0] = q0**2 + q1**2 - q2**2 - q3**2
    matrix[1, 1] = q0**2 - q1**2 + q2**2 - q3**2
    matrix[2, 2] = q0**2 - q1**2 - q2**2 + q3**2
    matrix[0, 1] = 2.0 * (q1*q2 - q0*q3)
    matrix[0, 2] = 2.0 * (q0*q2 + q1*q3)
    matrix[1, 2] = 2.0 * (q2*q3 - q0*q1)
    matrix[1, 0] = 2.0 * (q1*q2 + q0*q3)
    matrix[2, 0] = 2.0 * (q1*q3 - q0*q2)
    matrix[2, 1] = 2.0 * (q0*q1 + q2*q3)  
    return matrix
