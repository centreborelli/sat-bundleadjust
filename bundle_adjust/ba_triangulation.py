import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

from IS18 import utils
from IS18 import triangulation
from IS18 import rectification
from IS18 import stereo
from IS18 import vistools

from bundle_adjust import ba_core
from bundle_adjust import ba_utils

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
    K, R, vecT, C1 = ba_core.decompose_perspective_camera(P1)
    dir_vec_ray1 = np.linalg.inv(K @ R) @ np.expand_dims(np.hstack((pt1, np.ones(1))), axis=1)
    K, R, vecT, C2 = ba_core.decompose_perspective_camera(P2)
    dir_vec_ray2 = np.linalg.inv(K @ R) @ np.expand_dims(np.hstack((pt2, np.ones(1))), axis=1)
    n = np.cross(dir_vec_ray1, dir_vec_ray2, axis=0)
    d = np.dot((C2 - C1), n) / np.linalg.norm(n)
    return abs(d[0])

def initialize_3d_points(P_crop, C, pairs_to_triangulate, cam_model):
    '''
    Initialize the 3D point corresponding to each feature track.
    How? Pick the average value of all possible triangulated points within each track.
    Additionally, compute the sum of variances in each dimension for each set of candidates.
    If the total variance is larger than a certain threshold, then remove that 3d point due to low reliability.
    '''
    
    n_pts, n_cam = C.shape[1], int(C.shape[0]/2) 
    pts_3d = np.zeros((n_pts,3))
    
    verbose = True if n_pts > 30000 else False
    
    
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


def check_distance_between_projection_rays_matches(idx_cam1, idx_cam2, C, P_crop, P_crop_ba, \
                                                   plot_err_hist=True, save_err_dsm=False, \
                                                   output_dir='.', aoi_lonlat=None):
    
    print('Checking the distance between projection rays...')

    # get SIFT keypoints visible in both images
    visible_idx = np.logical_and(~np.isnan(C[idx_cam1*2,:]), ~np.isnan(C[idx_cam2*2,:]))
    pts1, pts2 = C[(idx_cam1*2):(idx_cam1*2+2), visible_idx], C[(idx_cam2*2):(idx_cam2*2+2), visible_idx]
    tr_err, tr_err_ba = [],[]
    pts_3d, pts_3d_ba = np.zeros((pts1.shape[1], 3)), np.zeros((pts1.shape[1], 3))

    triangulate_err_hist = True 
    triangulate_err_dsm  = False

    # triangulate and compute triangulation error (i.e. distance between projection rays)
    for n in range(pts1.shape[1]):
        pt1, pt2 = pts1[:,n].ravel(), pts2[:,n].ravel()
        # before bundle adjustment
        pts_3d[n,:] = linear_triangulation_single_pt(pt1, pt2, P_crop[idx_cam1], P_crop[idx_cam2])
        tr_err.append(dist_between_proj_rays(pt1, pt2, P_crop[idx_cam1], P_crop[idx_cam2]))
        # after bundle adjustment
        pts_3d_ba[n,:] = linear_triangulation_single_pt(pt1, pt2, P_crop_ba[idx_cam1], P_crop_ba[idx_cam2])
        tr_err_ba.append(dist_between_proj_rays(pt1, pt2, P_crop_ba[idx_cam1], P_crop_ba[idx_cam2]))
        
    if plot_err_hist:
        fig = plt.figure(figsize=(10,3))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.title.set_text('Triangulation error before BA')
        ax2.title.set_text('Triangulation error after  BA')
        ax1.hist(tr_err, bins=40); 
        ax2.hist(tr_err_ba, bins=40);
        
    # project dsm of the cloud but instead of projecting the height, project the triangulation error
    if save_err_dsm:
        emin, emax, nmin, nmax = utils.utm_bounding_box_from_lonlat_aoi(aoi_lonlat)
        # before bundle adjustment
        x , y, z = pts_3d[:,0], pts_3d[:,1], pts_3d[:,2]
        lat, lon, h = ba_utils.ecef_to_latlon_custom(x, y, z)
        east, north = utils.utm_from_lonlat(lon, lat)
        xyz = np.vstack((east, north, tr_err)).T
        _, dem_nan, _ = triangulation.project_cloud_into_utm_grid(xyz, emin, emax, nmin, nmax, resolution=1.0)
        im = Image.fromarray(dem_nan)
        im.save(output_dir+'/triangulate_{}_{}.tif'.format(idx_cam1, idx_cam2))
        # after bundle adjustment
        x , y, z = pts_3d_ba[:,0], pts_3d_ba[:,1], pts_3d_ba[:,2]
        lat, lon, h = ba_utils.ecef_to_latlon_custom(x, y, z)
        east, north = utils.utm_from_lonlat(lon, lat)
        xyz = np.vstack((east, north, tr_err_ba)).T
        _, dem_nan, _ = triangulation.project_cloud_into_utm_grid(xyz, emin, emax, nmin, nmax, resolution=1.0)
        im = Image.fromarray(dem_nan)
        im.save(output_dir+'/triangulate_{}_{}_ba.tif'.format(idx_cam1,idx_cam2))

    print('...done!\n')
    
def write_feature_tracks_stereo_point_clouds(pairs_to_triangulate, C, P_crop, P_crop_ba, output_dir='.'):

    print('Writing point clouds of SIFT keypoints...')

    os.makedirs(output_dir+'/sift_clouds_before', exist_ok=True)
    os.makedirs(output_dir+'/sift_clouds_after', exist_ok=True)

    for [im1,im2] in pairs_to_triangulate:

        # get SIFT keypoints visible in both images
        visible_idx = np.logical_and(~np.isnan(C[im1*2,:]), ~np.isnan(C[im2*2,:])) 
        pts1, pts2 = C[(im1*2):(im1*2+2), visible_idx], C[(im2*2):(im2*2+2), visible_idx]

        # triangulation of SIFT keypoints before bundle adjustment
        pts_3d_sift = triangulate_list_of_matches(pts1, pts2, P_crop[im1], P_crop[im2])
        x , y, z = pts_3d_sift[:,0], pts_3d_sift[:,1], pts_3d_sift[:,2]
        lat, lon, h = ba_utils.ecef_to_latlon_custom(x, y, z)
        east, north = utils.utm_from_lonlat(lon, lat)
        xyz = np.vstack((east, north, h)).T
        fn = output_dir+'/sift_clouds_before/{:02}_{:02}.ply'.format(im1, im2)
        ba_utils.write_point_cloud_ply(fn, xyz, color=np.random.choice(range(256), size=3))

        # triangulation of SIFT keypoints after bundle adjustment
        pts_3d_sift = triangulate_list_of_matches(pts1, pts2, P_crop_ba[im1], P_crop_ba[im2])
        x , y, z = pts_3d_sift[:,0], pts_3d_sift[:,1], pts_3d_sift[:,2]
        lat, lon, h = ba_utils.ecef_to_latlon_custom(x, y, z)
        east, north = utils.utm_from_lonlat(lon, lat)
        xyz = np.vstack((east, north, h)).T
        fn = output_dir+'/sift_clouds_after/{:02}_{:02}_ba.ply'.format(im1, im2)
        ba_utils.write_point_cloud_ply(fn, xyz, color=np.random.choice(range(256), size=3))
        
    print('...done!\n')

    
def project_xyz_bbx_on_map(xyz, map_zoom=12):
    
    lat, lon, alt = ba_utils.ecef_to_latlon_custom(xyz[:,0], xyz[:,1], xyz[:,2])

    mymap = vistools.clickablemap(zoom=map_zoom)
    ## set the coordinates of the area of interest as a GeoJSON polygon
    aoi = {'coordinates': [[[min(lon), min(lat)], [min(lon), max(lat)], 
                            [max(lon), max(lat)], [max(lon), min(lat)],
                            [min(lon), min(lat)]]], 'type': 'Polygon'}
    # set the center of the aoi
    aoi['center'] = np.mean(aoi['coordinates'][0][:4], axis=0).tolist()
    # display a polygon covering the aoi and center the map
    mymap.add_GeoJSON(aoi) 
    mymap.center = aoi['center'][::-1]         
    display(mymap)
    
def dense_cloud_from_pair(i, j, P1, P2, cam_model, myimages, crop_offsets, aoi):
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
    pts_im1_org[:,0] -= crop_offsets[i]['x0']
    pts_im1_org[:,1] -= crop_offsets[i]['y0']
    pts_im2_org[:,0] -= crop_offsets[j]['x0']
    pts_im2_org[:,1] -= crop_offsets[j]['y0']
    
    #build point cloud 
    if cam_model == 'Affine':
        x1, y1, x2, y2 = pts_im1_org[:,0], pts_im1_org[:,1], pts_im2_org[:,0], pts_im2_org[:,1]
        lon, lat, h, _ = triangulation.triangulation_affine(P1, P2, x1, y1, x2, y2)
        dense_cloud = np.vstack([lon, lat, h]).T
    else:
        dense_cloud = triangulate_list_of_matches(pts_im1_org.T, pts_im2_org.T, P1, P2)
    
    # convert to utm coordinates
    x , y, z = dense_cloud[:,0], dense_cloud[:,1], dense_cloud[:,2]
    lat, lon, h = ba_utils.ecef_to_latlon_custom(x, y, z)
    east, north = utils.utm_from_lonlat(lon, lat)
    xyz = np.vstack((east, north, h)).T
        
    return xyz
