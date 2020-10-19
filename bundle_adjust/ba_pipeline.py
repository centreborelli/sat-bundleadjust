"""
Bundle Adjustment for 3D Reconstruction from Multi-Date Satellite Images
This script implements the BundleAdjustmentPipeline class
This class takes all the input data for the problem and solves it following the next 8-step pipeline
(1) compute feature tracks
(2) initialize 3d tie-points associated to the observed tracks
(3) set BA variables
(4) first BA run - L1 loss
(5) remove outlier track observations based on reprojection error magnitude
(6) second BA run - L2 loss
(7) reconstruct output from variables
(8) save corrected cameras (either projection matrices or rpcs) and corrected 3d tie-points
by Roger Mari <roger.mari@ens-paris-saclay.fr>
"""

import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
from PIL import Image

from bundle_adjust import ba_utils
from bundle_adjust import ba_core
from bundle_adjust import rpc_fit
from bundle_adjust import ba_params
from bundle_adjust import camera_utils
from bundle_adjust import data_loader as loader


class Error(Exception):
    pass


class BundleAdjustmentPipeline:
    def __init__(self, ba_data, feature_detection=True, tracks_config=None, verbose=False):
        """
        Args:
            ba_data: dictionary specifying the bundle adjustment input data
            feature_detection (optional): boolean, set it False if you do not want to run feature detection
                                          because all necessary tracks are already available at ba_data['input_dir']
            tracks_config (optional): dictionary specifying the configuration for the feature detection step
        """

        self.feature_detection = feature_detection
        self.tracks_config = tracks_config
        self.verbose = verbose
        self.input_dir = ba_data['input_dir']
        self.output_dir = ba_data['output_dir']
        self.n_adj = ba_data['n_adj']
        self.n_new = ba_data['n_new']
        self.n_pts_fix = 0
        self.myimages = ba_data['image_fnames'].copy()
        self.crop_offsets = [{k: c[k] for k in ['col0', 'row0', 'width', 'height']} for c in ba_data['crops']]
        self.input_seq = [f['crop'] for f in ba_data['crops']]
        self.input_masks = ba_data['masks'].copy() if (ba_data['masks'] is not None) else None
        self.input_rpcs = ba_data['rpcs'].copy()
        self.cam_model = ba_data['cam_model']
        self.footprints = ba_utils.get_image_footprints(self.input_rpcs, self.crop_offsets)
        self.aoi = ba_data['aoi']

        print('Bundle Adjustment Pipeline created')
        print('-------------------------------------------------------------')
        print('Configuration:')
        print('    - input_dir:    {}'.format(self.input_dir))
        print('    - output_dir:   {}'.format(self.output_dir))
        print('    - n_new:        {}'.format(self.n_new))
        print('    - n_adj:        {}'.format(self.n_adj))
        print('    - cam_model:    {}'.format(self.cam_model))
        print('-------------------------------------------------------------\n')

        # stuff to be filled by 'run_feature_detection'
        self.features = []
        self.pairs_to_triangulate = []
        self.pairs_to_match = []
        self.C = None
        
        # stuff to be filled by 'initialize_pts3d'
        self.pts3d = None

        # stuff to be filled by 'define_ba_parameters'
        self.ba_params = None
        
        # stuff to be filled by 'run_ba_softL1'-'clean_outlier_observations'-'run_ba_L2'
        self.ba_e = None
        self.init_e = None
        self.corrected_cameras = None
        self.corrected_pts3d = None

        # set initial cameras
        ba_params.check_valid_cam_model(self.cam_model)
        if 'cameras' in ba_data.keys():
            self.cameras = ba_data['cameras'].copy()
        else:
            self.set_cameras(verbose=self.verbose)
        self.optical_centers = self.get_optical_centers(verbose=self.verbose)


    def approx_affine_projection_matrices(self, verbose=False):
        import srtm4
        projection_matrices, n_cam = [], len(self.input_rpcs)
        for im_idx, rpc, offset in zip(np.arange(n_cam), self.input_rpcs, self.crop_offsets):
            lon, lat = self.aoi['center'][0], self.aoi['center'][1]
            alt = srtm4.srtm4(lon, lat)
            x, y, z = ba_utils.latlon_to_ecef_custom(lat, lon, alt)
            projection_matrices.append(camera_utils.approx_rpc_as_affine_projection_matrix(rpc, x, y, z, offset))
            if verbose:
                print('\rApprox. rpcs as affine projection matrices... {} / {}'.format(im_idx + 1, n_cam), end='\r')
        if verbose:
            print('\n')
        return projection_matrices


    def approx_perspective_projection_matrices(self, verbose=False):
        projection_matrices, n_cam = [], len(self.input_rpcs)
        for im_idx, rpc, crop in zip(np.arange(n_cam), self.input_rpcs, self.crop_offsets):
            projection_matrices.append(camera_utils.approx_rpc_as_perspective_projection_matrix(rpc, crop))
            if verbose:
                print('\rApprox. rpcs as perspective projection matrices... {} / {}'.format(im_idx + 1, n_cam), end='\r')
        if verbose:
            print('\n')
        return projection_matrices


    def get_optical_centers(self, verbose=False):
        if verbose:
            print('\nEstimating camera positions...')
        if self.cam_model != 'perspective':
            tmp_perspective_cams = self.approx_perspective_projection_matrices(verbose=False)
            optical_centers = [camera_utils.get_perspective_optical_center(P) for P in tmp_perspective_cams]
        else:
            optical_centers = [camera_utils.get_perspective_optical_center(P) for P in self.cameras]
        if verbose:
            print('done!\n')
        return optical_centers


    def set_cameras(self, verbose=True):
        if self.cam_model == 'affine':
            self.cameras = self.approx_affine_projection_matrices(verbose=verbose)
        elif self.cam_model == 'perspective':
            self.cameras = self.approx_perspective_projection_matrices(verbose=verbose)
        else:
            self.cameras = self.input_rpcs.copy()


    def compute_feature_tracks(self):
        """
        Detect feature tracks in the input sequence of images
        """
        local_data = {'n_adj': self.n_adj, 'n_new': self.n_new, 'fnames': self.myimages, 'images': self.input_seq,
                      'rpcs': self.input_rpcs, 'offsets': self.crop_offsets,  'footprints': self.footprints,
                      'cameras': self.cameras, 'optical_centers': self.optical_centers,
                      'cam_model': self.cam_model, 'masks': self.input_masks}
        if not self.feature_detection:
            local_data['n_adj'], local_data['n_new'] = self.n_adj + self.n_new, 0

        from feature_tracks.ft_pipeline import FeatureTracksPipeline
        ft_pipeline = FeatureTracksPipeline(self.input_dir, self.output_dir, local_data,
                                            config=self.tracks_config, satellite=True)
        feature_tracks = ft_pipeline.build_feature_tracks()

        self.features = feature_tracks['features']
        self.pairwise_matches = feature_tracks['pairwise_matches']
        self.pairs_to_triangulate = feature_tracks['pairs_to_triangulate']
        self.pairs_to_match = feature_tracks['pairs_to_match']
        self.C = feature_tracks['C']
        self.C_v2 = feature_tracks['C_v2']

        ############### 
        # (optional) TO BE REVIEWED
        #put tracks with known 3d point coordinates before the rest
        ##############
        if self.tracks_config['tie_points']:
            track_idx_to_coords_3d = []
            for cam_idx, fname in enumerate(self.myimages):
                f_id = os.path.splitext(os.path.basename(fname))[0]
                if os.path.exists('{}/tie_points/{}.pickle'.format(self.input_dir, f_id)):
                    pickle_in = open('{}/tie_points/{}.pickle'.format(self.input_dir, f_id),'rb')
                    kp_idx_to_coords_3d = pickle.load(pickle_in)
            
                    tiepoints_kp_idx = kp_idx_to_coords_3d[:,0]
                    tiepoints_coords_3d = kp_idx_to_coords_3d[:,1:]
                    
                    not_nan = ~np.isnan(self.C_v2[cam_idx, :])
                    cam_track_idx = np.arange(self.C_v2.shape[1])[not_nan]
                    cam_kp_idx = self.C_v2[cam_idx,not_nan]
                    
                    _, cam_kp_indices_tiepoint_positions, tiepoints_indices_seen_in_cam =\
                    np.intersect1d(cam_kp_idx.astype(int), tiepoints_kp_idx.astype(int), return_indices=True)
                    
                    cam_track_idx_to_coords_3d = np.hstack((cam_track_idx[cam_kp_indices_tiepoint_positions][:, np.newaxis],
                                                             tiepoints_coords_3d[tiepoints_indices_seen_in_cam, :]))
                    
                    track_idx_to_coords_3d.append(cam_track_idx_to_coords_3d)
        
            if len(track_idx_to_coords_3d) > 0:
                track_idx_to_coords_3d = np.vstack(track_idx_to_coords_3d)
                track_idx_to_coords_3d = np.unique(track_idx_to_coords_3d, axis=0)
        
                true_if_known_3d_coords = np.zeros(self.C.shape[1]).astype(bool)
                true_if_known_3d_coords[track_idx_to_coords_3d[:,0].astype(int)] = True
                self.n_pts_fix = np.sum(1*true_if_known_3d_coords)
                new_C = self.C.copy()
                new_C[:,:self.n_pts_fix] = self.C[:,true_if_known_3d_coords]
                new_C[:,self.n_pts_fix:] = self.C[:,~true_if_known_3d_coords]
                self.C = new_C
                
                fixed_pts3d = np.zeros((self.C.shape[1], 3))
                fixed_pts3d[:] = np.nan
                fixed_pts3d[track_idx_to_coords_3d[:,0].astype(int), :] = track_idx_to_coords_3d[:,1:]
                self.fixed_pts3d = np.zeros((self.n_pts_fix, 3))
                self.fixed_pts3d[:] = np.nan
                self.fixed_pts3d = fixed_pts3d[true_if_known_3d_coords, :]
        
                print('Loaded {} known fixed 3d point coordinates !'.format(self.n_pts_fix))
        
        del feature_tracks


    def initialize_pts3d(self):
        """
        Initialize the ECEF coordinates of the 3d points that project into the detected tracks
        """
        from bundle_adjust.ba_triangulate import init_pts3d
        self.pts3d = init_pts3d(self.C, self.cameras, self.cam_model, self.pairs_to_triangulate)
        if self.n_pts_fix > 0:
            self.pts3d[:self.n_pts_fix, :] = self.fixed_pts3d


    def define_ba_parameters(self, verbose=False):
        """
        Define the necessary parameters to run the bundle adjustment optimization
        """
        args = [self.C, self.pts3d, self.cameras, self.cam_model, self.pairs_to_triangulate]
        self.ba_params = ba_params.BundleAdjustmentParameters(*args, self.n_adj, self.n_pts_fix, verbose=verbose)


    def run_ba_softL1(self, verbose=False):
        """
        Run bundle adjustment optimization with soft L1 norm for the reprojection errors
        """
        ls_params_L1 = {'loss': 'soft_l1', 'ftol': 1e-4, 'xtol': 1e-10, 'f_scale': 0.5}
        _, self.ba_sol, self.init_e, self.ba_e = ba_core.run_ba_optimization(self.ba_params, ls_params=ls_params_L1,
                                                                             verbose=verbose)


    def run_ba_L2(self, verbose=False):
        """
        Run the bundle adjustment optimization with classic L2 norm for the reprojection errors
        """
        ls_params_L2 = {'loss': 'linear', 'ftol': 1e-4, 'xtol': 1e-10, 'f_scale': 1.0}
        _, self.ba_sol, self.init_e, self.ba_e = ba_core.run_ba_optimization(self.ba_params, ls_params=ls_params_L2,
                                                                             verbose=verbose)


    def reconstruct_ba_result(self):
        """
        Recover the optimized 3d points and cameras after the bundle adjustment run(s)
        """
        args = [self.ba_sol, self.pts3d, self.cameras]
        self.corrected_pts3d, self.corrected_cameras = self.ba_params.reconstruct_vars(*args)


    def clean_outlier_observations(self, verbose=False):
        """
        Remove outliers from the available tracks according to their reprojection error
        """
        from bundle_adjust.ba_outliers import get_elbow_value, remove_outliers_from_reprojection_error
        elbow_value, success = get_elbow_value(self.ba_e, verbose=verbose)
        self.ba_params = remove_outliers_from_reprojection_error(self.ba_e, self.ba_params,
                                                                 thr=max(elbow_value,2.0), verbose=verbose)


    def save_initial_matrices(self):
        """
        Write initial projection matrices to json files
        """
        out_dir = os.path.join(self.output_dir, 'P_init')
        fnames = [os.path.join(out_dir, loader.get_id(fn)+'_pinhole.json') for fn in self.myimages]
        loader.save_projection_matrices(fnames, self.cameras, self.crop_offsets)
        print('\nInitial projection matrices successfully saved at {}\n'.format(out_dir))
    
    
    def save_corrected_matrices(self):
        """
        Write corrected projection matrices to json files
        """
        out_dir = os.path.join(self.output_dir, 'P_adj')
        fnames = [os.path.join(out_dir, loader.get_id(fn)+'_pinhole_adj.json') for fn in self.myimages]
        loader.save_projection_matrices(fnames, self.corrected_cameras, self.crop_offsets)
        print('\nBundle adjusted projection matrices successfully saved at {}\n'.format(out_dir))


    def save_corrected_rpcs(self): 
        """
        Write corrected rpc model coefficients to txt files
        """
        out_dir = os.path.join(self.output_dir, 'RPC_adj')
        fnames = [os.path.join(out_dir, loader.get_id(fn)+'_RPC_adj.txt') for fn in self.myimages]
        for fn, cam in zip(fnames, self.corrected_cameras):
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            if self.cam_model in ['perspective', 'affine']:
                rpc_calib, _ = rpc_fit.fit_rpc_from_projection_matrix(cam, self.corrected_pts3d)
                rpc_calib.write_to_file(fn)
            else:
                cam.write_to_file(fn)
        print('\nBundle adjusted RPCs successfully saved at {}\n'.format(out_dir))


    def save_corrected_points(self):
        """
        Write corrected 3d points produced by the bundle adjustment process into a ply file
        """
        pts_ba = self.corrected_pts3d[self.ba_params.pts_prev_indices, :]
        #np.savetxt(os.path.join(self.output_dir, 'pts3d_adj.txt'), pts_ba)
        ba_utils.write_point_cloud_ply(os.path.join(self.output_dir, 'pts3d_adj.ply'), pts_ba)
        print('\nBundle adjusted 3d points saved at {}\n'.format(self.output_dir))


    def visualize_feature_track(self, track_idx=None):
        """
         Visualize feature track before (and after, if optimization results are available) bundle adjustment
         """

        # load feature tracks and initial 3d pts + corrected 3d pts if available (also ignore outlier observations)
        from bundle_adjust.ba_triangulate import init_pts3d
        pts3d, C = self.pts3d.copy(), self.C.copy()
        print(self.pts3d)
        ba_available = self.corrected_pts3d is not None and self.corrected_cameras is not None
        if ba_available:
            C = C[:, self.ba_params.pts_prev_indices]
            pts3d = pts3d[self.ba_params.pts_prev_indices, :]
            pts3d_ba = self.corrected_pts3d[self.ba_params.pts_prev_indices, :]

        # pick a random track index in case none was specified
        true_where_track = np.sum(1*~np.isnan(C[np.arange(0, C.shape[0], 2), :])[-self.n_new:],axis=0).astype(bool)
        pt_ind = np.random.choice(np.arange(0, C.shape[1])[true_where_track]) if (track_idx is None) else track_idx
        # get indices of the images where the selected track is visible
        n_img = self.n_adj + self.n_new
        im_ind = [k for k, j in enumerate(range(n_img)) if not np.isnan(C[j*2, pt_ind])]
        print('Displaying feature track with index {}, length {}\n'.format(pt_ind, len(im_ind)))

        # get original xyz coordinates before and after BA
        pt3d = pts3d[pt_ind, :]
        if ba_available:
            pt3d_ba = pts3d_ba[pt_ind, :]
            print('3D location (initial)  :', pt3d.ravel())
            print('3D location (after BA) :', pt3d_ba.ravel(), '\n')

        # reproject feature track
        err_init, err_ba = [], []
        for i in im_ind:
            # feature track observation at current image
            pt2d_obs = C[(i*2):(i*2+2), pt_ind]
            # reprojection with initial variables
            pt2d_init = camera_utils.project_pts3d(self.cameras[i], self.cam_model, pt3d[np.newaxis, :])
            # reprojection with adjusted variables
            if ba_available:
                pt2d_ba = camera_utils.project_pts3d(self.corrected_cameras[i], self.cam_model, pt3d_ba[np.newaxis, :])
            # compute reprojection error before and after
            print(' ----> Real 2D loc in im', i, ' (yellow) = ', pt2d_obs)
            print(' ----> Proj 2D loc in im', i, ' before BA (red) = ', pt2d_init.ravel())
            if ba_available:
                print(' ----> Proj 2D loc in im', i, ' after  BA (green) = ', pt2d_ba.ravel())
            err_init.append(np.sqrt(np.sum((pt2d_init.ravel() - pt2d_obs)**2)))
            print('              Reprojection error beofre BA:', err_init[-1])
            if ba_available:
                err_ba.append(np.sqrt(np.sum((pt2d_ba.ravel() - pt2d_obs)**2)))
                print('              Reprojection error after  BA:', err_ba[-1])
            # display
            plt.figure(figsize=(10,20))
            plt.imshow(self.input_seq[i], cmap="gray")
            plt.plot(*pt2d_obs, "yo")
            plt.plot(*pt2d_init[0], "ro")
            if ba_available:
                plt.plot(*pt2d_ba[0], "go")
            plt.show()
            
        print('Mean reprojection error before BA: {}'.format(np.mean(err_init)))
        if ba_available:
            print('Mean reprojection error after BA: {}'.format(np.mean(err_ba)))


    def analyse_reprojection_error_per_image(self, im_idx):
        """
        Compute and analyse in detail the reprojection error of a specfic image
        """
    
        # pick all track observations and the corresponding 3d points visible in the selected image
        C = self.C[:, self.ba_params.pts_prev_indices]
        obs2d = C[(im_idx*2):(im_idx*2+2),~np.isnan(C[im_idx*2,:])].T
        pts3d = self.pts3d[self.ba_params.pts_prev_indices, :]
        pts3d_ba = self.corrected_pts3d[self.ba_params.pts_prev_indices, :]
        pts3d_before = pts3d[~np.isnan(C[im_idx*2, :]), :]
        pts3d_after = pts3d_ba[~np.isnan(C[im_idx*2, :]), :]

        # reproject and compute metrics
        from bundle_adjust.ba_metrics import reproject_pts3d_and_compute_errors
        reproject_pts3d_and_compute_errors(self.cameras[im_idx], self.corrected_cameras[im_idx],
                                           self.cam_model, obs2d, pts3d_before, pts3d_after,
                                           image_fname=self.myimages[im_idx], verbose=True)


    def save_crops(self, img_indices=None):
        """
        Write input crops to output directory
        """

        img_indices = np.arange(self.n_adj + self.n_new) if (img_indices is None) else img_indices
        images_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        # save geotiff crops
        for im_idx in img_indices:
            import rasterio
            from rpcm import utils as rpcm_utils
            from osgeo import gdal, gdalconst

            f_id = os.path.splitext(os.path.basename(self.myimages[im_idx]))[0]
            array = np.array([np.array(Image.open(self.myimages[im_idx]))])
            with rasterio.open(self.myimages[im_idx]) as src:
                #rpc_dict = src.tags(ns='RPC')
                rpcm_utils.rasterio_write('{}/{}.tif'.format(images_dir, f_id), array, profile=src.profile, tags=src.tags())
            
            #rpc_dict = self.input_rpcs[im_idx].to_geotiff_dict()
            #rpc_dict = {k: str(v) for k, v in rpc_dict.items()}
            rpc_dict = ba_utils.rpc_rpcm_to_geotiff_format(self.input_rpcs[im_idx].__dict__)
            tif_without_rpc = gdal.Open('{}/{}.tif'.format(images_dir, f_id), gdalconst.GA_Update)
            tif_without_rpc.SetMetadata(rpc_dict, 'RPC')
            del(tif_without_rpc)
        print('\nImage crops were saved at {}\n'.format(images_dir))


    def compute_image_weights_after_bundle_adjustment(self):
        """
        Compute image weights intended to measure the importance of each image in the bundle adjustment process
        """
        from feature_tracks import ft_ranking
        args = [self.ba_params.C, self.ba_params.pts3d_ba, self.ba_params.cameras_ba,
                self.ba_params.cam_model, self.ba_params.pairs_to_triangulate]
        C_reproj = ft_ranking.reprojection_error_from_C(*args)
        cam_weights = ft_ranking.compute_camera_weights(self.ba_params.C, C_reproj)
        return cam_weights


    def save_feature_tracks_as_svg(self, output_dir, img_indices=None, save_reprojected=True):
        
        from feature_tracks.ft_utils import save_pts2d_as_svg, save_sequence_features_svg
        
        if img_indices is None:
            n_img = self.n_adj + self.n_new
            img_indices = np.arange(n_img)
        
        #self.save_crops(output_dir, img_indices)
        save_sequence_features_svg(output_dir, np.array(self.myimages)[img_indices].tolist(), self.features)
        print('\nSIFT keypoints were saved at {}\n'.format(os.path.join(output_dir, 'sift')))
        
        before_dir = os.path.join(output_dir, 'feature_tracks/tracks_reproj_before')
        after_dir = os.path.join(output_dir, 'feature_tracks/tracks_reproj_after')
        original_dir = os.path.join(output_dir, 'feature_tracks/tracks_sift')
        os.makedirs(before_dir, exist_ok=True)
        os.makedirs(after_dir, exist_ok=True)
        os.makedirs(original_dir, exist_ok=True)
                
        
        for im_idx in img_indices:
        
            f_id = os.path.splitext(os.path.basename(self.myimages[im_idx]))[0]
            h,w = self.input_seq[im_idx].shape
            
            svg_fname_o = os.path.join(original_dir, '{}.svg'.format(f_id))
            svg_fname_b = os.path.join(before_dir, '{}.svg'.format(f_id))
            svg_fname_a = os.path.join(after_dir, '{}.svg'.format(f_id))
        
            P_before = self.input_P[im_idx]
            P_after = self.P_crop_ba[im_idx]
            
            # pick all points visible in the selected image
            pts2d = self.C[(im_idx*2):(im_idx*2+2),~np.isnan(self.C[im_idx*2,:])].T
            pts3d_before = self.pts_3d[~np.isnan(self.C[im_idx*2,:]),:]
            pts3d_after = self.pts_3d_ba[~np.isnan(self.C[im_idx*2,:]),:]
            n_pts = pts3d_before.shape[0]
            
            # reprojections before bundle adjustment
            proj = P_before @ np.hstack((pts3d_before, np.ones((n_pts,1)))).T
            pts_reproj_before = (proj[:2,:]/proj[-1,:]).T

            # reprojections after bundle adjustment
            proj = P_after @ np.hstack((pts3d_after, np.ones((n_pts,1)))).T
            pts_reproj_after = (proj[:2,:]/proj[-1,:]).T

            err_before = np.sum(abs(pts_reproj_before - pts2d), axis=1)
            err_after = np.sum(abs(pts_reproj_after - pts2d), axis=1)
            
            # draw pts on svg
            save_pts2d_as_svg(svg_fname_o, pts2d, w=w, h=h, c='green')
            save_pts2d_as_svg(svg_fname_b, pts_reproj_before, w=w, h=h, c='red')
            save_pts2d_as_svg(svg_fname_a, pts_reproj_after, w=w, h=h, c='yellow')
            

        print('\nFeature tracks and their reprojection were saved at {}\n'.format(output_dir))
    
    def get_number_of_matches_between_groups_of_views(self, img_indices_g1, img_indices_g2):
        
        img_indices_g1_s = sorted(img_indices_g1)
        img_indices_g2_s = sorted(img_indices_g2)
        n_matches = 0
        n_matches_inside_aoi = 0
        for im1 in img_indices_g1:
            for im2 in img_indices_g2:
                obs_im1 = 1*np.invert(np.isnan(self.C[2*im1,:]))
                obs_im2 = 1*np.invert(np.isnan(self.C[2*im2,:]))
                true_if_obs_seen_in_both_cams = np.sum(np.vstack((obs_im1, obs_im2)), axis=0) == 2
                n_matches += np.sum(1*true_if_obs_seen_in_both_cams)
                
                if self.input_masks is not None:
                    tmp = np.zeros(self.C.shape[1])
                    pts2d_colrow = (self.C[(2*im1):(2*im1+2),:][:,obs_im1.astype(bool)].T).astype(np.int)
                    tmp[obs_im1.astype(bool)] = 1*(self.input_masks[im1][pts2d_colrow[:,1], pts2d_colrow[:,0]] > 0)
                    true_if_obs_inside_aoi = tmp.astype(bool)
                    n_matches_inside_aoi += np.sum(1*np.logical_and(true_if_obs_seen_in_both_cams, \
                                                                    true_if_obs_inside_aoi))
                else:
                    n_matches_inside_aoi += None
        
        return n_matches, n_matches_inside_aoi
    
    def get_n_matches_within_group_of_views(self, img_indices_g1):
        
        img_indices_g1_s = sorted(img_indices_g1)
        n_matches = 0
        n_matches_inside_aoi = 0
        for im1 in img_indices_g1_s:
            for im2 in np.array(img_indices_g1_s[im1+1:]).tolist():
                obs_im1 = 1*np.invert(np.isnan(self.C[2*im1,:]))
                obs_im2 = 1*np.invert(np.isnan(self.C[2*im2,:]))
                true_if_obs_seen_in_both_cams = np.sum(np.vstack((obs_im1, obs_im2)), axis=0) == 2
                n_matches += np.sum(np.sum(np.vstack((obs_im1, obs_im2)), axis=0) == 2)
                
                if self.input_masks is not None:
                    tmp = np.zeros(self.C.shape[1])
                    pts2d_colrow = (self.C[(2*im1):(2*im1+2),:][:,obs_im1.astype(bool)].T).astype(np.int)
                    tmp[obs_im1.astype(bool)] = 1*(self.input_masks[im1][pts2d_colrow[:,1], pts2d_colrow[:,0]] > 0)
                    true_if_obs_inside_aoi = tmp.astype(bool)
                    n_matches_inside_aoi += np.sum(1*np.logical_and(true_if_obs_seen_in_both_cams, \
                                                                    true_if_obs_inside_aoi))
        return n_matches, n_matches_inside_aoi
    
    
    def get_n_tracks_within_group_of_views(self, img_indices_g1):
        
        # compute tracks within the specified cameras
        img_indices = sorted(img_indices_g1)
        true_if_track = (np.sum(~(np.isnan(self.C[np.arange(0,self.C.shape[0],2)[img_indices],:])),axis=0)>1).astype(bool)
        n_tracks = np.sum(1*true_if_track)
        
        if self.input_masks is not None:
        
            # compute tracks inside AOI within the specified cameras
            n_tracks_inside_aoi = 0
            n_tracks_in_C = self.C.shape[1]
            n_cam_in_C = int(self.C.shape[0]/2)
            true_if_cam = np.zeros(n_cam_in_C).astype(bool)
            true_if_cam[img_indices] = True
            true_if_cam = np.repeat(true_if_cam, 2)
            true_if_cam_2d = np.repeat(np.array([true_if_cam]), n_tracks_in_C, axis=0).T # same size as C
            true_if_track_2d =  np.repeat(np.array([true_if_track]), n_cam_in_C * 2, axis=0)
            cam_indices = np.repeat(np.array([np.arange(self.C.shape[0])/2]),n_tracks_in_C,axis=0).T
            cam_indices = cam_indices.astype(int).astype(float) # int removes decimals, float is necessary to use nan
            cam_indices[np.invert(true_if_track_2d * true_if_cam_2d)] = np.nan
            cam_indices[np.isnan(self.C)] = np.nan

            # take the first camera where the track is visible
            cam_indices_to_get_pts2d = np.nanmin(cam_indices[:, true_if_track],axis=0).astype(int) 
            track_indices_to_get_pts2d = np.arange(n_tracks_in_C)[true_if_track].astype(int)

            max_col, max_row = 0, 0
            for track_idx, cam_idx in zip(track_indices_to_get_pts2d, cam_indices_to_get_pts2d):
                col, row = self.C[2*cam_idx, track_idx].astype(int), self.C[2*cam_idx + 1, track_idx].astype(int)
                n_tracks_inside_aoi += 1*(self.input_masks[cam_idx][row, col] > 0)
        else:
            print('ba_pipeline.get_number_of_tracks_within_group_of_views cannot get number of tracks inside the aoi because aoi masks are not available !')
        
        return n_tracks, n_tracks_inside_aoi


    def run(self):

        print(self.verbose)

        # compute feature tracks
        self.compute_feature_tracks()
        self.initialize_pts3d()

        # run bundle adjustment
        self.define_ba_parameters(verbose=self.verbose)
        self.run_ba_softL1(verbose=self.verbose)
        self.clean_outlier_observations(verbose=self.verbose)
        self.run_ba_L2(verbose=self.verbose)
        self.reconstruct_ba_result()

        # save output
        if self.cam_model in ['perspective', 'affine']:
            self.save_corrected_matrices()
        self.save_corrected_rpcs()
        self.save_corrected_points()
