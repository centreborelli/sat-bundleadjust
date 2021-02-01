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
from bundle_adjust import ba_outliers
from bundle_adjust import camera_utils
from bundle_adjust import data_loader as loader

import timeit

class Error(Exception):
    pass

class BundleAdjustmentPipeline:
    def __init__(self, ba_data, feature_detection=True, tracks_config=None,
                 fix_ref_cam=False, clean_outliers=True, max_reproj_error=None, verbose=False):
        """
        Args:
            ba_data: dictionary specifying the bundle adjustment input data
            feature_detection (optional): boolean, set it False if you do not want to run feature detection
                                          because all necessary tracks are already available at ba_data['input_dir']
            tracks_config (optional): dictionary specifying the configuration for the feature detection step
        """

        # read configuration parameters
        self.display_plots = False
        self.verbose = verbose
        self.fix_ref_cam = fix_ref_cam
        self.max_init_reproj_error = max_reproj_error
        self.clean_outliers = clean_outliers
        self.feature_detection = feature_detection
        from feature_tracks.ft_utils import init_feature_tracks_config
        self.tracks_config = init_feature_tracks_config(tracks_config)

        # read input data
        self.input_dir = ba_data['input_dir']
        self.output_dir = ba_data['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        self.n_adj = ba_data['n_adj']
        self.n_new = ba_data['n_new']
        self.n_pts_fix = 0
        self.myimages = ba_data['image_fnames'].copy()
        self.crop_offsets = [{k: c[k] for k in ['col0', 'row0', 'width', 'height']} for c in ba_data['crops']]
        self.input_seq = [f['crop'] for f in ba_data['crops']]
        self.input_masks = ba_data['masks'].copy() if ba_data['masks'] is not None else None
        self.input_rpcs = ba_data['rpcs'].copy()
        self.cam_model = ba_data['cam_model']
        self.aoi = ba_data['aoi']
        self.correction_params = ba_data['correction_params'] if 'correction_params' in ba_data.keys() else ['R']


        print('Bundle Adjustment Pipeline created')
        print('-------------------------------------------------------------')
        print('Configuration:')
        print('    - input_dir:    {}'.format(self.input_dir))
        print('    - output_dir:   {}'.format(self.output_dir))
        print('    - n_new:        {}'.format(self.n_new))
        print('    - n_adj:        {}'.format(self.n_adj))
        print('    - cam_model:    {}'.format(self.cam_model))
        print('    - fix_ref_cam:  {}'.format(self.fix_ref_cam))
        print('-------------------------------------------------------------\n', flush=True)

        # stuff to be filled by 'run_feature_detection'
        self.features = []
        self.pairs_to_triangulate = []
        self.C = None

        # stuff to be filled by 'initialize_pts3d'
        self.pts3d = None

        # stuff to be filled by 'define_ba_parameters'
        self.ba_params = None

        # stuff to be filled by 'run_ba_softL1'-'clean_outlier_observations'-'run_ba_L2'
        self.ba_e = None
        self.init_e = None
        self.ba_iters = 0
        self.corrected_cameras = None
        self.corrected_pts3d = None

        # set initial cameras and image footprints
        ba_params.check_valid_cam_model(self.cam_model)
        if 'cameras' in ba_data.keys():
            self.cameras = ba_data['cameras'].copy()
        else:
            self.set_cameras(verbose=self.verbose)
        self.optical_centers = self.get_optical_centers(verbose=self.verbose)
        self.footprints = self.get_footprints(verbose=self.verbose)
        print('\n')


    def get_footprints(self, verbose=False):
        t0 = timeit.default_timer()
        if verbose:
            print('Getting image footprints...', flush=True)
        import srtm4
        z = srtm4.srtm4(self.aoi['center'][0], self.aoi['center'][1])
        footprints = ba_utils.get_image_footprints(self.input_rpcs, self.crop_offsets, z)
        if verbose:
            print('...done in {:.2f} seconds'.format(timeit.default_timer() - t0), flush=True)
        return footprints


    def check_projection_matrices(self, err, plot_errors=True, max_err=5.0):
        if plot_errors:
            plt.figure()
            plt.plot(err)
            plt.show()
        err_cams = np.arange(len(err))[np.array(err) > max_err]
        n_err_cams = len(err_cams)
        if n_err_cams > 0:
            args = [n_err_cams, ' '.join(['\nCamera {}, error = {:.3f}'.format(c, err[c]) for c in err_cams])]
            raise Error('Found {} perspective proj matrices with error larger than 1.0 px\n{}'.format(*args))

    def get_optical_centers(self, verbose=False):
        t0 = timeit.default_timer()
        if verbose:
            print('Estimating camera positions...', flush=True)
        if self.cam_model != 'perspective':
            args = [self.input_rpcs, self.crop_offsets]
            tmp_perspective_cams, err = loader.approx_perspective_projection_matrices(*args, verbose=verbose)
            #self.check_projection_matrices(err)
            optical_centers = [camera_utils.get_perspective_optical_center(P) for P in tmp_perspective_cams]
        else:
            optical_centers = [camera_utils.get_perspective_optical_center(P) for P in self.cameras]
        if verbose:
            print('...done in {:.2f} seconds'.format(timeit.default_timer() - t0), flush=True)
        return optical_centers


    def set_cameras(self, verbose=True):
        if self.cam_model == 'affine':
            args = [self.input_rpcs, self.crop_offsets, self.aoi]
            self.cameras, err = loader.approx_affine_projection_matrices(*args, verbose=verbose)
            self.check_projection_matrices(err)
        elif self.cam_model == 'perspective':
            args = [self.input_rpcs, self.crop_offsets]
            self.cameras, err = loader.approx_perspective_projection_matrices(*args, verbose=verbose)
            self.check_projection_matrices(err)
        else:
            self.cameras = self.input_rpcs.copy()


    def compute_feature_tracks(self):
        """
        Detect feature tracks in the input sequence of images
        """
        # s2p employs the rpcs to delimit a range for finding possible matches
        # knowing this, it is better that we use ALWAYS the original rpcs here
        # why? otherwise in the sequential model we could have X (previously adjusted rpcs using BA)
        # and Y (rpcs to adjust), whose ref systems could differ significantly due to the accumulation of drift
        # consequently, the number of matches of X and Y would be downgraded
        # note that the previous does not happen with opencv SIFT as rpcs are not employed for track construction

        if self.tracks_config['FT_sift_detection'] == 's2p' and os.path.exists(self.input_dir + '/../RPC_init'):
            args = [self.myimages, self.input_dir + '/../RPC_init', 'RPC', False]
            ft_rpcs = loader.load_rpcs_from_dir(*args)
        else:
            ft_rpcs = self.input_rpcs
        local_data = {'n_adj': self.n_adj, 'n_new': self.n_new, 'fnames': self.myimages, 'images': self.input_seq,
                      'rpcs': ft_rpcs, 'offsets': self.crop_offsets,  'footprints': self.footprints,
                      'optical_centers': self.optical_centers, 'masks': self.input_masks}
        if not self.feature_detection:
            local_data['n_adj'], local_data['n_new'] = self.n_adj + self.n_new, 0

        from feature_tracks.ft_pipeline import FeatureTracksPipeline
        ft_pipeline = FeatureTracksPipeline(self.input_dir, self.output_dir, local_data,
                                            config=self.tracks_config, satellite=True)
        feature_tracks, self.feature_tracks_running_time = ft_pipeline.build_feature_tracks()

        self.features = feature_tracks['features']
        self.pairs_to_triangulate = feature_tracks['pairs_to_triangulate']
        self.C = feature_tracks['C']
        if self.cam_model == 'rpc':
            for i in range(int(self.C.shape[0]/2)):
                self.C[2*i, :] += self.crop_offsets[i]['col0']
                self.C[2*i+1, :] += self.crop_offsets[i]['row0']
        self.C_v2 = feature_tracks['C_v2']
        self.n_pts_fix = feature_tracks['n_pts_fix']

        # sanity checks to verify if C looks good
        err_msg = 'Insufficient SIFT matches'
        n_cam = int(self.C.shape[0]/2)
        if n_cam > self.C.shape[1]:
            raise Error('{}: Found less tracks than cameras'.format(err_msg))
        obs_per_cam = np.sum(1 * ~np.isnan(self.C), axis=1)[::2]
        min_obs_cam = 10
        if np.sum(obs_per_cam < min_obs_cam) > 0:
            err_msg = [err_msg, np.sum(obs_per_cam < min_obs_cam),
                       min_obs_cam, np.arange(n_cam)[obs_per_cam < min_obs_cam]]
            raise Error('{}: Found {} cameras with less than {} tie point observations (nodes: {})'.format(*err_msg))

        del feature_tracks


    def initialize_pts3d(self, verbose=False):
        """
        Initialize the ECEF coordinates of the 3d points that project into the detected tracks
        """
        from bundle_adjust.ba_triangulate import init_pts3d
        self.pts3d = np.zeros((self.C.shape[1], 3), dtype=np.float32)
        n_pts_opt = self.C.shape[1] - self.n_pts_fix
        if self.n_pts_fix > 0:
            if verbose:
                print('Initializing {} fixed 3d point coords !'.format(self.n_pts_fix), flush=True)
            self.pts3d[:self.n_pts_fix, :] = init_pts3d(self.C[:self.n_adj*2, :self.n_pts_fix], self.cameras,
                                                        self.cam_model, self.pairs_to_triangulate, verbose=verbose)
        if verbose:
            print('Initializing {} 3d point coords to optimize !'.format(n_pts_opt), flush=True)
        self.pts3d[-n_pts_opt:, :] = init_pts3d(self.C[:, -n_pts_opt:], self.cameras, self.cam_model,
                                                self.pairs_to_triangulate, verbose=verbose)


    def define_ba_parameters(self, freeze_all_cams=False, verbose=False):
        """
        Define the necessary parameters to run the bundle adjustment optimization
        """
        n_cam_fix = int(self.C.shape[0]/2) if freeze_all_cams else self.n_adj
        args = [self.C, self.pts3d, self.cameras, self.cam_model, self.pairs_to_triangulate, self.optical_centers]
        self.ba_params = ba_params.BundleAdjustmentParameters(*args, n_cam_fix, self.n_pts_fix, verbose=verbose,
                                                              cam_params_to_optimize=self.correction_params)


    def run_ba_softL1(self, verbose=False):
        """
        Run bundle adjustment optimization with soft L1 norm for the reprojection errors
        """
        ls_params_L1 = {'loss': 'soft_l1', 'f_scale': 0.5, 'max_iter': 50}
        _, self.ba_sol, err, iters = ba_core.run_ba_optimization(self.ba_params, ls_params=ls_params_L1,
                                                                 verbose=verbose, plots=self.display_plots)
        self.init_e, self.ba_e, self.init_e_cam, self.ba_e_cam = err
        self.ba_iters += iters


    def run_ba_L2(self, verbose=False):
        """
        Run the bundle adjustment optimization with classic L2 norm for the reprojection errors
        """
        _, self.ba_sol, err, iters = ba_core.run_ba_optimization(self.ba_params, ls_params=None,
                                                                 verbose=verbose, plots=self.display_plots)
        self.init_e, self.ba_e, self.init_e_cam, self.ba_e_cam = err
        self.ba_iters += iters


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

        #from bundle_adjust.ba_outliers import rm_outliers_based_on_reprojection_error_global
        #self.ba_params = rm_outliers_based_on_reprojection_error_global(self.ba_e, self.ba_params, verbose=verbose)

        start = timeit.default_timer()
        pts_ind_rm, cam_ind_rm, cam_thr = ba_outliers.compute_obs_to_remove(self.ba_e, self.ba_params)
        self.ba_params = ba_outliers.rm_outliers(self.ba_params, pts_ind_rm, cam_ind_rm, cam_thr,
                                                 verbose=verbose, correction_params=self.correction_params)
        running_time = timeit.default_timer() - start
        print('Removal of outliers based on reprojection error completed in {:.2f} seconds'.format(running_time))

    def save_initial_matrices(self):
        """
        Write initial projection matrices to json files
        """
        out_dir = os.path.join(self.output_dir, 'P_init')
        fnames = [os.path.join(out_dir, loader.get_id(fn)+'_pinhole.json') for fn in self.myimages]
        loader.save_projection_matrices(fnames, self.cameras, self.crop_offsets)
        print('\nInitial projection matrices written at {}\n'.format(out_dir), flush=True)
    
    
    def save_corrected_matrices(self):
        """
        Write corrected projection matrices to json files
        """
        out_dir = os.path.join(self.output_dir, 'P_adj')
        fnames = [os.path.join(out_dir, loader.get_id(fn)+'_pinhole_adj.json') for fn in self.myimages]
        loader.save_projection_matrices(fnames, self.corrected_cameras, self.crop_offsets)
        print('Bundle adjusted projection matrices written at {}\n'.format(out_dir), flush=True)


    def save_corrected_rpcs(self): 
        """
        Write corrected rpc model coefficients to txt files
        """
        out_dir = os.path.join(self.output_dir, 'RPC_adj')
        fnames = [os.path.join(out_dir, loader.get_id(fn)+'_RPC_adj.txt') for fn in self.myimages]
        for cam_idx, (fn, cam) in enumerate(zip(fnames, self.corrected_cameras)):
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            if self.cam_model in ['perspective', 'affine']:
                rpc_calib, err = rpc_fit.fit_rpc_from_projection_matrix(cam, self.ba_params.pts3d_ba)
                to_print = [cam_idx, 1e4*err.max(), 1e4*err.mean()]
                print('cam {:2} - RPC fit error per obs [1e-4 px] (max / avg): {:.2f} / {:.2f}'.format(*to_print))
                rpc_calib.write_to_file(fn)
            else:
                cam.write_to_file(fn)
        print('Bundle adjusted rpcs written at {}\n'.format(out_dir), flush=True)


    def save_corrected_points(self):
        """
        Write corrected 3d points produced by the bundle adjustment process into a ply file
        """
        pts3d_adj_ply_path = os.path.join(self.output_dir, 'pts3d_adj.ply')
        ba_utils.write_point_cloud_ply(pts3d_adj_ply_path, self.ba_params.pts3d_ba)
        print('Bundle adjusted 3d points written at {}\n'.format(pts3d_adj_ply_path), flush=True)


    def visualize_feature_track(self, track_idx=None):
        """
         Visualize feature track before (and after, if optimization results are available) bundle adjustment
         """

        # load feature tracks and initial 3d pts + corrected 3d pts if available (also ignore outlier observations)
        from bundle_adjust.ba_triangulate import init_pts3d
        pts3d, C = self.pts3d.copy(), self.C.copy()
        ba_available = self.corrected_pts3d is not None and self.corrected_cameras is not None
        if ba_available:
            C = C[:, self.ba_params.pts_prev_indices]
            pts3d = pts3d[self.ba_params.pts_prev_indices, :]
            #pts3d_ba = self.corrected_pts3d[self.ba_params.pts_prev_indices, :]
            pts3d_ba = self.ba_params.pts3d_ba

        # pick a random track index in case none was specified
        true_where_track = np.sum(1*~np.isnan(C[::2, :])[-self.n_new:],axis=0).astype(bool)
        pt_ind = np.random.choice(np.arange(C.shape[1])[true_where_track]) if (track_idx is None) else track_idx
        # get indices of the images where the selected track is visible
        n_img = self.n_adj + self.n_new
        im_ind = [k for k, j in enumerate(range(n_img)) if not np.isnan(C[j*2, pt_ind])]
        print('Displaying feature track with index {}, length {}\n'.format(pt_ind, len(im_ind)))

        # get original xyz coordinates before and after BA
        pt3d = pts3d[pt_ind, :]
        if ba_available:
            pt3d_ba = pts3d_ba[pt_ind, :]
            print('3D location (before BA): ', pt3d.ravel())
            print('3D location (after BA):  ', pt3d_ba.ravel(), '\n')

        # reproject feature track
        err_init, err_ba = [], []
        for i in im_ind:
            # feature track observation at current image
            pt2d_obs = C[(i*2):(i*2+2), pt_ind]
            if self.cam_model == 'rpc':
                pt2d_obs[0] -= self.crop_offsets[i]['col0']
                pt2d_obs[1] -= self.crop_offsets[i]['row0']

            # reprojection with initial variables
            pt2d_init = camera_utils.project_pts3d(self.cameras[i], self.cam_model, pt3d[np.newaxis, :])
            if self.cam_model == 'rpc':
                pt2d_init[0][0] -= self.crop_offsets[i]['col0']
                pt2d_init[0][1] -= self.crop_offsets[i]['row0']

            err_init.append(np.sqrt(np.sum((pt2d_init.ravel() - pt2d_obs) ** 2)))
            # reprojection with adjusted variables
            if ba_available:
                pt2d_ba = camera_utils.project_pts3d(self.corrected_cameras[i], self.cam_model, pt3d_ba[np.newaxis, :])
                if self.cam_model == 'rpc':
                    pt2d_ba[0][0] -= self.crop_offsets[i]['col0']
                    pt2d_ba[0][1] -= self.crop_offsets[i]['row0']
                err_ba.append(np.sqrt(np.sum((pt2d_ba.ravel() - pt2d_obs) ** 2)))
            # compute reprojection error before and after
            print(' --> Real 2D loc in im {} (yellow):         {}'.format(i, pt2d_obs))
            args = [i, pt2d_init.ravel(), err_init[-1]]
            print(' --> Proj 2D loc in im {} before BA (red):  {} (reprojection err: {:.3f})'.format(*args))
            if ba_available:
                args = [i, pt2d_ba.ravel(), err_ba[-1]]
                print(' --> Proj 2D loc in im {} after BA (green): {} (reprojection err: {:.3f})'.format(*args))
            # display
            plt.figure(figsize=(10,20))
            plt.imshow(loader.custom_equalization(self.input_seq[i]), cmap="gray")
            plt.plot(*pt2d_obs, "yo")
            plt.plot(*pt2d_init[0], "ro")
            if ba_available:
                plt.plot(*pt2d_ba[0], "go")
            plt.show()
            
        print('Mean reprojection error before BA: {:.3f}'.format(np.mean(err_init)))
        if ba_available:
            print('Mean reprojection error after BA:  {:.3f}'.format(np.mean(err_ba)))


    def analyse_reprojection_error_per_image(self, im_idx):
        """
        Compute and analyse in detail the reprojection error of a specfic image
        """

        # pick all track observations and the corresponding 3d points visible in the selected image
        true_if_3d_pt_seen_in_cam = ~np.isnan(self.ba_params.C[im_idx*2, :])
        obs2d = self.ba_params.C[(im_idx*2):(im_idx*2+2), true_if_3d_pt_seen_in_cam].T
        pts3d = self.pts3d[self.ba_params.pts_prev_indices, :]
        pts3d_before = pts3d[true_if_3d_pt_seen_in_cam, :]
        pts3d_after = self.ba_params.pts3d_ba[true_if_3d_pt_seen_in_cam, :]
        #pts3d_ba = self.corrected_pts3d[self.ba_params.pts_prev_indices, :]
        #pts3d_after = pts3d_ba[true_if_3d_pt_seen_in_cam, :]

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
                self.ba_params.cam_model, self.ba_params.pairs_to_triangulate, self.optical_centers]
        C_reproj = ft_ranking.compute_C_reproj(*args)
        cam_weights = ft_ranking.compute_camera_weights(self.ba_params.C, C_reproj)
        return cam_weights


    def save_feature_tracks_as_svg(self, output_dir, img_indices=None, save_reprojected=True):

        from feature_tracks.ft_utils import save_pts2d_as_svg, save_sequence_features_svg
        img_indices = np.arange(self.n_adj + self.n_new) if img_indices is None else img_indices

        # save all sift keypoints
        sift_dir = os.path.join(output_dir, 'sift')
        save_sequence_features_svg(sift_dir, np.array(self.myimages)[img_indices].tolist(), self.features)
        print('\nSIFT keypoints were saved at {}'.format(os.path.join(output_dir, 'sift')))

        # save all track observations
        tracks_dir = os.path.join(output_dir, 'feature_tracks')
        os.makedirs(tracks_dir, exist_ok=True)
        for im_idx in img_indices:
            h, w = self.input_seq[im_idx].shape
            svg_fname = os.path.join(tracks_dir, '{}.svg'.format(loader.get_id(self.myimages[im_idx])))
            # pick all feature track observations visible in the current image and save them as svg
            pts2d = self.C[(im_idx*2):(im_idx*2+2),~np.isnan(self.C[im_idx*2,:])].T
            save_pts2d_as_svg(svg_fname, pts2d, w=w, h=h, c='yellow')

        print('\nFeature track observations were saved at {}\n'.format(tracks_dir ))


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
        true_if_track = (np.sum(~(np.isnan(self.C[2*np.array(img_indices), :])),axis=0)>1).astype(bool)
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


    def select_best_tracks(self, priority=['length', 'scale', 'cost'], verbose=False):

        if self.tracks_config['FT_K'] > 0:
            from feature_tracks import ft_ranking
            args_C_scale = [self.C_v2, self.features]
            C_scale = ft_ranking.compute_C_scale(*args_C_scale)
            if self.pts3d is not None:
                args_C_reproj = [self.C, self.pts3d, self.cameras, self.cam_model, self.pairs_to_triangulate]
                C_reproj = ft_ranking.compute_C_reproj(*args_C_reproj)
            else:
                C_reproj = np.zeros(C_scale.shape)

            true_if_new_track = np.sum(~np.isnan(self.C[::2, :])[-self.n_new:] * 1, axis=0).astype(bool)
            C_new = self.C[:, true_if_new_track]
            C_scale_new = C_scale[:, true_if_new_track]
            C_reproj_new = C_reproj[:, true_if_new_track]
            prev_track_indices = np.arange(len(true_if_new_track))[true_if_new_track]
            selected_track_indices = ft_ranking.select_best_tracks(C_new, C_scale_new, C_reproj_new,
                                                                   K=self.tracks_config['FT_K'], priority=priority,
                                                                   verbose=verbose)
            selected_track_indices = prev_track_indices[np.array(selected_track_indices)]

            self.C = self.C[:, selected_track_indices]
            self.C_v2 = self.C_v2[:, selected_track_indices]
            self.n_pts_fix = len(selected_track_indices[selected_track_indices < self.n_pts_fix])
            if self.pts3d is not None:
                self.pts3d = self.pts3d[selected_track_indices, :]

    def check_connectivity_graph(self, min_matches=10, verbose=False):
        from bundle_adjust.ba_utils import build_connectivity_graph
        _, n_cc, _, _, missing_cams = build_connectivity_graph(self.C, min_matches=min_matches, verbose=verbose)
        err_msg = 'Insufficient SIFT matches'
        if n_cc > 1:
            args = [err_msg, n_cc, min_matches]
            raise Error('{}: Connectivity graph has {} connected components (min_matches = {})'.format(*args))
        if len(missing_cams) > 0:
            args = [err_msg, len(missing_cams), missing_cams]
            raise Error('{}: Found {} cameras missing in the connectivity graph (nodes: {})'.format(*args))


    def fix_reference_camera(self):

        # part 1: identify the camera connected to more cameras with highest number of observations

        from feature_tracks import ft_ranking
        neighbor_nodes_per_cam = np.sum(ft_ranking.build_connectivity_matrix(self.C, 10)>0, axis=1)
        obs_per_cam = np.sum(1 * ~np.isnan(self.C), axis=1)[::2]

        n_cam = int(self.C.shape[0]/2)
        cams_dtype = [('neighbor_nodes', int), ('obs', int)]
        cam_values = np.array(list(zip(neighbor_nodes_per_cam, obs_per_cam)), dtype=cams_dtype)
        ranked_cams = dict(list(zip(np.argsort(cam_values, order=['neighbor_nodes', 'obs'])[::-1], np.arange(n_cam))))

        ordered_cam_indices = sorted(np.arange(n_cam), key=lambda idx: ranked_cams[idx])
        ref_cam_idx = ordered_cam_indices[0]

        def rearange_corresp_matrix(C, ref_cam_idx):
            C = np.vstack([C[2*ref_cam_idx:2*ref_cam_idx+2, :], C])
            C = np.delete(C, 2*(ref_cam_idx+1), axis=0)
            C = np.delete(C, 2*(ref_cam_idx+1), axis=0)
            return C

        def rearange_list(input_list, new_indices):
            new_list = [input_list[idx] for idx in np.argsort(new_indices)]
            return new_list

        # part 2: the reference camera will be fix so it goes on top of all lists to work with the code
        #         rearange input rpcs, myimages, crops, footprints, C, C_v2, pairs_to_triangulate, etc.

        self.n_adj += 1
        self.n_new -= 1
        self.C = rearange_corresp_matrix(self.C, ref_cam_idx)
        C_v2 = np.vstack([self.C_v2[ref_cam_idx, :], self.C_v2])
        self.C_v2 = np.delete(C_v2, ref_cam_idx+1, axis=0)
        new_cam_indices = np.arange(n_cam)
        new_cam_indices[new_cam_indices < ref_cam_idx] += 1
        new_cam_indices[ref_cam_idx] = 0
        new_pairs_to_triangulate = []
        for (cam_i, cam_j) in self.pairs_to_triangulate:
            new_cam_i, new_cam_j = new_cam_indices[cam_i], new_cam_indices[cam_j]
            new_pairs_to_triangulate.append((min(new_cam_i, new_cam_j), max(new_cam_i, new_cam_j)))
        self.pairs_to_triangulate = new_pairs_to_triangulate
        self.input_rpcs = rearange_list(self.input_rpcs, new_cam_indices)
        self.input_seq = rearange_list(self.input_seq, new_cam_indices)
        self.myimages = rearange_list(self.myimages, new_cam_indices)
        self.crop_offsets = rearange_list(self.crop_offsets, new_cam_indices)
        self.optical_centers = rearange_list(self.optical_centers, new_cam_indices)
        self.footprints = rearange_list(self.footprints, new_cam_indices)
        self.cameras = rearange_list(self.cameras, new_cam_indices)
        self.features = rearange_list(self.features, new_cam_indices)
        if self.input_masks is not None:
            self.input_masks = rearange_list(self.input_masks, new_cam_indices)

        print('Using input image {} as reference image of the set'.format(ref_cam_idx))
        print('Reference geotiff: {}'.format(self.myimages[0]))
        print('After this step the camera indices are modified to put the ref. camera at position 0,')
        print('so they are not coincident anymore with the indices from the feature tracking step')


    def plot_reprojection_error_over_aoi(self, before_ba=False, thr=1.0, r=1.0, s=2):

        if before_ba:
            pts3d_ecef, err = self.ba_params.pts3d, self.init_e
        else:
            pts3d_ecef, err = self.corrected_pts3d[self.ba_params.pts_prev_indices, :], self.ba_e
        args = [err, pts3d_ecef, self.ba_params.cam_ind, self.ba_params.pts_ind, self.aoi, r, thr, s]
        ba_utils.plot_heatmap_reprojection_error(*args)


    def remove_all_obs_with_reprojection_error_higher_than(self, thr):

        t0 = timeit.default_timer()
        self.define_ba_parameters(verbose=False)
        _, _, err, _ = ba_core.run_ba_optimization(self.ba_params, ls_params={'max_iter': 1, 'verbose': 0},
                                                   verbose=False, plots=self.display_plots)
        _, ba_e, _, _ = err

        #plt.plot(sorted(ba_e))
        #plt.show()

        to_rm = ba_e > thr
        self.C_v2[self.ba_params.cam_ind[to_rm], self.ba_params.pts_ind[to_rm]] = np.nan
        p = ba_outliers.rm_outliers(self.ba_params, self.ba_params.pts_ind[to_rm], self.ba_params.cam_ind[to_rm],
                                    thr, correction_params=self.correction_params, verbose=False)
        self.C = p.C
        self.pts3d = p.pts3d
        self.n_pts_fix = p.n_pts_fix
        t = timeit.default_timer() - t0
        print('\nSanity check:')
        print('Removed {} obs with reprojection error above {} px ({:.2f} seconds)\n'.format(sum(1*to_rm), thr, t))


    def save_estimated_params(self):
        for cam_idx, cam_prev_idx in enumerate(self.ba_params.cam_prev_indices):
            cam_id = loader.get_id(self.myimages[cam_prev_idx])
            params_fname = '{}/ba_params/{}.params'.format(self.output_dir, cam_id)
            os.makedirs(os.path.dirname(params_fname), exist_ok=True)
            params_file = open(params_fname, 'w')
            for k in self.ba_params.estimated_params[cam_idx].keys():
                params_file.write('{}\n'.format(k))
                params_file.write(' '.join(['{:.16f}'.format(v) for v in self.ba_params.estimated_params[cam_idx][k]]))
                params_file.write('\n')
            params_file.close()
        print('All BA estimated camera parameters written at {}'.format(os.path.dirname(params_fname)))


    def run(self):

        pipeline_start = timeit.default_timer()
        # feature tracking stage
        self.compute_feature_tracks()
        self.initialize_pts3d(verbose=self.verbose)
        if self.max_init_reproj_error is not None:
            self.remove_all_obs_with_reprojection_error_higher_than(thr=self.max_init_reproj_error)
        self.select_best_tracks(priority=self.tracks_config['FT_priority'], verbose=self.verbose)
        self.check_connectivity_graph(verbose=self.verbose)

        # bundle adjustment stage
        if self.fix_ref_cam:
            self.fix_reference_camera()
        t0 = timeit.default_timer()
        self.define_ba_parameters(verbose=self.verbose)
        if self.clean_outliers:
            self.run_ba_softL1(verbose=self.verbose)
            self.clean_outlier_observations(verbose=self.verbose)
        self.run_ba_L2(verbose=self.verbose)
        optimization_time = loader.get_time_in_hours_mins_secs(timeit.default_timer() - t0)
        print('Optimization problem solved in {} ({} iterations)\n'.format(optimization_time, self.ba_iters))

        # save output
        self.reconstruct_ba_result()
        if self.cam_model in ['perspective', 'affine']:
            self.save_corrected_matrices()
        self.save_corrected_rpcs()
        self.save_corrected_points()
        self.save_estimated_params()

        pipeline_time = loader.get_time_in_hours_mins_secs(timeit.default_timer() - pipeline_start)
        print('BA pipeline completed in {}\n'.format(pipeline_time))
