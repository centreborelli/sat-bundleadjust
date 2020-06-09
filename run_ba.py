import numpy as np
import ba_utils
import pickle
import os
import time
import json
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from PIL import Image


class BundleAdjustmentPipeline:
    def __init__(self, ba_input_data, use_masks=False, use_opencv_sift=True):
        
        self.input_dir = ba_input_data['input_dir']
        self.n_adj = ba_input_data['n_adj']
        self.n_new = ba_input_data['n_new']
        self.myimages = ba_input_data['myimages'].copy()
        self.crop_offsets = [{'x0':f['x0'], 'y0':f['y0']} for f in ba_input_data['input_crops']] 
        self.input_seq = [f['crop'] for f in ba_input_data['input_crops']]
        self.input_masks = ba_input_data['input_masks'].copy() if ba_input_data['input_masks'] is not None else None
        self.input_rpcs = ba_input_data['input_rpcs'].copy()
        self.input_P = ba_input_data['input_P'].copy()
        self.footprints = ba_input_data['input_footprints'].copy()
        self.cam_model = ba_input_data['cam_model']
        
        # stuff to be filled by 'run_feature_detection'
        self.features = []
        self.all_pairwise_matches = []
        self.pairs_to_match = []
        self.pairs_to_triangulate = []
        self.matching_thr = 0.6
        self.feature_detection_use_masks = use_masks
        self.use_opencv_sift = use_opencv_sift
        self.C = None
        
        # stuff to be filled by 'define_ba_parameters'
        self.params_opt = None
        self.cam_params = None 
        self.pts_3d = None 
        self.pts_2d = None 
        self.cam_ind = None 
        self.pts_ind = None 
        self.ba_params = None
        
        # stuff to be filled by 'run_ba_optimization'
        self.pts_3d_ba = None
        self.cam_params_ba = None
        self.P_crop_ba = None
        self.ba_e = None
        
        
        
        
        
        
    def run_feature_detection(self):


        # FEATURE DETECTION + MATCHING ON THE NEW IMAGES
        import timeit

        print('Running OpenCV based feature detection...\n')
        print('Parameters:')
        print('      use_masks:    {}'.format(self.feature_detection_use_masks))
        print('      matching_thr: {}'.format(self.matching_thr))
        print('\n')
        start = timeit.default_timer()
        last_stop = start

        if self.feature_detection_use_masks and self.input_masks is None:
            print('Feature detection is set to use masks to restrict the search of keypoints, but no masks were found !')
            print('No masks will be used\n')
              
        # feature detection on the new view(s)
        kp_cont, self.features = 0, [] 
        # load previous feature tracks if existent
        if self.n_adj > 0:
            pickle_in = open(self.input_dir+'/features.pickle','rb')
            self.features = pickle.load(pickle_in)
            kp_cont = self.features[-1]['id'][-1] + 1
            print('Previous features loaded!')
        new_indices = np.arange(self.n_adj, self.n_adj + self.n_new)
        new_input_seq = [self.input_seq[idx].astype(np.uint8) for idx in new_indices]
        new_input_rpcs = [self.input_rpcs[idx] for idx in new_indices]
        new_footprints = [self.footprints[idx] for idx in new_indices]
        if self.input_masks is not None and self.feature_detection_use_masks:
            new_masks = [self.input_masks[idx] for idx in new_indices]
            new_features = ba_utils.feature_detection_skysat(new_input_seq, new_input_rpcs, new_footprints, new_masks)
        else:
            new_features = ba_utils.feature_detection_skysat(new_input_seq, new_input_rpcs, new_footprints)
        for current_features in new_features:
            current_features['id'] += kp_cont
        self.features.extend(new_features)

        pickle_out = open(self.input_dir+'/features.pickle','wb')
        pickle.dump(self.features, pickle_out)
        pickle_out.close()
        print('\nDetected features saved!')
        stop = timeit.default_timer()
        print('\n...done in {} seconds'.format(stop - last_stop))
        last_stop = stop

        print('\nComputing pairs to be matched...\n')
        # load previous matches and list of paris to be matched/triangulate if existent
        if self.n_adj > 0:
            pickle_in = open(self.input_dir+'/matches.pickle','rb')
            self.all_pairwise_matches, self.pairs_to_match, self.pairs_to_triangulate = pickle.load(pickle_in)
            print('Previous matches loaded!')

        # possible new pairs to match are composed by 1 + 2 
        # 1. each of the previously adjusted images with the new ones
        possible_pairs = []
        for i in np.arange(self.n_adj):
            for j in np.arange(self.n_adj, self.n_adj + self.n_new):
                possible_pairs.append((i, j))       
        # 2. each of the new images with the rest of the new images
        for i in np.arange(self.n_adj, self.n_adj + self.n_new):
            for j in np.arange(i+1, self.n_adj + self.n_new):
                possible_pairs.append((i, j))

        # filter stereo pairs that are not overlaped
        # stereo pairs with small baseline should not be used to triangulate 
        pairs2match, pairs2triangulate = ba_utils.filter_pairs_to_match_skysat(possible_pairs, \
                                                                               self.footprints, \
                                                                               self.input_P)
        self.pairs_to_match.extend(pairs2match)
        self.pairs_to_triangulate.extend(pairs2triangulate)
        print('{} new pairs to be matched'.format(len(pairs2match)))

        stop = timeit.default_timer()
        print('\n...done in {} seconds'.format(stop - last_stop))
        last_stop = stop

        print('\nMatching...\n')
        new_pairwise_matches = ba_utils.matching_skysat(pairs2match, self.features, self.matching_thr) 
        self.all_pairwise_matches.extend(new_pairwise_matches)

        pickle_out = open(self.input_dir+'/matches.pickle','wb')
        pickle.dump([self.all_pairwise_matches, self.pairs_to_match, self.pairs_to_triangulate], pickle_out)
        pickle_out.close()
        print('\nPairwise matches saved!')      
        stop = timeit.default_timer()
        print('\n...done in {} seconds'.format(stop - last_stop))
        last_stop = stop

        print('\nBuilding feature tracks...\n')
        self.C = ba_utils.feature_tracks_from_pairwise_matches(self.features, \
                                                               self.all_pairwise_matches, \
                                                               self.pairs_to_triangulate)
        pickle_out = open(self.input_dir+'/Cmatrix.pickle','wb')
        pickle.dump(self.C, pickle_out)
        pickle_out.close()
        print('\nCorrespondence matrix saved!')
        stop = timeit.default_timer()
        print('\n...done in {} seconds'.format(stop - last_stop))
        last_stop = stop

        hours, rem = divmod(last_stop - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print('\nTotal time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds))
    
    
    
    
    def run_feature_detection_s2p(self):


        # FEATURE DETECTION + MATCHING ON THE NEW IMAGES
        import timeit

        print('Running s2p based feature detection...\n')
        start = timeit.default_timer()
        last_stop = start

        # feature detection on the new view(s)
        kp_cont, self.features = 0, [] 
        # load previous feature tracks if existent
        if self.n_adj > 0:
            pickle_in = open(self.input_dir+'/features.pickle','rb')
            self.features = pickle.load(pickle_in)
            kp_cont = self.features[-1]['id'][-1] + 1
            print('Previous features loaded!')
        new_indices = np.arange(self.n_adj, self.n_adj + self.n_new)
        new_input_seq = [self.input_seq[idx] for idx in new_indices]
        new_input_rpcs = [self.input_rpcs[idx] for idx in new_indices]
        new_footprints = [self.footprints[idx] for idx in new_indices]
        if self.input_masks is not None and self.feature_detection_use_masks:
            new_masks = [self.input_masks[idx] for idx in new_indices]
            new_features = ba_utils.feature_detection_skysat_s2p(new_input_seq, new_input_rpcs, new_footprints, new_masks)
        else:
            new_features = ba_utils.feature_detection_skysat_s2p(new_input_seq, new_input_rpcs, new_footprints)
        for current_features in new_features:
            current_features['id'] += kp_cont
        self.features.extend(new_features)

        pickle_out = open(self.input_dir+'/features.pickle','wb')
        pickle.dump(self.features, pickle_out)
        pickle_out.close()
        print('\nDetected features saved!')
        stop = timeit.default_timer()
        print('\n...done in {} seconds'.format(stop - last_stop))
        last_stop = stop

        print('\nComputing pairs to be matched...\n')
        # load previous matches and list of paris to be matched/triangulate if existent
        if self.n_adj > 0:
            pickle_in = open(self.input_dir+'/matches.pickle','rb')
            self.all_pairwise_matches, self.pairs_to_match, self.pairs_to_triangulate = pickle.load(pickle_in)
            print('Previous matches loaded!')

        # possible new pairs to match are composed by 1 + 2 
        # 1. each of the previously adjusted images with the new ones
        possible_pairs = []
        for i in np.arange(self.n_adj):
            for j in np.arange(self.n_adj, self.n_adj + self.n_new):
                possible_pairs.append((i, j))       
        # 2. each of the new images with the rest of the new images
        for i in np.arange(self.n_adj, self.n_adj + self.n_new):
            for j in np.arange(i+1, self.n_adj + self.n_new):
                possible_pairs.append((i, j))

        # filter stereo pairs that are not overlaped
        # stereo pairs with small baseline should not be used to triangulate 
        pairs2match, pairs2triangulate = ba_utils.filter_pairs_to_match_skysat(possible_pairs, \
                                                                               self.footprints, \
                                                                               self.input_P)
        self.pairs_to_match.extend(pairs2match)
        self.pairs_to_triangulate.extend(pairs2triangulate)
        print('{} new pairs to be matched'.format(len(pairs2match)))

        stop = timeit.default_timer()
        print('\n...done in {} seconds'.format(stop - last_stop))
        last_stop = stop

        print('\nMatching...\n')
        new_pairwise_matches = ba_utils.matching_skysat_s2p(pairs2match, self.features, self.input_seq, self.input_rpcs) 
        self.all_pairwise_matches.extend(new_pairwise_matches)

        pickle_out = open(self.input_dir+'/matches.pickle','wb')
        pickle.dump([self.all_pairwise_matches, self.pairs_to_match, self.pairs_to_triangulate], pickle_out)
        pickle_out.close()
        print('\nPairwise matches saved!')      
        stop = timeit.default_timer()
        print('\n...done in {} seconds'.format(stop - last_stop))
        last_stop = stop

        print('\nBuilding feature tracks...\n')
        self.C = ba_utils.feature_tracks_from_pairwise_matches(self.features, \
                                                               self.all_pairwise_matches, \
                                                               self.pairs_to_triangulate)
        pickle_out = open(self.input_dir+'/Cmatrix.pickle','wb')
        pickle.dump(self.C, pickle_out)
        pickle_out.close()
        print('\nCorrespondence matrix saved!')
        stop = timeit.default_timer()
        print('\n...done in {} seconds'.format(stop - last_stop))
        last_stop = stop

        hours, rem = divmod(last_stop - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print('\nTotal time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds))
        
        
        
        
    
    def define_ba_parameters(self, verbose=False):

        '''
        INPUT PARAMETERS FOR BUNDLE ADJUSTMENT
        'cam_params': (n_cam, 12), initial projection matrices. 1 row = 1 camera estimate.
                      first 3 elements of each row = R vector, next 3 = T vector, then f and two dist. coef.
        'pts_3d'    : (n_pts, 3) contains the initial estimates of the 3D points in the world frame.
        'cam_ind'   : (n_observations,), indices of cameras (from 0 to n_cam - 1) involved in each observation.
        'pts_ind'   : (n_observations,) indices of points (from 0 to n_points - 1) involved in each observation.
        'pts_2d'    : (n_observations, 2) 2-D coordinates of points projected on images in each observations.
        '''

        print('Defining BA input parameters...')
        self.params_opt, self.cam_params, self.pts_3d, self.pts_2d, self.cam_ind, self.pts_ind, self.ba_params \
        = ba_utils.set_ba_params(self.input_P, self.C, self.cam_model, self.n_adj, self.n_new, self.pairs_to_triangulate)
        print('...done!\n')

        if verbose:
            print('pts_2d.shape:{}  pts_ind.shape:{}  cam_ind.shape:{}'.format(self.pts_2d.shape, \
                                                                               self.pts_ind.shape, \
                                                                               self.cam_ind.shape))
            print('pts_3d.shape:{}  cam_params.shape:{}\n'.format(self.pts_3d.shape, \
                                                                  self.cam_params.shape))
            print('Bundle Adjustment parameters defined')

            if self.ba_params['n_params'] > 0 and self.ba_params['opt_X']:
                print('  -> Both camera parameters and 3D points will be optimized')
            elif self.ba_params['n_params'] > 0 and not self.ba_params['opt_X']:
                print('  -> Only the camera parameters will be optimized')
            else:
                print('  -> Only 3D points will be optimized')
        

        
    def run_ba_optimization(self, input_loss='linear', input_f_scale=1.0, input_ftol=1e-8, input_xtol=1e-8):
        
        # assign a weight to each observation
        pts_2d_w = np.ones(self.pts_2d.shape[0])

        # compute loss value and plot residuals at the initial parameters
        f0 = ba_utils.fun(self.params_opt, self.cam_ind, self.pts_ind, self.pts_2d, \
                          self.cam_params, self.pts_3d, self.ba_params, pts_2d_w)
        plt.plot(f0)

        # define jacobian
        A = ba_utils.bundle_adjustment_sparsity(self.cam_ind, self.pts_ind, self.ba_params)

        # run bundle adjustment
        t0 = time.time()
        res = least_squares(ba_utils.fun, self.params_opt, jac_sparsity=A, verbose=1, x_scale='jac',
                            method='trf', ftol=input_ftol, xtol=input_xtol, loss=input_loss, f_scale=input_f_scale,
                            args=(self.cam_ind, self.pts_ind, self.pts_2d, self.cam_params, \
                                  self.pts_3d, self.ba_params, pts_2d_w))

        t1 = time.time()
        print("Optimization took {0:.0f} seconds\n".format(t1 - t0))

        #plot residuals at the found solution
        plt.plot(res.fun);

        # recover BA output
        self.pts_3d_ba, self.cam_params_ba, self.P_crop_ba \
        = ba_utils.get_ba_output(res.x, self.ba_params, self.cam_params, self.pts_3d)

        # check BA error performance
        self.ba_e = ba_utils.check_ba_error(f0, res.fun, pts_2d_w)
        
    def run_ba_softL1(self):
        input_loss = 'soft_l1'
        input_f_scale = 0.5
        input_ftol = 1e-4
        input_xtol = 1e-8
        self.run_ba_optimization(input_loss, input_f_scale, input_ftol, input_xtol)
        
    def run_ba_L2(self):
        self.run_ba_optimization()
    
    def clean_outlier_obs(self):

        elbow_value = ba_utils.get_elbow_value(self.ba_e, 95)
        fig = plt.figure()
        plt.plot(np.sort(self.ba_e))
        plt.axhline(y=elbow_value, color='r', linestyle='-')
        plt.show()
        self.C = ba_utils.remove_outlier_obs(self.ba_e, self.pts_ind, self.cam_ind, \
                                           self.C, self.pairs_to_triangulate, thr=max(elbow_value,2.0))

        pickle_out = open(self.input_dir+'/Cmatrix2.pickle','wb')
        pickle.dump(self.C, pickle_out)
        pickle_out.close()
        print('Correspondence matrix saved!')
        self.define_ba_parameters()

        
    def save_corrected_matrices(self):
        
        os.makedirs(self.input_dir+'/P_adj', exist_ok=True)

        for im_idx in np.arange(self.n_adj, self.n_adj + self.n_new):

            P_calib_fn = os.path.basename(os.path.splitext(self.myimages[im_idx])[0])+'_pinhole_adj.json'
            to_write = {
                # 'P_camera'
                # 'P_extrinsic'
                # 'P_intrinsic'
                "P_projective": [self.P_crop_ba[im_idx][0,:].tolist(), 
                                 self.P_crop_ba[im_idx][1,:].tolist(),
                                 self.P_crop_ba[im_idx][2,:].tolist()],
                # 'exterior_orientation'
                "height": self.input_seq[im_idx].shape[0],
                "width": self.input_seq[im_idx].shape[1]        
            }

            with open(self.input_dir+'/P_adj/'+P_calib_fn, 'w') as json_file:
                json.dump(to_write, json_file, indent=4)


    def save_corrected_rpcs(self, check_rpc_fitting_error=False, verbose=False): 
        
        #fit rpc

        import rpc_fit
        import copy
        os.makedirs(self.input_dir+'/RPC_adj', exist_ok=True)
        
        # rpc fitting starts here
        myrpcs_calib = []
        if self.n_adj > 0:
            for im_idx in np.arange(self.n_adj):
                im_idx = int(im_idx)
                myrpcs_calib.append(copy.copy(self.input_rpcs[im_idx]))
        
        for im_idx in np.arange(self.n_adj, self.n_adj + self.n_new):
            im_idx = int(im_idx)
            
            # calibrate and get error
            rpc_init = copy.copy(self.input_rpcs[im_idx])                  
            current_P = self.P_crop_ba[im_idx].copy()
            current_im = self.input_seq[im_idx].copy()
            current_ecef = self.pts_3d_ba.copy()
            rpc_calib, err_calib = rpc_fit.fit_rpc_from_projection_matrix(rpc_init, current_P, current_im, current_ecef)
            print('image {}, RMSE calibrated RPC = {}'.format(im_idx, err_calib))

            rpc_calib_fn = os.path.basename(os.path.splitext(self.myimages[im_idx])[0])+'_RPC_adj.txt'
            rpc_calib.write_to_file(self.input_dir+'/RPC_adj/'+rpc_calib_fn)
            myrpcs_calib.append(rpc_calib)

            # check the histogram of errors if the RMSE error is above subpixel
            if err_calib > 1.0 and verbose:
                col_pred, row_pred = rpc_calib.projection(lon, lat, alt)
                err = np.sum(abs(np.hstack([col_pred.reshape(-1, 1), row_pred.reshape(-1, 1)]) - target), axis=1)
                plt.figure()
                plt.hist(err, bins=30);
                plt.show()
           
        if verbose:
            for im_idx in range(int(self.C.shape[0]/2)):
                for p_idx in range(self.pts_3d_ba.shape[0]):
                        p_2d_gt = self.C[(im_idx*2):(im_idx*2+2),p_idx]
                        current_p = self.pts_3d_ba[p_idx,:]
                        lat, lon, alt = ba_utils.ecef_to_latlon_custom(current_p[0], current_p[1], current_p[2])
                        proj = self.input_P[im_idx] @ np.expand_dims(np.hstack((current_p, np.ones(1))), axis=1)
                        p_2d_proj = (proj[0:2,:] / proj[-1,-1]).ravel()
                        col, row = self.input_rpcs[im_idx].projection(lon, lat, alt)
                        p_2d_proj_rpc = np.hstack([col - self.crop_offsets[im_idx]['x0'], \
                                                   row - self.crop_offsets[im_idx]['y0']]).ravel()
                        proj = self.P_crop_ba[im_idx] @ np.expand_dims(np.hstack((current_p, np.ones(1))), axis=1)
                        p_2d_proj_ba = (proj[0:2,:] / proj[-1,-1]).ravel()
                        col, row = myrpcs_calib[im_idx].projection(lon, lat, alt)
                        p_2d_proj_rpc_ba = np.hstack([col, row])

                        reprojection_error_P = np.sum(abs(p_2d_proj_ba - p_2d_gt))
                        reprojection_error_RPC = np.sum(abs(p_2d_proj_rpc_ba - p_2d_gt))

                        if abs(reprojection_error_RPC - reprojection_error_P) > 0.001:
                            print('GT location   : {:.4f} , {:.4f}'.format(p_2d_gt[0], p_2d_gt[1])) 
                            print('RPC proj      : {:.4f} , {:.4f}'.format(p_2d_proj_rpc[0], p_2d_proj_rpc[1]))
                            print('P proj        : {:.4f} , {:.4f}'.format(p_2d_proj[0], p_2d_proj[1]))
                            print('P proj   (BA) : {:.4f} , {:.4f}'.format(p_2d_proj_ba[0], p_2d_proj_ba[1]))
                            print('RPC proj (BA) : {:.4f} , {:.4f}'.format(p_2d_proj_rpc_ba[0], p_2d_proj_rpc_ba[1]))

                print('Finished checking image {}'.format(im_idx))
        
    
    def visualize_feature_track(self, feature_track_index=None):
    
        pts_3d_ba_available = self.pts_3d_ba is not None
        n_img = self.n_adj + self.n_new
        hC, wC = self.C.shape
        true_where_track = np.sum(np.invert(np.isnan(self.C[np.arange(0, hC, 2), :]))[-self.n_new:]*1,axis=0).astype(bool) 
        
        if feature_track_index is None:
            feature_track_index = np.random.choice(np.arange(0, wC)[true_where_track])
        p_ind = feature_track_index
        im_ind = [k for k, j in enumerate(range(n_img)) if not np.isnan(self.C[j*2,p_ind])]

        reprojection_error, reprojection_error_ba  = 0., 0.
        cont = -1

        print('Displaying feature track with index {}\n'.format(p_ind))
        
        for i in im_ind:   
            cont += 1

            p_2d_gt = self.C[(i*2):(i*2+2),p_ind]

            proj = self.input_P[i] @ np.expand_dims(np.hstack((self.pts_3d[p_ind,:], np.ones(1))), axis=1)
            p_2d_proj = proj[0:2,:] / proj[-1,-1]  # col, row
            
            if pts_3d_ba_available:
                proj = self.P_crop_ba[i] @ np.expand_dims(np.hstack((self.pts_3d_ba[p_ind,:], np.ones(1))), axis=1)
                p_2d_proj_ba = proj[0:2,:] / proj[-1,-1]

            if cont == 0 and pts_3d_ba_available:
                print('3D location (initial)  :', self.pts_3d[p_ind,:].ravel())
                print('3D location (after BA) :', self.pts_3d_ba[p_ind,:].ravel(), '\n')

            print(' ----> Real 2D loc in im', i, ' (sol) = ', p_2d_gt)
            print(' ----> Proj 2D loc in im', i, ' before BA = ', p_2d_proj.ravel())
            if pts_3d_ba_available:
                print(' ----> Proj 2D loc in im', i, ' after  BA = ', p_2d_proj_ba.ravel())
            print('              Reprojection error beofre BA:', np.sum(abs(p_2d_proj.ravel() - p_2d_gt)))
            if pts_3d_ba_available:
                print('              Reprojection error after  BA:', np.sum(abs(p_2d_proj_ba.ravel() - p_2d_gt)))

            fig = plt.figure(figsize=(10,20))

            plt.imshow(self.input_seq[i], cmap="gray")
            plt.plot(*p_2d_gt, "yo")
            plt.plot(*p_2d_proj, "ro")
            plt.plot(*p_2d_proj_ba, "go")
            plt.show()
    
    
    def save_crops(self, output_dir, img_indices=None):
        
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        if img_indices is None:
            n_img = self.n_adj + self.n_new
            img_indices = np.arange(n_img)
            
        for im_idx in img_indices:
            f_id = os.path.splitext(os.path.basename(self.myimages[im_idx]))[0]
            Image.fromarray(self.input_seq[im_idx]).save(os.path.join(images_dir, '{}.tif'.format(f_id)))

        print('\nImage crops were saved at {}\n'.format(images_dir))
    
    
    def save_sift_kp_as_svg(self, output_dir, img_indices=None):
        
        sift_dir = os.path.join(output_dir, 'features/sift_all_kp')
        os.makedirs(sift_dir, exist_ok=True)
        
        if img_indices is None:
            n_img = self.n_adj + self.n_new
            img_indices = np.arange(n_img)
            
        for im_idx in img_indices:
            
            f_id = os.path.splitext(os.path.basename(self.myimages[im_idx]))[0]
            h,w = self.input_seq[im_idx].shape
            svg_fname = os.path.join(sift_dir, '{}.svg'.format(f_id))
            ba_utils.save_pts2d_as_svg(svg_fname, self.features[im_idx]['kp'], w, h, 'green')

        print('\nSIFT keypoints were saved at {}\n'.format(sift_dir))
    
    
    def save_feature_tracks_as_svg(self, output_dir, img_indices=None, save_reprojected=True):
        
        self.save_crops(output_dir, img_indices)
        self.save_sift_kp_as_svg(output_dir, img_indices)
        
        before_dir = os.path.join(output_dir, 'features/tracks_reproj_before')
        after_dir = os.path.join(output_dir, 'features/tracks_reproj_after')
        original_dir = os.path.join(output_dir, 'features/tracks_sift')
        os.makedirs(before_dir, exist_ok=True)
        os.makedirs(after_dir, exist_ok=True)
        os.makedirs(original_dir, exist_ok=True)
                
        if img_indices is None:
            n_img = self.n_adj + self.n_new
            img_indices = np.arange(n_img)
        
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
            ba_utils.save_pts2d_as_svg(svg_fname_o, pts2d, w, h, 'green')
            ba_utils.save_pts2d_as_svg(svg_fname_b, pts_reproj_before, w, h, 'red')
            ba_utils.save_pts2d_as_svg(svg_fname_a, pts_reproj_after, w, h, 'yellow')
            

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
    
    def get_number_of_matches_within_group_of_views(self, img_indices_g1):
        
        img_indices_g1_s = sorted(img_indices_g1)
        n_matches = 0
        for im1 in img_indices_g1_s:
            for im2 in np.array(img_indices_g1_s[im1:]).tolist():
                obs_im1 = 1*np.invert(np.isnan(self.C[2*im1,:]))
                obs_im2 = 1*np.invert(np.isnan(self.C[2*im2,:]))
                n_matches += np.sum(np.sum(np.vstack((obs_im1, obs_im2)), axis=0) == 2)
        return n_matches
            
    
    def run(self):
        
        os.makedirs(self.input_dir, exist_ok=True)
        pickle_out = open(self.input_dir+'/myimages.pickle','wb')
        pickle.dump([os.path.basename(imagefn) for imagefn in self.myimages], pickle_out)
        pickle_out.close()
        
        if self.use_opencv_sift:
            self.run_feature_detection()
        else:
            self.run_feature_detection_s2p()
        self.define_ba_parameters()
        self.run_ba_softL1()
        self.clean_outlier_obs()
        self.run_ba_L2()
        self.save_corrected_matrices()
        self.save_corrected_rpcs()
        
