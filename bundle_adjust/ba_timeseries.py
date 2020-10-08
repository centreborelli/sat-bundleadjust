import numpy as np
import matplotlib.pyplot as plt

import glob
import rpcm
import os
import pickle
import subprocess
import srtm4
from PIL import Image

from IS18 import vistools
from IS18 import utils

from bundle_adjust import ba_outofcore
from bundle_adjust import ba_core
from bundle_adjust import ba_utils
from bundle_adjust.ba_pipeline import BundleAdjustmentPipeline
from bundle_adjust import data_loader as loader
from bundle_adjust import ba_metrics
from bundle_adjust import geojson_utils

from contextlib import contextmanager
import sys


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout           


class Error(Exception):
    pass


class Scene:
    
    def __init__(self, scene_config):
        
        args = loader.load_dict_from_json(scene_config)
        
        # read scene args
        self.images_dir = args['geotiff_dir']
        self.s2p_configs_dir = args.get('s2p_configs_dir', '')
        self.rpc_src = args['rpc_src']
        self.dst_dir = args['output_dir']

        # camera model to adjust is currently fixed
        self.projmats_model = 'Perspective'
        
        # optional arguments    
        if 'dsm_resolution' in args.keys():
            self.dsm_resolution = args['dsm_resolution']
        else:
            self.dsm_resolution = 1
            
        if 'compute_aoi_masks' in args.keys():
            self.compute_aoi_masks = args['compute_aoi_masks']
        else:
            self.compute_aoi_masks = False
            
        if 'use_aoi_masks_to_equalize_crops' in args.keys():
            self.use_aoi_masks_to_equalize_crops = args['use_aoi_masks_to_equalize_crops']
        else:
            self.use_aoi_masks_to_equalize_crops = False

        if 'geotiff_label' in args.keys():
            self.geotiff_label = args['geotiff_label']
        else:
            self.geotiff_label = None
        
        # check geotiff_dir and s2p_configs_dir exist
        if not os.path.isdir(self.images_dir):
            raise Error('geotiff_dir does not exist')
        
        if self.s2p_configs_dir != '' and not os.path.isdir(self.s2p_configs_dir):
            raise Error('s2p_config_dir does not exist')

        # create output path
        os.makedirs(self.dst_dir, exist_ok=True)
        
        # needed to run bundle adjustment
        self.init_ba_input_data()
        self.tracks_config = None
        
        self.tracks_config = {'s2p': False,
                              'matching_thr': 0.6,
                              'use_masks': False,
                              'filter_pairs': True,
                              'max_kp': 3000,
                              'optimal_subset': False,
                              'K': 30,
                              'continue': True,
                              'tie_points': False}
        
        print('#############################################################')
        print('\nLoading scene from {}\n'.format(scene_config))
        print('-------------------------------------------------------------')
        print('Configuration:')
        print('    - images_dir:      {}'.format(self.images_dir))
        print('    - s2p_configs_dir: {}'.format(self.s2p_configs_dir))
        print('    - rpc_src:         {}'.format(self.rpc_src))
        print('    - output_dir:      {}'.format(self.dst_dir))
        print('-------------------------------------------------------------\n')
        
        if self.s2p_configs_dir == '':
            self.timeline, self.aoi_lonlat = loader.load_scene_from_geotiff_dir(self.images_dir, self.dst_dir,
                                                                                rpc_src=self.rpc_src,
                                                                                geotiff_label=self.geotiff_label)
            
            # TODO: create default s2p configs to reconstruct everything possible at a given resolution
            
        else:
            self.timeline, self.aoi_lonlat = loader.load_scene_from_s2p_configs(self.images_dir, 
                                                                                self.s2p_configs_dir, self.dst_dir,
                                                                                rpc_src=self.rpc_src,
                                                                                geotiff_label=self.geotiff_label)
        
        self.utm_bbx = loader.get_utm_bbox_from_aoi_lonlat(self.aoi_lonlat)
        self.mask = loader.get_binary_mask_from_aoi_lonlat_within_utm_bbx(self.utm_bbx, 1., self.aoi_lonlat)
        #self.display_dsm_mask()
        
        n_dates = len(self.timeline)
        start_date = self.timeline[0]['datetime'].date() 
        end_date = self.timeline[-1]['datetime'].date()
        print('Number of acquisition dates: {} (from {} to {})'.format(n_dates,
                                                                      start_date, end_date))
        print('Number of images: {}'.format(np.sum([d['n_images'] for d in self.timeline])))
        sq_km = geojson_utils.measure_squared_km_from_lonlat_geojson(self.aoi_lonlat)
        print('The aoi covers a total of {:.2f} squared km:'.format(sq_km))

        print('\n#############################################################\n\n')


        
        
        
    def get_timeline_attributes(self, timeline_indices, attributes):
        loader.get_timeline_attributes(self.timeline, timeline_indices, attributes)
    
    
    def display_aoi(self):
        geojson_utils.display_lonlat_geojson_list_over_map([self.aoi_lonlat], zoom_factor=14)
    
    def display_crops(self):
        mycrops = self.mycrops_adj + self.mycrops_new
        if len(mycrops) > 0:
            vistools.display_gallery([f['crop'] for f in mycrops])
        else:
            print('No crops have been loaded. Use load_data_from_date() to load them.')
    
    def display_image_masks(self):
        if not self.compute_aoi_masks:
            print('compute_aoi_masks is False')
        else:
            mycrops = self.mycrops_adj + self.mycrops_new
            if len(mycrops) > 0:
                vistools.display_gallery([255*f['mask'] for f in mycrops])
            else:
                print('No crops have been loaded. Use load_data_from_date() to load them.')
    
    def display_dsm_mask(self):
        if self.mask is None:
            print('The mask of this scene is not defined')
        else:
            fig = plt.figure(figsize=(10,10))
            plt.imshow(self.mask)
            plt.show()
            
    def check_adjusted_dates(self, input_dir):
        
        dir_adj_rpc = os.path.join(input_dir, 'RPC_adj')
        if os.path.exists(input_dir + '/filenames.pickle') and os.path.isdir(dir_adj_rpc):

            # read tiff images 
            adj_fnames = pickle.load(open(input_dir+'/filenames.pickle','rb'))
            print('Found {} previously adjusted images in {}\n'.format(len(adj_fnames), self.dst_dir))
            
            datetimes_adj = [loader.get_acquisition_date(img_geotiff_path) for img_geotiff_path in adj_fnames]
            timeline_adj = loader.group_files_by_date(datetimes_adj, adj_fnames)
            for d in timeline_adj:
                adj_id = d['id']
                adj_date = d['datetime']
                for idx in range(len(self.timeline)):
                    if self.timeline[idx]['id'] == adj_id:
                        self.timeline[idx]['adjusted'] = True
                        
            prev_adj_data_found=True      
        else:
            print('No previously adjusted data was found in {}\n'.format(self.dst_dir))
            prev_adj_data_found=False
            
        return prev_adj_data_found
            

    def load_data_from_dates(self, timeline_indices, input_dir, adjusted=False):
        
        im_fnames = []
        for t_idx in timeline_indices:
            im_fnames.extend(self.timeline[t_idx]['fnames'])
        print(len(im_fnames), '{} images for bundle adjustment ! \n'.format('adjusted' if adjusted else 'new'))
        
        if len(im_fnames) > 0:
            # get rpcs            
            rpc_dir = os.path.join(input_dir, 'RPC_adj') if adjusted else os.path.join(self.dst_dir, 'RPC_init')  
            rpc_suffix = 'RPC_adj' if adjusted else 'RPC'
            im_rpcs = loader.load_rpcs_from_dir(im_fnames, rpc_dir, suffix=rpc_suffix)
            # load previously adjusted projection matrices
            #self.myprojmats_adj = load_matrices_from_dir(im_fnames, os.path.join(input_dir, 'P_adj'))

            # get image crops
            im_crops = loader.load_image_crops(im_fnames, rpcs = im_rpcs,
                                               aoi = self.aoi_lonlat, 
                                               get_aoi_mask = self.compute_aoi_masks,
                                               use_aoi_mask_for_equalization = self.use_aoi_masks_to_equalize_crops)
            
        if adjusted:
            self.n_adj += len(im_fnames)
            self.myimages_adj.extend(im_fnames.copy())
            self.myrpcs_adj.extend(im_rpcs.copy())
            self.mycrops_adj.extend(im_crops.copy())
        else:
            self.n_new += len(im_fnames)
            self.myimages_new.extend(im_fnames.copy())
            self.myrpcs_new.extend(im_rpcs.copy())
            self.mycrops_new.extend(im_crops.copy())
    
    
    def load_previously_adjusted_dates(self, t_idx, input_dir, n_previous_dates=1):
        
        # t_idx = timeline index of the new date to adjust
        
        found_adj_dates = self.check_adjusted_dates(input_dir)
        if found_adj_dates:
        
            # get closest date in time
            prev_adj_timeline_indices = [idx for idx, d in enumerate(self.timeline) if d['adjusted']==True]
            closest_adj_timeline_indices = sorted(prev_adj_timeline_indices, key=lambda x:abs(x-t_idx))

            self.load_data_from_dates(closest_adj_timeline_indices[:n_previous_dates], input_dir, adjusted=True)
        
    
    def init_ba_input_data(self):
        self.n_adj = 0
        self.myimages_adj = []
        self.mycrops_adj = [] 
        self.myrpcs_adj = [] 
        self.n_new = 0
        self.myimages_new = []
        self.mycrops_new = []
        self.myrpcs_new = [] 
    
    def set_ba_input_data(self, t_indices, input_dir, output_dir, n_previous_dates=0):
        
        # init
        self.init_ba_input_data()
        # load previously adjusted data (if existent) relevant for the current date
        if n_previous_dates > 0:
            self.load_previously_adjusted_dates(min(t_indices), input_dir, n_previous_dates=n_previous_dates)
        # load new data to adjust
        self.load_data_from_dates(t_indices, input_dir)
        
        self.ba_input_data = {}
        self.ba_input_data['input_dir'] = input_dir
        self.ba_input_data['output_dir'] = output_dir
        self.ba_input_data['n_new'] = self.n_new
        self.ba_input_data['n_adj'] = self.n_adj
        self.ba_input_data['image_fnames'] = self.myimages_adj + self.myimages_new
        self.ba_input_data['crops'] = self.mycrops_adj + self.mycrops_new
        self.ba_input_data['rpcs'] = self.myrpcs_adj + self.myrpcs_new
        self.ba_input_data['cam_model'] = self.projmats_model
        self.ba_input_data['aoi'] = self.aoi_lonlat
        
        if self.compute_aoi_masks:
            self.ba_input_data['masks'] = [f['mask'] for f in self.mycrops_adj] + [f['mask'] for f in self.mycrops_new]
        else:
            self.ba_input_data['masks'] = None
        
        print('\nBundle Adjustment input data is ready !\n')
            
    
    def bundle_adjust(self, time_indices, input_dir=None, output_dir=None, n_previous_dates=0, ba_input_data=None,
                      feature_detection=True, tracks_config=None, verbose=True, extra_outputs=False):

        if input_dir is None:
            input_dir = self.dst_dir
        if output_dir is None:
            output_dir = self.dst_dir    
        
        import timeit
        start = timeit.default_timer()
        
        # run bundle adjustment
        if verbose:
            if ba_input_data is None:
                self.set_ba_input_data(time_indices, input_dir, output_dir, n_previous_dates=n_previous_dates)
            else:
                self.ba_input_data = ba_input_data
            self.tracks_config = tracks_config
            self.ba_pipeline = BundleAdjustmentPipeline(self.ba_input_data, 
                                                        feature_detection=feature_detection,
                                                        tracks_config=self.tracks_config)
            self.ba_pipeline.run()

        else:
            with suppress_stdout():
                if ba_input_data is None:
                    self.set_ba_input_data(time_indices, input_dir, output_dir, n_previous_dates=n_previous_dates)
                else:
                    self.ba_input_data = ba_input_data
                self.tracks_config = tracks_config
                self.ba_pipeline = BundleAdjustmentPipeline(self.ba_input_data,
                                                            feature_detection=feature_detection,
                                                            tracks_config=self.tracks_config)
                self.ba_pipeline.run()
        
        n_cams = int(self.ba_pipeline.C.shape[0]/2)
        #n_tracks_employed = self.ba_pipeline.get_n_tracks_within_group_of_views(np.arange(n_cams))
        n_tracks_employed = self.ba_pipeline.C.shape[1]
        elapsed_time = int(timeit.default_timer() - start)
        ba_e = np.round(np.mean(self.ba_pipeline.ba_e), 3) 
        init_e = np.round(np.mean(self.ba_pipeline.init_e), 3)
        
        if extra_outputs:
            image_weights = self.ba_pipeline.get_image_weights()
            #base_node_idx = np.argmax(image_weights)
            #base_pair_candidates = [p for p in self.ba_pipeline.pairs_to_triangulate if base_node_idx in p]
            #base_pair_idx = np.argmax([image_weights[p[0]] + image_weights[p[1]] for p in base_pair_candidates])
            #base_pair = base_pair_candidates[base_pair_idx]
            return elapsed_time, n_tracks_employed, ba_e, init_e, image_weights, self.ba_pipeline.pairs_to_triangulate
        else:
            return elapsed_time, n_tracks_employed, ba_e, init_e
    
    
    def reset_ba_params(self, method):
        if os.path.exists('{}/ba_{}'.format(self.dst_dir, method)):
            os.system('rm -r {}/ba_{}'.format(self.dst_dir, method))
        for t_idx in range(len(self.timeline)):
            self.timeline[t_idx]['adjusted'] = False
    
    def print_ba_headline(self, timeline_indices):
        print('Chosen {} dates of the timeline to bundle adjust:'.format(len(timeline_indices)))
        for idx, t_idx in enumerate(timeline_indices):
            print('({}) {} --> {} views'.format(idx+1, self.timeline[t_idx]['datetime'], self.timeline[t_idx]['n_images']))
        print('\n')
    
    def print_running_time(self, in_seconds):
        hours, rem = divmod(in_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        print('\nTOTAL TIME: {:0>2}:{:0>2}:{:05.2f}\n\n\n'.format(int(hours),int(minutes),seconds))  
        
    
    def run_sequential_bundle_adjustment(self, timeline_indices, n_previous=1, reset=False, verbose=True):

        if reset:
            self.reset_ba_params('sequential')
        self.print_ba_headline(timeline_indices)
        
        ba_dir = os.path.join(self.dst_dir, 'ba_sequential')
        os.makedirs(ba_dir, exist_ok=True)
        
        print('\nRunning bundle ajustment sequentially, each date aligned with {} previous date(s) !'.format(n_previous))
        time_per_date = []
        for idx, t_idx in enumerate(timeline_indices):
            if verbose:
                print('Bundle adjusting date {}...'.format(self.timeline[t_idx]['datetime']))
            running_time, n_tracks, ba_e, init_e = self.bundle_adjust([t_idx],
                                                                      input_dir=ba_dir, output_dir=ba_dir,
                                                                      n_previous_dates=n_previous,
                                                                      ba_input_data=None, feature_detection=True,
                                                                      tracks_config=self.tracks_config, verbose=verbose,
                                                                      extra_outputs=False)
            os.makedirs(ba_dir+'/pts3d_adj', exist_ok=True)
            os.system('mv {} {}'.format(ba_dir+'/pts3d_adj.ply', 
                                        ba_dir+'/pts3d_adj/{}_pts3d_adj.ply'.format(self.timeline[t_idx]['id'])))
            
            time_per_date.append(running_time)
            print_args = [idx+1, self.timeline[t_idx]['datetime'], running_time, n_tracks, init_e, ba_e]
            print('({}) {} adjusted in {} seconds, {} ({}, {})'.format(*print_args))
        print('\n')
        
        self.print_running_time(np.sum(time_per_date))

            
    
    def run_global_bundle_adjustment(self, timeline_indices, reset=False, verbose=True):
    
        if reset:
            self.reset_ba_params('global')
        self.print_ba_headline(timeline_indices)
        
        ba_dir = os.path.join(self.dst_dir, 'ba_global')
        os.makedirs(ba_dir, exist_ok=True)
        
        print('\nRunning bundle ajustment all at once !')
        running_time, n_tracks, ba_e, init_e  = self.bundle_adjust(timeline_indices,
                                                                   input_dir=ba_dir, output_dir=ba_dir,
                                                                   n_previous_dates=0,
                                                                   ba_input_data=None, feature_detection=True,
                                                                   tracks_config=self.tracks_config, verbose=verbose,
                                                                   extra_outputs=False)
        print('All dates adjusted in {} seconds, {} ({}, {})'.format(running_time, n_tracks, init_e, ba_e))
        
        self.print_running_time(running_time)
    
    
    
    def run_out_of_core_bundle_adjustment(self, timeline_indices, reset=False, verbose=True,
                                          parallelize=True, tie_points=False):
    
        if parallelize:
            verbose = False
    
        import pickle
        import timeit
        from multiprocessing import Pool
        
        
        if reset:
            self.reset_ba_params('out-of-core')
        self.print_ba_headline(timeline_indices)
                  
        print('########################\n Running out of core BA \n########################\n')
        abs_start = timeit.default_timer()
        

        ###############
        # local sweep
        ###############
        
        ba_dir = os.path.join(self.dst_dir, 'ba_out-of-core')
        os.makedirs(ba_dir, exist_ok=True)
        
        all_filenames = []
        for t_idx in timeline_indices:
            all_filenames.extend(self.timeline[t_idx]['fnames'])
        pickle_out = open(ba_dir+'/filenames.pickle','wb')
        pickle.dump(all_filenames, pickle_out)
        pickle_out.close()
        

        local_sweep_args = [([t_idx], ba_dir, ba_dir, 0, None, True,
                              self.tracks_config, self.tracks_config['verbose local'], True) for t_idx in timeline_indices]
        
        time_per_date = []
        
        #base_local = []
        local_output = []
        start = timeit.default_timer()  
        if parallelize:          
            #with Pool(processes=2, maxtasksperchild=1000) as p:
            with Pool() as p:
                local_output = p.starmap(self.bundle_adjust, local_sweep_args)
        else:
            for idx, t_idx in enumerate(timeline_indices):
                running_time, n_tracks, ba_e, init_e, im_w, p_triangulate = self.bundle_adjust(*local_sweep_args[idx])
                local_output.append([running_time, n_tracks, ba_e, init_e, im_w, p_triangulate])
                
        stop = timeit.default_timer()
        total_time = int(stop-start)
        
        for idx, t_idx in enumerate(timeline_indices):
            
            ba_e, init_e, n_tracks = local_output[idx][2], local_output[idx][3], local_output[idx][1]
            print('({}) {}, {}, ({}, {}) '.format(idx + 1, self.timeline[t_idx]['datetime'], n_tracks, init_e, ba_e))
            
            self.timeline[t_idx]['image_weights'] = local_output[idx][4]
            self.timeline[t_idx]['pairs_to_triangulate'] = local_output[idx][5]
            
            #self.timeline[t_idx]['base_node'] = self.timeline[t_idx]['fnames'][np.argmax(iw)]
            #self.timeline[t_idx]['base_pair'] = bp

            #im_dir = '{}/images/{}'.format(self.dst_dir, self.timeline[t_idx]['id'])
            #os.makedirs(im_dir, exist_ok=True)
            #for fn in self.timeline[t_idx]['fnames']:
            #    os.system('cp {} {}'.format(fn, os.path.join(im_dir, os.path.basename(fn)))) 

            #print('image weights', self.timeline[t_idx]['image_weights'])
            #print('base node', self.timeline[t_idx]['base_node'])
            #print('base pair', self.timeline[t_idx]['base_pair'])

            #base_pair_fnames = [self.timeline[t_idx]['fnames'][bp[0]]] + [self.timeline[t_idx]['fnames'][bp[1]]]
            #P_dir = os.path.join(ba_dir, 'P_adj')
            #base_local.extend(load_matrices_from_dir(base_pair_fnames, P_dir, suffix='pinhole_adj', verbose=verbose))
        
        print('\n##############################################################')
        print('- Local sweep done in {} seconds (avg per date: {:.2f} s)'.format(total_time,
                                                                                 total_time/len(timeline_indices)))
        print('##############################################################\n')
        
        ##### get base nodes for the current consecutive dates
        
        base_pair_per_date = ba_outofcore.get_base_pairs_complete_sequence(self.timeline, timeline_indices, ba_dir)
        
        ##### express camera positions in local submaps in terms of the corresponding base node
        relative_poses_dir = '{}/relative_local_poses'.format(ba_dir)
        os.makedirs(relative_poses_dir, exist_ok=True)
        for t_idx, bp in zip(timeline_indices, base_pair_per_date):
            P_dir = os.path.join(ba_dir, 'P_adj')
            P_crop_ba = loader.load_matrices_from_dir(self.timeline[t_idx]['fnames'], P_dir, 
                                                      suffix='pinhole_adj', verbose=verbose)
            for P1, fn in zip(P_crop_ba, self.timeline[t_idx]['fnames']):
                P_relative = ba_utils.relative_extrinsic_matrix_between_two_proj_matrices(P1, P_crop_ba[bp[0]])
                f_id = os.path.splitext(os.path.basename(fn))[0]
                np.savetxt(os.path.join(relative_poses_dir,  f_id + '.txt'), P_relative, fmt='%.6f')
        
        ###############
        # global sweep
        ###############
        
        start = timeit.default_timer()
        
        self.n_adj = 0
        self.myimages_adj = []
        self.myrpcs_adj = []
        self.mycrops_adj = []
        
        self.myimages_new = []
        self.myrpcs_new = []
        rpc_dir = os.path.join(ba_dir, 'RPC_adj')
        for t_idx, bp in zip(timeline_indices, base_pair_per_date):
            base_fnames = (np.array(self.timeline[t_idx]['fnames'])[np.array(bp)]).tolist()
            self.myimages_new.extend(base_fnames)  
            self.myrpcs_new.extend(loader.load_rpcs_from_dir(base_fnames, rpc_dir, suffix='RPC_adj', verbose=verbose))
        self.n_new = len(self.myimages_new)
        self.mycrops_new = loader.load_image_crops(self.myimages_new, get_aoi_mask = self.compute_aoi_masks, \
                                                   rpcs = self.myrpcs_new, aoi = self.aoi_lonlat, \
                                                   use_mask_for_equalization = self.use_aoi_masks_to_equalize_crops,
                                                   verbose=verbose)
        
        self.ba_input_data = {}
        self.ba_input_data['input_dir'] = ba_dir
        self.ba_input_data['output_dir'] = ba_dir
        self.ba_input_data['n_new'] = self.n_new
        self.ba_input_data['n_adj'] = self.n_adj
        self.ba_input_data['image_fnames'] = self.myimages_adj + self.myimages_new
        self.ba_input_data['crops'] = self.mycrops_adj + self.mycrops_new
        self.ba_input_data['rpcs'] = self.myrpcs_adj + self.myrpcs_new
        self.ba_input_data['cam_model'] = self.projmats_model
        self.ba_input_data['aoi'] = self.aoi_lonlat
        
        if self.compute_aoi_masks:
            self.ba_input_data['masks'] = [f['mask'] for f in self.mycrops_adj] + [f['mask'] for f in self.mycrops_new]
        else:
            self.ba_input_data['masks'] = None
        
        running_time, n_tracks, ba_e, init_e = self.bundle_adjust(timeline_indices, ba_dir, ba_dir, 0, self.ba_input_data,
                                                                  True, self.tracks_config, verbose, False)
        
        #base_global = [P for P in self.ba_pipeline.P_crop_ba]
        #print('from local to global. Did base Ps change?', not np.allclose(np.array(base_local), np.array(base_global)))
        

        if tie_points:
            #### save coordinates of the base 3D points
            os.makedirs(ba_dir + '/tie_points', exist_ok=True)
            for im_idx, fn in enumerate(self.ba_pipeline.myimages):

                f_id = os.path.splitext(os.path.basename(fn))[0]

                true_if_track_seen_in_current_cam = ~np.isnan(self.ba_pipeline.C_v2[im_idx,:])
                kp_index_per_track_observation = np.array([self.ba_pipeline.C_v2[im_idx,true_if_track_seen_in_current_cam]])
                pts_3d_per_track_observation = self.ba_pipeline.pts_3d_ba[true_if_track_seen_in_current_cam, :].T

                kp_idx_to_3d_coords = np.vstack((kp_index_per_track_observation, pts_3d_per_track_observation)).T

                np.savetxt(os.path.join(ba_dir + '/tie_points',  f_id + '.txt'), kp_idx_to_3d_coords, fmt='%.6f')

                pickle_out = open(os.path.join(ba_dir + '/tie_points',  f_id + '.pickle'),'wb')
                pickle.dump(kp_idx_to_3d_coords, pickle_out)
                pickle_out.close()

        
        print('All dates adjusted in {} seconds, {} ({}, {})'.format(running_time, n_tracks, init_e, ba_e))
        
        stop = timeit.default_timer()
        total_time = int(stop-start)
        
        print('\n##############################################################')
        print('- Global sweep completed in {} seconds'.format(total_time))
        print('##############################################################\n')
    
        
        ###############
        # freeze base variables and update local systems
        ###############
        
        os.makedirs(ba_dir, exist_ok=True)
        
        update_args = [(t_idx, bp, verbose) for t_idx, bp in zip(timeline_indices, base_pair_per_date)]
        
        start = timeit.default_timer()
        
        if tie_points:
            self.tracks_config['tie_points'] = True
        
        #base_update = []
        
        update_output = []
        
        #parallelize = False
        
        if parallelize:
            with Pool() as p:
                update_output = p.starmap(self.out_of_core_update_local_system, update_args)
        else:
            for idx, t_idx in enumerate(timeline_indices):
                running_time, n_tracks, ba_e, init_e = self.out_of_core_update_local_system(*update_args[idx])
                update_output.append([running_time, n_tracks, ba_e, init_e])
                
                #base_update.append(self.ba_pipeline.P_crop_ba[0])
                #base_update.append(self.ba_pipeline.P_crop_ba[1])
        
        self.tracks_config['tie_points'] = False     
        
        #print('from global to update. Did base Ps change?', not np.allclose(np.array(base_update), np.array(base_global)))
        
        for idx, t_idx in enumerate(timeline_indices):
            
            ba_e, init_e, n_tracks = update_output[idx][2], update_output[idx][3], update_output[idx][1]
            print('({}) {}, {}, ({}, {}) '.format(idx + 1, self.timeline[t_idx]['datetime'], n_tracks, init_e, ba_e))
        
        stop = timeit.default_timer()
        total_time = int(stop-start)
        
        print('\n##############################################################')
        print('- Local update done in {} seconds (avg per date: {:.2} s)'.format(total_time,
                                                                                 total_time/len(timeline_indices)))
        print('##############################################################\n')
        
        abs_stop = timeit.default_timer()
        self.print_running_time(int(abs_stop-abs_start))

       
    def out_of_core_update_local_system(self, t_idx, base_pair, verbose):
        
        ba_dir = os.path.join(self.dst_dir, 'ba_out-of-core')
        
        myimages_adj = (np.array(self.timeline[t_idx]['fnames'])[np.array(base_pair)]).tolist()
        n_adj = len(myimages_adj)
        rpc_dir = os.path.join(ba_dir, 'RPC_adj')
        myrpcs_adj = loader.load_rpcs_from_dir(myimages_adj, rpc_dir, suffix='RPC_adj', verbose=verbose)
        mycrops_adj = loader.load_image_crops(myimages_adj, get_aoi_mask = self.compute_aoi_masks, \
                                              rpcs = myrpcs_adj, aoi = self.aoi_lonlat, \
                                              use_mask_for_equalization = self.use_aoi_masks_to_equalize_crops, \
                                              verbose=verbose)

        myimages_new = list(set(self.timeline[t_idx]['fnames']) - set(myimages_adj))
        n_new = len(myimages_new)
        myrpcs_new = loader.load_rpcs_from_dir(myimages_new, rpc_dir, suffix='RPC_adj', verbose=verbose)            
        mycrops_new = loader.load_image_crops(myimages_new, get_aoi_mask = self.compute_aoi_masks, \
                                              rpcs = myrpcs_new, aoi = self.aoi_lonlat, \
                                              use_mask_for_equalization = self.use_aoi_masks_to_equalize_crops, \
                                              verbose=verbose)

        ba_input_data = {}
        ba_input_data['input_dir'] = ba_dir
        ba_input_data['output_dir'] = ba_dir
        ba_input_data['n_new'] = n_new
        ba_input_data['n_adj'] = n_adj
        ba_input_data['image_fnames'] = myimages_adj + myimages_new
        ba_input_data['crops'] = mycrops_adj + mycrops_new
        ba_input_data['rpcs'] = myrpcs_adj + myrpcs_new
        ba_input_data['cam_model'] = self.projmats_model
        ba_input_data['aoi'] = self.aoi_lonlat
        
        #print('BASE NODE', self.timeline[t_idx]['base_node'])
        #print('FNAMES', ba_input_data['image_fnames'])
        
        relative_poses_dir = os.path.join(ba_dir, 'relative_local_poses')
        base_node_fn = self.timeline[t_idx]['fnames'][base_pair[0]]
        P_base = loader.load_matrices_from_dir([base_node_fn], ba_dir+'/P_adj',
                                        suffix='pinhole_adj', verbose=verbose)
        k_b, r_b, t_b, o_b = ba_core.decompose_perspective_camera(P_base[0])
        ext_b = np.hstack(( r_b, t_b[:, np.newaxis] ))
        P_init = loader.load_matrices_from_dir(ba_input_data['image_fnames'], ba_dir+'/P_adj', suffix='pinhole_adj',
                                       verbose=verbose)
        ba_input_data['input_P'] = []
        for cam_idx, fname in enumerate(myimages_adj):
            ba_input_data['input_P'].append(P_init[cam_idx])                 
        for cam_idx, fname in enumerate(myimages_new):
            f_id = os.path.splitext(os.path.basename(fname))[0]
            k_cam, r_cam, t_cam, o_cam = ba_core.decompose_perspective_camera(P_init[n_adj + cam_idx])
            ba_input_data['input_P'].append(k_cam @ ext_b @ np.loadtxt('{}/{}.txt'.format(relative_poses_dir, f_id)))

        
        #print('P_base:', P_init[0])
        #print(self.ba_input_data['input_P'])
        
        if self.compute_aoi_masks:
            ba_input_data['masks'] = [f['mask'] for f in mycrops_adj] + [f['mask'] for f in mycrops_new]
        else:
            ba_input_data['masks'] = None
        
        running_time, n_tracks, ba_e, init_e = self.bundle_adjust([t_idx], ba_dir, ba_dir, 0, ba_input_data, 
                                                                  False, self.tracks_config, verbose, False)
        
        return running_time, n_tracks, ba_e, init_e
    
    
    
    
    def extract_ba_data_indices_according_to_date(self, t_idx):
        
        # returns the indices of the files in ba_input_data['myimages'] that belong to a common timeline index
        ba_data_indices_t_id = [self.ba_input_data['image_fnames'].index(fn) for fn in self.timeline[t_idx]['fnames']]
        return ba_data_indices_t_id
    
    
    def compare_timeline_instances(self, timeline_indices_g1, timeline_indices_g2, ba_pipeline):
               
        # (3) Draw the graph and save it as a .pgf image
        import networkx as nx
        
        # get dates
        node_labels = {}
        nodes_img_indices_in_ba_data = []
        for node_idx, timeline_indices_g in enumerate([timeline_indices_g1, timeline_indices_g2]):
            g_dates = sorted([self.timeline[t_idx]['datetime'] for t_idx in timeline_indices_g])
            g_n_img = sorted([self.timeline[t_idx]['n_images'] for t_idx in timeline_indices_g])
            label = ''
            n_views = []
            for dt, n_img in zip(g_dates, g_n_img):
                n_views.append(n_img)
                label += '{}-{}-{} {}h\n'.format(dt.day, dt.month, str(dt.year)[-2:], dt.hour)
            label += '{} {}'.format(sum(n_views), n_views)
            node_labels[node_idx] = label
            tmp = [self.extract_ba_data_indices_according_to_date(t_idx) for t_idx in timeline_indices_g]
            g_img_indices = [item for sublist in tmp for item in sublist]
            nodes_img_indices_in_ba_data.append(g_img_indices)
        
        
        #mandatory edges
        pairs_to_draw = [[0, 1]]
        
        # init graph
        G=nx.Graph()
        
        # add edges
        edge_labels = {}
        for edge_idx, node_indices in enumerate(pairs_to_draw):
            
            # add current edge
            i, j = node_indices[0], node_indices[1]
            G.add_edge(i, j)
    
            # get number of matches
            g1_img_indices = nodes_img_indices_in_ba_data[i]
            g2_img_indices = nodes_img_indices_in_ba_data[j]
            n_matches, n_matches_aoi = ba_pipeline.get_number_of_matches_between_groups_of_views(g1_img_indices, \
                                                                                                 g2_img_indices)

            current_edge = (i,j)
            if n_matches_aoi is not None:
                edge_labels[current_edge] = '{} matches\n({} aoi)'.format(n_matches, n_matches_aoi)
            else:
                edge_labels[current_edge] = '{} matches'.format(n_matches)
            
            
        # init node positions from layout
        G_pos = nx.circular_layout(G)
        
        # draw nodes
        nx.draw_networkx_nodes(G, G_pos, node_shape='s', node_size=3000, node_color='#FFFFFF', edgecolors='#000000')
        # draw edges and labels
        nx.draw_networkx_edges(G, G_pos)
        nx.draw_networkx_labels(G, G_pos, node_labels, font_size=8, font_family='sans-serif', with_labels=True)
        nx.draw_networkx_edge_labels(G, G_pos, edge_labels, font_size=8, font_family='sans-serif', with_labels=True)
        plt.axis('off')
        plt.show()
    
    def get_datetime_diff_in_days_hours(self, dt_i, dt_j):
        import time
        di_ts = time.mktime(dt_i.timetuple())
        dj_ts = time.mktime(dt_j.timetuple())
        diff_di_dj_in_mins = int(abs(di_ts - dj_ts) / 60)
        days = diff_di_dj_in_mins / 1440     
        leftover_minutes = diff_di_dj_in_mins % 1440
        hours = leftover_minutes / 60
        total_diff_minutes = diff_di_dj_in_mins
        return days, hours, total_diff_minutes
        
    def timeline_instances_diagram(self, timeline_indices, ba_pipeline, neighbors=1):
        
        # (3) Draw the graph and save it as a .pgf image
        import networkx as nx
        
        # get dates
        n_nodes = len(timeline_indices)
        nodes_dates = sorted([self.timeline[t_idx]['datetime'] for t_idx in timeline_indices])
        nodes_n_img = sorted([self.timeline[t_idx]['n_images'] for t_idx in timeline_indices])
        node_labels = {}
        for node_idx, dt in enumerate(nodes_dates):
            node_labels[node_idx] = '{}-{}\n{}\n{}h {}v'.format(dt.day, dt.month, dt.year, dt.hour, nodes_n_img[node_idx])
        nodes_img_indices_in_ba_data = [self.extract_ba_data_indices_according_to_date(t_idx) for t_idx in timeline_indices] 
        
        
        #mandatory edges
        pairs_to_draw = [[i, i+1] for i in range(n_nodes-1)]
        n_mandatory_edges = len(pairs_to_draw)
        # edges of interest
        pairs_to_draw.extend([[i, i+neighbors] for i in range(n_nodes-neighbors)])
        
        # init graph
        G=nx.Graph()
        
        # add edges
        edge_labels = {}
        edges_of_interest = []
        for edge_idx, node_indices in enumerate(pairs_to_draw):
            
            # add current edge
            i, j = node_indices[0], node_indices[1]
            G.add_edge(i, j)
            
            if edge_idx >= n_mandatory_edges:
                # get number of matches
                g1_img_indices = nodes_img_indices_in_ba_data[i]
                g2_img_indices = nodes_img_indices_in_ba_data[j]
                n_matches, _ = ba_pipeline.get_number_of_matches_between_groups_of_views(g1_img_indices, g2_img_indices)

                # get temporal difference between nodes
                days, hours, _ = self.get_datetime_diff_in_days_hours(nodes_dates[i], nodes_dates[j])
                
                current_edge = (i,j)
                edges_of_interest.append(current_edge)
                edge_labels[current_edge] = '{}m\n{}d {}h'.format(n_matches, int(days), int(hours))
            
            
        # init node positions from layout
        G_pos = nx.circular_layout(G)
        
        # draw nodes
        nx.draw_networkx_nodes(G, G_pos, node_shape='s', node_size=1000, node_color='#FFFFFF', edgecolors='#000000')
        # draw edges and labels
        nx.draw_networkx_edges(G, G_pos, edge_color='w')
        nx.draw_networkx_edges(G, G_pos, edgelist=edges_of_interest)
        nx.draw_networkx_labels(G, G_pos, node_labels, font_size=8, font_family='sans-serif', with_labels=True)
        nx.draw_networkx_edge_labels(G, G_pos, edge_labels, font_size=8, font_family='sans-serif', with_labels=True)
        plt.axis('off')
        plt.show()
        
    
    
    
    def reconstruct_date(self, timeline_index, ba_method=None, compute_std=True, verbose=False):
        
        use_corrected_rpcs = True
        #use_corrected_rpcs = ba_method == 'global' or ba_method == 'sequential' or ba_method == 'out-of-core'
        
        t_id, t_date =  self.timeline[timeline_index]['id'], self.timeline[timeline_index]['datetime']
        
        rec4D_dir = '{}/{}/4D'.format(self.dst_dir, ba_method if ba_method is not None else 'init')
        os.makedirs(rec4D_dir, exist_ok=True)
        
        # load configs
        src_config_fnames = loader.load_s2p_configs_from_image_filenames(self.timeline[timeline_index]['fnames'],
                                                                         self.s2p_configs_dir)
        
        dst_config_fnames = [os.path.join(rec4D_dir, fn.replace(self.s2p_configs_dir, 
                                                                's2p/{}'.format(t_id)))  for fn in src_config_fnames]
        n_dsms = len(src_config_fnames)
        
        print('\n###################################################################################')
        print('Reconstructing scene at time {}'.format(t_date))
        print('Number of dsms to compute: {}'.format(n_dsms))
        print('Output directory: {}'.format(rec4D_dir))
        print('Timeline id: {}'.format(t_id))
        print('###################################################################################\n')
        
        # run s2p
        print('Running s2p...\n')
        err_indices = []
        if use_corrected_rpcs:
            adj_rpc_dir = os.path.join(self.dst_dir, '{}/RPC_adj'.format(ba_method))
        else:
            adj_rpc_dir = os.path.join(self.dst_dir, 'RPC_init')
        
        
        
        for dsm_idx, src_config_fname, dst_config_fname in zip(np.arange(n_dsms), src_config_fnames, dst_config_fnames):
            
            s2p_out_dir = os.path.dirname(dst_config_fname)
            os.makedirs(s2p_out_dir, exist_ok=True)
            
            config_s2p = loader.load_dict_from_json(src_config_fname)
            loader.save_dict_to_json(config_s2p, dst_config_fname.replace('config.json', 'config_src.json'))

            # set s2p dirs
            config_s2p['out_dir'] = '.' #s2p_out_dir 
            config_s2p['temporary_dir'] = 'tmp' #os.path.join(s2p_out_dir, 'tmp')
        
            # correct roi_geojson
            img_rpc_path = os.path.join(adj_rpc_dir, loader.get_id(config_s2p['images'][0]['img']) + '_RPC_adj.txt')
            correct_rpc = rpcm.rpc_from_rpc_file(img_rpc_path)
            initial_rpc = rpcm.RPCModel(config_s2p['images'][0]['rpc'], dict_format = "rpcm")
            roi_lons_init = np.array(config_s2p['roi_geojson']['coordinates'][0])[:,0]
            roi_lats_init = np.array(config_s2p['roi_geojson']['coordinates'][0])[:,1]
            alt = srtm4.srtm4(np.mean(roi_lons_init), np.mean(roi_lats_init))           
            roi_cols_init, roi_rows_init = initial_rpc.projection(roi_lons_init, roi_lats_init, [alt]*roi_lons_init.shape[0])
            roi_lons_ba, roi_lats_ba = correct_rpc.localization(roi_cols_init, roi_rows_init, [alt]*roi_lons_init.shape[0])
            config_s2p['roi_geojson'] = {'coordinates': np.array([np.vstack([roi_lons_ba, roi_lats_ba]).T]).tolist(), 
                                         'type': 'Polygon'}
            

                
            '''
            # correct utm bbx
            roi_easts_ba, roi_norths_ba = utils.utm_from_lonlat(roi_lons_ba, roi_lats_ba)            
            roi_xmin, roi_xmax = min(roi_easts_ba), max(roi_easts_ba)
            roi_ymin, roi_ymax = min(roi_norths_ba + 10000000), max(roi_norths_ba + 10000000)
            config_s2p['utm_bbx'] = [roi_xmin, roi_xmax, roi_ymin, roi_ymax]
            '''
            
            # correct global aoi
            if dsm_idx == 0:
                
                prev_dsms = glob.glob(os.path.join(rec4D_dir, 'dsms/*.tif'))
                
                aoi_lons_init = np.array(self.aoi_lonlat['coordinates'][0])[:,0]
                aoi_lats_init = np.array(self.aoi_lonlat['coordinates'][0])[:,1]
                alt = srtm4.srtm4(np.mean(aoi_lons_init), np.mean(aoi_lats_init))
                aoi_cols_init, aoi_rows_init = initial_rpc.projection(aoi_lons_init, aoi_lats_init,
                                                                     [alt]*aoi_lons_init.shape[0])
                aoi_lons_ba, aoi_lats_ba = correct_rpc.localization(aoi_cols_init, aoi_rows_init,
                                                                    [alt]*aoi_lons_init.shape[0])
                lonlat_coords = np.vstack((aoi_lons_ba, aoi_lats_ba)).T
                lonlat_geojson = {'coordinates': [lonlat_coords.tolist()], 'type': 'Polygon'}
                lonlat_geojson['center'] = np.mean(lonlat_geojson['coordinates'][0][:4], axis=0).tolist()
                self.corrected_aoi_lonlat = lonlat_geojson
                
                if len(prev_dsms) == 0:
                    aoi_easts_ba, aoi_norths_ba = utils.utm_from_lonlat(aoi_lons_ba, aoi_lats_ba)
                    aoi_norths_ba[aoi_norths_ba < 0] = aoi_norths_ba[aoi_norths_ba < 0] + 10000000
                    xmin, xmax = min(aoi_easts_ba), max(aoi_easts_ba)
                    ymin, ymax = min(aoi_norths_ba), max(aoi_norths_ba)
                    self.corrected_utm_bbx = {'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}
                    self.dsm_resolution = float(config_s2p['dsm_resolution'])
                    self.h = int(np.floor((ymax - ymin)/self.dsm_resolution) + 1)
                    self.w = int(np.floor((xmax - xmin)/self.dsm_resolution) + 1)
                    
                else:
                    self.corrected_utm_bbx,_, self.dsm_resolution, self.h, self.w = loader.read_geotiff_metadata(prev_dsms[0])
            
            
            # correct image filenames
            for i in [0,1]:
                img_basename = os.path.basename(config_s2p['images'][i]['img'])
                file_id = os.path.splitext(img_basename)[0]
                img_geotiff_path = glob.glob('{}/**/{}'.format(self.images_dir, img_basename), recursive=True)[0]
                config_s2p['images'][i]['img'] = img_geotiff_path
            
            # DEBUG: print roi over input images
            if verbose:
                for i, c in zip([0,1], ['r', 'b']):
                    img_rpc_path = os.path.join(adj_rpc_dir, loader.get_id(config_s2p['images'][i]['img'])+'_RPC_adj.txt')
                    correct_rpc = rpcm.rpc_from_rpc_file(img_rpc_path)

                    roi_cols_ba, roi_rows_ba = correct_rpc.projection(roi_lons_ba, roi_lats_ba, [alt]*roi_lons_init.shape[0])
                    x_min, y_min = min(roi_cols_ba), min(roi_rows_ba)
                    x_max, y_max = max(roi_cols_ba), max(roi_rows_ba)
                    current_roi = {'x': int(np.floor(x_min)), 'y': int(np.floor(y_min)), 
                                   'w': int(np.floor(x_max-x_min)+1), 'h': int(np.floor(y_max-y_min)+1)}       
                    #current_roi = config_s2p['roi']

                    import matplotlib.patches as patches
                    fig,ax = plt.subplots(1, figsize=(30,10))
                    # Display the image
                    im = np.array(Image.open(config_s2p['images'][i]['img']))[10:-10,10:-10]
                    h,w = im.shape[:2]
                    ax.imshow(im, cmap='gray')
                    # Create a Rectangle patch
                    rect = patches.Rectangle((current_roi['x'],current_roi['y']),current_roi['w'],current_roi['h'],
                                             linewidth=2,edgecolor='y',facecolor='none')
                    # Add the patch to the Axes
                    ax.add_patch(rect)
                    
                    rect = patches.Rectangle((0,0),w,h,
                                             linewidth=5,edgecolor=c,facecolor='none')
                    ax.add_patch(rect)
 
                    plt.show()

            # DEBUG: collect utm polygons before and after bundle adjustment (composition should be the same)
            if verbose:
                
                from shapely.geometry import shape
                utm_polys_init, utm_polys_ba = [], []
                
                for i in [0, 1]:
                    initial_rpc = rpcm.RPCModel(config_s2p['images'][i]['rpc'], dict_format = "rpcm")
                    height, width = np.array(Image.open(config_s2p['images'][i]['img'])).shape
                    current_offset = {'col0': 0., 'row0': 0., 'width': width, 'height': height}
                    current_footprint = ba_utils.get_image_footprints([initial_rpc], [current_offset])[0]
                    utm_polys_init.append(current_footprint['poly'])

                    img_rpc_path = os.path.join(adj_rpc_dir, loader.get_id(config_s2p['images'][i]['img']) + '_RPC_adj.txt')
                    correct_rpc = rpcm.rpc_from_rpc_file(img_rpc_path)
                    current_footprint = ba_utils.get_image_footprints([correct_rpc], [current_offset])[0]
                    utm_polys_ba.append(current_footprint['poly'])
            
                
                aoi_easts_init, aoi_norths_init = utils.utm_from_lonlat(aoi_lons_init, aoi_lats_init)
                aoi_utm_init = shape({'type': 'Polygon', 
                                      'coordinates': [(np.vstack((aoi_easts_init, aoi_norths_init)).T).tolist()]})
  
                
                aoi_utm_ba = shape({'type': 'Polygon', 
                                    'coordinates': [(np.vstack((aoi_easts_ba, aoi_norths_ba)).T).tolist()]})
                #utm_polys_init.append(aoi_utm_init)
                #utm_polys_ba.append(aoi_utm_ba)
                
                
                roi_easts_init, roi_norths_init = utils.utm_from_lonlat(roi_lons_init, roi_lats_init)
                roi_easts_ba, roi_norths_ba = utils.utm_from_lonlat(roi_lons_ba, roi_lats_ba)
                roi_utm_init = shape({'type': 'Polygon',
                                      'coordinates': [(np.vstack((roi_easts_init, roi_norths_init)).T).tolist()]})
                roi_utm_ba = shape({'type': 'Polygon',
                                    'coordinates': [(np.vstack((roi_easts_ba, roi_norths_ba)).T).tolist()]})
                utm_polys_init.append(roi_utm_init)
                utm_polys_ba.append(roi_utm_ba)
                
                fig, ax = plt.subplots(1, 2, figsize=(2*8,8))
                for (this_ax, utm_polys_to_display) in zip(ax, [utm_polys_init, utm_polys_ba]):
                    for shapely_poly, color in zip(utm_polys_to_display, ['r', 'b', 'g', 'r']):
                        aoi_utm = (np.array(shapely_poly.boundary.coords.xy).T)[:-1,:]
                        this_ax.fill(aoi_utm[:,0], aoi_utm[:,1], facecolor='none', edgecolor=color, linewidth=1)
                plt.show();
            
                   
            # correct rpcs
            for i in [0,1]:
                img_basename = os.path.basename(config_s2p['images'][i]['img'])
                file_id = os.path.splitext(img_basename)[0]
                if use_corrected_rpcs:
                    img_rpc_path = os.path.join(adj_rpc_dir, file_id + '_RPC_adj.txt')
                    config_s2p['images'][i]['rpc'] = rpcm.rpc_from_rpc_file(img_rpc_path).__dict__
                else:
                    img_rpc_path = os.path.join(adj_rpc_dir, file_id + '_RPCgnegne.txt')
                    config_s2p['images'][i]['rpc'] = rpcm.rpc_from_rpc_file(img_rpc_path).__dict__
          
            if 'utm_bbx' in config_s2p.keys():
                del config_s2p['utm_bbx'] 
            if 'roi' in config_s2p.keys():
                del config_s2p['roi']
        
            # save updated config.json
            loader.save_dict_to_json(config_s2p, dst_config_fname)
            
            if os.path.exists(os.path.join(s2p_out_dir, 'dsm.tif')):
                continue
            
            # RUN S2P
            log_file = os.path.join(os.path.dirname(dst_config_fname), 'log.txt')
            with open(log_file, 'w') as outfile:
                subprocess.run(['s2p', dst_config_fname], stdout=outfile, stderr=outfile)
        
            if not os.path.exists(os.path.join(s2p_out_dir, 'dsm.tif')):
                print(dst_config_fname)
                err_indices.append(dsm_idx)
                with open(os.path.join(rec4D_dir, 's2p_crashes.txt'), 'a') as outfile:
                    outfile.write('{}\n\n'.format(dst_config_fname))
               
            tmp = loader.load_dict_from_json(dst_config_fname)
            
            print('\rComputed {} dsms / {} ({} err)'.format(dsm_idx+1, n_dsms, len(err_indices)),end='\r')
                
        
        # merge dsms and save std
        print('\n\nMerging dsms...\n')

        dsm_path = os.path.join(rec4D_dir, 'dsms/{}.tif'.format(t_id))
        os.makedirs(os.path.dirname(dsm_path), exist_ok=True)
         
        
        xoff = np.floor(self.corrected_utm_bbx['xmin'] / self.dsm_resolution) * self.dsm_resolution
        #xsize = int(1 + np.floor((self.corrected_utm_bbx['xmax'] - xoff) / self.dsm_resolution))
        xsize = int(self.w)
        yoff = np.ceil(self.corrected_utm_bbx['ymax']  / self.dsm_resolution) * self.dsm_resolution
        #ysize = int(1 - np.floor((self.corrected_utm_bbx['ymin']  - yoff) / self.dsm_resolution))
        ysize = int(self.h)
        dsm_roi = (xoff, yoff, xsize, ysize)
        
        from plyflatten import plyflatten_from_plyfiles_list
        
        ply_list = glob.glob('{}/s2p/{}/**/cloud.ply'.format(rec4D_dir, t_id), recursive=True)
        raster, profile = plyflatten_from_plyfiles_list(ply_list, self.dsm_resolution, roi=dsm_roi, std=compute_std)
        
        self.mask = loader.get_binary_mask_from_aoi_lonlat_within_utm_bbx(self.corrected_utm_bbx,
                                                                          self.dsm_resolution, self.corrected_aoi_lonlat)

        import rasterio
        
        profile["dtype"] = raster.dtype
        profile["height"] = raster.shape[0]
        profile["width"] = raster.shape[1]
        profile["count"] = 1
        profile["driver"] = "GTiff"
        
        with rasterio.open(dsm_path, "w", **profile) as f:
            f.write(loader.apply_mask_to_raster(raster[:, :, 0], self.mask), 1)

        if compute_std:

            if raster.shape[2] % 2 == 1:
                cnt_path = dsm_path.replace('/dsms/', '/cnt_per_date/')
                os.makedirs(os.path.dirname(cnt_path), exist_ok=True)
                with rasterio.open(cnt_path, "w", **profile) as f:
                    f.write(loader.apply_mask_to_raster(raster[:, :, -1], self.mask), 1)
                    raster = raster[:, :, :-1]

            std_path = dsm_path.replace('/dsms/', '/std_per_date/')
            os.makedirs(os.path.dirname(std_path), exist_ok=True)
            n = raster.shape[2]
            assert n % 2 == 0
            with rasterio.open(std_path, "w", **profile) as f:
                f.write(loader.apply_mask_to_raster(raster[:, :, n // 2], self.mask), 1)
        
        print('Done!\n\n')
        
    
    def load_reconstructed_DSMs(self, timeline_indices, ba_method, use_mask=False):
        
        rec4D_dir = '{}/{}/4D'.format(self.dst_dir, ba_method if ba_method is not None else 'init')
        
        dsm_timeseries = []
        for t_idx in timeline_indices:
            masked_dsm_fname = os.path.join(rec4D_dir, 'dsms/masked_dsms/{}.tif'.format(self.timeline[t_idx]['id']))
            full_dsm_fname = os.path.join(rec4D_dir, 'dsms/{}.tif'.format(self.timeline[t_idx]['id']))
            if os.path.exists(masked_dsm_fname) and use_mask:
                 dsm_timeseries.append(np.array(Image.open(masked_dsm_fname)))
            else:
                dsm_timeseries.append(np.array(Image.open(full_dsm_fname)))       
        return np.dstack(dsm_timeseries)
    
    
    def compute_3D_statistics_over_time(self, timeline_indices, ba_method):
        
        print('\nComputing 4D statistics of the timeseries! Chosen dates:')
        for t_idx in timeline_indices:
            print('{}'.format(self.timeline[t_idx]['datetime']))
        
        rec4D_dir = '{}/{}/4D'.format(self.dst_dir, ba_method if ba_method is not None else 'init')
        
        output_dir = os.path.join(rec4D_dir, '4Dstats')
        os.makedirs(output_dir, exist_ok=True)
        
        # get timeseries
        dsm_timeseries_ndarray = self.load_reconstructed_DSMs(timeline_indices, ba_method)    
        
        # extract mean
        mean_dsm = np.nanmean(dsm_timeseries_ndarray, axis=2)
        Image.fromarray(mean_dsm).save(output_dir + '/mean_along_time.tif')
        
        # extract std over time (consider only points seen at least 2 times)
        counts_per_coord = np.sum(1*~np.isnan(dsm_timeseries_ndarray), axis=2)
        overlapping_coords_mask = counts_per_coord >= 2
        std_along_time = np.nanstd(dsm_timeseries_ndarray, axis=2)
        std_along_time[~overlapping_coords_mask] = np.nan
        Image.fromarray(std_along_time).save(output_dir + '/std_along_time.tif')
        
        # save log of the dates employed to compute the statistics
        with open(os.path.join(output_dir, 'dates.txt'), 'w') as f:
            for t_idx in timeline_indices:
                f.write('{}\n'.format(self.timeline[t_idx]['datetime']))
         
        for t_idx in timeline_indices:
            t_id = self.timeline[t_idx]['id']
            warp_dir = os.path.join(self.dst_dir, '{}/4D/warp/{}'.format(ba_method, t_id)) 
            fnames = glob.glob(os.path.join(warp_dir,'**/*.tif'), recursive=True)
            stacked_warps = np.dstack([np.array(Image.open(fn)) for fn in fnames])
            counts_per_coord = np.sum(1*~np.isnan(stacked_warps), axis=2)
            overlapping_coords_mask = counts_per_coord >= 2

            std_current_date = np.nanstd(stacked_warps, axis=2)
            std_current_date[~overlapping_coords_mask] = np.nan

            #median_std_per_date.append(np.nanmedian(std_current_date, axis=(0, 1)))

            out_fn = os.path.join(self.dst_dir, '{}/4D/4Dstats/std_per_date/{}.tif'.format(ba_method, t_id))
            os.makedirs(os.path.dirname(out_fn), exist_ok=True)
            Image.fromarray(std_current_date).save(out_fn)
                                  
        print('\nDone! Results were saved at {}'.format(output_dir))
        
        
    def compute_stat_per_date(self, timeline_indices, ba_method=None, stat='std',
                              use_cdsms=False, clean_tmp=True, mask=None):

        if stat not in ['std', 'avg']:
            raise Error('stat is not valid')

        for t_idx in timeline_indices:
        
            t_id = self.timeline[t_idx]['id']

            rec4D_dir = '{}/{}/4D'.format(self.dst_dir, ba_method if ba_method is not None else 'init')

            complete_dsm_fname = os.path.join(rec4D_dir, 'dsms/{}.tif'.format(t_id))

            stereo_dsms_fnames = loader.load_s2p_dsm_fnames_from_dir(os.path.join(rec4D_dir, 's2p/{}'.format(t_id)))

            if use_cdsms:
                stereo_dsms_fnames = [fn.replace('dsm.tif', 'cdsm.tif') for fn in stereo_dsms_fnames]

            out_dir = os.path.join(rec4D_dir, 'metrics/{}_per_date'.format(stat))
        
            ba_metrics.compute_stat_for_specific_date_from_tiles(complete_dsm_fname, stereo_dsms_fnames,
                                                                 output_dir = out_dir, stat=stat,
                                                                 clean_tmp=clean_tmp, mask=mask)

            print('done computing {} for date {}'.format(stat, self.timeline[t_idx]['datetime']))


    def project_pts3d_adj_onto_dsms(self, timeline_indices, ba_method):
        
        if ba_method not in ['ba_global', 'ba_sequential']:
            raise Error('ba_method is not valid')
            
        rec4D_dir = '{}/{}/4D'.format(self.dst_dir, ba_method)
                                      
        for t_idx in timeline_indices:
            t_id = self.timeline[t_idx]['id']
            if ba_method == 'ba_sequential':
                ply_path = '{}/{}/pts3d_adj/{}_pts3d_adj.ply'.format(self.dst_dir, ba_method, t_id)
            else:
                ply_path = '{}/{}/pts3d_adj.ply'.format(self.dst_dir, ba_method)
            dsm_path = '{}/dsms/{}.tif'.format(rec4D_dir, t_id)
            svg_path = '{}/pts3d_adj/{}.svg'.format(rec4D_dir, t_id)
            os.makedirs(os.path.dirname(svg_path), exist_ok=True)
            ba_utils.save_ply_pts_projected_over_geotiff_as_svg(dsm_path, ply_path, svg_path)
        
            print('done computing svg for date {}'.format(self.timeline[t_idx]['datetime']))


    def close_small_holes(self, timeline_indices, ba_method, imscript_bin_dir):


        rec4D_dir = '{}/{}/4D'.format(self.dst_dir, ba_method)

        for t_idx in timeline_indices:
            t_id = self.timeline[t_idx]['id']

            s2p_dir = '{}/s2p/{}'.format(rec4D_dir, t_id)
            dsm_fnames = loader.load_s2p_dsm_fnames_from_dir(s2p_dir)

            for dsm_idx, dsm_fn in enumerate(dsm_fnames):
                cdsm_fn = dsm_fn.replace('/dsm.tif', '/cdsm.tif')
                ba_utils.close_small_holes_from_dsm(dsm_fn, cdsm_fn, imscript_bin_dir)

                print('\rdone computing {}/{} cdsms for date {}'.format(dsm_idx+1, len(dsm_fnames),
                                                                        self.timeline[t_idx]['datetime']), end='\r')
            print('\n')
