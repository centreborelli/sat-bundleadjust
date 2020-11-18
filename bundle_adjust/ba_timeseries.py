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


def run_s2p(thread_idx, geotiff_label, input_config_files):
    dsms_computed, crashes = 0, []
    for dst_config_fn in input_config_files:
        to_print = [thread_idx, dsms_computed, len(input_config_files)]
        print('...reconstructing stereo DSMs (thread {} -> {}/{})'.format(*to_print), flush=True)
        dsms_computed += 1

        # check if dsm already exists
        dst_dsm_fn = '{}/dsm.tif'.format(os.path.dirname(dst_config_fn))
        if os.path.exists(dst_dsm_fn):  # check if dsm has already been computed
            continue
        if geotiff_label is not None:  # check if dsm is available in the rec4D dir for all images
            dst_dsm_fn_all_ims = dst_dsm_path.replace(rec4D_dir, self.set_rec4D_dir(ba_method))
            if os.path.exists(dst_dsm_fn_all_ims):
                os.system(
                    'cp -r {} {}'.format(os.path.dirname(dst_dsm_fn_all_ims), os.path.dirname(dst_dsm_fn)))
                continue

        # run s2p if not
        log_file = os.path.join(os.path.dirname(dst_config_fn), 'log.txt')
        with open(log_file, 'w') as outfile:
            subprocess.run(['s2p', dst_config_fn], stdout=outfile, stderr=outfile)

        # check if output dsm was successfully created
        if not os.path.exists(dst_dsm_fn):
            crashes.append(dst_config_fn)

    to_print = [thread_idx, dsms_computed, len(input_config_files)]
    print('...reconstructing stereo DSMs (thread {} -> {}/{})'.format(*to_print), flush=True)
    return crashes


class Scene:
    
    def __init__(self, scene_config):
        
        args = loader.load_dict_from_json(scene_config)
        
        # read scene args
        self.geotiff_dir = args['geotiff_dir']
        self.s2p_configs_dir = args.get('s2p_configs_dir', '')
        self.rpc_src = args['rpc_src']
        self.dst_dir = args['output_dir']

        # optional arguments
        self.dsm_resolution = args['dsm_resolution'] if 'dsm_resolution' in args.keys() else 1.0
        self.compute_aoi_masks = args['compute_aoi_masks'] if 'compute_aoi_masks' in args.keys() else False
        self.use_aoi_equalization = args['use_aoi_equalization'] if 'use_aoi_equalization' in args.keys() else False
        self.geotiff_label = args['geotiff_label'] if 'geotiff_label' in args.keys() else None
        self.pc3dr = args['pc3dr'] if 'pc3dr' in args.keys() else False
        self.correction_params = args['correction_params'] if 'correction_params' in args.keys() else ['R']

        # cam_model can be 'perspective', 'affine' or 'rpc'
        if 'cam_model' in args.keys():
            self.cam_model = args['cam_model']
        else:
            self.cam_model = 'rpc'

        # check geotiff_dir and s2p_configs_dir exist
        if not os.path.isdir(self.geotiff_dir):
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
                              'K': 0,
                              'continue': True,
                              'tie_points': False,
                              'predefined_pairs': None}
        
        print('\n###################################################################################')
        print('\nLoading scene from {}\n'.format(scene_config))
        print('-------------------------------------------------------------')
        print('Configuration:')
        print('    - geotiff_dir:      {}'.format(self.geotiff_dir))
        print('    - s2p_configs_dir:  {}'.format(self.s2p_configs_dir))
        print('    - rpc_src:          {}'.format(self.rpc_src))
        print('    - output_dir:       {}'.format(self.dst_dir))
        print('    - cam_model:        {}'.format(self.cam_model))
        print('-------------------------------------------------------------\n', flush=True)
        
        if self.s2p_configs_dir == '':
            self.timeline, self.aoi_lonlat = loader.load_scene_from_geotiff_dir(self.geotiff_dir, self.dst_dir,
                                                                                rpc_src=self.rpc_src,
                                                                                geotiff_label=self.geotiff_label)
            
            # TODO: create default s2p configs to reconstruct everything possible at a given resolution
            
        else:
            self.timeline, self.aoi_lonlat = loader.load_scene_from_s2p_configs(self.geotiff_dir,
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
        print('The aoi covers a surface of {:.2f} squared km'.format(sq_km))

        print('\n###################################################################################\n\n', flush=True)

    def get_timeline_attributes(self, timeline_indices, attributes):
        loader.get_timeline_attributes(self.timeline, timeline_indices, attributes)
    
    
    def display_aoi(self, zoom=14):
        geojson_utils.display_lonlat_geojson_list_over_map([self.aoi_lonlat], zoom_factor=zoom)
    
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
            
    def check_adjusted_dates(self, input_dir, verbose=True):
        
        dir_adj_rpc = os.path.join(input_dir, 'RPC_adj')
        if os.path.exists(input_dir + '/filenames.pickle') and os.path.isdir(dir_adj_rpc):

            # read tiff images 
            adj_fnames = pickle.load(open(input_dir+'/filenames.pickle','rb'))
            if verbose:
                print('Found {} previously adjusted images in {}\n'.format(len(adj_fnames), self.dst_dir))
            
            datetimes_adj = [loader.get_acquisition_date(img_geotiff_path) for img_geotiff_path in adj_fnames]
            timeline_adj = loader.group_files_by_date(datetimes_adj, adj_fnames)
            for d in timeline_adj:
                adj_id = d['id']
                for idx in range(len(self.timeline)):
                    if self.timeline[idx]['id'] == adj_id:
                        self.timeline[idx]['adjusted'] = True
                        
            prev_adj_data_found = True
        else:
            if verbose:
                print('No previously adjusted data was found in {}\n'.format(self.dst_dir))
            prev_adj_data_found = False
            
        return prev_adj_data_found
            

    def load_data_from_dates(self, timeline_indices, input_dir, adjusted=False, verbose=True):
        
        im_fnames = []
        for t_idx in timeline_indices:
            im_fnames.extend(self.timeline[t_idx]['fnames'])
        n_cam = len(im_fnames)
        if verbose:
            print(n_cam, '{} images for bundle adjustment !'.format('adjusted' if adjusted else 'new'))
        
        if n_cam > 0:
            # get rpcs            
            rpc_dir = os.path.join(input_dir, 'RPC_adj') if adjusted else os.path.join(self.dst_dir, 'RPC_init')  
            rpc_suffix = 'RPC_adj' if adjusted else 'RPC'
            im_rpcs = loader.load_rpcs_from_dir(im_fnames, rpc_dir, suffix=rpc_suffix, verbose=verbose)

            # get image crops
            im_crops = loader.load_image_crops(im_fnames, rpcs=im_rpcs, aoi=self.aoi_lonlat,
                                               get_aoi_mask=self.compute_aoi_masks,
                                               use_aoi_mask_for_equalization=self.use_aoi_equalization,
                                               verbose=verbose)
            
        if adjusted:
            self.n_adj += n_cam
            self.myimages_adj.extend(im_fnames.copy())
            self.myrpcs_adj.extend(im_rpcs.copy())
            self.mycrops_adj.extend(im_crops.copy())
        else:
            self.n_new += n_cam
            self.myimages_new.extend(im_fnames.copy())
            self.myrpcs_new.extend(im_rpcs.copy())
            self.mycrops_new.extend(im_crops.copy())
    
    
    def load_prev_adjusted_dates(self, t_idx, input_dir, previous_dates=1, verbose=True):
        
        # t_idx = timeline index of the new date to adjust
        dt2str = lambda t: t.strftime("%Y-%m-%d %H:%M:%S")
        found_adj_dates = self.check_adjusted_dates(input_dir, verbose=verbose)
        if found_adj_dates:
            # load data from closest date in time
            all_prev_adj_t_indices = [idx for idx, d in enumerate(self.timeline) if d['adjusted']]
            closest_adj_t_indices = sorted(all_prev_adj_t_indices, key=lambda x:abs(x-t_idx))
            adj_t_indices_to_use = closest_adj_t_indices[:previous_dates]
            adj_dates_to_use = ', '.join([dt2str(self.timeline[k]['datetime']) for k in adj_t_indices_to_use])
            if verbose:
                print('Using {} previously adjusted date(s): {}\n'.format(len(adj_t_indices_to_use), adj_dates_to_use))
            self.load_data_from_dates(adj_t_indices_to_use, input_dir, adjusted=True, verbose=verbose)
        
    
    def init_ba_input_data(self):
        self.n_adj = 0
        self.myimages_adj = []
        self.mycrops_adj = [] 
        self.myrpcs_adj = [] 
        self.n_new = 0
        self.myimages_new = []
        self.mycrops_new = []
        self.myrpcs_new = [] 
    
    def set_ba_input_data(self, t_indices, input_dir, output_dir, previous_dates, verbose):

        if verbose:
            print('\n\n\nSetting bundle adjustment input data...\n')
        # init
        self.init_ba_input_data()
        # load previously adjusted data (if existent) relevant for the current date
        if previous_dates > 0:
            self.load_prev_adjusted_dates(min(t_indices), input_dir, previous_dates=previous_dates, verbose=verbose)
        # load new data to adjust
        self.load_data_from_dates(t_indices, input_dir, verbose=verbose)

        self.ba_data = {}
        self.ba_data['input_dir'] = input_dir
        self.ba_data['output_dir'] = output_dir
        self.ba_data['n_new'] = self.n_new
        self.ba_data['n_adj'] = self.n_adj
        self.ba_data['image_fnames'] = self.myimages_adj + self.myimages_new
        self.ba_data['crops'] = self.mycrops_adj + self.mycrops_new
        self.ba_data['rpcs'] = self.myrpcs_adj + self.myrpcs_new
        self.ba_data['cam_model'] = self.cam_model
        self.ba_data['aoi'] = self.aoi_lonlat
        self.ba_data['correction_params'] = self.correction_params

        if self.compute_aoi_masks:
            self.ba_data['masks'] = [f['mask'] for f in self.mycrops_adj] + [f['mask'] for f in self.mycrops_new]
        else:
            self.ba_data['masks'] = None
        if verbose:
            print('\n...bundle adjustment input data is ready !\n\n', flush=True)
            
    
    def bundle_adjust(self, feature_detection=True, verbose=True, extra_outputs=False):

        import timeit
        t0 = timeit.default_timer()
        
        # run bundle adjustment
        if verbose:
            self.ba_pipeline = BundleAdjustmentPipeline(self.ba_data, tracks_config=self.tracks_config,
                                                        feature_detection=feature_detection, verbose=verbose)
            self.ba_pipeline.run()
        else:
            with suppress_stdout():
                self.ba_pipeline = BundleAdjustmentPipeline(self.ba_data, tracks_config=self.tracks_config,
                                                            feature_detection=feature_detection, verbose=verbose)
                self.ba_pipeline.run()

        # retrieve some stuff for verbose
        n_tracks, elapsed_time = self.ba_pipeline.ba_params.pts3d_ba.shape[0], timeit.default_timer() - t0
        ba_e, init_e = np.mean(self.ba_pipeline.ba_e), np.mean(self.ba_pipeline.init_e)
        
        if extra_outputs:
            image_weights = self.ba_pipeline.compute_image_weights_after_bundle_adjustment()
            return elapsed_time, n_tracks, ba_e, init_e, image_weights, self.ba_pipeline.pairs_to_triangulate
        else:
            return elapsed_time, n_tracks, ba_e, init_e
    
    
    def reset_ba_params(self, ba_method):
        ba_dir = '{}/{}'.format(self.dst_dir, ba_method)
        if os.path.exists(ba_dir):
            os.system('rm -r {}'.format(ba_dir))
        for t_idx in range(len(self.timeline)):
            self.timeline[t_idx]['adjusted'] = False
    
    def print_ba_headline(self, timeline_indices, ba_method, previous_dates=0, next_dates=0):
        print('{} dates of the timeline were selected for bundle adjustment:'.format(len(timeline_indices)))
        for idx, t_idx in enumerate(timeline_indices):
            args = [idx+1, self.timeline[t_idx]['datetime'], self.timeline[t_idx]['n_images']]
            print('({}) {} --> {} views'.format(*args), flush=True)
        if ba_method == 'ba_sequential':
            print('\nRunning sequential bundle adjustment !')
            print('Each date aligned with {} previous date(s)\n'.format(previous_dates), flush=True)
        elif ba_method == 'ba_global':
            print('\nRunning global bundle ajustment !')
            print('All dates will be adjusted together at once')
            print('Track pairs restricted to the same date and the next {} dates\n'.format(next_dates), flush=True)
        else:
            print('\nRunning bruteforce bundle ajustment !')
            print('All dates will be adjusted together at once\n', flush=True)

    def run_sequential_bundle_adjustment(self, timeline_indices,
                                         previous_dates=1, fix_1st_cam=True, reset=False, verbose=True):

        ba_method = 'ba_sequential'
        if reset:
            self.reset_ba_params(ba_method)
        self.print_ba_headline(timeline_indices, ba_method, previous_dates=previous_dates)
        ba_dir = os.path.join(self.dst_dir, ba_method)
        os.makedirs(ba_dir, exist_ok=True)

        n_dates = len(timeline_indices)
        self.tracks_config['predefined_pairs'] = None

        time_per_date, tracks_per_date, init_e_per_date, ba_e_per_date = [], [], [], []
        for idx, t_idx in enumerate(timeline_indices):
            self.set_ba_input_data([t_idx], ba_dir, ba_dir, previous_dates, verbose)
            if (idx == 0 and fix_1st_cam) or (previous_dates == 0 and fix_1st_cam):
                self.ba_data['n_adj'] += 1
                self.ba_data['n_new'] -= 1
            running_time, n_tracks, _, _ = self.bundle_adjust(verbose=verbose, extra_outputs=False)
            pts_out_fn = '{}/pts3d_adj/{}_pts3d_adj.ply'.format(ba_dir, self.timeline[t_idx]['id'])
            os.makedirs(os.path.dirname(pts_out_fn), exist_ok=True)
            os.system('mv {} {}'.format(ba_dir+'/pts3d_adj.ply', pts_out_fn))
            init_e, ba_e = self.compute_reprojection_error_after_bundle_adjust(ba_method)
            time_per_date.append(running_time)
            tracks_per_date.append(n_tracks)
            init_e_per_date.append(init_e)
            ba_e_per_date.append(ba_e)
            args = [idx+1, n_dates, self.timeline[t_idx]['datetime'], running_time, n_tracks, init_e, ba_e]
            print('({}/{}) {} adjusted in {:.2f} seconds, {} ({:.3f}, {:.3f})'.format(*args), flush=True)

        self.update_aoi_after_bundle_adjustment(ba_dir)
        args = [sum(time_per_date), sum(tracks_per_date), np.mean(init_e_per_date), np.mean(ba_e_per_date)]
        print('All dates adjusted in {:.2f} seconds, {} ({:.3f}, {:.3f})'.format(*args), flush=True)
        print('\nTOTAL TIME: {}\n'.format(loader.get_time_in_hours_mins_secs(sum(time_per_date))), flush=True)


    def run_global_bundle_adjustment(self, timeline_indices,
                                     next_dates=1, fix_1st_cam=True, reset=False, verbose=True):
    
        ba_method = 'ba_global'
        if reset:
            self.reset_ba_params(ba_method)
        self.print_ba_headline(timeline_indices, ba_method, next_dates=next_dates)
        ba_dir = os.path.join(self.dst_dir, ba_method)
        os.makedirs(ba_dir, exist_ok=True)

        # only pairs from the same date or consecutive dates are allowed
        args = [self.timeline, timeline_indices, next_dates]
        self.tracks_config['predefined_pairs'] = loader.load_pairs_from_same_date_and_next_dates(*args)

        # load bundle adjustment data and run bundle adjustment
        self.set_ba_input_data(timeline_indices, ba_dir, ba_dir, 0, verbose)
        if fix_1st_cam:
            self.ba_data['n_adj'] += 1
            self.ba_data['n_new'] -= 1
        running_time, n_tracks, ba_e, init_e = self.bundle_adjust(verbose=verbose, extra_outputs=False)
        self.update_aoi_after_bundle_adjustment(ba_dir)

        args = [running_time, n_tracks, init_e, ba_e]
        print('All dates adjusted in {:.2f} seconds, {} ({:.3f}, {:.3f})'.format(*args), flush=True)
        print('\nTOTAL TIME: {}\n'.format(loader.get_time_in_hours_mins_secs(running_time)), flush=True)


    def run_bruteforce_bundle_adjustment(self, timeline_indices, fix_1st_cam=True, reset=False, verbose=True):

        ba_method = 'ba_bruteforce'
        if reset:
            self.reset_ba_params(ba_method)
        self.print_ba_headline(timeline_indices, ba_method)
        ba_dir = os.path.join(self.dst_dir, ba_method)
        os.makedirs(ba_dir, exist_ok=True)

        self.tracks_config['predefined_pairs'] = None
        self.set_ba_input_data(timeline_indices, ba_dir, ba_dir, 0, verbose)
        if fix_1st_cam:
            self.ba_data['n_adj'] += 1
            self.ba_data['n_new'] -= 1
        running_time, n_tracks, ba_e, init_e = self.bundle_adjust(verbose=verbose, extra_outputs=False)
        self.update_aoi_after_bundle_adjustment(ba_dir)

        args = [running_time, n_tracks, init_e, ba_e]
        print('All dates adjusted in {:.2f} seconds, {} ({:.3f}, {:.3f})'.format(*args), flush=True)
        print('\nTOTAL TIME: {}\n'.format(loader.get_time_in_hours_mins_secs(running_time)), flush=True)


    def extract_ba_data_indices_according_to_date(self, t_idx):
        
        # returns the indices of the files in ba_input_data['myimages'] that belong to a common timeline index
        ba_data_indices_t_id = [self.ba_data['image_fnames'].index(fn) for fn in self.timeline[t_idx]['fnames']]
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


    def set_rec4D_dir(self, ba_method, geotiff_label=None):

        rec4D_dir = '{}/{}/4D'.format(self.dst_dir, ba_method if ba_method is not None else 'init')
        if geotiff_label is not None:
            rec4D_dir += '_{}'.format(geotiff_label)
        os.makedirs(rec4D_dir, exist_ok=True)
        return rec4D_dir


    def reconstruct_dates(self, timeline_indices,
                          ba_method=None, std=True, geotiff_label=None, n_s2p=7, verbose=False):
        # n_s2p = amount of s2p processes that will be launched in parallel

        n_dates = len(timeline_indices)
        configs = []
        for t_idx in timeline_indices:
            src_configs = loader.load_s2p_configs_from_image_filenames(self.timeline[t_idx]['fnames'],
                                                                       self.s2p_configs_dir,
                                                                       geotiff_label=geotiff_label)
            configs.append(src_configs)
        n_dsms = len([item for sublist in configs for item in sublist])
        rec4D_dir = self.set_rec4D_dir(ba_method, geotiff_label=geotiff_label)

        print('\n###################################################################################')
        print('Reconstructing {} dates with s2p'.format(n_dates))
        print('Timeline indices: {}'.format(timeline_indices))
        print('Number of DSMs to reconstruct: {}'.format(n_dsms))
        print('S2P parallel processes: {}'.format(n_s2p))
        print('Output DSMs directory: {}'.format(os.path.join(rec4D_dir, 'dsms')))
        print('###################################################################################\n', flush=True)

        import timeit
        start = timeit.default_timer()
        n_crashes = 0
        for counter, t_idx in enumerate(timeline_indices):
            t0 = timeit.default_timer()
            args = [counter + 1, len(timeline_indices), self.timeline[t_idx]['id'], len(configs[counter])]
            print('...({}/{}) reconstructing {} [{} DSMs]'.format(*args), flush=True)
            print('-----------------------------------------------------------------------------------', flush=True)
            if os.path.exists('{}/dsms/{}.tif'.format(rec4D_dir, self.timeline[t_idx]['id'])):
                print('...the complete DSM of this area already exists ! Skipping\n', flush=True)
                continue
            t_idx_crashes = self.reconstruct_date(t_idx, ba_method=ba_method, std=std,
                                                  geotiff_label=geotiff_label, n_s2p=n_s2p, verbose=verbose)
            n_crashes += t_idx_crashes
            running_time_current_date = loader.get_time_in_hours_mins_secs(timeit.default_timer() - t0)
            print('...done in {}\n'.format(running_time_current_date), flush=True)

        print('-----------------------------------------------------------------------------------', flush=True)
        total_running_time = loader.get_time_in_hours_mins_secs(timeit.default_timer() - start)
        print('All {} DSMs done in {}'.format(n_dsms, total_running_time), flush=True)
        print('{} DSMs crashed\n\n'.format(n_crashes), flush=True)


    def reconstruct_date(self, t_idx,
                         ba_method=None, std=True, geotiff_label=None, n_s2p=7, verbose=False):

        t_id = self.timeline[t_idx]['id']
        rec4D_dir = self.set_rec4D_dir(ba_method, geotiff_label=geotiff_label)

        # load source configs
        src_configs = loader.load_s2p_configs_from_image_filenames(self.timeline[t_idx]['fnames'],
                                                                   self.s2p_configs_dir,
                                                                   geotiff_label=geotiff_label)
        set_dst_s2p_dir = lambda t: t.replace(self.s2p_configs_dir, 's2p/{}'.format(t_id))
        dst_configs = ['{}/{}'.format(rec4D_dir, set_dst_s2p_dir(fn)) for fn in src_configs]
        n_dsms = len(src_configs)

        # prepare new s2p configs
        for dsm_idx, (src_config_fn, dst_config_fn) in enumerate(zip(src_configs, dst_configs)):
            self.update_config_json_after_bundle_adjustment(src_config_fn, dst_config_fn, ba_method, verbose=verbose)

        n = int(np.ceil(n_dsms / n_s2p))
        args_s2p = [(k, geotiff_label, dst_configs[i:i+n]) for k, i in enumerate(np.arange(0, len(dst_configs), n))]
        from multiprocessing import Pool
        with Pool(len(args_s2p)) as p:
            crashes = p.starmap(run_s2p, args_s2p)
        flatten_list = lambda t: [item for sublist in t for item in sublist]
        crashes = flatten_list(crashes)
        n_crashes = len(crashes)

        with open(os.path.join(rec4D_dir, 's2p_crashes.log'), 'a') as outfile:
            outfile.write(' \n\n'.join(crashes))

        # set resolution and utm_bbx for output multi-view dsm
        if ba_method is None:
            aoi_lonlat_ba = self.aoi_lonlat
        else:
            aoi_lonlat_ba = loader.load_pickle(os.path.join(self.dst_dir, '{}/AOI_adj.pickle'.format(ba_method)))
        utm_bbx_ba = loader.get_utm_bbox_from_aoi_lonlat(aoi_lonlat_ba)
        dsm_resolution = float(loader.load_dict_from_json(src_configs[0])['dsm_resolution'])

        # merge stereo dsms into output muti-view dsm
        dsm_path = os.path.join(rec4D_dir, 'dsms/{}.tif'.format(t_id))
        ply_list = glob.glob('{}/s2p/{}/**/cloud.ply'.format(rec4D_dir, t_id), recursive=True)
        ba_utils.run_plyflatten(ply_list, dsm_resolution, utm_bbx_ba, dsm_path, aoi_lonlat=aoi_lonlat_ba, std=std)
        print('...merging with plyflatten completed!', flush=True)

        return n_crashes


    def load_reconstructed_DSMs(self, timeline_indices, ba_method, pc3dr=False):
        
        rec4D_dir = self.set_rec4D_dir(ba_method)
        dsms_dir = os.path.join(rec4D_dir, 'pc3dr/dsms' if pc3dr else 'dsms')

        dsm_timeseries = []
        for t_idx in timeline_indices:
            dsm_fname = os.path.join(dsms_dir, '{}.tif'.format(self.timeline[t_idx]['id']))
            dsm_timeseries.append(np.array(Image.open(dsm_fname)))
        return np.dstack(dsm_timeseries)
    
    
    def compute_stats_over_time(self, timeline_indices, ba_method, pc3dr=False):
        
        print('\nComputing 4D statistics of the timeseries! Chosen dates:', flush=True)
        for t_idx in timeline_indices:
            print('{}'.format(self.timeline[t_idx]['datetime']), flush=True)
        
        rec4D_dir = self.set_rec4D_dir(ba_method)
        
        output_dir = os.path.join(rec4D_dir, 'metrics_over_time_pc3dr' if pc3dr else 'metrics_over_time')
        os.makedirs(output_dir, exist_ok=True)
        
        # get timeseries
        dsm_timeseries_ndarray = self.load_reconstructed_DSMs(timeline_indices, ba_method, pc3dr=pc3dr)
        
        # extract mean
        mean_dsm = np.nanmean(dsm_timeseries_ndarray, axis=2)
        Image.fromarray(mean_dsm).save(output_dir + '/avg_over_time.tif')
        
        # extract std over time (consider only points seen at least 2 times)
        counts_per_coord = np.sum(1*~np.isnan(dsm_timeseries_ndarray), axis=2)
        overlapping_coords_mask = counts_per_coord >= 2
        std_along_time = np.nanstd(dsm_timeseries_ndarray, axis=2)
        std_along_time[~overlapping_coords_mask] = np.nan
        Image.fromarray(std_along_time).save(output_dir + '/std_over_time.tif')
        
        # save log of the dates employed to compute the statistics
        with open(os.path.join(output_dir, 'dates.txt'), 'w') as f:
            for t_idx in timeline_indices:
                f.write('{}\n'.format(self.timeline[t_idx]['id']))
                                  
        print('\nDone! Results were saved at {}'.format(output_dir))
        
        
    def compute_stat_per_date(self, timeline_indices, ba_method=None, stat='std', tile_size=500, use_cdsms=False,
                              geotiff_label=None, clean_tmp_warps=True, clean_tmp_tiles=True):

        if stat not in ['std', 'avg']:
            raise Error('stat is not valid')

        print('\n###################################################################################')
        print('Computing {} per date...'.format(stat))
        print('  - timeline_indices: {}'.format(timeline_indices))
        print('  - tile_size: {}'.format(tile_size), flush=True)
        print('  - use_cdsms: {}'.format(use_cdsms), flush=True)
        print('###################################################################################\n', flush=True)

        for d_idx, t_idx in enumerate(timeline_indices):
        
            t_id = self.timeline[t_idx]['id']
            rec4D_dir = self.set_rec4D_dir(ba_method, geotiff_label=geotiff_label)
            complete_dsm_fname = os.path.join(rec4D_dir, 'dsms/{}.tif'.format(t_id))
            stereo_dsms_fnames = loader.load_s2p_dsm_fnames_from_dir('{}/s2p/{}'.format(rec4D_dir, t_id))
            if use_cdsms:
                stereo_dsms_fnames = [fn.replace('dsm.tif', 'cdsm.tif') for fn in stereo_dsms_fnames]
            out_dir = os.path.join(rec4D_dir, 'metrics/{}_per_date'.format(stat))

            ba_metrics.compute_stat_for_specific_date_from_tiles(complete_dsm_fname, stereo_dsms_fnames,
                                                                 output_dir=out_dir, stat=stat, tile_size=tile_size,
                                                                 clean_tmp_warps=clean_tmp_warps,
                                                                 clean_tmp_tiles=clean_tmp_tiles)
            args = [t_id, stat, d_idx+1, len(timeline_indices)]
            print('{} - done computing {} multi-view DSM ({}/{})\n'.format(*args), flush=True)


    def is_ba_method_valid(self, ba_method):
        return ba_method in ['ba_global', 'ba_sequential', 'ba_bruteforce']

    def project_pts3d_adj_onto_dsms(self, timeline_indices, ba_method):

        if not self.is_ba_method_valid(ba_method):
            raise Error('ba_method is not valid')
            
        rec4D_dir = self.set_rec4D_dir(ba_method)
        for d_idx, t_idx in enumerate(timeline_indices):
            t_id = self.timeline[t_idx]['id']
            if ba_method == 'ba_sequential':
                ply_path = '{}/{}/pts3d_adj/{}_pts3d_adj.ply'.format(self.dst_dir, ba_method, t_id)
            else:
                ply_path = '{}/{}/pts3d_adj.ply'.format(self.dst_dir, ba_method)
            dsm_path = '{}/dsms/{}.tif'.format(rec4D_dir, t_id)
            svg_path = '{}/pts3d_adj/{}.svg'.format(rec4D_dir, t_id)
            os.makedirs(os.path.dirname(svg_path), exist_ok=True)
            ba_utils.save_ply_pts_projected_over_geotiff_as_svg(dsm_path, ply_path, svg_path)
            args = [d_idx+1, len(timeline_indices)]
            print('Projecting adjusted 3d points onto output multi-view DSMs... {}/{}'.format(*args), flush=True)
        print('\n')


    def interpolate_small_holes(self, timeline_indices, imscript_bin_dir, ba_method=None, geotiff_label=None):

        print('\n###################################################################################')
        print('Closing small holes from s2p DSMs...')
        print('  - timeline_indices: {}'.format(timeline_indices))
        print('###################################################################################\n', flush=True)
        rec4D_dir = self.set_rec4D_dir(ba_method, geotiff_label=geotiff_label)
        for t_idx in timeline_indices:
            s2p_dir = '{}/s2p/{}'.format(rec4D_dir, self.timeline[t_idx]['id'])
            dsm_fnames = loader.load_s2p_dsm_fnames_from_dir(s2p_dir)
            for dsm_idx, dsm_fn in enumerate(dsm_fnames):
                cdsm_fn = dsm_fn.replace('/dsm.tif', '/cdsm.tif')
                ba_utils.close_small_holes_from_dsm(dsm_fn, cdsm_fn, imscript_bin_dir)
                args = [self.timeline[t_idx]['id'], dsm_idx+1, len(dsm_fnames)]
                print('\r{} - interpolating stereo DSMs: {}/{}'.format(*args), end='\r', flush=True)
            print('\n')


    def update_aoi_after_bundle_adjustment(self, ba_dir):

        ba_rpc_fn = glob.glob('{}/RPC_adj/*'.format(ba_dir))[0]
        init_rpc_fn = '{}/RPC_init/{}'.format(self.dst_dir, os.path.basename(ba_rpc_fn).replace('_RPC_adj', '_RPC'))
        init_rpc = rpcm.rpc_from_rpc_file(init_rpc_fn)
        ba_rpc = rpcm.rpc_from_rpc_file(ba_rpc_fn)
        corrected_aoi = geojson_utils.reestimate_lonlat_geojson_after_rpc_correction(init_rpc, ba_rpc, self.aoi_lonlat)
        loader.save_pickle('{}/AOI_adj.pickle'.format(ba_dir), corrected_aoi)


    def update_config_json_after_bundle_adjustment(self, src_config_fname, dst_config_fname,
                                                   ba_method, verbose=False):

        use_corrected_rpcs = True if self.is_ba_method_valid(ba_method) else False

        if use_corrected_rpcs:
            adj_rpc_dir = os.path.join(self.dst_dir, '{}/RPC_adj'.format(ba_method))
        else:
            adj_rpc_dir = os.path.join(self.dst_dir, 'RPC_init')


        config_s2p = loader.load_dict_from_json(src_config_fname)
        os.makedirs(os.path.dirname(dst_config_fname), exist_ok=True)
        loader.save_dict_to_json(config_s2p, dst_config_fname.replace('config.json', 'config_src.json'))

        # set s2p dirs
        config_s2p['out_dir'] = '.'
        config_s2p['temporary_dir'] = 'tmp'

        # correct roi_geojson
        if use_corrected_rpcs:
            img_rpc_path = os.path.join(adj_rpc_dir, loader.get_id(config_s2p['images'][0]['img']) + '_RPC_adj.txt')
            corrected_rpc = rpcm.rpc_from_rpc_file(img_rpc_path)
            initial_rpc = rpcm.RPCModel(config_s2p['images'][0]['rpc'], dict_format = "rpcm")
            roi_lonlat_init = config_s2p['roi_geojson']
            roi_lonlat_ba = geojson_utils.reestimate_lonlat_geojson_after_rpc_correction(initial_rpc, corrected_rpc,
                                                                                         roi_lonlat_init)
            config_s2p['roi_geojson'] = roi_lonlat_ba

        # correct image filenames
        for i in [0,1]:
            img_basename = os.path.basename(config_s2p['images'][i]['img'])
            file_id = os.path.splitext(img_basename)[0]
            img_geotiff_path = glob.glob('{}/**/{}'.format(self.geotiff_dir, img_basename), recursive=True)[0]
            config_s2p['images'][i]['img'] = img_geotiff_path

        # DEBUG: print roi over input images
        if verbose:
            for i, c in zip([0,1], ['r', 'b']):
                img_rpc_path = os.path.join(adj_rpc_dir, loader.get_id(config_s2p['images'][i]['img'])+'_RPC_adj.txt')
                correct_rpc = rpcm.rpc_from_rpc_file(img_rpc_path)

                roi_lons_ba = np.array(roi_lonlat_ba['coordinates'][0])[:,0]
                roi_lats_ba = np.array(roi_lonlat_ba['coordinates'][0])[:,1]
                alt = srtm4.srtm4(np.mean(roi_lons_ba), np.mean(roi_lats_ba))
                roi_cols_ba, roi_rows_ba = correct_rpc.projection(roi_lons_ba, roi_lats_ba, [alt]*roi_lons_ba.shape[0])
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
                height, width = loader.read_image_size(config_s2p['images'][i]['img'])
                current_offset = {'col0': 0., 'row0': 0., 'width': width, 'height': height}
                current_footprint = ba_utils.get_image_footprints([initial_rpc], [current_offset])[0]
                utm_polys_init.append(current_footprint['poly'])

                img_rpc_path = os.path.join(adj_rpc_dir, loader.get_id(config_s2p['images'][i]['img']) + '_RPC_adj.txt')
                correct_rpc = rpcm.rpc_from_rpc_file(img_rpc_path)
                current_footprint = ba_utils.get_image_footprints([correct_rpc], [current_offset])[0]
                utm_polys_ba.append(current_footprint['poly'])

            aoi_utm_init = shape(geojson_utils.utm_geojson_from_lonlat_geojson(self.aoi_lonlat))
            aoi_utm_ba = shape(geojson_utils.utm_geojson_from_lonlat_geojson(self.corrected_aoi_lonlat))
            #utm_polys_init.append(aoi_utm_init)
            #utm_polys_ba.append(aoi_utm_ba)

            roi_utm_init = shape(geojson_utils.utm_geojson_from_lonlat_geojson(roi_lonlat_init))
            utm_polys_init.append(roi_utm_init)
            roi_utm_ba = shape(geojson_utils.utm_geojson_from_lonlat_geojson(roi_lonlat_ba))
            utm_polys_ba.append(roi_utm_ba)

            fig, ax = plt.subplots(1, 2, figsize=(2*8,8))
            for (this_ax, utm_polys_to_display) in zip(ax, [utm_polys_init, utm_polys_ba]):
                for shapely_poly, color in zip(utm_polys_to_display, ['r', 'b', 'g', 'r']):
                    aoi_utm = (np.array(shapely_poly.boundary.coords.xy).T)[:-1,:]
                    this_ax.fill(aoi_utm[:,0], aoi_utm[:,1], facecolor='none', edgecolor=color, linewidth=1)
            plt.show()


        # correct rpcs
        for i in [0,1]:
            img_basename = os.path.basename(config_s2p['images'][i]['img'])
            file_id = os.path.splitext(img_basename)[0]
            if use_corrected_rpcs:
                img_rpc_path = os.path.join(adj_rpc_dir, file_id + '_RPC_adj.txt')
                config_s2p['images'][i]['rpc'] = rpcm.rpc_from_rpc_file(img_rpc_path).__dict__
            else:
                img_rpc_path = os.path.join(adj_rpc_dir, file_id + '_RPC.txt')
                config_s2p['images'][i]['rpc'] = rpcm.rpc_from_rpc_file(img_rpc_path).__dict__

        if 'utm_bbx' in config_s2p.keys():
            del config_s2p['utm_bbx']
        if 'roi' in config_s2p.keys():
            del config_s2p['roi']
        if not 'out_crs' in config_s2p.keys() and 'utm_zone' in config_s2p.keys():
            config_s2p['out_crs'] = 'epsg:{}'.format(loader.epsg_from_utm_zone(config_s2p['utm_zone']))

        loader.save_dict_to_json(config_s2p, dst_config_fname)


    def run_pc3dr(self, timeline_indices, ba_method=None, std=True):

        """
        This function applies the pc3dr alignment to the reconstructed dsms of some input timeline_indices
        The dsms with the aligned points are written in the rec4D_dir/pc3dr output directory
        This function should be called right after reconstruct_dates
        """

        rec4D_dir = self.set_rec4D_dir(ba_method)
        s2p_dirs = ['{}/s2p/{}'.format(rec4D_dir, self.timeline[t_idx]['id']) for t_idx in timeline_indices]
        flatten_list = lambda t: [item for sublist in t for item in sublist]
        s2p_dsm_fnames = flatten_list([loader.load_s2p_dsm_fnames_from_dir(path) for path in s2p_dirs])
        input_dirs_pc3dr = [os.path.dirname(fn) for fn in s2p_dsm_fnames]
        output_dir_pc3dr = os.path.join(rec4D_dir, 'pc3dr')
        n_dates = len(timeline_indices)

        # set resolution and utm_bbx for output multi-view dsm
        if ba_method is None:
            corrected_aoi_lonlat = self.aoi_lonlat
        else:
            corrected_aoi_fn = os.path.join(self.dst_dir, '{}/AOI_adj.pickle'.format(ba_method))
            corrected_aoi_lonlat = loader.load_pickle(corrected_aoi_fn)
        corrected_utm_bbx = loader.get_utm_bbox_from_aoi_lonlat(corrected_aoi_lonlat)
        src_config_fnames = [fn.replace('dsm.tif', 'config.json') for fn in s2p_dsm_fnames]
        dsm_resolution = float(loader.load_dict_from_json(src_config_fnames[0])['dsm_resolution'])

        print('\n###################################################################################')
        print('Applying pc3dr to {} dates'.format(n_dates))
        print('Timeline indices: {}'.format(timeline_indices))
        print('Number of DSMs to align: {}'.format(len(input_dirs_pc3dr)))
        print('Output DSMs directory: {}'.format(output_dir_pc3dr))
        print('###################################################################################\n', flush=True)

        # run pc3dr
        import timeit
        start = timeit.default_timer()
        tmp_dir = 'pc3dr_tmpfiles'
        log_file = os.path.join(rec4D_dir, 'pc3dr.log')
        with open(log_file, 'w') as outfile:
            subprocess.run(['pc3dr'] + input_dirs_pc3dr + ['--outdir', tmp_dir], stdout=outfile, stderr=outfile)
        os.system('rm -r {}'.format(tmp_dir))
        running_time = timeit.default_timer() - start
        print('pc3dr run completed in {}\n'.format(loader.get_time_in_hours_mins_secs(running_time)), flush=True)

        # merge the cloud_registered.ply files created by pc3dr
        for d_idx, (t_idx, path_to_registered_ply_files) in enumerate(zip(timeline_indices, s2p_dirs)):
            ply_list = glob.glob(path_to_registered_ply_files + '/**/cloud_registered.ply', recursive=True)
            output_dsm_path = os.path.join('{}/{}.tif'.format(output_dir_pc3dr, self.timeline[t_idx]['id']))
            ba_utils.run_plyflatten(ply_list, dsm_resolution, corrected_utm_bbx, output_dsm_path,
                                    aoi_lonlat=corrected_aoi_lonlat, std=std)
            args = [d_idx + 1, n_dates, self.timeline[t_idx]['id']]
            print('\rmerging registered DSMs... ({}/{}) {} done'.format(*args), end='\r', flush=True)
        print('\n', flush=True)


    def run_pc3dr_datewise(self, timeline_indices, ba_method=None, std=True, reset=False, intradate=True):

        """
        This function applies the pc3dr alignment to the reconstructed dsms of some input timeline_indices
        The dsms with the aligned points are written in the rec4D_dir/pc3dr output directory
        This function should be called right after reconstruct_dates
        """

        rec4D_dir = self.set_rec4D_dir(ba_method)
        s2p_dirs = ['{}/s2p/{}'.format(rec4D_dir, self.timeline[t_idx]['id']) for t_idx in timeline_indices]
        flatten_list = lambda t: [item for sublist in t for item in sublist]
        s2p_dsm_fnames = flatten_list([loader.load_s2p_dsm_fnames_from_dir(path) for path in s2p_dirs])
        out_dir_pc3dr = os.path.join(rec4D_dir, 'pc3dr')
        n_dates = len(timeline_indices)

        if reset:
            if os.path.exists(out_dir_pc3dr):
                os.system('rm -r {}'.format(out_dir_pc3dr))
        os.makedirs(out_dir_pc3dr, exist_ok=True)

        # set resolution and utm_bbx for output multi-view dsm
        if ba_method is None:
            corrected_aoi_lonlat = self.aoi_lonlat
        else:
            corrected_aoi_fn = os.path.join(self.dst_dir, '{}/AOI_adj.pickle'.format(ba_method))
            corrected_aoi_lonlat = loader.load_pickle(corrected_aoi_fn)
        corrected_utm_bbx = loader.get_utm_bbox_from_aoi_lonlat(corrected_aoi_lonlat)
        src_config_fnames = [fn.replace('dsm.tif', 'config.json') for fn in s2p_dsm_fnames]
        dsm_res = float(loader.load_dict_from_json(src_config_fnames[0])['dsm_resolution'])
        mask = loader.get_binary_mask_from_aoi_lonlat_within_utm_bbx(corrected_utm_bbx, dsm_res, corrected_aoi_lonlat)

        print('\n###################################################################################')
        print('Applying pc3dr to {} dates'.format(n_dates))
        print('Timeline indices: {}'.format(timeline_indices))
        print('Number of DSMs to align: {}'.format(len(s2p_dsm_fnames)))
        print('Output DSMs directory: {}'.format(out_dir_pc3dr))
        print('###################################################################################\n', flush=True)

        def merge_s2p_ply(ply_fnames, out_ply):
            from s2p import ply
            ply_comments = ply.read_3d_point_cloud_from_ply(ply_fnames[0])[1]
            super_xyz = np.vstack([ply.read_3d_point_cloud_from_ply(fn)[0] for fn in ply_fnames])
            ply.write_3d_point_cloud_to_ply(out_ply, super_xyz[:, :3],
                                            colors=super_xyz[:, 3:6].astype('uint8'), comments=ply_comments)

        # run pc3dr at intra-date level
        import timeit
        t0 = timeit.default_timer()
        tmp_dir = 'pc3dr_tmpfiles'

        if intradate:
            crashes = []
            os.makedirs(os.path.join(out_dir_pc3dr, 'dsms'), exist_ok=True)
            os.makedirs(os.path.join(out_dir_pc3dr, 'ply'), exist_ok=True)
            os.makedirs(os.path.join(out_dir_pc3dr, 'log'), exist_ok=True)
            for k, (t_idx, s2p_dir) in enumerate(zip(timeline_indices, s2p_dirs)):
                t_id = self.timeline[t_idx]['id']
                s2p_dsm_fnames = loader.load_s2p_dsm_fnames_from_dir(s2p_dir)
                n_dsms = len(s2p_dsm_fnames)
                print('...({}/{}) pc3dr to align {} [{} DSMs]'.format(k + 1, n_dates, t_id, n_dsms), flush=True)
                log_file = '{}/log/{}.log'.format(out_dir_pc3dr, t_id)
                in_dirs_pc3dr = [os.path.dirname(fn) for fn in s2p_dsm_fnames]
                out_dsm_fn = os.path.join('{}/dsms/{}.tif'.format(out_dir_pc3dr, t_id))
                if os.path.exists(out_dsm_fn):
                    print('...already available!\n', flush=True)
                    continue

                start = timeit.default_timer()
                with open(log_file, 'w') as outfile:
                    subprocess.run(['pc3dr'] + in_dirs_pc3dr + ['--outdir', tmp_dir], stdout=outfile, stderr=outfile)

                ply_list = glob.glob(s2p_dir + '/**/cloud_registered.ply', recursive=True)
                if len(ply_list) > 0:
                    merge_s2p_ply(ply_list, '{}/ply/{}.ply'.format(out_dir_pc3dr, t_id))
                    ba_utils.run_plyflatten(ply_list, dsm_res, corrected_utm_bbx, out_dsm_fn, aoi_lonlat=None, std=std)
                else:
                    crashes.append(log_file)
                os.system('rm -r {}'.format(tmp_dir)) # clean temp files, otherwise next run of pc3dr will crash
                time_to_print = loader.get_time_in_hours_mins_secs(timeit.default_timer() - start)
                err = '' if os.path.exists(out_dsm_fn) else ' ---> FAIL !!!'
                print('...done in {}{}\n'.format(time_to_print, err), flush=True)

            with open(os.path.join(out_dir_pc3dr, 'pc3dr_crashes.log'), 'a') as outfile:
                outfile.write(' \n\n'.join(crashes))
            os.system('rm -r {}/ply'.format(out_dir_pc3dr))
            time_to_print = loader.get_time_in_hours_mins_secs(timeit.default_timer() - t0)
            print('\nAll pc3dr runs completed in {} ({} crashes)'.format(time_to_print, len(crashes)), flush=True)
        else:
            print('Skipping intra-date pc3dr alignment !')
            os.makedirs(os.path.join(out_dir_pc3dr, 'dsms'), exist_ok=True)
            os.system('cp -r {}/dsms/* {}/dsms'.format(rec4D_dir, out_dir_pc3dr))


        # run max ncc alignment at inter-date level
        import rasterio
        t1 = timeit.default_timer()
        os.makedirs(os.path.join(out_dir_pc3dr, 'cdsms'), exist_ok=True)
        os.makedirs(os.path.join(out_dir_pc3dr, 'mcdsms'), exist_ok=True)
        os.makedirs(os.path.join(out_dir_pc3dr, 'rcdsms'), exist_ok=True)
        os.makedirs(os.path.join(out_dir_pc3dr, 'rcdsms_mask'), exist_ok=True)
        os.makedirs(os.path.join(out_dir_pc3dr, 'ncc_transforms'), exist_ok=True)
        t_ids = [self.timeline[t_idx]['id'] for t_idx in timeline_indices]
        all_dsms = ['{}/dsms/{}.tif'.format(out_dir_pc3dr, t_id) for t_id in t_ids]
        dsm_fnames = [fn for fn in all_dsms if os.path.exists(fn)]
        in_dir = out_dir_pc3dr
        for iter_cont, dsm in enumerate(dsm_fnames):
            filename = os.path.basename(dsm)

            # small hole interpolation by closing
            cdsm = in_dir + '/cdsms/c' + filename  # dsm after closing
            cmd_1 = 'bin/morsi square closing {}'.format(dsm)
            cmd_2 = 'bin/plambda {} - "x isfinite x y isfinite y nan if if" -o {}'.format(dsm, cdsm)
            os.system('{} | {}'.format(cmd_1, cmd_2))

            # larger holes with min interpolation
            mcdsm = in_dir + '/mcdsms/mc' + filename  # dsm after closing and min interpolation
            os.system('bin/bdint5pc -a min {} {}'.format(cdsm, mcdsm))

        ref_dsm = dsm_fnames[0]
        ref_filename = os.path.basename(ref_dsm)
        ref_cdsm = in_dir + '/cdsms/c' + ref_filename
        ref_mcdsm = in_dir + '/mcdsms/mc' + ref_filename
        for iter_cont, dsm in enumerate(dsm_fnames):
            filename = os.path.basename(dsm)

            cdsm = in_dir + '/cdsms/c' + filename
            mcdsm = in_dir + '/mcdsms/mc' + filename
            rcdsm = in_dir + '/rcdsms/' + filename
            rcdsm_mask = in_dir + '/rcdsms_mask/' + filename
            trans = in_dir + '/ncc_transforms/t_' + os.path.splitext(filename)[0] + '.txt'

            # compute horizontal registration on the interpolated DSMs
            os.system('bin/ncc_compute_shift {} {} 5 > {}'.format(ref_mcdsm, mcdsm, trans))
            dx, dy = np.loadtxt(trans)[:2]
            # compute vertical registration on the original DSMs
            os.system('bin/ncc_compute_shift {} {} 5 {} {} > {}'.format(ref_cdsm, cdsm, dx, dy, trans))
            # apply the registration
            os.system('bin/ncc_apply_shift {} `cat {}` {}'.format(cdsm, trans, rcdsm))

            # apply mask to dsm
            with rasterio.open(os.path.join(rec4D_dir, 'dsms/{}'.format(filename))) as src_dataset:
                kwds = src_dataset.profile
            with rasterio.open(rcdsm) as src:
                raster = src.read()[0, :, :]
            with rasterio.open(rcdsm_mask, "w", **kwds) as f:
                f.write(loader.apply_mask_to_raster(raster, mask), 1)
            # apply mask to std
            std_file = os.path.join(out_dir_pc3dr, 'dsms/std/{}'.format(filename))
            with rasterio.open(std_file) as src:
                raster = src.read()[0, :, :]
            with rasterio.open(std_file, "w", **kwds) as f:
                f.write(loader.apply_mask_to_raster(raster, mask), 1)

        os.system('rm -r {}/cdsms'.format(out_dir_pc3dr))
        os.system('rm -r {}/mcdsms'.format(out_dir_pc3dr))
        os.system('rm -r {}/rcdsms'.format(out_dir_pc3dr))
        os.system('rm -r {}/dsms/*.tif'.format(out_dir_pc3dr))
        os.system('rm -r {}/ncc_transforms'.format(out_dir_pc3dr))
        os.system('cp {}/rcdsms_mask/*.tif {}/dsms'.format(out_dir_pc3dr, out_dir_pc3dr))
        os.system('rm -r {}/rcdsms_mask'.format(out_dir_pc3dr))
        time_to_print = loader.get_time_in_hours_mins_secs(timeit.default_timer() - t1)
        print('\nAll inter-date alignments completed in {}\n'.format(time_to_print), flush=True)
        time_to_print = loader.get_time_in_hours_mins_secs(timeit.default_timer() - t0)
        print('\nTOTAL TIME: {}\n'.format(time_to_print), flush=True)


    def compute_reprojection_error_after_bundle_adjust(self, ba_method):

        im_fnames, C = self.ba_pipeline.myimages, self.ba_pipeline.ba_params.C
        pairs_to_triangulate, cam_model = self.ba_pipeline.ba_params.pairs_to_triangulate, 'rpc'

        # get init and bundle adjusted rpcs
        rpcs_init_dir = os.path.join(self.dst_dir, 'RPC_init')
        rpcs_init = loader.load_rpcs_from_dir(im_fnames, rpcs_init_dir, suffix='RPC', verbose=False)
        rpcs_ba_dir = os.path.join(self.dst_dir, ba_method + '/RPC_adj')
        rpcs_ba = loader.load_rpcs_from_dir(im_fnames, rpcs_ba_dir, suffix='RPC_adj', verbose=False)

        # triangulate
        from bundle_adjust.ba_triangulate import init_pts3d
        pts3d_before = init_pts3d(C, rpcs_init, cam_model, pairs_to_triangulate, verbose=False)
        #pts3d_after = init_pts3d(C, rpcs_ba, cam_model, pairs_to_triangulate, verbose=False)
        pts3d_after = self.ba_pipeline.ba_params.pts3d_ba

        # reproject
        n_pts, n_cam = C.shape[1], int(C.shape[0] / 2)
        err_before, err_after = [], []
        for cam_idx in range(n_cam):
            pt_indices = np.arange(n_pts)[~np.isnan(C[2 * cam_idx, :])]
            obs2d = C[(cam_idx * 2):(cam_idx * 2 + 2), pt_indices].T
            pts3d_init = pts3d_before[pt_indices, :]
            pts3d_ba = pts3d_after[pt_indices, :]
            args = [rpcs_init[cam_idx], rpcs_ba[cam_idx], cam_model, obs2d, pts3d_init, pts3d_ba]
            _, _, err_b, err_a, _ = ba_metrics.reproject_pts3d_and_compute_errors(*args)
            err_before.extend(err_b.tolist())
            err_after.extend(err_a.tolist())
        return np.mean(err_before), np.mean(err_after)


    def plot_timeline(self, timeline_indices, filename=None, date_label_freq=2):

        # plot distribution of temporal distances between consecutive dates
        # and plot also the number of images available per date

        n_dates = len(timeline_indices)
        dt2str = lambda t: t.strftime("%d %b\n%Hh") # %b to get month abreviation
        dates = [dt2str(self.timeline[timeline_indices[0]]['datetime'])]
        diff_in_days = []
        for i in range(n_dates - 1):
            d1 = self.timeline[timeline_indices[i]]['datetime']
            d2 = self.timeline[timeline_indices[i + 1]]['datetime']
            delta_days = abs((d1 - d2).total_seconds() / (24.0 * 3600))
            diff_in_days.append(delta_days)
            dates.append(dt2str(d2))

        n_ims = [self.timeline[i]['n_images'] for i in timeline_indices]

        fontsize = 14
        plt.rcParams['xtick.labelsize'] = fontsize
        fig_w = 1*n_dates/float(date_label_freq)
        if fig_w < 5:
            fig_w = fig_w*2
        fig, ax1 = plt.subplots(figsize=(fig_w, 5))

        color = 'tab:blue'
        l1, = ax1.plot(np.arange(1, n_dates), diff_in_days, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(bottom=np.floor(min(diff_in_days)) - 0.2, top=np.ceil(max(diff_in_days))+0.7)
        ax1_yticks = np.arange(0, np.ceil(max(diff_in_days))+0.6, 0.5).astype(float)
        ax1.set_yticks(ax1_yticks)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:orange'
        l2, = ax2.plot(np.arange(n_dates), n_ims, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(bottom=min(n_ims)-1.2, top=max(n_ims)+1.2)
        ax2_yticks = np.arange(min(n_ims)-1, max(n_ims)+2).astype(int)
        ax2.set_yticks(ax2_yticks)

        fontweight = 'bold'
        fontproperties = {'weight': fontweight, 'size': fontsize}
        ax1.set_yticklabels(['{:.1f}'.format(v) for v in ax1_yticks], fontproperties)
        ax2.set_yticklabels(ax2_yticks.astype(str).tolist(), fontproperties)

        plt.xticks(np.arange(n_dates)[::date_label_freq], np.array(dates)[::date_label_freq])
        legend_labels = ['distance to previous date in day units', 'number of images per date']
        plt.legend([l1, l2], legend_labels, fontsize=fontsize)
        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename, dpi=300)
        plt.show()


    def compute_dsm_registration_metrics(self, timeline_indices, ba_method=None,
                                         over_time=True, pc3dr=False, use_cdsms=False):

        rec4D_dir = self.set_rec4D_dir(ba_method)
        t_ids = [self.timeline[t_idx]['id'] for t_idx in timeline_indices]
        n_dates = len(timeline_indices)

        # get mean std per date
        if pc3dr:
            std_per_date_files = ['{}/pc3dr/dsms/std/{}.tif'.format(rec4D_dir, t_id) for t_id in t_ids]
        else:
            if use_cdsms:
                std_per_date_files = ['{}/metrics/std_per_date/{}.tif'.format(rec4D_dir, t_id) for t_id in t_ids]
            else:
                std_per_date_files = ['{}/dsms/std/{}.tif'.format(rec4D_dir, t_id) for t_id in t_ids]
        stacked_std_per_date = np.dstack([np.array(Image.open(fn)) for fn in std_per_date_files])
        avg_std_per_date = [np.nanmean(stacked_std_per_date[:, :, i], axis=(0, 1)) for i in range(n_dates)]

        if over_time:
            if pc3dr:
                std_over_time_file = '{}/metrics_over_time_pc3dr/std_over_time.tif'.format(rec4D_dir)
            else:
                if use_cdsms:
                    std_over_time_file = '{}/metrics_over_time_dense/std_over_time.tif'.format(rec4D_dir)
                else:
                    std_over_time_file = '{}/metrics_over_time/std_over_time.tif'.format(rec4D_dir)
            std_over_time = np.array(Image.open(std_over_time_file))

        print('\n\n******************** DSM registration metrics ********************', flush=True)
        print('  - ba_method: {}'.format(ba_method), flush=True)
        print('  - over_time: {}'.format(over_time), flush=True)
        print('  - pc3dr: {}'.format(pc3dr), flush=True)
        print('  - use_cdsms: {}\n'.format(use_cdsms), flush=True)
        if over_time:
            print('mean std over time: {:.3f}'.format(np.nanmean(std_over_time, axis=(0, 1))), flush=True)
        print('mean std per date: {:.3f}\n'.format(np.mean(avg_std_per_date)), flush=True)
        print('detailed std per date:', flush=True)
        for k, (t_id, v) in enumerate(zip(t_ids, avg_std_per_date)):
            print('({}/{}) {}:  {:.3f}'.format(k+1, n_dates, t_id, v), flush=True)
        print('******************************************************************\n\n', flush=True)
