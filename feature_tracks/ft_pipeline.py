import numpy as np
import os
import timeit
import pickle

from feature_tracks import ft_ranking
from feature_tracks import ft_utils
from feature_tracks import ft_sat
from feature_tracks import ft_opencv
from feature_tracks import ft_s2p


class FeatureTracksPipeline:
    def __init__(self, input_dir, output_dir, local_data, config=None, satellite=True, display_plots=False):
        
        
        self.config = config
        self.satellite = satellite
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.local_data = local_data
        self.global_data = {}
    
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
    
        # initialize parameters
        if self.satellite and self.config is None:
            self.config = {'s2p': False,
                           'matching_thr': 0.6,
                           'use_masks': False,
                           'filter_pairs': True,
                           'max_kp': 3000,
                           'optimal_subset': False,
                           'K': 30,
                           'tie_points': False,
                           'continue': False}
            
    def save_feature_detection_results(self):

        features_dir = os.path.join(self.output_dir, 'features')
        os.makedirs(features_dir, exist_ok=True)
        
        if self.satellite:
            features_utm_dir = os.path.join(self.output_dir, 'features_utm')
            os.makedirs(features_utm_dir, exist_ok=True)
        
        pickle_out = open(self.output_dir+'/filenames.pickle','wb')
        pickle.dump(self.global_data['fnames'], pickle_out)
        pickle_out.close()

        for idx in self.new_images_idx:
            
            f_id = os.path.splitext(os.path.basename(self.local_data['fnames'][idx]))[0]

            pickle_out = open(features_dir+'/{}.pickle'.format(f_id),'wb')
            pickle.dump(self.local_data['features'][idx], pickle_out)
            pickle_out.close()

            if self.satellite:   
                pickle_out = open(features_utm_dir+'/{}.pickle'.format(f_id),'wb')
                pickle.dump(self.local_data['features_utm'][idx], pickle_out)
                pickle_out.close()      


    def save_feature_matching_results(self):
        
        pickle_out = open(self.output_dir+'/matches.pickle','wb')
        pickle.dump(self.global_data['pairwise_matches'], pickle_out)
        pickle_out.close()
        pickle_out = open(self.output_dir+'/pairs_matching.pickle','wb')
        pickle.dump(self.global_data['pairs_to_match'], pickle_out)
        pickle_out.close()
        pickle_out = open(self.output_dir+'/pairs_triangulation.pickle','wb')
        pickle.dump(self.global_data['pairs_to_triangulate'], pickle_out)
        pickle_out.close()
    
                                     
    def init_feature_matching(self):
    

    
        # load previous matches and list of paris to be matched/triangulate if existent
        self.local_data['pairwise_matches'] = []
        self.local_data['pairs_to_triangulate'] = []
        self.local_data['pairs_to_match'] = []
        
        self.global_data['pairwise_matches'] = []
        self.global_data['pairs_to_match'] = []
        self.global_data['pairs_to_triangulate'] = []
        
        found_prev_matches = os.path.exists(self.input_dir+'/matches.pickle')
        found_prev_m_pairs = os.path.exists(self.input_dir+'/pairs_matching.pickle') 
        found_prev_t_pairs = os.path.exists(self.input_dir+'/pairs_triangulation.pickle')
        
        
        if np.sum(1*self.true_if_seen) > 0 and found_prev_matches and found_prev_m_pairs and found_prev_t_pairs:
            
            pickle_in = open(self.input_dir+'/matches.pickle','rb')
            self.global_data['pairwise_matches'].append(pickle.load(pickle_in))
            pickle_in = open(self.input_dir+'/pairs_matching.pickle','rb')
            self.global_data['pairs_to_match'].extend(pickle.load(pickle_in))
            pickle_in = open(self.input_dir+'/pairs_triangulation.pickle','rb')
            self.global_data['pairs_to_triangulate'].extend(pickle.load(pickle_in))
                      
            # load pairwise matches (if existent) within the images in use
            total_cams = len(self.global_data['fnames'])
            true_where_im_in_use = np.zeros(total_cams).astype(bool)
            true_where_im_in_use[self.local_idx_to_global_idx] = True
            
            true_where_prev_match = np.logical_and(true_where_im_in_use[self.global_data['pairwise_matches'][0][:,2]],
                                                   true_where_im_in_use[self.global_data['pairwise_matches'][0][:,3]])
            prev_pairwise_matches_in_use_global = self.global_data['pairwise_matches'][0][true_where_prev_match]
            prev_pairwise_matches_in_use_local = prev_pairwise_matches_in_use_global.copy()
            
            prev_pairwise_matches_in_use_local[:,2] = self.global_idx_to_local_idx[prev_pairwise_matches_in_use_local[:,2]]
            prev_pairwise_matches_in_use_local[:,3] = self.global_idx_to_local_idx[prev_pairwise_matches_in_use_local[:,3]]
            self.local_data['pairwise_matches'].append(prev_pairwise_matches_in_use_local)

            # incorporate triangulation pairs composed by pairs of previously seen images now in use
            for pair in self.global_data['pairs_to_triangulate']:
                local_im_idx_i = self.global_idx_to_local_idx[pair[0]]
                local_im_idx_j = self.global_idx_to_local_idx[pair[1]]
                if local_im_idx_i > -1  and local_im_idx_j > -1:
                    if self.true_if_seen[local_im_idx_i] and self.true_if_seen[local_im_idx_j]:
                        self.local_data['pairs_to_triangulate'].append(( min(local_im_idx_i, local_im_idx_j),
                                                                         max(local_im_idx_i, local_im_idx_j)))
            
             # incorporate matching pairs composed by pairs of previously seen images now in use
            for pair in self.global_data['pairs_to_match']:
                local_im_idx_i = self.global_idx_to_local_idx[pair[0]]
                local_im_idx_j = self.global_idx_to_local_idx[pair[1]]
                if local_im_idx_i > -1  and local_im_idx_j > -1:
                    if self.true_if_seen[local_im_idx_i] and self.true_if_seen[local_im_idx_j]:
                        self.local_data['pairs_to_match'].append(( min(local_im_idx_i, local_im_idx_j),
                                                                   max(local_im_idx_i, local_im_idx_j)))
    
    
    def init_feature_detection(self):

        
        def get_id(fn):
            return os.path.splitext(os.path.basename(fn))[0]
        
        import pickle

        n_adj = self.local_data['n_adj']
        n_new = self.local_data['n_new']
        local_fnames = self.local_data['fnames']
        self.global_data['fnames'] = []
        
        # load previous features if existent and list of previously adjusted filenames
        feats_dir = os.path.join(self.input_dir, 'features')
        feats_utm_dir = os.path.join(self.input_dir, 'features_utm')

        self.local_data['features'] = []
        if self.satellite:
            self.local_data['features_utm'] = []

        g_adj = []
        g_new = []

        if self.config['continue'] and os.path.exists(self.input_dir+'/filenames.pickle'):
            seen_fn = pickle.load(open(self.input_dir+'/filenames.pickle','rb')) # previously seen filenames
            self.global_data['fnames'] = seen_fn
            #print('LOADED PREVIOUS FILENAMES')
        else:
            seen_fn = []
            #print('STARTING FROM ZERO')

        n_cams_so_far = len(seen_fn)

        # check if files in use have been previously seen or not
        self.true_if_seen = np.array([fn in seen_fn for fn in local_fnames])

        # global indices of previously adjusted cameras (i.e. true_if_seen MUST be always true here)
        g_adj = []
        if n_adj > 0:
            adj_fnames = np.array(local_fnames)[:n_adj].tolist()
            for k, fn in enumerate(adj_fnames):
                if self.true_if_seen[k]:
                    g_idx = seen_fn.index(fn)
                    g_adj.append(g_idx)
                    f_id = get_id(seen_fn[g_idx])
                    self.local_data['features'].append(pickle.load(open(feats_dir+'/{}.pickle'.format(f_id),'rb')))
                    self.local_data['features_utm'].append(pickle.load(open(feats_utm_dir+'/{}.pickle'.format(f_id),'rb')))
                else:
                    print('something is very wrong if we fell here')


        # global indices of cameras to be adjusted that have been previously seen
        g_new = []
        n_cams_never_seen_before = 0
        self.new_images_idx = []

        new_fnames = np.array(local_fnames)[n_adj:].tolist()
        for k, fn in enumerate(new_fnames):
            if self.true_if_seen[n_adj+k]: 
                g_idx = seen_fn.index(fn)
                g_new.append(g_idx)
                f_id = get_id(seen_fn[g_idx])
                if os.path.exists(feats_dir+'/{}.pickle'.format(f_id)):
                    self.local_data['features'].append(pickle.load(open(feats_dir+'/{}.pickle'.format(f_id),'rb')))
                    self.local_data['features_utm'].append(pickle.load(open(feats_utm_dir+'/{}.pickle'.format(f_id),'rb')))
                else:
                    self.local_data['features'].append(np.array([np.nan]))
                    self.local_data['features_utm'].append(np.array([np.nan]))
                    self.new_images_idx.append(n_adj+k)
            else:
                n_cams_never_seen_before += 1
                g_new.append(n_cams_so_far + n_cams_never_seen_before - 1)
                self.local_data['features'].append(np.array([np.nan]))
                self.local_data['features_utm'].append(np.array([np.nan]))
                self.global_data['fnames'].append(fn)
                self.new_images_idx.append(n_adj+k)
                

        self.local_idx_to_global_idx = g_adj + g_new

        n_cams_in_use = len(self.local_idx_to_global_idx)

        self.global_idx_to_local_idx = -1 * np.ones(len(self.global_data['fnames']))
        self.global_idx_to_local_idx[self.local_idx_to_global_idx] = np.arange(n_cams_in_use)
        self.global_idx_to_local_idx = self.global_idx_to_local_idx.astype(int)
        
        
    def run_feature_detection(self):

        n_adj = self.local_data['n_adj']
        n_new = self.local_data['n_new']
        
        new_images = [self.local_data['images'][idx] for idx in self.new_images_idx]

        if self.local_data['masks'] is not None and self.config['use_masks']:
            new_masks = [self.local_data['masks'][idx] for idx in self.new_images_idx]
        else:
            new_masks = None

        if self.config['s2p']:
            new_features = ft_s2p.detect_features_image_sequence(new_images, masks=new_masks, max_kp=self.config['max_kp'])
        else:
            new_features = ft_opencv.detect_features_image_sequence(new_images, masks=new_masks, max_kp=self.config['max_kp']) 
        
        if self.satellite:
            new_rpcs = [self.local_data['rpcs'][idx] for idx in self.new_images_idx]
            new_footprints = [self.local_data['footprints'][idx] for idx in self.new_images_idx]
            new_offsets = [self.local_data['offsets'][idx] for idx in self.new_images_idx]
            new_features_utm = ft_sat.keypoints_to_utm_coords(new_features, new_rpcs, new_footprints, new_offsets)
            
        for k, idx in enumerate(self.new_images_idx):
            self.local_data['features'][idx] = new_features[k]
            self.local_data['features_utm'][idx] = new_features_utm[k]

    
    def get_stereo_pairs_to_match(self):
        
        n_adj = self.local_data['n_adj']
        n_new = self.local_data['n_new']
        
        init_pairs = []
        # possible new pairs to match are composed by 1 + 2 
        # 1. each of the previously adjusted images with the new ones
        for i in np.arange(n_adj):
            for j in np.arange(n_adj, n_adj + n_new):
                init_pairs.append((i, j))       
        # 2. each of the new images with the rest of the new images
        for i in np.arange(n_adj, n_adj + n_new):
            for j in np.arange(i+1, n_adj + n_new):
                init_pairs.append((i, j))
        
        # filter stereo pairs that are not overlaped
        # stereo pairs with small baseline should not be used to triangulate
        new_pairs_to_match, new_pairs_to_triangulate =\
        ft_sat.compute_pairs_to_match(init_pairs, self.local_data['footprints'], self.local_data['proj_matrices'])
        
        # remove pairs to match or to triangulate already in local_data
        new_pairs_to_triangulate = list(set(new_pairs_to_triangulate) - set(self.local_data['pairs_to_triangulate']))
        new_pairs_to_match = list(set(new_pairs_to_match) - set(self.local_data['pairs_to_match']))
        
        print('{} new pairs to be matched'.format(len(new_pairs_to_match)))
        
        # convert image indices from local to global (global indices consider all images, not only the ones in use)
        # and update all_pairs_to_match and all_pairs_to_triangulate
        for pair in new_pairs_to_triangulate:
            global_idx_i = self.local_idx_to_global_idx[pair[0]]
            global_idx_j = self.local_idx_to_global_idx[pair[1]]                                          
            self.global_data['pairs_to_triangulate'].append((min(global_idx_i, global_idx_j),
                                                             max(global_idx_i, global_idx_j)))

        for pair in new_pairs_to_match:
            global_idx_i = self.local_idx_to_global_idx[pair[0]]
            global_idx_j = self.local_idx_to_global_idx[pair[1]]
            self.global_data['pairs_to_match'].append((min(global_idx_i, global_idx_j),
                                                       max(global_idx_i, global_idx_j)))   
            
        self.local_data['pairs_to_match'] = new_pairs_to_match
        self.local_data['pairs_to_triangulate'].extend(new_pairs_to_triangulate)
            
        #print('PAIRS TO TRIANGULATE ', self.local_data['pairs_to_triangulate'])
        #print('PAIRS TO MATCH ', self.local_data['pairs_to_match'])
        
    
    def run_feature_matching(self):
        
        features_utm = self.local_data['features_utm'] if self.satellite else None
        footprints_utm = self.local_data['footprints'] if self.satellite else None

        if self.config['s2p'] and self.satellite:
            new_pairwise_matches = ft_s2p.match_stereo_pairs(self.local_data['pairs_to_match'],
                                                             self.local_data['features'],
                                                             footprints_utm,
                                                             features_utm,
                                                             self.local_data['rpcs'],
                                                             self.local_data['images'], 
                                                             threshold=self.config['matching_thr'])
        else:
            new_pairwise_matches = ft_opencv.match_stereo_pairs(self.local_data['images'],
                                                                self.local_data['pairs_to_match'],
                                                                self.local_data['features'],
                                                                footprints=footprints_utm,
                                                                utm_coords=features_utm,
                                                                threshold=self.config['matching_thr']) 
        
        print('Found {} new pairwise matches'.format(new_pairwise_matches.shape[0]))
        
        # add the newly found pairwise matches to local and global data
        self.local_data['pairwise_matches'].append(new_pairwise_matches)
        self.local_data['pairwise_matches'] = np.vstack(self.local_data['pairwise_matches'])
        
        new_pairwise_matches[:,2] = np.array(self.local_idx_to_global_idx)[new_pairwise_matches[:,2]]
        new_pairwise_matches[:,3] = np.array(self.local_idx_to_global_idx)[new_pairwise_matches[:,3]]
        
        self.global_data['pairwise_matches'].append(new_pairwise_matches)
        self.global_data['pairwise_matches'] = np.vstack(self.global_data['pairwise_matches']) 
                                           
    
    def get_feature_tracks(self):        
        
        C, C_v2 = ft_utils.feature_tracks_from_pairwise_matches(self.local_data['features'],
                                                                self.local_data['pairwise_matches'],
                                                                self.local_data['pairs_to_triangulate'])
        if self.config['optimal_subset']:
            selected_track_indices = ft_ranking.select_best_tracks_adj_cams(self.local_data['n_adj'],
                                                                            self.local_data['n_new'],
                                                                            C,
                                                                            self.local_data['proj_matrices'],
                                                                            self.local_data['pairs_to_triangulate'],
                                                                            self.local_data['cam_model'],
                                                                            K=self.config['K'])
            C = C[:, selected_track_indices]
            C_v2 = C_v2[:, selected_track_indices]


        feature_tracks = {}
        feature_tracks['features'] = self.local_data['features']
        feature_tracks['pairwise_matches'] = self.local_data['pairwise_matches']
        feature_tracks['pairs_to_triangulate'] = self.local_data['pairs_to_triangulate']
        feature_tracks['pairs_to_match'] = self.local_data['pairs_to_match']
        feature_tracks['C'] = C
        feature_tracks['C_v2'] = C_v2
        
        return feature_tracks
    
    
    def build_feature_tracks(self):

        # FEATURE DETECTION + MATCHING ON THE NEW IMAGES

        args = ['satellite' if self.satellite else 'generic', 's2p' if self.config['s2p'] else 'opencv']
        print('Building feature tracks - {} scenario - using {} SIFT\n'.format(*args))
        print('Parameters:')
        print('      use_masks:    {}'.format(self.config['use_masks']))
        print('      matching_thr: {}'.format(self.config['matching_thr']))
        print('\n')

        if self.config['use_masks'] and self.local_data['masks'] is None:
            print('Feature detection is set to use masks to restrict the search of keypoints, but no masks were found !')
            print('No masks will be used\n')

        start = timeit.default_timer()
        last_stop = start

        ############### 
        #feature detection
        ##############
        
        self.init_feature_detection()
        
        if self.local_data['n_new'] > 0:
            print('\nRunning feature detection...\n')
            self.run_feature_detection()
            self.save_feature_detection_results() 
        
            stop = timeit.default_timer()
            print('\n...done in {} seconds'.format(stop - last_stop))
            last_stop = stop
        else:  
            print('\nSkipping feature detection (no new images)')
         
        ############### 
        #compute stereo pairs to match
        ##############

        print('\nComputing pairs to match...\n')
        self.init_feature_matching()
        self.get_stereo_pairs_to_match()

        stop = timeit.default_timer()
        print('\n...done in {} seconds'.format(stop - last_stop))
        last_stop = stop

        ############### 
        #feature matching
        ##############
        
        if len(self.local_data['pairs_to_match']) > 0:
            print('\nMatching...\n')
            self.run_feature_matching()
            self.save_feature_matching_results()
            stop = timeit.default_timer()
            print('\n...done in {} seconds'.format(stop - last_stop))
            last_stop = stop
        else:  
            self.local_data['pairwise_matches'] = np.vstack(self.local_data['pairwise_matches'])
            self.global_data['pairwise_matches'] = np.vstack(self.global_data['pairwise_matches']) 
            print('\nSkipping matching (no pairs to match)')     
        
        print('\nPAIRS TO TRIANGULATE:\n{}'.format('\n'.join([str(x) for x in self.local_data['pairs_to_triangulate']])))
        nodes_in_pairs_to_triangulate = np.unique(np.array(self.local_data['pairs_to_triangulate']).flatten()).tolist()
        new_nodes = np.arange(self.local_data['n_adj'], self.local_data['n_adj'] + self.local_data['n_new']).tolist()
        sanity_check = len(list(set(new_nodes) - set(nodes_in_pairs_to_triangulate))) == 0
        print('do all new nodes appear at least once in pairs to triangulate?', sanity_check)
        
        ############### 
        #construct tracks
        ##############
                            
        print('\nExtracting feature tracks...\n') 
        feature_tracks = self.get_feature_tracks()
        print('Found {} tracks in total'.format(feature_tracks['C'].shape[1]))
        
        stop = timeit.default_timer()
        print('\n...done in {} seconds'.format(stop - last_stop))
        last_stop = stop

        
        hours, rem = divmod(last_stop - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print('\nTotal time: {:0>2}:{:0>2}:{:05.2f}\n\n\n'.format(int(hours),int(minutes),seconds))

        return feature_tracks
