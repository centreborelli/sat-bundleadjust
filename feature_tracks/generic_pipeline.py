import numpy as np
import os

from feature_tracks import feature_detection as fd
from feature_tracks import s2p_warp as fd_s2p

import timeit
import pickle

def run_feature_detection_generic(data_dir,
                                  n_adj,
                                  n_new,
                                  input_fnames,
                                  input_seq,
                                  input_masks=None,
                                  use_masks=False,
                                  matching_thr=0.6,
                                  feature_detection_lib='opencv'):
                                  

        features = []
        all_pairwise_matches = []
        all_pairs_to_match = []
        all_pairs_to_triangulate = []
        pairs_to_triangulate = []
                                 
                                 
        # FEATURE DETECTION + MATCHING ON THE NEW IMAGES

        print('Running generic {} feature detection...\n'.format(feature_detection_lib))
        print('Parameters:')
        print('      use_masks:    {}'.format(use_masks))
        print('      matching_thr: {}'.format(matching_thr))
        print('\n')
        
        start = timeit.default_timer()
        last_stop = start

        if use_masks and input_masks is None:
            print('Feature detection is set to use masks to restrict the search of keypoints, but no masks were found !')
            print('No masks will be used\n')
        
        # load previous features if existent and update myimages.pickle
        all_features, kp_cont = [], 0
        all_img_fnames, indices_adj_img_in_use = [], []
        if n_adj > 0:
            all_features = pickle.load(open(data_dir+'/features.pickle','rb'))
            all_adj_img_fnames = pickle.load(open(data_dir+'/myimages.pickle','rb'))
            all_img_fnames.extend(all_adj_img_fnames)
            adj_img_fnames_in_use = [input_fnames[idx] for idx in np.arange(n_adj)]
            indices_adj_img_in_use = [all_adj_img_fnames.index(fn) for fn in adj_img_fnames_in_use]
            features = np.array(all_features)[indices_adj_img_in_use].tolist()
            kp_cont = all_features[-1]['id'][-1] + 1
            print('Previous features loaded!')
        indices_new_img_in_use = np.arange(len(all_features), len(all_features) + n_new).tolist()
        indices_img_global = indices_adj_img_in_use + indices_new_img_in_use
        all_img_fnames.extend(np.array(input_fnames)[n_adj:])
        os.makedirs(data_dir, exist_ok=True)
        pickle_out = open(data_dir+'/myimages.pickle','wb')
        pickle.dump(all_img_fnames, pickle_out)
        pickle_out.close()
        
        
        # feature detection on the new view(s)
        new_indices = np.arange(n_adj, n_adj + n_new)
        if feature_detection_lib == 's2p':
            new_input_seq = [input_seq[idx] for idx in new_indices]
        else:
            new_input_seq = [input_seq[idx].astype(np.uint8) for idx in new_indices]
        if input_masks is not None and use_masks:
            new_masks = [input_masks[idx] for idx in new_indices]
        else:
            new_masks = None  
        new_features = fd.opencv_feature_detection(new_input_seq, masks=new_masks)
        for current_features in new_features:
            current_features['id'] += kp_cont
        features.extend(new_features)
        all_features.extend(new_features)
        total_cams = len(all_features)
        n_cams_in_use = n_adj + n_new
        pickle_out = open(data_dir+'/features.pickle','wb')
        pickle.dump(all_features, pickle_out)
        pickle_out.close()
        print('\nDetected features saved!')
        stop = timeit.default_timer()
        print('\n...done in {} seconds'.format(stop - last_stop))
        last_stop = stop
        
        
        print('\nComputing pairs to be matched...\n')
        # load previous matches and list of paris to be matched/triangulate if existent
        if n_adj > 0:
            pickle_in = open(data_dir+'/matches.pickle','rb')
            all_pairwise_matches, all_pairs_to_match, all_pairs_to_triangulate = pickle.load(pickle_in)
            print('Previous matches loaded!')

        # possible new pairs to match are composed by 1 + 2 
        # 1. each of the previously adjusted images with the new ones
        possible_pairs = []
        for i in np.arange(n_adj):
            for j in np.arange(n_adj, n_adj + n_new):
                possible_pairs.append((i, j))       
        # 2. each of the new images with the rest of the new images
        for i in np.arange(n_adj, n_adj + n_new):
            for j in np.arange(i+1, n_adj + n_new):
                possible_pairs.append((i, j))

        # filter stereo pairs that are not overlaped
        # stereo pairs with small baseline should not be used to triangulate 
        pairs2match, pairs2triangulate = possible_pairs.copy(), possible_pairs.copy()
        print('{} new pairs to be matched'.format(len(pairs2match)))
        # incorporate pairs composed by pairs of previously adjusted images now in use)
        true_where_im_in_use = np.zeros(total_cams).astype(bool)
        true_where_im_in_use[indices_img_global] = True
        local_im_idx = dict(zip((np.arange(total_cams)[true_where_im_in_use]).astype(np.uint8), \
                                np.arange(n_cams_in_use).astype(np.uint8)))
        for pair in all_pairs_to_triangulate:
            if true_where_im_in_use[pair[0]] and true_where_im_in_use[pair[1]]:
                pairs_to_triangulate.append((local_im_idx[pair[0]], local_im_idx[pair[1]]))
        pairs_to_triangulate.extend(pairs2triangulate)
        # convert image indices from local to global (global indices consider all images, not only the ones in use)
        # and update all_pairs_to_match and all_pairs_to_triangulate
        for pair in pairs2match:
            all_pairs_to_match.append((indices_img_global[pair[0]], indices_img_global[pair[1]]))
        for pair in pairs2triangulate: 
            all_pairs_to_triangulate.append((indices_img_global[pair[0]], indices_img_global[pair[1]]))
        
        stop = timeit.default_timer()
        print('\n...done in {} seconds'.format(stop - last_stop))
        last_stop = stop

        print('\nMatching...\n')
        new_pairwise_matches = fd.opencv_matching(pairs2match, features, matching_thr) 
        all_pairwise_matches.extend(new_pairwise_matches)

        pickle_out = open(data_dir+'/matches.pickle','wb')
        pickle.dump([all_pairwise_matches, all_pairs_to_match, all_pairs_to_triangulate], pickle_out)
        pickle_out.close()
        print('\nPairwise matches saved!')      
        stop = timeit.default_timer()
        print('\n...done in {} seconds'.format(stop - last_stop))
        last_stop = stop

        print('\nBuilding feature tracks...\n') 
        C = fd.feature_tracks_from_pairwise_matches(all_features, all_pairwise_matches, \
                                                    pairs_to_triangulate, indices_img_global)
        pickle_out = open(data_dir+'/Cmatrix.pickle','wb')
        pickle.dump(C, pickle_out)
        pickle_out.close()
        print('\nCorrespondence matrix saved!')
        stop = timeit.default_timer()
        print('\n...done in {} seconds'.format(stop - last_stop))
        last_stop = stop

        hours, rem = divmod(last_stop - start, 3600)
        minutes, seconds = divmod(rem, 60)
        print('\nTotal time: {:0>2}:{:0>2}:{:05.2f}\n\n\n'.format(int(hours),int(minutes),seconds))
        
        feature_tracks = {}
        feature_tracks['features'] = features
        feature_tracks['pairs_to_triangulate'] = pairs_to_triangulate
        feature_tracks['C'] = C
                                 
        return feature_tracks
