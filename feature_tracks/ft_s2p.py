import numpy as np
import s2p
from feature_tracks import ft_sat

def detect_features_image_sequence(input_seq, masks=None, max_kp=None, parallelize=True):

    # default parameters
    thresh_dog = 0.0133
    nb_octaves = 8
    nb_scales = 3
    offset = None

    n_img = len(input_seq)
    sift_args = [(input_seq[i], thresh_dog, nb_octaves, nb_scales, offset) for i in range(n_img)]
    
    if parallelize:
        from multiprocessing import Pool
        with Pool() as p:
            features_p = p.starmap(s2p.sift.keypoints_from_nparray, sift_args)
    
    features = []
    for i in range(n_img):
        if parallelize:
            features_i = features_p[i]
        else: 
            features_i = s2p.sift.keypoints_from_nparray(input_seq[i], thresh_dog, nb_octaves, nb_scales, offset)
        
        if masks is not None:
            pts2d_colrow = features_i[:,:2].astype(np.int)
            true_if_obs_inside_aoi = 1*masks[i][pts2d_colrow[:,1], pts2d_colrow[:,0]] > 0
            features_i = features_i[true_if_obs_inside_aoi,:]
        
        features_i = np.array(sorted(features_i.tolist(), key=lambda kp: kp[2], reverse=True))  # reverse= True?
        if max_kp is not None:
            features_i = features_i[:max_kp]
        features.append(features_i)
        print('Found', features_i.shape[0], 'keypoints in image', i)
        
    return features
    

def s2p_match_SIFT(s2p_features_i, s2p_features_j, Fij, dst_thr=0.6):
    '''
    Match SIFT keypoints from an stereo pair
    '''
    # set matching parameters
    method = 'relative'
    sift_thr = dst_thr
    epipolar_thr = 10
    model = 'fundamental'
    ransac_max_err = 0.3
    
    matching_args = [s2p_features_i, s2p_features_j, \
                     method, sift_thr, Fij, epipolar_thr, model, ransac_max_err]
    
    matching_output = s2p.sift.keypoints_match(*matching_args)
    
    m_pts_i, m_pts_j = matching_output[:,:2].tolist(), matching_output[:,2:].tolist()
    
    all_pts_i = s2p_features_i[:,:2].tolist()
    all_pts_j = s2p_features_j[:,:2].tolist()
    matches_ij = np.array([np.array([all_pts_i.index(pt_i), all_pts_j.index(pt_j)]) for pt_i, pt_j in zip(m_pts_i, m_pts_j)])
    return matches_ij


def match_stereo_pairs(pairs_to_match, features, utm_coords, rpcs, input_seq, threshold=0.6, parallelize=True):
    
    def init_F_pair_to_match(h,w, rpc_i, rpc_j):
        import s2p
        rpc_matches = s2p.rpc_utils.matches_from_rpc(rpc_i, rpc_j, 0, 0, w, h, 5)
        Fij = s2p.estimation.affine_fundamental_matrix(rpc_matches)
        return Fij
    
    pairwise_matches_kp_indices = []
    pairwise_matches_im_indices = []
    
    matching_args = []
    for idx, pair in enumerate(pairs_to_match):
        i, j = pair['im_i'], pair['im_j']  
        h, w = input_seq[i].shape
        Fij = init_F_pair_to_match(h, w, rpcs[i], rpcs[j])
        utm_polygon = pair['intersection_poly']
        
        matching_args.append((features[i], features[j], utm_coords[i], utm_coords[j], utm_polygon, True, threshold, Fij))
            
    if parallelize:
        from multiprocessing import Pool
        with Pool() as p:
            matching_output = p.starmap(ft_sat.match_kp_within_utm_polygon, matching_args)
    
    for idx, pair in enumerate(pairs_to_match):
        i, j = pair['im_i'], pair['im_j']  
        # pick only those keypoints within the intersection area
        if parallelize:
            matches_ij = matching_output[idx]
        else:
            matches_ij = ft_sat.match_kp_within_utm_polygon(*matching_args[idx])
        n_matches = 0 if matches_ij is None else matches_ij.shape[0]
        print('Pair ({},{}) -> {} matches'.format(i,j,n_matches))

        if n_matches > 0:
            im_indices = np.vstack((np.array([i]*n_matches),np.array([j]*n_matches))).T
            pairwise_matches_kp_indices.extend(matches_ij.tolist())
            pairwise_matches_im_indices.extend(im_indices.tolist())
            
    pairwise_matches = np.hstack((np.array(pairwise_matches_kp_indices), np.array(pairwise_matches_im_indices)))
    
    print('\n')
    # filter matches with inconsistent utm coordinates
    pairwise_matches = ft_sat.filter_pairwise_matches_inconsistent_utm_coords(pairwise_matches, utm_coords)
    
    return pairwise_matches

