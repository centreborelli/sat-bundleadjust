import numpy as np

import s2p


def s2p_feature_detection(input_seq, masks=None, max_kp=np.inf, parallelize=True):

    # finds SIFT keypoints in a sequence of grayscale images
    # saves the keypoints coordinates, the descriptors, and assigns a unique id to each keypoint that is found
    thresh_dog = 0.0133
    nb_octaves = 8
    nb_scales = 3
    offset = None

    n_img = len(input_seq)
    sift_args = [(input_seq[im_idx], thresh_dog, nb_octaves, nb_scales, offset) for im_idx in range(n_img)]
    
    if parallelize:
        from multiprocessing import Pool
        with Pool() as p:
            features_output = p.starmap(s2p.sift.keypoints_from_nparray, sift_args)
    
    kp_cont = 0
    features, all_vertices, all_keypoints = [], [], []
    for idx in range(n_img):
        
        if parallelize:
            s2p_features = features_output[idx]
        else: 
            s2p_features = s2p.sift.keypoints_from_nparray(input_seq[idx], thresh_dog, nb_octaves, nb_scales, offset)
        
        if masks is not None:
            pts2d_colrow = s2p_features[:,:2].astype(np.int)
            true_if_obs_inside_aoi = 1*masks[idx][pts2d_colrow[:,1], pts2d_colrow[:,0]] > 0
            s2p_features = s2p_features[true_if_obs_inside_aoi,:]
       
        if s2p_features.shape[0] > max_kp:
            prev_idx = np.arange(s2p_features.shape[0])
            new_idx = np.random.choice(prev_idx,max_kp_per_im,replace=False)
            s2p_features = s2p_features[new_idx, :]
            
        des, pts = s2p_features[:,4:], s2p_features[:,:2]
        kp_id = np.arange(kp_cont, kp_cont + pts.shape[0]).tolist()
        features.append({ 'kp': pts, 'des':des, 'id': np.array(kp_id), 's2p': s2p_features})
        all_keypoints.extend(pts.tolist())
        tmp = np.vstack((np.ones(pts.shape[0]).astype(int)*idx, kp_id)).T
        all_vertices.extend( tmp.tolist() )
        print('Found', pts.shape[0], 'keypoints in image', idx)
        kp_cont += pts.shape[0]
    return features


def s2p_match_pair(s2p_features_i, s2p_features_j, Fij, dst_thr=0.6):
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
    
    matched_pt_i, matched_pt_j = matching_output[:,:2], matching_output[:,2:]
    
    all_pts_i = s2p_features_i[:,:2].tolist()
    idx_matched_kp1 = [all_pts_i.index(pt) for pt in matched_pt_i.tolist()]
    all_pts_j = s2p_features_j[:,:2].tolist()
    idx_matched_kp2 = [all_pts_j.index(pt) for pt in matched_pt_j.tolist()]

    return idx_matched_kp1, idx_matched_kp2

