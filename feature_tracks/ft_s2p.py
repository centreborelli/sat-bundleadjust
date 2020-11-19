import numpy as np
import s2p
from feature_tracks import ft_sat
import matplotlib.pyplot as plt

def detect_features_image_sequence(input_seq, masks=None, max_kp=None, parallel=True):

    # default parameters
    thresh_dog = 0.0133
    nb_octaves = 8
    nb_scales = 3
    offset = None

    n_img = len(input_seq)
    sift_args = [(input_seq[i], thresh_dog, nb_octaves, nb_scales, offset) for i in range(n_img)]
    
    if parallel:
        from multiprocessing import Pool
        with Pool() as p:
            features_p = p.starmap(s2p.sift.keypoints_from_nparray, sift_args)
    
    features = []
    for i in range(n_img):
        if parallel:
            features_i = features_p[i]
        else: 
            features_i = s2p.sift.keypoints_from_nparray(input_seq[i], thresh_dog, nb_octaves, nb_scales, offset)
        
        # features_i is a list of 132 floats, the first four elements are the keypoint (x, y, scale, orientation),
        # the 128 following values are the coefficients of the SIFT descriptor, i.e. integers between 0 and 255

        if masks is not None:
            pts2d_colrow = features_i[:, :2].astype(np.int)
            true_if_obs_inside_aoi = 1*masks[i][pts2d_colrow[:, 1], pts2d_colrow[:, 0]] > 0
            features_i = features_i[true_if_obs_inside_aoi, :]

        features_i = np.array(sorted(features_i.tolist(), key=lambda kp: kp[2], reverse=True))
        if max_kp is not None:
            features_i = features_i[:max_kp]
        features.append(features_i)
        print('{:3} keypoints in image {}'.format(features_i.shape[0], i), flush=True)

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

    matching_args = [s2p_features_i, s2p_features_j, method, sift_thr, Fij, epipolar_thr, model, ransac_max_err]
    
    matching_output = s2p.sift.keypoints_match(*matching_args)

    if len(matching_output) > 0:
        m_pts_i, m_pts_j = matching_output[:, :2].tolist(), matching_output[:, 2:].tolist()
        all_pts_i = s2p_features_i[:, :2].tolist()
        all_pts_j = s2p_features_j[:, :2].tolist()
        matches_ij = np.array([[all_pts_i.index(pt_i), all_pts_j.index(pt_j)] for pt_i, pt_j in zip(m_pts_i, m_pts_j)])
        n = matches_ij.shape[0]
    else:
        matches_ij = None
        n = 0
    return matches_ij, [n]


def match_stereo_pairs(pairs_to_match, features, footprints, utm_coords, rpcs, input_seq, threshold=0.6, parallel=True):
    
    def init_F_pair_to_match(h,w, rpc_i, rpc_j):
        import s2p
        rpc_matches = s2p.rpc_utils.matches_from_rpc(rpc_i, rpc_j, 0, 0, w, h, 5)
        Fij = s2p.estimation.affine_fundamental_matrix(rpc_matches)
        return Fij
    
    pairwise_matches_kp_indices = []
    pairwise_matches_im_indices = []
    
    matching_args = []
    for idx, pair in enumerate(pairs_to_match):
        i, j = pair[0], pair[1]  
        h, w = input_seq[i].shape
        Fij = init_F_pair_to_match(h, w, rpcs[i], rpcs[j])
        utm_polygon = footprints[i]['poly'].intersection(footprints[j]['poly'])
        
        matching_args.append((features[i], features[j], utm_coords[i], utm_coords[j], utm_polygon, True, threshold, Fij))
            
    if parallel:
        from multiprocessing import Pool
        with Pool() as p:
            matching_output = p.starmap(ft_sat.match_kp_within_utm_polygon, matching_args)
    
    for idx, pair in enumerate(pairs_to_match):
        i, j = pair[0], pair[1]  
        # pick only those keypoints within the intersection area
        if parallel:
            matches_ij, n = matching_output[idx]
        else:
            matches_ij, n = ft_sat.match_kp_within_utm_polygon(*matching_args[idx])

        n_matches = 0 if matches_ij is None else matches_ij.shape[0]
        args = [n_matches, n[0], n[1], (i, j)]
        print('{:4} matches (s2p: {:4}, utm: {:4}) in pair {}'.format(*args), flush=True)

        if n_matches > 0:
            im_indices = np.vstack((np.array([i]*n_matches), np.array([j]*n_matches))).T
            pairwise_matches_kp_indices.extend(matches_ij.tolist())
            pairwise_matches_im_indices.extend(im_indices.tolist())
            
    pairwise_matches = np.hstack((np.array(pairwise_matches_kp_indices), np.array(pairwise_matches_im_indices)))
    return pairwise_matches

