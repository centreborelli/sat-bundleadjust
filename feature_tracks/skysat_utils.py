import numpy as np

from IS18 import utils
from bundle_adjust import ba_core
from feature_tracks import feature_detection as fd
from feature_tracks import s2p_warp as fd_s2p


def filter_pairs_to_match_skysat(init_pairs, footprints, projection_matrices):

    # get optical centers and footprints
    optical_centers, n_img = [], len(footprints)
    for current_P in projection_matrices:
        _, _, _, current_optical_center = ba_core.decompose_perspective_camera(current_P)
        optical_centers.append(current_optical_center)
        
    pairs_to_match, pairs_to_triangulate = [], []
    for (i, j) in init_pairs:
        
            # check if the baseline between both cameras is large enough
            baseline = np.linalg.norm(optical_centers[i] - optical_centers[j])
            baseline_ok = baseline > 150000
            
            # check there is enough overlap between the images (at least 10% w.r.t image 1)
            intersection_polygon = footprints[i]['poly'].intersection(footprints[j]['poly']) 
            overlap_ok = intersection_polygon.area/footprints[i]['poly'].area >= 0.1
            
            if overlap_ok:    
                pairs_to_match.append({'im_i' : i, 'im_j' : j,
                       'footprint_i' : footprints[i], 'footprint_j' : footprints[j],
                       'baseline' : baseline, 'intersection_poly': intersection_polygon})
                 
                if baseline_ok:
                    pairs_to_triangulate.append((i,j))
                    
    print('{} / {} pairs to be matched'.format(len(pairs_to_match),int((n_img*(n_img-1))/2)))  
    return pairs_to_match, pairs_to_triangulate



def match_kp_within_utm_polygon(kp_i, kp_j, des_i, des_j, pair, matching_method='opencv', thr=0.8, F=None):
       
    east_i, north_i, east_j, north_j = kp_i[:,0], kp_i[:,1], kp_j[:,0], kp_j[:,1]
    
    # get instersection polygon utm coords
    east_poly, north_poly = pair['intersection_poly'].exterior.coords.xy
    east_poly, north_poly = np.array(east_poly), np.array(north_poly)
        
    # get centroid of the intersection polygon in utm coords
    #east_centroid, north_centroid = pair['intersection_poly'].centroid.coords.xy # centroid = baricenter ?
    #east_centroid, north_centroid = np.array(east_centroid), np.array(north_centroid)    
    #centroid_utm = np.array([east_centroid[0], north_centroid[0]])
    
    # use the rectangle containing the intersection polygon as AOI 
    min_east, max_east, min_north, max_north = min(east_poly), max(east_poly), min(north_poly), max(north_poly)
    
    east_ok_i = np.logical_and(east_i > min_east, east_i < max_east)
    north_ok_i = np.logical_and(north_i > min_north, north_i < max_north)
    indices_i_poly_bool, all_indices_i = np.logical_and(east_ok_i, north_ok_i), np.arange(kp_i.shape[0])
    indices_i_poly_int = all_indices_i[indices_i_poly_bool]
    
    if not any(indices_i_poly_bool):
        return [], []
    
    east_ok_j = np.logical_and(east_j > min_east, east_j < max_east)
    north_ok_j = np.logical_and(north_j > min_north, north_j < max_north)
    indices_j_poly_bool, all_indices_j = np.logical_and(east_ok_j, north_ok_j), np.arange(kp_j.shape[0])
    indices_j_poly_int = all_indices_j[indices_j_poly_bool]
    
    if not any(indices_j_poly_bool):
        return [], []
    
    # pick kp in overlap area and the descriptors
    kp_i_poly, des_i_poly = kp_i[indices_i_poly_bool], des_i[indices_i_poly_bool] 
    kp_j_poly, des_j_poly = kp_j[indices_j_poly_bool], des_j[indices_j_poly_bool]   
    
    '''
    import matplotlib.patches as patches
    fig, ax = plt.subplots(figsize=(10,6))
    plt.scatter(kp_i[:,0], kp_i[:,1], c='b')
    plt.scatter(kp_j[:,0], kp_j[:,1], c='r')
    plt.scatter(kp_i[indices_i_poly_int,0], kp_i[indices_i_poly_int,1], c='g')
    plt.scatter(kp_j[indices_j_poly_int,0], kp_j[indices_j_poly_int,1], c='g')
    rect = patches.Rectangle((min_east,min_north),max_east-min_east, max_north-min_north, facecolor='g', alpha=0.4)
    #plt.scatter(east_centroid, north_centroid, s=100, c='g')
    #plt.scatter(east_poly, north_poly, s=100, c='g')
    ax.add_patch(rect)
    plt.show()   
    '''
    
    if matching_method == 'opencv':
        indices_m_kp_i_poly, indices_m_kp_j_poly = fd.opencv_match_pair(kp_i_poly, kp_j_poly, \
                                                                        des_i_poly, des_j_poly, dst_thr=thr)
    else:
        indices_m_kp_i_poly, indices_m_kp_j_poly = fd_s2p.s2p_match_pair(des_i_poly, des_j_poly, F, dst_thr=thr)
        
    
    # go back from the filtered indices inside the polygon to the original indices of all the kps in the image
    if indices_m_kp_i_poly is None:
        indices_m_kp_i, indices_m_kp_j = [], []
    else:
        indices_m_kp_i, indices_m_kp_j = indices_i_poly_int[indices_m_kp_i_poly], indices_j_poly_int[indices_m_kp_j_poly]

    return indices_m_kp_i, indices_m_kp_j 


    
def opencv_feature_detection_skysat(input_seq, input_rpcs, footprints, input_masks=None):
    if input_masks is None:
        features = fd.opencv_feature_detection(input_seq, enforce_large_size = True, min_kp_size = 4.)
    else:
        features = fd.opencv_feature_detection(input_seq, masks=input_masks, enforce_large_size = True, min_kp_size = 4.) 
    for idx, features_current_im in enumerate(features):
        # convert im coords to utm coords
        pts = features[idx]['kp']
        cols, rows, alts = pts[:,0].tolist(), pts[:,1].tolist(), [footprints[idx]['z']] * pts.shape[0]
        lon, lat = input_rpcs[idx].localization(cols, rows, alts)
        east, north = utils.utm_from_lonlat(lon, lat)
        features[idx]['kp_utm'] = np.vstack((east, north)).T
    return features


def s2p_feature_detection_skysat(input_seq, input_rpcs, footprints, input_masks=None):
    
    features = fd_s2p.s2p_feature_detection(input_seq, masks=input_masks) 
    for idx, features_current_im in enumerate(features):
        # convert im coords to utm coords
        pts = features[idx]['kp']
        cols, rows, alts = pts[:,0].tolist(), pts[:,1].tolist(), [footprints[idx]['z']] * pts.shape[0]
        lon, lat = input_rpcs[idx].localization(cols, rows, alts)
        east, north = utils.utm_from_lonlat(lon, lat)
        features[idx]['kp_utm'] = np.vstack((east, north)).T
    return features


def opencv_matching_skysat(pairs_to_match, features, threshold=0.8):

    all_pairwise_matches = []
    for idx, pair in enumerate(pairs_to_match):
        i, j = pair['im_i'], pair['im_j']  
        kp_i, des_i, kp_i_id = features[i]['kp'], features[i]['des'], features[i]['id']
        kp_j, des_j, kp_j_id = features[j]['kp'], features[j]['des'], features[j]['id']
        kp_i_utm, kp_j_utm = features[i]['kp_utm'], features[j]['kp_utm']
        
        # pick only those keypoints within the intersection area
        indices_m_kp_i,indices_m_kp_j=match_kp_within_utm_polygon(kp_i_utm, kp_j_utm, \
                                                                           des_i, des_j, pair, 'opencv', thr=threshold)
        n_matches = 0 if indices_m_kp_i is None else len(indices_m_kp_i)
        print('Pair ({},{}) -> {} matches'.format(i,j,n_matches))

        if indices_m_kp_i is not None:
            matches_i_j = np.vstack((kp_i_id[indices_m_kp_i], kp_j_id[indices_m_kp_j])).T
            all_pairwise_matches.extend(matches_i_j.tolist())

    return all_pairwise_matches


def s2p_matching_skysat(pairs_to_match, features, input_seq, rpcs, threshold=0.6, parallelize=True):
    
    def init_F_pair_to_match(h,w, rpc_i, rpc_j):
        import s2p
        rpc_matches = s2p.rpc_utils.matches_from_rpc(rpc_i, rpc_j, 0, 0, w, h, 5)
        Fij = s2p.estimation.affine_fundamental_matrix(rpc_matches)
        return Fij
    
    all_pairwise_matches = []
    matching_args = []
    for idx, pair in enumerate(pairs_to_match):
        i, j = pair['im_i'], pair['im_j']  
        kp_i, des_i = features[i]['kp'], features[i]['des'] 
        kp_j, des_j = features[j]['kp'], features[j]['des']
        kp_i_utm, kp_j_utm = features[i]['kp_utm'], features[j]['kp_utm']
        s2p_i, s2p_j = features[i]['s2p'], features[j]['s2p']
        h, w = input_seq[i].shape
        
        Fij = init_F_pair_to_match(h, w, rpcs[i], rpcs[j])
        
        matching_args.append((kp_i_utm,kp_j_utm,s2p_i,s2p_j,pair,'s2p',threshold,Fij))
     
    
    if parallelize:
        from multiprocessing import Pool
        with Pool() as p:
            matching_output = p.starmap(match_kp_within_utm_polygon, matching_args)
    
    for idx, pair in enumerate(pairs_to_match):
        i, j = pair['im_i'], pair['im_j']  
        # pick only those keypoints within the intersection area
        if parallelize:
            indices_m_kp_i,indices_m_kp_j = matching_output[idx][0], matching_output[idx][1]
        else:
            indices_m_kp_i,indices_m_kp_j = match_kp_within_utm_polygon(*matching_args[idx])
        n_matches = 0 if indices_m_kp_i is None else len(indices_m_kp_i)
        print('Pair ({},{}) -> {} matches'.format(i,j,n_matches))

        kp_i_id, kp_j_id = features[i]['id'], features[j]['id']
        if indices_m_kp_i is not None:
            matches_i_j = np.vstack((kp_i_id[indices_m_kp_i], kp_j_id[indices_m_kp_j])).T
            all_pairwise_matches.extend(matches_i_j.tolist())

    return all_pairwise_matches
