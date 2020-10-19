import numpy as np

from IS18 import utils
from bundle_adjust import ba_core
from feature_tracks import ft_utils as fd

from feature_tracks import ft_s2p
from feature_tracks import ft_opencv

def keypoints_to_utm_coords(features, rpcs, footprints, offsets):
    
    features_utm = []
    for features_i, rpc_i, footprint_i, offset_i in zip(features, rpcs, footprints, offsets):

        # convert image coords to utm coords (remember to deal with the nan pad...)
        n_kp = np.sum(1*~np.isnan(features_i[:,0])) 
        cols = (features_i[:n_kp,0] + offset_i['col0']).tolist()
        rows = (features_i[:n_kp,1] + offset_i['row0']).tolist()
        alts = [footprint_i['z']] * n_kp
        lon, lat = rpc_i.localization(cols, rows, alts)
        east, north = utils.utm_from_lonlat(lon, lat)
        utm_coords = np.vstack((east, north)).T
        rest = features_i[n_kp:, :2].copy()
        features_utm.append(np.vstack((utm_coords, rest)))
        
    return features_utm
    

def compute_pairs_to_match(init_pairs, footprints, optical_centers, no_filter=False, verbose=True):

    pairs_to_match, pairs_to_triangulate = [], []
    for (i, j) in init_pairs:
        
            # check there is enough overlap between the images (at least 10% w.r.t image 1)
            intersection_polygon = footprints[i]['poly'].intersection(footprints[j]['poly'])
            
            # check if the baseline between both cameras is large enough
            baseline = np.linalg.norm(optical_centers[i] - optical_centers[j])
  
            if no_filter:
                overlap_ok = True
                baseline_ok = True
            else:
                overlap_ok = intersection_polygon.area/footprints[i]['poly'].area >= 0.1
                baseline_ok = baseline > 125000 #150000
            
            if overlap_ok:    
                pairs_to_match.append((i,j))
                 
                if baseline_ok:
                    pairs_to_triangulate.append((i,j))
                    
    # total number of possible pairs given n_imgs is int((n_img*(n_img-1))/2)
    if verbose:
        print('     {} / {} pairs suitable to match'.format(len(pairs_to_match), len(init_pairs)))
        print('     {} / {} pairs suitable to triangulate'.format(len(pairs_to_triangulate), len(init_pairs)))
    return pairs_to_match, pairs_to_triangulate
    

def match_kp_within_utm_polygon(features_i, features_j, utm_i, utm_j, utm_polygon, s2p=False, thr=0.8, F=None):
        
    east_i, north_i, east_j, north_j = utm_i[:,0], utm_i[:,1], utm_j[:,0], utm_j[:,1]
    
    # get instersection polygon utm coords
    east_poly, north_poly = utm_polygon.exterior.coords.xy
    east_poly, north_poly = np.array(east_poly), np.array(north_poly)

    # get centroid of the intersection polygon in utm coords
    #east_centroid, north_centroid = pair['intersection_poly'].centroid.coords.xy # centroid = baricenter ?
    #east_centroid, north_centroid = np.array(east_centroid), np.array(north_centroid)    
    #centroid_utm = np.array([east_centroid[0], north_centroid[0]])
    
    # use the rectangle containing the intersection polygon as AOI 
    min_east, max_east, min_north, max_north = min(east_poly), max(east_poly), min(north_poly), max(north_poly)
    
    east_ok_i = np.logical_and(east_i > min_east, east_i < max_east)
    north_ok_i = np.logical_and(north_i > min_north, north_i < max_north)
    indices_i_poly_bool, all_indices_i = np.logical_and(east_ok_i, north_ok_i), np.arange(utm_i.shape[0])
    indices_i_poly_int = all_indices_i[indices_i_poly_bool]
    if not any(indices_i_poly_bool):
        return None
    
    east_ok_j = np.logical_and(east_j > min_east, east_j < max_east)
    north_ok_j = np.logical_and(north_j > min_north, north_j < max_north)
    indices_j_poly_bool, all_indices_j = np.logical_and(east_ok_j, north_ok_j), np.arange(utm_j.shape[0])
    indices_j_poly_int = all_indices_j[indices_j_poly_bool]
    if not any(indices_j_poly_bool):
        return None
    
    # pick kp in overlap area and the descriptors
    features_i_poly, features_j_poly = features_i[indices_i_poly_bool], features_j[indices_j_poly_bool]
    
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
    
    if s2p:
        matches_ij_poly = ft_s2p.s2p_match_SIFT(features_i_poly, features_j_poly, F, dst_thr=thr)
    else:
        matches_ij_poly = ft_opencv.opencv_match_SIFT(features_i_poly, features_j_poly, dst_thr=thr)
    
    # go back from the filtered indices inside the polygon to the original indices of all the kps in the image
    if matches_ij_poly is None:
        matches_ij = None
    else:
        indices_m_kp_i, indices_m_kp_j = indices_i_poly_int[matches_ij_poly[:,0]], indices_j_poly_int[matches_ij_poly[:,1]]
        matches_ij = np.vstack((indices_m_kp_i, indices_m_kp_j)).T
    return matches_ij


def filter_pairwise_matches_inconsistent_utm_coords(matches_ij, features_utm_i, features_utm_j):
 
    n_init = matches_ij.shape[0]
    pt_i_utm = features_utm_i[matches_ij[:,0]]
    pt_j_utm = features_utm_j[matches_ij[:,1]]
    
    all_utm_distances = np.linalg.norm(pt_i_utm - pt_j_utm, axis=1)
    from bundle_adjust.ba_outliers import get_elbow_value
    utm_thr, success = get_elbow_value(all_utm_distances, verbose=False)
    utm_thr = utm_thr + 10 if success else np.max(all_utm_distances)
    matches_ij = matches_ij[all_utm_distances <= utm_thr]
    
    n_filt = matches_ij.shape[0]
    
    removed = n_init - n_filt
    percent = (float(removed)/n_init) * 100.
    
    '''
    print('UTM consistency distance threshold set to {:.2f} m'.format(utm_thr))
    print('Removed {} pairwise matches ({:.2f}%) due to inconsistent UTM coords ({} left)'.format(removed, percent, n_filt))
    ''' 
    return matches_ij