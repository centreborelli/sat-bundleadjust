import numpy as np
import cv2

from IS18 import utils
from bundle_adjust import ba_utils

def opencv_find_SIFT_kp(im, mask = None, enforce_large_size = False, min_kp_size = 3., max_kp = np.inf):
    '''
    Detect SIFT keypoints in an input image
    Requirement: pip3 install opencv-contrib-python==3.4.0.12
    '''
    
    sift = cv2.xfeatures2d.SIFT_create()
    if mask is not None:
        kp, des = sift.detectAndCompute(im,(1*mask).astype(np.uint8))
    else:
        kp, des = sift.detectAndCompute(im, None)
    
    # reduce number of keypoints if there are more than allowed
    if len(kp) > max_kp:
        prev_idx = np.arange(len(kp))
        new_idx = np.random.choice(prev_idx,max_kp_per_im,replace=False)
        kp, des = kp[new_idx], des[new_idx]
    
    # pick only keypoints from the first scale (satellite imagery)
    if enforce_large_size:
        kp, des = np.array(kp), np.array(des) 
        large_size_indices = np.array([current_kp.size > min_kp_size for current_kp in kp])
        kp, des = kp[large_size_indices].tolist(), des[large_size_indices].tolist()
    
    pts = np.array([kp[idx].pt for idx in range(len(kp))])
    
    return kp, des, pts


def opencv_match_pair(kp1, kp2, des1, des2, dst_thr=0.8):
    '''
    Match SIFT keypoints from an stereo pair
    '''
    
    # bruteforce matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)
    
    # FLANN parameters
    # from https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
    #FLANN_INDEX_KDTREE = 1
    #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #search_params = dict(checks=50)   # or pass empty dictionary
    #flann = cv2.FlannBasedMatcher(index_params,search_params)
    #matches = flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)
    
    # Apply ratio test as in Lowe's paper
    pts1, pts2, idx_matched_kp1, idx_matched_kp2 = [], [], [], []
    for m,n in matches:
        if m.distance < dst_thr*n.distance:
            pts2.append(kp2[m.trainIdx])
            pts1.append(kp1[m.queryIdx])
            idx_matched_kp1.append(m.queryIdx)
            idx_matched_kp2.append(m.trainIdx)
            
    # Geometric filtering using the Fundamental matrix
    pts1, pts2 = np.array(pts1), np.array(pts2)
    idx_matched_kp1, idx_matched_kp2 = np.array(idx_matched_kp1), np.array(idx_matched_kp2)
    if pts1.shape[0] > 0 and pts2.shape[0] > 0:
        F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

        # We select only inlier points
        if mask is None:
            # no matches after geometric filtering
            idx_matched_kp1, idx_matched_kp2 = None, None
        else:
            mask_bool = mask.ravel()==1
            idx_matched_kp1, idx_matched_kp2 = idx_matched_kp1[mask_bool], idx_matched_kp2[mask_bool]
    else:
        # no matches were left after ratio test
        idx_matched_kp1, idx_matched_kp2 = None, None
    
    return idx_matched_kp1, idx_matched_kp2


def opencv_matching(pairs_to_match, features, threshold=0.8):
    all_pairwise_matches = []
    for idx, [i,j] in enumerate(pairs_to_match):
        kp_i, des_i, kp_i_id = features[i]['kp'], features[i]['des'], np.array(features[i]['id'])
        kp_j, des_j, kp_j_id = features[j]['kp'], features[j]['des'], np.array(features[j]['id'])
        indices_m_kp_i, indices_m_kp_j = opencv_match_pair(kp_i, kp_j, des_i, des_j, dst_thr=threshold)

        n_matches = 0 if indices_m_kp_i is None else len(indices_m_kp_i)
        print('Pair ({},{}) -> {} matches'.format(i,j,n_matches))

        # display matches for pair (i,j)
        #im_matches = cv2.drawMatches(mycrops[i]['crop'],kp_i,mycrops[j]['crop'],kp_j,m_filt,outImg=np.array([]))
        #vistools.display_imshow(utils.simple_equalization_8bit(im_matches))

        if indices_m_kp_i is not None:
            matches_i_j = np.vstack((kp_i_id[indices_m_kp_i], kp_j_id[indices_m_kp_j])).T
            all_pairwise_matches.extend(matches_i_j.tolist())
    return all_pairwise_matches

def opencv_feature_detection(input_seq, masks=None, enforce_large_size = False, min_kp_size = 3.):
    # finds SIFT keypoints in a sequence of grayscale images
    # saves the keypoints coordinates, the descriptors, and assigns a unique id to each keypoint that is found
    kp_cont, n_img = 0, len(input_seq)
    features, all_keypoints, all_vertices = [], [], []
    for idx in range(n_img):
        if masks is None:
            kp, des, pts = opencv_find_SIFT_kp(input_seq[idx], None, enforce_large_size, min_kp_size)
        else:
            kp, des, pts = opencv_find_SIFT_kp(input_seq[idx], masks[idx], enforce_large_size, min_kp_size)
        kp_id = np.arange(kp_cont, kp_cont + pts.shape[0]).tolist()
        features.append({ 'kp': pts, 'des': np.array(des), 'id': np.array(kp_id) })
        all_keypoints.extend(pts.tolist())
        tmp = np.vstack((np.ones(pts.shape[0]).astype(int)*idx, kp_id)).T
        all_vertices.extend( tmp.tolist() )
        print('Found', pts.shape[0], 'keypoints in image', idx)
        kp_cont += pts.shape[0]
        #im_kp=cv2.drawKeypoints(input_seq[idx],kp,outImage=np.array([]))
        #vistools.display_image(im_kp) 
    return features


def feature_tracks_from_pairwise_matches(features, pairwise_matches, pairs_to_triangulate, indices_img_global):

    '''
    TO DO:
    This function has a drawback: everytime we build the feature tracks we load ALL features of the scene
    and ALL pairwise matches. When a reasonable amount of images is used this will be fine but for 1000 
    images the computation time may increase in a relevant way.
    The solution would be to save matches separately per image and directly load those of interest 
    (instead of filtering them from all_pairwise_matches).
    '''
    
    n_cams_in_use = len(indices_img_global)
    total_cams = len(features)

    # prepreate data to build correspondence matrix
    keypoints_coord_all, keypoints_im_idx_all = {}, {}
    for im_idx, features_current_im in enumerate(features):
        pts = features_current_im['kp']
        ids = features_current_im['id']
        keypoints_coord_all.update(dict(zip(ids, pts.tolist())))
        tmp = {}.fromkeys(ids, im_idx)
        keypoints_im_idx_all.update(tmp)  #{feature id: image idx}
    
    def find(parents, feature_id):
        p = parents[feature_id]
        return feature_id if not p else find(parents, p)

    def union(parents, feature_i_id, feature_j_id):
        p_1, p_2 = find(parents, feature_i_id), find(parents, feature_j_id)
        if p_1 != p_2: 
            parents[p_1] = p_2
    
    # get pairwise matches of interest, i.e. with matched features located in at least one image currently in use
    keypoints_im_idx, keypoints_coord = {}, {}
    true_where_im_in_use = np.zeros(total_cams).astype(bool)
    true_where_im_in_use[indices_img_global] = True
    pairwise_matches_of_interest = []
    for feature_i_id, feature_j_id in pairwise_matches:
        im_idx_i, im_idx_j = keypoints_im_idx_all[feature_i_id], keypoints_im_idx_all[feature_j_id]
        if true_where_im_in_use[im_idx_i] or true_where_im_in_use[im_idx_j]:
            pairwise_matches_of_interest.append([feature_i_id, feature_j_id])
            keypoints_im_idx[feature_i_id] = keypoints_im_idx_all[feature_i_id]
            keypoints_im_idx[feature_j_id] = keypoints_im_idx_all[feature_j_id]
            keypoints_coord[feature_i_id] = keypoints_coord_all[feature_i_id]
            keypoints_coord[feature_j_id] = keypoints_coord_all[feature_j_id]
    # associate a track index to each feature id
    feature_ids_in_use = list(keypoints_coord.keys())
    feature_ids_to_t_idx = dict(zip(feature_ids_in_use, np.arange(len(feature_ids_in_use)).tolist()))
    
    parents = [None]*(len(keypoints_im_idx))
    for feature_i_id, feature_j_id in pairwise_matches_of_interest:
        union(parents, feature_ids_to_t_idx[feature_i_id], feature_ids_to_t_idx[feature_j_id])

    # handle parents without None
    parents = [find(parents, feature_id) for feature_id, v in enumerate(parents)]
    
    # parents = track_id
    _, parents_indices, parents_counts = np.unique(parents, return_inverse=True, return_counts=True)
    n_tracks = np.sum(1*(parents_counts>1))
    track_parents = np.array(parents)[parents_counts[parents_indices] > 1]
    _, track_idx_from_parent, _ = np.unique(track_parents, return_inverse=True, return_counts=True)
    
    # t_idx, parent_id
    track_indices = np.zeros(len(parents))
    track_indices[:] = np.nan
    track_indices[parents_counts[parents_indices] > 1] = track_idx_from_parent
        
    '''
    Build a correspondence matrix C from a set of input feature tracks
    
    C = x11 ... x1n
        y11 ... y1n
        x21 ... x2n
        y21 ... y2n
        ... ... ...
        xm1 ... xmn
        ym1 ... ymn
 
        where (x11, y11) is the observation of feature track 1 in camera 1
              (xm1, ym1) is the observation of feature track 1 in camera m
              (x1n, y1n) is the observation of feature track n in camera 1
              (xmn, ymn) is the observation of feature track n in camera m
              
    Consequently, the shape of C is  (2*number of cameras) x number of feature tracks
    '''  
    
    # build correspondence matrix
    C = np.zeros((2*n_cams_in_use, n_tracks))
    C[:] = np.nan
    local_im_idx = dict(zip((np.arange(total_cams)[true_where_im_in_use]).astype(np.uint8), \
                            np.arange(n_cams_in_use).astype(np.uint8)))
    for (feature_i_id, feature_j_id) in pairwise_matches_of_interest:
        t_idx = int(track_indices[feature_ids_to_t_idx[feature_i_id]])
        # t_idx2 = int(track_indices[feature_ids_to_t_idx[feature_j_id]]) #sanity check: t_idx2 must be equal to t_idx
        im_idx_i, im_idx_j = keypoints_im_idx[feature_i_id], keypoints_im_idx[feature_j_id]
        if true_where_im_in_use[im_idx_i] and true_where_im_in_use[im_idx_j]:
            im_idx_i, im_idx_j = local_im_idx[im_idx_i], local_im_idx[im_idx_j]
            C[(2*im_idx_i):(2*im_idx_i+2), t_idx] = np.array(keypoints_coord[feature_i_id])
            C[(2*im_idx_j):(2*im_idx_j+2), t_idx] = np.array(keypoints_coord[feature_j_id])
    
    # remove matches found in pairs with short baseline that were not extended to more images
    # since these columns of C will not be triangulated
    columns_to_preserve = []
    for i in range(C.shape[1]):
        im_ind = [k for k, j in enumerate(range(n_cams_in_use)) if not np.isnan(C[j*2,i])]
        all_pairs = [(im_i, im_j) for im_i in im_ind for im_j in im_ind if im_i != im_j and im_i<im_j]
        good_pairs = [pair for pair in all_pairs if pair in pairs_to_triangulate]
        columns_to_preserve.append( len(good_pairs) > 0 )
    C = C[:, columns_to_preserve]
    
    print('Found {} tracks in total'.format(C.shape[1]))
    return C

def corresp_matrix_from_tracks(feature_tracks, r):
    '''
    Build a correspondence matrix C from a set of input feature tracks
    
    C = x11 ... x1n
        y11 ... y1n
        x21 ... x2n
        y21 ... y2n
        ... ... ...
        xm1 ... xmn
        ym1 ... ymn
 
        where (x11, y11) is the observation of feature track 1 in camera 1
              (xm1, ym1) is the observation of feature track 1 in camera m
              (x1n, y1n) is the observation of feature track n in camera 1
              (xmn, ymn) is the observation of feature track n in camera m
              
    Consequently, the shape of C is  (2*number of cameras) x number of feature tracks
    '''
    
    N = feature_tracks.shape[0]
    M = r.shape[1]
    C = np.zeros((2*M,N))
    C[:] = np.nan
    for i in range(N):
        im_ind = [k for k, j in enumerate(range(M)) if r[i,j]!=0]
        for ind in im_ind:
            C[ind*2:(ind*2)+2,i] = feature_tracks[i,:,ind]        
    return C  

