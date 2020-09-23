import numpy as np
import cv2
from feature_tracks import ft_sat

def opencv_detect_SIFT(im, mask=None, max_nb=3000):
    '''
    Detect SIFT keypoints in an input image
    Requirement: pip3 install opencv-contrib-python==3.4.0.12
    Documentation of opencv keypoint class: https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html
    
    Args:
        im: 2D array, input image
        mask (optional): binary mask to restrict the area of search in the image
        max_nb (optional): maximal number of keypoints. If more are detected, those at smallest scales are discarded
    
    Returns:
        features: Nx132 array containing the output N features. Each row-keypoint is represented by 132 values:
                  (col, row, scale, orientation) in positions 0-3 and (sift_descriptor) in the following 128 positions
    '''
    
    sift = cv2.xfeatures2d.SIFT_create()
    if mask is not None:
        kp, des = sift.detectAndCompute(im.astype(np.uint8),(1*mask).astype(np.uint8))
    else:
        kp, des = sift.detectAndCompute(im.astype(np.uint8), None)
    detections = len(kp)
    
    
    # pick only the largest keypoints if max_nb is different from None  
    if max_nb is None:
        max_nb = detections
    
    
    features = np.zeros((max_nb, 132))
    features[:] = np.nan
    sorted_indices = sorted(np.arange(len(kp)), key=lambda i: kp[i].size, reverse=True)
    kp = np.array(kp)[sorted_indices]
    des = np.array(des)[sorted_indices]
    kp = kp[:max_nb].tolist()
    des = des[:max_nb].tolist()
    
    # write result in the features format
    features[:min(detections, max_nb)] = np.array([np.array([*k.pt, k.size, k.angle, *d]) for k, d in zip(kp, des)])
    
    return features


def detect_features_image_sequence(input_seq, masks=None, max_kp=None):
    '''
    Finds SIFT features in a sequence of grayscale images
    Saves features per image and assigns a unique id to each keypoint that is found in the sequence
    
    Args:
        input_seq: list of 2D arrays with the input images
        masks (optional): list of 2D boolean arrays with the masks corresponding to the input images
        max_kp (optional): float, maximum number of features allowed per image
        
    Returns:
        features: list of N arrays containing the feature of each imge
    '''

    n_img = len(input_seq)
    features = []
    for i in range(n_img):
        mask_i = None if masks is None else masks[i]
        features_i = opencv_detect_SIFT(input_seq[i], mask_i, max_nb=max_kp)
        features.append(features_i)
        n_kp = np.sum(1*~np.isnan(features_i[:,0])) 
        print('Found {} keypoints in image {}'.format(n_kp, i))

    return features


def opencv_match_SIFT(features_i, features_j, dst_thr=0.8):
    '''
    Matches SIFT keypoints
    
    Args:
        features_i: Nix132 array representing the Ni sift keypoints from image i
        features_j: Njx132 array representing the Nj sift keypoints from image j
        dst_thr (optional): distance threshold for the distance ratio test
        
    Returns:
        matches_ij: Mx2 array where each row represents a match that is found
                    1st col indicates keypoint index in features_i; 2nd col indicates keypoint index in features_j
    '''
    
    # Bruteforce matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(features_i[:,4:].astype(np.float32),features_j[:,4:].astype(np.float32),k=2)
    
    # FLANN parameters
    # from https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
    #FLANN_INDEX_KDTREE = 1
    #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #search_params = dict(checks=50)   # or pass empty dictionary
    #flann = cv2.FlannBasedMatcher(index_params,search_params)
    #matches = flann.knnMatch(np.asarray(des1,np.float32),np.asarray(des2,np.float32),k=2)
    
    # Apply ratio test as in Lowe's paper
    matches_ij = [np.array([m.queryIdx, m.trainIdx]) for m,n in matches if m.distance < dst_thr*n.distance]
    n_matches = len(matches_ij)

    # Geometric filtering using the Fundamental matrix
    if len(matches_ij) > 0:
        matches_ij = np.array(matches_ij)
        F, mask = cv2.findFundamentalMat(features_i[matches_ij[:,0],:2], features_j[matches_ij[:,1],:2], cv2.FM_LMEDS)

        # We select only inlier points
        if mask is None:
            # no matches after geometric filtering
            matches_ij = None
        else:
            matches_ij = matches_ij[mask.ravel()==1,:]
    else:
        # no matches were left after ratio test
        matches_ij = None
    
    return matches_ij


    
def match_stereo_pairs(images, pairs_to_match, features, footprints=None, utm_coords=None, threshold=0.8):
    '''
    Given a list of features per image, matches the stereo pairs defined by pairs_to_match
    
    Args:
        pairs_to_match: the list of stereo pairs to match, in the format output by filter_pairs_to_match_utm
        features: a list of features from an image sequence, in the format output by detect_features_image_sequence
        threshold (optional): distance threshold for the distance ratio test
        utm_coords (optional): the utm coordinates of the feature keypoints
        
    Returns:
        pairwise_matches_kp_indices: Mx2 array where each row identifies a match that is found at image scale
                                     columns store the keypoint index within the corresponding image features
        pairwise_matches_im_indices: Mx2 array where each row identifies a match that is found at sequence scale
                                     columns store the indices of the images from the sequence where the match was found
    '''
    
    pairwise_matches_kp_indices = []
    pairwise_matches_im_indices = []
    
    for idx, pair in enumerate(pairs_to_match):
        i, j = pair[0], pair[1]
        
        if utm_coords is not None and footprints is not None:
            # pick only those keypoints within the utm intersection area between the satellite images
            utm_polygon = footprints[i]['poly'].intersection(footprints[j]['poly'])
            matches_ij = ft_sat.match_kp_within_utm_polygon(features[i], features[j], utm_coords[i], utm_coords[j],
                                                            utm_polygon, thr=threshold)
            
            n_matches_init = 0 if matches_ij is None else matches_ij.shape[0]
            if n_matches_init > 0:
                matches_ij = ft_sat.filter_pairwise_matches_inconsistent_utm_coords(matches_ij,
                                                                                    utm_coords[i],
                                                                                    utm_coords[j])
            n_matches = 0 if matches_ij is None else matches_ij.shape[0]
            print('Pair ({},{}) -> {} matches ({} after utm consistency check)'.format(i,j,n_matches_init,n_matches))
            
        else:
            matches_ij = ft_opencv.opencv_match_SIFT(features[i], features[j], dst_thr=threshold)
            n_matches = 0 if matches_ij is None else matches_ij.shape[0]
            print('Pair ({},{}) -> {} matches'.format(i,j,n_matches))
        
        if n_matches > 0:
            im_indices = np.vstack((np.array([i]*n_matches),np.array([j]*n_matches))).T
            pairwise_matches_kp_indices.extend(matches_ij.tolist())
            pairwise_matches_im_indices.extend(im_indices.tolist())
            
            '''
            tmp = np.hstack((np.array(pairwise_matches_kp_indices), np.array(pairwise_matches_im_indices)))
            from feature_tracks import ft_utils
            ft_utils.plot_pairwise_matches_stereo_pair(i, j, features, tmp, images)
            '''
    
    # pairwise match format is a 1x4 vector
    # position 1 corresponds to the kp index in image 1, that links to features[im1_index]
    # position 2 corresponds to the kp index in image 2, that links to features[im2_index]
    # position 3 is the index of image 1 within the sequence of images, i.e. im1_index
    # position 4 is the index of image 2 within the sequence of images, i.e. im2_index
    pairwise_matches = np.hstack((np.array(pairwise_matches_kp_indices), np.array(pairwise_matches_im_indices)))
    

    
    return pairwise_matches