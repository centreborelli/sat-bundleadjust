import numpy as np
import cv2

from bundle_adjust.loader import flush_print


def opencv_detect_SIFT(im, mask=None, max_nb=3000):
    """
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
    """

    sift = cv2.xfeatures2d.SIFT_create()
    if mask is not None:
        kp, des = sift.detectAndCompute(im.astype(np.uint8), (1 * mask).astype(np.uint8))
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
    features[: min(detections, max_nb)] = np.array([[*k.pt, k.size, k.angle, *d] for k, d in zip(kp, des)])

    return features


def detect_features_image_sequence(input_seq, masks=None, max_kp=None):
    """
    Finds SIFT features in a sequence of grayscale images
    Saves features per image and assigns a unique id to each keypoint that is found in the sequence
    Args:
        input_seq: list of 2D arrays with the input images
        masks (optional): list of 2D boolean arrays with the masks corresponding to the input images
        max_kp (optional): float, maximum number of features allowed per image
    Returns:
        features: list of N arrays containing the feature of each imge
    """

    n_img = len(input_seq)
    features = []
    for i in range(n_img):
        mask_i = None if masks is None else masks[i]
        features_i = opencv_detect_SIFT(input_seq[i], mask_i, max_nb=max_kp)
        features.append(features_i)
        n_kp = int(np.sum(~np.isnan(features_i[:, 0])))
        flush_print("{} keypoints in image {}".format(n_kp, i))

    return features


def opencv_match_SIFT(features_i, features_j, dst_thr=0.8, ransac_thr=0.3, matcher="flann"):
    """
    Matches SIFT keypoints
    Args:
        features_i: Nix132 array representing the Ni sift keypoints from image i
        features_j: Njx132 array representing the Nj sift keypoints from image j
        dst_thr (optional): distance threshold for the distance ratio test
    Returns:
        matches_ij: Mx2 array where each row represents a match that is found
                    1st col indicates keypoint index in features_i; 2nd col indicates keypoint index in features_j
    """

    descriptors_i = features_i[:, 4:].astype(np.float32)
    descriptors_j = features_j[:, 4:].astype(np.float32)
    if matcher == "bruteforce":
        # Bruteforce matcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_i, descriptors_j, k=2)
    elif matcher == "flann":
        # FLANN parameters
        # from https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors_i, descriptors_j, k=2)
    else:
        flush_print('ERROR: OpenCV matcher is not recognized ! Valid values are "flann" or "bruteforce"')

    # Apply ratio test as in Lowe's paper
    matches_ij = np.array([[m.queryIdx, m.trainIdx] for m, n in matches if m.distance < dst_thr * n.distance])
    n_matches_after_ratio_test = matches_ij.shape[0]

    # Geometric filtering using the Fundamental matrix
    if n_matches_after_ratio_test > 0:
        matches_ij = geometric_filtering(features_i, features_j, matches_ij, ransac_thr)
    else:
        # no matches were left after ratio test
        matches_ij = None
    n_matches_after_geofilt = 0 if matches_ij is None else matches_ij.shape[0]

    return matches_ij, [n_matches_after_ratio_test, n_matches_after_geofilt]


def geometric_filtering(features_i, features_j, matches_ij, ransac_thr=None):
    """
    Given a list of matches, fit a fundamental matrix using RANSAC and remove outliers
    Documentation here:
    https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#findfundamentalmat
    """
    kp_coords_i = features_i[matches_ij[:, 0], :2]
    kp_coords_j = features_j[matches_ij[:, 1], :2]
    if ransac_thr is None:
        F, mask = cv2.findFundamentalMat(kp_coords_i, kp_coords_j, cv2.FM_RANSAC)
    else:
        F, mask = cv2.findFundamentalMat(kp_coords_i, kp_coords_j, cv2.FM_RANSAC, ransac_thr)

    if mask is None:
        # no matches after geometric filtering
        matches_ij = None
    else:
        matches_ij = matches_ij[mask.ravel().astype(bool), :]
    return matches_ij
