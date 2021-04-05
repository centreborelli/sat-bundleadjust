"""
A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery
author: Roger Mari <roger.mari@ens-paris-saclay.fr>
year: 2021

This script implements functions dedicated to detect and match SIFT keypoints using OpenCV
"""

import numpy as np
import cv2

from bundle_adjust.loader import flush_print


def opencv_detect_SIFT(im, mask=None, max_kp=None):
    """
    Detect SIFT keypoints in a single input grayscale image using OpenCV
    Requirement: pip3 install opencv-contrib-python==3.4.0.12
    Documentation of opencv keypoint class: https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html

    Args:
        im: 2d array, input image
        mask (optional): binary mask to restrict the detection of keypoints to a certain area of the image
                         (parts of the mask with 0s are not explored)
        max_kp (optional): integer, the maximum number of keypoints that is allowed to detect for an image
                           if the detections exceed max_kp then points with larger scale are given priority

    Returns:
        features: list of N[i]x132 arrays, where N is the number of SIFT keypoints detected in image i
                  each row/keypoint is represented by 132 values:
                  (col, row, scale, orientation) in columns 0-3 and (sift_descriptor) in the following 128 columns
    """

    sift = cv2.xfeatures2d.SIFT_create()
    if mask is not None:
        kp, des = sift.detectAndCompute(im.astype(np.uint8), mask.astype(np.uint8))
    else:
        kp, des = sift.detectAndCompute(im.astype(np.uint8), None)
    detections = len(kp)

    # pick only the largest keypoints if max_nb is different from None
    if max_kp is None:
        max_kp = detections

    features = np.zeros((max_kp, 132))
    features[:] = np.nan
    sorted_indices = sorted(np.arange(len(kp)), key=lambda i: kp[i].size, reverse=True)
    kp = np.array(kp)[sorted_indices]
    des = np.array(des)[sorted_indices]
    kp = kp[:max_kp].tolist()
    des = des[:max_kp].tolist()

    # write result in the features format
    features[: min(detections, max_kp)] = np.array([[*k.pt, k.size, k.angle, *d] for k, d in zip(kp, des)])

    return features


def detect_features_image_sequence(input_seq, masks=None, max_kp=None):
    """
    Detect SIFT keypoints in each image of a collection of input grayscale images using OpenCV
    This function iterates over the input sequence of images (input_seq) and calls opencv_detect_SIFT
    """

    n_img = len(input_seq)
    features = []
    for i in range(n_img):
        mask_i = None if masks is None else masks[i]
        features_i = opencv_detect_SIFT(input_seq[i], mask_i, max_kp=max_kp)
        features.append(features_i)
        n_kp = int(np.sum(~np.isnan(features_i[:, 0])))
        flush_print("{} keypoints in image {}".format(n_kp, i))

    return features


def opencv_match_SIFT(features_i, features_j, dst_thr=0.8, ransac_thr=0.3, matcher="flann"):
    """
    Match SIFT keypoints using OpenCV matchers

    Args:
        features_i: N[i]x132 array representing the N[i] keypoints from image i
        features_j: N[j]x132 array representing the N[j] keypoints from image j
        dst_thr (optional): float, threshold for SIFT distance ratio test
        ransac_thr (optional): float, threshold for RANSAC geometric filtering using the fundamental matrix
        matcher (optional): string, identifies the OpenCV matcher to use: either "flann" or "bruteforce"
    Returns:
        matches_ij: Mx2 array representing M matches. Each match is represented by two values (i, j)
                    which means that the i-th kp/row in s2p_features_i matches the j-th kp/row in s2p_features_j
        n_matches_after_ratio_test: integer, the number of matches left after the SIFT distance ratio test
        n_matches_after_geofilt: integer, the number of matches left after RANSAC filtering
    """

    descriptors_i = features_i[:, 4:].astype(np.float32)
    descriptors_j = features_j[:, 4:].astype(np.float32)
    if matcher == "bruteforce":
        # Bruteforce matcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_i, descriptors_j, k=2)
    elif matcher == "flann":
        # FLANN matcher
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

    return matches_ij, n_matches_after_ratio_test, n_matches_after_geofilt


def geometric_filtering(features_i, features_j, matches_ij, ransac_thr=None):
    """
    Given a series of pairwise matches, use OpenCV to fit a fundamental matrix using RANSAC to filter outliers
    https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#findfundamentalmat

    Args:
        features_i: N[i]x132 array representing the N[i] keypoints from image i
        features_j: N[j]x132 array representing the N[j] keypoints from image j
        matches_ij: Mx2 array representing M matches between features_i and features_j
        ransac_thr (optional): float, RANSAC outlier rejection threshold

    Returns:
        matches_ij: filtered version of matches_ij (will contain same amount of rows or less)
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
