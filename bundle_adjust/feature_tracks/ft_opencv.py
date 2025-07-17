"""
A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery
author: Roger Mari <roger.mari@ens-paris-saclay.fr>
year: 2021

This script implements functions dedicated to detect and match SIFT keypoints using OpenCV
"""

import numpy as np
import cv2
import os

from bundle_adjust.loader import flush_print, get_id
from bundle_adjust import loader
from bundle_adjust.feature_tracks import ft_utils

def opencv_detect_SIFT(geotiff_path, mask_path=None, offset=None, tracks_config=None):
    """
    Detect SIFT keypoints in a single input grayscale image using OpenCV
    Requirement: pip3 install opencv-contrib-python==3.4.0.12
    Documentation of opencv keypoint class: https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html

    Args:
        geotiff_path: path to npy 2d array, input image
        mask_path (optional): path to npy binary mask, to restrict the search of keypoints to a certain area,
                              parts of the mask with 0s are not explored
        offset (optional): dictionary that specifies a subwindow of the input geotiff,
                           this should be used in case we do not want to treat the entire image

    Returns:
        features: Nx132 array, where N is the number of SIFT keypoints detected in image i
                  each row/keypoint is represented by 132 values:
                  (col, row, scale, orientation) in columns 0-3 and (sift_descriptor) in the following 128 columns
        n_kp: integer, number of keypoints detected
    """

    config = ft_utils.init_feature_tracks_config(tracks_config)
    max_kp = None if tracks_config is None else config["FT_kp_max"]
    mask = None if mask_path is None else np.load(mask_path, mmap_mode='r').astype(np.uint8)

    found_existing_file = False
    if not config["FT_reset"] and "in_dir" in config.keys():
        npy_path_in = os.path.join(config["in_dir"], "features/{}.npy".format(get_id(geotiff_path)))
        if os.path.exists(npy_path_in):
            features_i = np.load(npy_path_in)
            found_existing_file = True
    if not found_existing_file:
        im = loader.load_image(geotiff_path, offset=offset, equalize=True)
        sift = cv2.SIFT_create() # cv2.xfeatures2d.SIFT_create() for older opencv versions
        kp, des = sift.detectAndCompute(im.astype(np.uint8), mask)
        features_i = np.array([[*k.pt, k.size, k.angle, *d] for k, d in zip(kp, des)])

    if mask_path is not None:
        mask = np.load(mask_path)
        pts2d_colrow = features_i[:, :2].astype(np.int)
        true_if_obs_inside_aoi = mask[pts2d_colrow[:, 1], pts2d_colrow[:, 0]] > 0
        features_i = features_i[true_if_obs_inside_aoi, :]

    # pick only the largest keypoints if max_nb is different from None
    features_i = np.array(sorted(features_i.tolist(), key=lambda kp: kp[2], reverse=True))
    if max_kp is not None:
        features_i_final = np.zeros((max_kp, 132))
        features_i_final[:] = np.nan
        features_i_final[: min(features_i.shape[0], max_kp)] = features_i[:max_kp]
    else:
        features_i_final = features_i
    n_kp = int(np.sum(~np.isnan(features_i_final[:, 0])))

    if config["FT_save"] and "out_dir" in config.keys():
        npy_path_out = os.path.join(config["out_dir"], "features/{}.npy".format(get_id(geotiff_path)))
        os.makedirs(os.path.dirname(npy_path_out), exist_ok=True)
        np.save(npy_path_out, features_i_final)

    return features_i_final, n_kp


def detect_features_image_sequence(geotiff_paths, mask_paths=None, offsets=None, tracks_config=None):
    """
    Detect SIFT keypoints in each image of a collection of input grayscale images using OpenCV
    This function iterates over the input sequence of images (input_seq) and calls opencv_detect_SIFT
    """

    n_img = len(geotiff_paths)
    features = []
    for i in range(n_img):
        mask_i = None if mask_paths is None else mask_paths[i]
        offset_i = None if offsets is None else offsets[i]
        features_i, n_kp = opencv_detect_SIFT(geotiff_paths[i], mask_i, offset_i, tracks_config)
        features.append(features_i)
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


def inliers_mask_from_fundamental_matrix(F, m1, m2, ransac_thr):
    """
    This function returns the error of a fundamental matrix F that links two sets of matching points,
    according to the epipolar equation: x'.T * F * x = 0 (where x' = m2 and x = m1)
    The output error is similar to the Symmetric Epipolar Distance, which is defined in formula 11.10
    in the Hartley and Zisserman book Multiple View Geometry in Computer Vision (second edition)
    but instead of adding the 2 distances of the points to its epipolar lines, it takes the maximum one
    https://github.com/opencv/opencv/blob/master/modules/calib3d/src/fundam.cpp#L796

     Args:
        m1, m2: input matches, i.e. arrays of corresponding 2d points with shape Nx2
        F: fundamental matrix with shape 3x3
        ransac_thr: float, RANSAC outlier rejection threshold
    Returns:
        inliers_mask: vector of length N, True if inlier False if outlier
    """

    F_ = F.ravel() if F.shape == (3, 3) else F
    assert m1.shape[1] == 2 and m2.shape[1] == 2 and m1.shape == m2.shape

    a = F_[0] * m1[:, 0] + F_[1] * m1[:, 1] + F_[2]
    b = F_[3] * m1[:, 0] + F_[4] * m1[:, 1] + F_[5]
    c = F_[6] * m1[:, 0] + F_[7] * m1[:, 1] + F_[8]

    s2 = 1. / (a * a + b * b)
    d2 = m2[:, 0] * a + m2[:, 1] * b + c  # m2.T * F * m1


    a = F_[0] * m2[:, 0] + F_[3] * m2[:, 1] + F_[6]
    b = F_[1] * m2[:, 0] + F_[4] * m2[:, 1] + F_[7]
    c = F_[2] * m2[:, 0] + F_[5] * m2[:, 1] + F_[8]

    s1 = 1. / (a * a + b * b)
    d1 = m1[:, 0] * a + m1[:, 1] * b + c  # m1.T * F * m2

    # vector of length N with the error associated to each match
    err = np.max(np.vstack((d1 * d1 * s1, d2 * d2 * s2)), axis=0)

    # build mask
    inliers_mask = err < ransac_thr ** 2
    if sum(inliers_mask.ravel()) == 0:
        inliers_mask = None
    return inliers_mask


def geometric_filtering(features_i, features_j, matches_ij, ransac_thr=0.3, return_mask=False):
    """
    Given a series of pairwise matches, use OpenCV to fit a fundamental matrix using RANSAC to filter outliers
    The 7-point algorithm is used to derive the fundamental matrix
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

    #mask = inliers_mask_from_fundamental_matrix(F, kp_coords_i, kp_coords_j, ransac_thr)
    matches_ij = matches_ij[mask.ravel().astype(bool), :] if mask is not None else None

    if return_mask:
        return matches_ij, mask

    return matches_ij
