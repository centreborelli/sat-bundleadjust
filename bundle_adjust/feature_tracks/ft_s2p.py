"""
A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery
author: Roger Mari <roger.mari@ens-paris-saclay.fr>
year: 2021

This script implements functions dedicated to detect and match SIFT keypoints using s2p (satellite stereo pipeline)
Code of the s2p: https://github.com/cmla/s2p
"""

import numpy as np

from bundle_adjust.loader import flush_print
from bundle_adjust import loader


def detect_features_image_sequence(geotiff_paths, mask_paths=None, offsets=None, npy_paths=None,
                                   max_kp=None, image_indices=None, thread_idx=None):
    """
    Detects SIFT keypoints in each image of a collection of input grayscale images using s2p

    Args:
        geotiff_paths: list of paths to npy 2d arrays, input images (floats allowed)
        mask_paths (optional): list of paths to npy 2d arrays binary masks, to restrict the search of keypoints
                               to a certain area, parts of the mask with 0s are not explored
        offsets (optional): list of dictionaries that specify a subwindow of the input geotiffs
        npy_paths (optional): list of paths to the npy array files where the detected keypoints will be stored
        max_kp (optional): integer, the maximum number of keypoints that is allowed to detect for an image
                           if the detections exceed max_kp then points with larger scale are given priority
        image_indices (optional): the index of the image with respect to the input sequence,
                                  only interesting for verbose when multiprocessing is used
        thread_idx (optional): integer, the thread index, only interesting for verbose when multiprocessing is used

    Returns:
        features: list of N[i]x132 arrays, where N is the number of SIFT keypoints detected in image i
                  each row/keypoint is represented by 132 values:
                  (col, row, scale, orientation) in columns 0-3 and (sift_descriptor) in the following 128 columns
    """
    from bundle_adjust.s2p.sift import keypoints_from_nparray

    # default parameters
    thresh_dog = 0.0133
    nb_octaves = 8
    nb_scales = 3
    offset = None

    multiproc = False if thread_idx is None else True
    image_indices = np.arange(len(geotiff_paths)) if image_indices is None else image_indices

    features = []
    for i, geotiff_path in enumerate(geotiff_paths):

        image = loader.load_image(geotiff_path, offset=None if offsets is None else offsets[i])
        features_i = keypoints_from_nparray(image, thresh_dog, nb_octaves, nb_scales, offset)

        # features_i is a list of 132 floats, the first four elements are the keypoint (x, y, scale, orientation),
        # the 128 following values are the coefficients of the SIFT descriptor, i.e. integers between 0 and 255

        if mask_paths is not None:
            mask = np.load(mask_paths[i], mmap_mode='r')
            pts2d_colrow = features_i[:, :2].astype(np.int)
            true_if_obs_inside_aoi = mask[pts2d_colrow[:, 1], pts2d_colrow[:, 0]] > 0
            features_i = features_i[true_if_obs_inside_aoi, :]

        features_i = np.array(sorted(features_i.tolist(), key=lambda kp: kp[2], reverse=True))

        if max_kp is not None:
            features_i_final = np.zeros((max_kp, 132))
            features_i_final[:] = np.nan
            features_i_final[: min(features_i.shape[0], max_kp)] = features_i[:max_kp]
        else:
            features_i_final = features_i

        n_kp = int(np.sum(~np.isnan(features_i_final[:, 0])))
        tmp = ""
        if multiproc:
            tmp = " (thread {} -> {}/{})".format(thread_idx, i + 1, len(geotiff_paths))
        flush_print("{} keypoints in image {}{}".format(n_kp, image_indices[i] if multiproc else i, tmp))

        if npy_paths is None:
            features.append(features_i_final)
        else:
            np.save(npy_paths[i], features_i_final)

    if npy_paths is None:
        return features


def detect_features_image_sequence_multiprocessing(geotiff_paths, mask_paths=None, offsets=None, npy_paths=None,
                                                   max_kp=None, n_proc=5):
    """
    This function is just a wrapper to call detect_features_image_sequence using multiprocessing
    The inputs and outputs are therefore the ones defined in detect_features_image_sequence
    The number of independent threads is given by the additional input field n_proc (integer)
    """
    n_img = len(geotiff_paths)
    n = int(np.ceil(n_img / n_proc))
    args = []
    for k, i in enumerate(np.arange(0, n_img, n)):
        mask_paths_ = None if mask_paths is None else mask_paths[i : i + n]
        offsets_ = None if offsets is None else offsets[i : i + n]
        npy_paths_ = None if npy_paths is None else npy_paths[i : i + n]
        tmp = [geotiff_paths[i : i + n], mask_paths_, offsets_, npy_paths_, max_kp, np.arange(i, i + n).astype(int), k]
        args.append(tmp)

    parallel_lib = "pool"
    if parallel_lib == "joblib":
        from joblib import Parallel, delayed, parallel_backend
        with parallel_backend(n_jobs=n_proc, backend='multiprocessing'):
            detection_output = Parallel()(delayed(detect_features_image_sequence)(*a) for a in args)
    else:
        from multiprocessing import Pool
        with Pool(len(args)) as p:
            detection_output = p.starmap(detect_features_image_sequence, args)

    if npy_paths is None:
        flatten_list = lambda t: [item for sublist in t for item in sublist]
        return flatten_list(detection_output)


def s2p_match_SIFT(s2p_features_i, s2p_features_j, Fij, dst_thr=0.6, ransac_thr=0.3):
    """
    Match SIFT keypoints from two images using s2p

    Args:
        s2p_features_i: N[i]x132 array representing the N[i] keypoints from image i
        s2p_features_j: N[j]x132 array representing the N[j] keypoints from image j
        Fij: 3x3 array with the fundamental matrix between image i and j
        dst_thr (optional): float, threshold for SIFT distance ratio test
        ransac_thr (optional): float, threshold for RANSAC geometric filtering using the fundamental matrix

    Returns:
        matches_ij: Mx2 array representing M matches. Each match is represented by two values (i, j)
                    which means that the i-th kp/row in s2p_features_i matches the j-th kp/row in s2p_features_j
        n: integer, the number of matches that were found (M)
    """
    # set matching parameters
    method = "relative"
    epipolar_thr = 20
    model = "fundamental"

    from bundle_adjust.s2p.sift import keypoints_match

    matching_args = [
        s2p_features_i,
        s2p_features_j,
        method,
        dst_thr,
        Fij,
        epipolar_thr,
        model,
        ransac_thr,
    ]

    matching_output = keypoints_match(*matching_args)

    if len(matching_output) > 0:
        m_pts_i = matching_output[:, :2].tolist()
        m_pts_j = matching_output[:, 2:].tolist()
        all_pts_i = s2p_features_i[:, :2].tolist()
        all_pts_j = s2p_features_j[:, :2].tolist()
        matches_ij = np.array([[all_pts_i.index(i), all_pts_j.index(j)] for i, j in zip(m_pts_i, m_pts_j)])
        n = matches_ij.shape[0]
    else:
        matches_ij = None
        n = 0
    return matches_ij, n
