import matplotlib.pyplot as plt
import numpy as np

from .feature_tracks import ft_sat


def detect_features_image_sequence(
    input_seq, masks=None, max_kp=None, image_indices=None, thread_idx=None
):

    from s2p.sift import keypoints_from_nparray

    # default parameters
    thresh_dog = 0.0133
    nb_octaves = 8
    nb_scales = 3
    offset = None

    multiproc = False if thread_idx is None else True
    n_img = len(input_seq)

    features = []
    for i in range(n_img):

        features_i = keypoints_from_nparray(
            input_seq[i], thresh_dog, nb_octaves, nb_scales, offset
        )

        # features_i is a list of 132 floats, the first four elements are the keypoint (x, y, scale, orientation),
        # the 128 following values are the coefficients of the SIFT descriptor, i.e. integers between 0 and 255

        if masks is not None:
            pts2d_colrow = features_i[:, :2].astype(np.int)
            true_if_obs_inside_aoi = (
                1 * masks[i][pts2d_colrow[:, 1], pts2d_colrow[:, 0]] > 0
            )
            features_i = features_i[true_if_obs_inside_aoi, :]

        features_i = np.array(
            sorted(features_i.tolist(), key=lambda kp: kp[2], reverse=True)
        )

        if max_kp is not None:
            features_i_final = np.zeros((max_kp, 132))
            features_i_final[:] = np.nan
            features_i_final[: min(features_i.shape[0], max_kp)] = features_i[:max_kp]
            features.append(features_i_final)
        else:
            features.append(features_i)
        n_kp = int(np.sum(1 * ~np.isnan(features[-1][:, 0])))
        tmp = ""
        if multiproc:
            tmp = " (thread {} -> {}/{})".format(thread_idx, i + 1, n_img)
        print(
            "{} keypoints in image {}{}".format(
                n_kp, image_indices[i] if multiproc else i, tmp
            ),
            flush=True,
        )

    return features


def detect_features_image_sequence_multiprocessing(
    input_seq, masks=None, max_kp=None, n_proc=5
):

    n_img = len(input_seq)

    n = int(np.ceil(n_img / n_proc))
    args = [
        (
            input_seq[i : i + n],
            None if masks is None else masks[i : i + n],
            max_kp,
            np.arange(i, i + n).astype(int),
            k,
        )
        for k, i in enumerate(np.arange(0, n_img, n))
    ]

    from multiprocessing import Pool

    with Pool(len(args)) as p:
        detection_output = p.starmap(detect_features_image_sequence, args)
    flatten_list = lambda t: [item for sublist in t for item in sublist]
    return flatten_list(detection_output)


def s2p_match_SIFT(s2p_features_i, s2p_features_j, Fij, dst_thr=0.6, ransac_thr=0.3):
    """
    Match SIFT keypoints from an stereo pair
    """
    # set matching parameters
    method = "relative"
    epipolar_thr = 20
    model = "fundamental"

    from s2p.sift import keypoints_match

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
        m_pts_i, m_pts_j = (
            matching_output[:, :2].tolist(),
            matching_output[:, 2:].tolist(),
        )
        all_pts_i = s2p_features_i[:, :2].tolist()
        all_pts_j = s2p_features_j[:, :2].tolist()
        matches_ij = np.array(
            [
                [all_pts_i.index(pt_i), all_pts_j.index(pt_j)]
                for pt_i, pt_j in zip(m_pts_i, m_pts_j)
            ]
        )
        n = matches_ij.shape[0]
    else:
        matches_ij = None
        n = 0
    return matches_ij, [n]
