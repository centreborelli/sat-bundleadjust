import os
import pickle

from bundle_adjust import data_loader as loader
from feature_tracks import ft_sat
import matplotlib.pyplot as plt
import numpy as np

from bundle_adjust import ba_utils, data_loader
from .feature_tracks import ft_sat


def get_fname_id(fname):
    return os.path.splitext(os.path.basename(fname))[0]


def plot_features_stereo_pair(i, j, features, input_seq):

    # i, j : indices of the images
    pts1, pts2 = features[i][:, :2], features[j][:, :2]
    to_print = [pts1.shape[0], i, pts2.shape[0], j]
    print(
        "Found {} keypoints in image {} and {} keypoints in image {}".format(*to_print)
    )

    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(loader.custom_equalization(input_seq[i]), cmap="gray")
    ax2.imshow(loader.custom_equalization(input_seq[j]), cmap="gray")
    if pts1.shape[0] > 0:
        ax1.scatter(x=pts1[:, 0], y=pts1[:, 1], c="r", s=40)
    if pts2.shape[0] > 0:
        ax2.scatter(x=pts2[:, 0], y=pts2[:, 1], c="r", s=40)
    plt.show()


def plot_track_observations_stereo_pair(i, j, C, input_seq):

    # i, j : indices of the images
    visible_idx = np.logical_and(~np.isnan(C[i * 2, :]), ~np.isnan(C[j * 2, :]))
    pts1, pts2 = (
        C[(i * 2) : (i * 2 + 2), visible_idx],
        C[(j * 2) : (j * 2 + 2), visible_idx],
    )
    pts1, pts2 = pts1.T, pts2.T
    n_pts = pts1.shape[0]
    print("{} track observations to display for pair ({},{})".format(n_pts, i, j))
    print("List of track indices: {}".format(np.arange(C.shape[1])[visible_idx]))

    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(loader.custom_equalization(input_seq[i]), cmap="gray")
    ax2.imshow(loader.custom_equalization(input_seq[j]), cmap="gray")
    if n_pts > 0:
        ax1.scatter(x=pts1[:, 0], y=pts1[:, 1], c="r", s=40)
        ax2.scatter(x=pts2[:, 0], y=pts2[:, 1], c="r", s=40)
    plt.show()


def plot_pairwise_matches_stereo_pair(i, j, features, pairwise_matches, input_seq):

    # i, j : indices of the images
    pairwise_matches_kp_indices = pairwise_matches[:, :2]
    pairwise_matches_im_indices = pairwise_matches[:, 2:]

    true_where_matches = np.all(pairwise_matches_im_indices == np.array([i, j]), axis=1)
    matched_kps_i = features[i][pairwise_matches_kp_indices[true_where_matches, 0]]
    matched_kps_j = features[j][pairwise_matches_kp_indices[true_where_matches, 1]]

    print(
        "{} pairwise matches to display for pair ({},{})".format(
            matched_kps_i.shape[0], i, j
        )
    )

    h, w = input_seq[i].shape
    max_v = max(input_seq[i].max(), input_seq[j].max())
    margin = 100
    fig = plt.figure(figsize=(42, 6))
    complete_im = np.hstack([input_seq[i], np.ones((h, margin)) * max_v, input_seq[j]])
    ax = plt.gca()
    ax.imshow((complete_im), cmap="gray")
    if matched_kps_i.shape[0] > 0:
        ax.scatter(x=matched_kps_i[:, 0], y=matched_kps_i[:, 1], c="r", s=30)
        ax.scatter(
            x=w + margin + matched_kps_j[:, 0], y=matched_kps_j[:, 1], c="r", s=30
        )
        for k in range(matched_kps_i.shape[0]):
            ax.plot(
                [matched_kps_i[k, 0], w + margin + matched_kps_j[k, 0]],
                [matched_kps_i[k, 1], matched_kps_j[k, 1]],
                "y--",
                lw=3,
            )
    plt.show()

    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(loader.custom_equalization(input_seq[i]), cmap="gray")
    ax2.imshow(loader.custom_equalization(input_seq[j]), cmap="gray")
    if matched_kps_i.shape[0] > 0:
        ax1.scatter(x=matched_kps_i[:, 0], y=matched_kps_i[:, 1], c="r", s=10)
        ax2.scatter(x=matched_kps_j[:, 0], y=matched_kps_j[:, 1], c="r", s=10)
    plt.show()


def filter_C_using_pairs_to_triangulate(C, pairs_to_triangulate):

    # remove matches found in pairs with short baseline that were not extended to more images
    # since these columns of C will not be triangulated

    columns_to_preserve = []
    mask = ~np.isnan(C[::2])
    pairs_to_triangulate_set = set(pairs_to_triangulate)
    for i in range(C.shape[1]):
        im_ind = np.where(mask[:, i])[0]
        all_pairs_current_track = set([(im_i, im_j) for im_i in im_ind for im_j in im_ind if im_i < im_j])
        triangulation_pairs_current_track = pairs_to_triangulate_set & all_pairs_current_track
        found_at_least_one_triangulation_pair = len(triangulation_pairs_current_track) > 0
        columns_to_preserve.append(found_at_least_one_triangulation_pair)
    return columns_to_preserve


def feature_tracks_from_pairwise_matches(
    features, pairwise_matches, pairs_to_triangulate
):

    """
    warning:
    This function has a drawback: everytime we build the feature tracks we load ALL features of the scene
    and ALL pairwise matches. When a reasonable amount of images is used this will be fine but for 1000
    images the computation time may increase in a relevant way.
    The solution would be to save matches separately per image and directly load those of interest
    (instead of filtering them from all_pairwise_matches).
    """

    n_cams = len(features)

    # prepreate data to build correspondence matrix
    feature_ids = []
    id_count = 0
    for im_idx, features_i in enumerate(features):
        ids = np.arange(id_count, id_count + features_i.shape[0])
        feature_ids.append(ids)
        id_count += features_i.shape[0]
    feature_ids = np.array(feature_ids)

    def find(parents, feature_id):
        p = parents[feature_id]
        return feature_id if not p else find(parents, p)

    def union(parents, feature_i_id, feature_j_id):
        p_1, p_2 = find(parents, feature_i_id), find(parents, feature_j_id)
        if p_1 != p_2:
            parents[p_1] = p_2

    # get pairwise matches of interest, i.e. with matched features located in at least one image currently in use
    pairwise_matches_of_interest = pairwise_matches.tolist()

    parents = [None] * (id_count)
    for kp_i, kp_j, im_i, im_j in pairwise_matches_of_interest:
        feature_i_id, feature_j_id = feature_ids[im_i, kp_i], feature_ids[im_j, kp_j]
        union(parents, feature_i_id, feature_j_id)

    # handle parents without None
    parents = [find(parents, feature_id) for feature_id, v in enumerate(parents)]

    # parents = track_id
    _, parents_indices, parents_counts = np.unique(
        parents, return_inverse=True, return_counts=True
    )
    n_tracks = np.sum(1 * (parents_counts > 1))
    track_parents = np.array(parents)[parents_counts[parents_indices] > 1]
    _, track_idx_from_parent, _ = np.unique(
        track_parents, return_inverse=True, return_counts=True
    )

    # t_idx, parent_id
    track_indices = np.zeros(len(parents))
    track_indices[:] = np.nan
    track_indices[parents_counts[parents_indices] > 1] = track_idx_from_parent

    """
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
    """

    # build correspondence matrix
    C = np.zeros((2 * n_cams, n_tracks))
    C[:] = np.nan

    C_v2 = np.zeros((n_cams, n_tracks))
    C_v2[:] = np.nan

    kp_i, kp_j = pairwise_matches[:, 0], pairwise_matches[:, 1]
    im_i, im_j = pairwise_matches[:, 2], pairwise_matches[:, 3]
    feature_i_id, feature_j_id = feature_ids[im_i, kp_i], feature_ids[im_j, kp_j]
    t_idx = track_indices[feature_i_id].astype(int)
    features_tmp = np.moveaxis(np.dstack(features), 2, 0)
    C[2 * im_i, t_idx] = features_tmp[im_i, kp_i, 0]
    C[2 * im_i + 1, t_idx] = features_tmp[im_i, kp_i, 1]
    C[2 * im_j, t_idx] = features_tmp[im_j, kp_j, 0]
    C[2 * im_j + 1, t_idx] = features_tmp[im_j, kp_j, 1]
    C_v2[im_i, t_idx] = kp_i
    C_v2[im_j, t_idx] = kp_j

    print("C.shape before baseline check {}".format(C.shape))
    tracks_to_preserve = filter_C_using_pairs_to_triangulate(C, pairs_to_triangulate)
    C = C[:, tracks_to_preserve]
    C_v2 = C_v2[:, tracks_to_preserve]
    print("C.shape after baseline check {}".format(C.shape))

    return C, C_v2


def save_pts2d_as_svg(output_filename, pts2d, c, r=5, w=None, h=None):
    def boundaries_ok(col, row):
        return col > 0 and col < w - 1 and row > 0 and row < h - 1

    def svg_header(w, h):
        svg_header = (
            '<?xml version="1.0" standalone="no"?>\n'
            + '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n'
            + ' "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n'
            + '<svg width="{}px" height="{}px" version="1.1"\n'.format(w, h)
            + ' xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">\n'
        )
        return svg_header

    def svg_pt(col, row, color, pt_r, im_w=None, im_h=None):

        col, row = int(col), int(row)

        l1_x1, l1_y1, l1_x2, l1_y2 = col - pt_r, row - pt_r, col + pt_r, row + pt_r
        l2_x1, l2_y1, l2_x2, l2_y2 = col + pt_r, row - pt_r, col - pt_r, row + pt_r

        if (im_w is not None) and (im_h is not None):
            l1_boundaries_ok = boundaries_ok(l1_x1, l1_y1) and boundaries_ok(
                l1_x2, l1_y2
            )
            l2_boundaries_ok = boundaries_ok(l2_x1, l2_y1) and boundaries_ok(
                l2_x2, l2_y2
            )
        else:
            l1_boundaries_ok = True
            l2_boundaries_ok = True

        if l1_boundaries_ok and l2_boundaries_ok:
            l1_args = [l1_x1, l1_y1, l1_x2, l1_y2, color]
            l2_args = [l2_x1, l2_y1, l2_x2, l2_y2, color]
            svg_pt_str = '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="5" />\n'.format(
                *l1_args
            ) + '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="5" />\n'.format(
                *l2_args
            )
        else:
            svg_pt_str = ""
        return svg_pt_str

    # write the svg
    f_svg = open(output_filename, "w+")
    f_svg.write(svg_header(w, h))
    for p_idx in range(pts2d.shape[0]):
        f_svg.write(
            svg_pt(pts2d[p_idx, 0], pts2d[p_idx, 1], pt_r=r, color=c, im_w=w, im_h=h)
        )
    f_svg.write("</svg>")


def save_sequence_features_svg(output_dir, seq_fnames, seq_features):
    os.makedirs(output_dir, exist_ok=True)
    for fname, features in zip(seq_fnames, seq_features):
        f_id = get_fname_id(fname)
        not_nan = np.sum(~np.isnan(features[:, 0]))
        save_pts2d_as_svg(
            os.path.join(output_dir, f_id + ".svg"), features[:not_nan, :2], c="yellow"
        )


def save_sequence_features_txt(
    output_dir, seq_fnames, seq_features, seq_features_utm=None
):

    do_utm = seq_features_utm is not None
    n_img = len(seq_fnames)

    kps_dir = os.path.join(output_dir, "kps")
    des_dir = os.path.join(output_dir, "des")
    os.makedirs(kps_dir, exist_ok=True)
    os.makedirs(des_dir, exist_ok=True)

    if do_utm:
        utm_dir = os.path.join(output_dir, "utm")
        os.makedirs(utm_dir, exist_ok=True)

    for i in range(n_img):
        f_id = get_fname_id(seq_fnames[i])
        np.savetxt(
            os.path.join(kps_dir, f_id + ".txt"), seq_features[i][:, :4], fmt="%.6f"
        )
        np.savetxt(
            os.path.join(des_dir, f_id + ".txt"), seq_features[i][:, 4:], fmt="%d"
        )
        if do_utm:
            np.savetxt(
                os.path.join(utm_dir, f_id + ".txt"), seq_features_utm[i], fmt="%.6f"
            )


def init_feature_tracks_config(config=None):

    """
    Decription of all paramters involved in the creation of feature tracks

        - FT_preprocess        bool     - if True the image histograms are equalized to within 0-255
        - FT_preprocess_aoi    bool     - if True, the preprocessing considers pixels inside the aoi
        - FT_sift_detection    string   - 'opencv' or 's2p'
        - FT_sift_matching     string   - 'bruteforce', 'flann', 'epipolar_based' or 'local_window'
        - FT_rel_thr           float    - distance ratio threshold for matching
        - FT_abs_thr           float    - absolute distance threshold for matching
        - FT_ransac            float    - ransac threshold for matching
        - FT_kp_max            int      - maximum number of keypoints allowed per image
                                          keypoints with larger scale are given higher priority
        - FT_kp_aoi            bool     - when True only keypoints inside the aoi are considered
        - FT_K                 int      - number of spanning trees to cover if feature track selection
                                          if K = 0 then no feature track selection takes place
        - FT_priority          list     - list of strings containing the order to rank tracks
                                          most important criterion goes first
        - FT_predefined_pairs  list     - list of predefined pairs that it is allowed to match
        - FT_filter_pairs      bool     - filter pairs using the stereo pair selection algorithm
        - FT_n_proc            int      - number of processes to launch in parallel when possible
        - FT_reset             bool     - if False, the pipeline tries to reuse previously detected features,
                                          if True keypoints will be extracted from all images regardless
                                          of any previous detections that may be available
    '''

    keys = ['FT_preprocess', 'FT_preprocess_aoi', 'FT_sift_detection', 'FT_sift_matching',
            'FT_rel_thr', 'FT_abs_thr', 'FT_ransac', 'FT_kp_max', 'FT_kp_aoi',
            'FT_K', 'FT_priority', 'FT_predefined_pairs', 'FT_filter_pairs', 'FT_n_proc', 'FT_reset']

    default_values = [False, False, 's2p', 'epipolar_based', 0.6, 250, 0.3,
                      60000, False, 0, ['length', 'scale', 'cost'], [], True, 1, False]

    output_config = {}
    if config is not None:
        for v, k in zip(default_values, keys):
            output_config[k] = config[k] if k in config.keys() else v
    else:
        output_config = dict(zip(keys, default_values))

    if output_config["FT_sift_detection"] == "opencv":
        output_config["FT_preprocess"] = True

    return output_config


def save_matching_to_light_format(ba_data_dir):

    import glob
    features_fnames = glob.glob(ba_data_dir + '/features/*.npy')
    os.makedirs(ba_data_dir + '/features_light', exist_ok=True)
    for fn in features_fnames:
        features_light = np.load(fn)[:, :3] # we take only the first 3 columns corresponding to (col, row, scale)
        np.save(fn.replace('/features/', '/features_light/'), features_light)
    print('features conversion to light format done')


def load_tracks_from_predefined_matches_light_format(local_data, tracks_config, predefined_matches_dir, output_dir):

    import timeit
    start = timeit.default_timer()

    print('Loading predefined matches from {}'.format(predefined_matches_dir), flush=True)
    source_im_paths = loader.load_list_of_paths(predefined_matches_dir + '/filenames.txt')
    source_im_bn = [os.path.basename(p) for p in source_im_paths]
    target_im_bn = [os.path.basename(p) for p in local_data['fnames']]

    target_im_indices = []
    for t_bn in target_im_bn:
        if t_bn not in source_im_bn:
            # sanity check: are all target images present in the predefined_matches_dir ?
            print('ERROR ! Input image {} is not listed in predefined_matches_dir'.format(t_bn))
        else:
            target_im_indices.append(source_im_bn.index(t_bn))
    target_im_indices = np.array(target_im_indices)

    ####
    #### load predefined features
    ####

    features = []
    for idx in target_im_indices:
        path_to_npy = '{}/features_light/{}.npy'.format(predefined_matches_dir, loader.get_id(source_im_paths[idx]))
        kp_coords = np.load(path_to_npy)  #Nx3 array
        current_im_features = np.hstack([kp_coords, np.ones((kp_coords.shape[0], 129))]) #Nx132 array
        features.append(current_im_features)

    ####
    #### compute pairs to match and to triangulate
    ####

    n_adj = local_data['n_adj']
    n_new = local_data['n_new']
    if len(tracks_config['FT_predefined_pairs']) == 0:
        init_pairs = []
        # possible new pairs to match are composed by 1 + 2
        # 1. each of the previously adjusted images with the new ones
        for i in np.arange(n_adj):
            for j in np.arange(n_adj, n_adj + n_new):
                init_pairs.append((i, j))
        # 2. each of the new images with the rest of the new images
        for i in np.arange(n_adj, n_adj + n_new):
            for j in np.arange(i+1, n_adj + n_new):
                init_pairs.append((i, j))
    else:
        init_pairs = tracks_config['FT_predefined_pairs']

    args = [init_pairs, local_data['footprints'], local_data['optical_centers']]
    pairs_to_match, pairs_to_triangulate = ft_sat.compute_pairs_to_match(*args)

    ####
    #### load predefined matches
    ####

    predefined_stereo_matches = np.load(predefined_matches_dir + '/matches.npy')
    total_cams = len(source_im_paths)
    true_where_im_in_use = np.zeros(total_cams).astype(bool)
    true_where_im_in_use[target_im_indices] = True
    true_where_prev_match = np.logical_and(true_where_im_in_use[predefined_stereo_matches[:, 2]],
                                           true_where_im_in_use[predefined_stereo_matches[:, 3]])
    predefined_stereo_matches = predefined_stereo_matches[true_where_prev_match, :]

    src_im_indices_to_target_im_indices = np.array([np.nan]*total_cams)
    src_im_indices_to_target_im_indices[target_im_indices] = np.arange(len(target_im_indices))

    # regorganize all_predefined_matches
    # pairwise match format is a 1x4 vector
    # position 1 corresponds to the kp index in image 1, that links to features[im1_index]
    # position 2 corresponds to the kp index in image 2, that links to features[im2_index]
    # position 3 is the index of image 1 within the sequence of images, i.e. im1_index
    # position 4 is the index of image 2 within the sequence of images, i.e. im2_index
    for col_idx in [2, 3]:
        predefined_stereo_matches[:, col_idx] = src_im_indices_to_target_im_indices[predefined_stereo_matches[:, col_idx]]

    # the idx of the 4th row (2nd image) must be always larger than the idx of the 3rd row (1st image)
    # all the code follows this convention for encoding paris of image indices
    rows_where_wrong_pair_format = predefined_stereo_matches[:, 2] > predefined_stereo_matches[:, 3]
    tmp = predefined_stereo_matches.copy()
    predefined_stereo_matches[rows_where_wrong_pair_format, 2] = tmp[rows_where_wrong_pair_format, 3]
    predefined_stereo_matches[rows_where_wrong_pair_format, 3] = tmp[rows_where_wrong_pair_format, 2]
    predefined_stereo_matches[rows_where_wrong_pair_format, 0] = tmp[rows_where_wrong_pair_format, 1]
    predefined_stereo_matches[rows_where_wrong_pair_format, 1] = tmp[rows_where_wrong_pair_format, 0]
    del tmp
    print('Using {} predefined stereo matches !'.format(predefined_stereo_matches.shape[0]), flush=True)


    C, C_v2 = feature_tracks_from_pairwise_matches(features,
                                                   predefined_stereo_matches,
                                                   pairs_to_triangulate)
    # n_pts_fix = amount of columns with no observations in the new cameras to adjust
    # these columns have to be put at the beginning of C
    where_fix_pts = np.sum(1 * ~np.isnan(C[::2, :])[-local_data['n_new']:], axis=0) == 0
    n_pts_fix = np.sum(1 * where_fix_pts)
    if n_pts_fix > 0:
        C = np.hstack([C[:, where_fix_pts], C[:, ~where_fix_pts]])
        C_v2 = np.hstack([C_v2[:, where_fix_pts], C_v2[:, ~where_fix_pts]])
    print('Found {} tracks in total'.format(C.shape[1]), flush=True)

    feature_tracks = {'C': C, 'C_v2': C_v2, 'features': features,
                      'pairwise_matches': predefined_stereo_matches,
                      'pairs_to_triangulate': pairs_to_triangulate,
                      'pairs_to_match': pairs_to_match, 'n_pts_fix': n_pts_fix}

    np.save(output_dir + '/matches.npy', predefined_stereo_matches)
    loader.save_list_of_pairs(output_dir + '/pairs_matching.npy', pairs_to_match)
    loader.save_list_of_pairs(output_dir + '/pairs_triangulation.npy', pairs_to_triangulate)

    stop = timeit.default_timer()
    print('\nFeature tracks computed in {}\n'.format(loader.get_time_in_hours_mins_secs(stop - start)), flush=True)

    return feature_tracks, stop - start
