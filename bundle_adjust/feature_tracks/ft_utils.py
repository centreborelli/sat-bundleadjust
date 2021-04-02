import os

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from bundle_adjust import loader

from . import ft_match


def plot_connectivity_graph(C, min_matches, save_pgf=False):

    G, _, _, _, _ = build_connectivity_graph(C, min_matches=min_matches)

    if save_pgf:
        fig_width_pt = 229.5  # CVPR
        inches_per_pt = 1.0 / 72.27  # Convert pt to inches
        golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_width = fig_width_pt * inches_per_pt  # width in inches
        fig_height = fig_width * golden_mean  # height in inches
        fig_size = [fig_width, fig_height]
        params = {
            "backend": "pgf",
            "axes.labelsize": 8,
            "font.size": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
            "text.usetex": True,
            "figure.figsize": fig_size,
        }
        plt.rcParams.update(params)

    fig = plt.gcf()
    fig.set_size_inches(8, 8)

    # draw all nodes in a circle
    G_pos = nx.circular_layout(G)

    # draw nodes
    nx.draw_networkx_nodes(G, G_pos, node_size=600, node_color="#FFFFFF", edgecolors="#000000")

    # paint subgroup of nodes
    # nx.draw_networkx_nodes(G, G_pos, nodelist=[41,42, 43, 44, 45], node_size=600,
    #                        node_color="#FF6161", edgecolors="#000000")

    # draw edges and labels
    nx.draw_networkx_edges(G, G_pos)
    nx.draw_networkx_labels(G, G_pos, font_size=12, font_family="sans-serif")

    # show graph and save it as .pgf
    plt.axis("off")
    if save_pgf:
        plt.savefig("graph.pgf", pad_inches=0, bbox_inches="tight", dpi=200)
    plt.show()


def build_connectivity_graph(C, min_matches, verbose=True):
    def connected_component_subgraphs(G):
        for c in nx.connected_components(G):
            yield G.subgraph(c)

    # (1) Build connectivity matrix A, where position (i,j) contains the number of matches between images i and j
    n_cam = C.shape[0] // 2
    A, n_correspondences_filt, tmp_pairs = np.zeros((n_cam, n_cam)), [], []
    not_nan_C = ~np.isnan(C)
    for im1 in range(n_cam):
        for im2 in range(im1 + 1, n_cam):
            n_matches = np.sum(not_nan_C[2 * im1] & not_nan_C[2 * im2])
            n_correspondences_filt.append(n_matches)
            tmp_pairs.append((im1, im2))
            A[im1, im2] = n_matches
            A[im2, im1] = n_matches

    # (2) Filter graph edges according to the threshold on the number of matches
    pairs_to_draw = []
    matches_per_pair = []
    for i in range(len(tmp_pairs)):
        if n_correspondences_filt[i] >= min_matches:
            pairs_to_draw.append(tmp_pairs[i])
            matches_per_pair.append(n_correspondences_filt[i])

    # (3) Create networkx graph
    G = nx.Graph()
    # add edges
    for edge in pairs_to_draw:
        G.add_edge(edge[0], edge[1])

    # get list of connected components (to see if there is any disconnected subgroup)
    G_cc = list(connected_component_subgraphs(G))
    n_cc = len(G_cc)
    missing_cams = list(set(np.arange(n_cam)) - set(G_cc[0].nodes))

    obs_per_cam = np.sum(1 * ~np.isnan(C), axis=1)[::2]

    if verbose:
        print("Connectivity graph: {} missing cameras: {}".format(len(missing_cams), missing_cams))
        print("                    {} connected components".format(n_cc))
        print("                    {} edges".format(len(pairs_to_draw)))
        print("                    {} min n_matches in an edge".format(min(matches_per_pair)))
        print("                    {} min obs per camera\n".format(min(obs_per_cam)))

    return G, n_cc, pairs_to_draw, matches_per_pair, missing_cams


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


def feature_tracks_from_pairwise_matches(features, pairwise_matches, pairs_to_triangulate):

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
    _, parents_indices, parents_counts = np.unique(parents, return_inverse=True, return_counts=True)
    n_tracks = np.sum(1 * (parents_counts > 1))
    track_parents = np.array(parents)[parents_counts[parents_indices] > 1]
    _, track_idx_from_parent, _ = np.unique(track_parents, return_inverse=True, return_counts=True)

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


def init_feature_tracks_config(config=None):

    """
    Decription of all paramters involved in the creation of feature tracks

        - FT_preprocess        bool     - if True the image histograms are equalized to within 0-255
        - FT_preprocess_aoi    bool     - if True, the preprocessing considers pixels inside the aoi
        - FT_sift_detection    string   - 'opencv' or 's2p'
        - FT_sift_matching     string   - 'bruteforce', 'flann' or 'epipolar_based'
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
    """

    keys = [
        "FT_preprocess",
        "FT_preprocess_aoi",
        "FT_sift_detection",
        "FT_sift_matching",
        "FT_rel_thr",
        "FT_ransac",
        "FT_kp_max",
        "FT_kp_aoi",
        "FT_K",
        "FT_priority",
        "FT_predefined_pairs",
        "FT_filter_pairs",
        "FT_n_proc",
        "FT_reset",
    ]

    default_values = [
        False,
        False,
        "s2p",
        "epipolar_based",
        0.6,
        0.3,
        60000,
        False,
        0,
        ["length", "scale", "cost"],
        [],
        True,
        1,
        False,
    ]

    output_config = {}
    if config is not None:
        for v, k in zip(default_values, keys):
            output_config[k] = config[k] if k in config.keys() else v
    else:
        output_config = dict(zip(keys, default_values))

    if output_config["FT_sift_detection"] == "opencv":
        output_config["FT_preprocess"] = True

    return output_config


def load_tracks_from_predefined_matches(local_data, tracks_config, predefined_matches_dir, output_dir):

    import timeit

    start = timeit.default_timer()

    print("Loading predefined matches from {}".format(predefined_matches_dir))
    src_im_paths = loader.load_list_of_paths(predefined_matches_dir + "/filenames.txt")
    src_im_bn = [os.path.basename(p) for p in src_im_paths]
    target_im_bn = [os.path.basename(p) for p in local_data["fnames"]]

    target_im_indices = []
    for t_bn in target_im_bn:
        if t_bn not in src_im_bn:
            # sanity check: are all target images present in the predefined_matches_dir ?
            print("ERROR ! Input image {} is not listed in predefined_matches_dir".format(t_bn))
        else:
            target_im_indices.append(src_im_bn.index(t_bn))
    target_im_indices = np.array(target_im_indices)

    ####
    #### load predefined features
    ####

    features = []
    features_dir = os.path.join(output_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    for idx in target_im_indices:
        file_id = loader.get_id(src_im_paths[idx])
        path_to_npy = "{}/keypoints/{}.npy".format(predefined_matches_dir, file_id)
        kp_coords = np.load(path_to_npy)  # Nx3 array
        current_im_features = np.hstack([kp_coords, np.ones((kp_coords.shape[0], 129))])  # Nx132 array
        features.append(current_im_features)
        np.save(features_dir + "/" + file_id + ".npy", current_im_features)

    ####
    #### compute pairs to match and to triangulate
    ####

    n_adj = local_data["n_adj"]
    n_new = len(local_data["fnames"]) - n_adj
    if len(tracks_config["FT_predefined_pairs"]) == 0:
        init_pairs = []
        # possible new pairs to match are composed by 1 + 2
        # 1. each of the previously adjusted images with the new ones
        for i in np.arange(n_adj):
            for j in np.arange(n_adj, n_adj + n_new):
                init_pairs.append((i, j))
        # 2. each of the new images with the rest of the new images
        for i in np.arange(n_adj, n_adj + n_new):
            for j in np.arange(i + 1, n_adj + n_new):
                init_pairs.append((i, j))
    else:
        init_pairs = tracks_config["FT_predefined_pairs"]

    args = [init_pairs, local_data["footprints"], local_data["optical_centers"]]
    pairs_to_match, pairs_to_triangulate = ft_match.compute_pairs_to_match(*args)

    ####
    #### load predefined matches
    ####

    predefined_stereo_matches = np.load(predefined_matches_dir + "/matches.npy")
    total_cams = len(src_im_paths)
    true_where_im_in_use = np.zeros(total_cams).astype(bool)
    true_where_im_in_use[target_im_indices] = True
    true_where_prev_match = np.logical_and(
        true_where_im_in_use[predefined_stereo_matches[:, 2]],
        true_where_im_in_use[predefined_stereo_matches[:, 3]],
    )
    predefined_stereo_matches = predefined_stereo_matches[true_where_prev_match, :]

    src_im_indices_to_target_im_indices = np.array([np.nan] * total_cams)
    src_im_indices_to_target_im_indices[target_im_indices] = np.arange(len(target_im_indices))

    # regorganize all_predefined_matches
    # pairwise match format is a 1x4 vector
    # position 1 corresponds to the kp index in image 1, that links to features[im1_index]
    # position 2 corresponds to the kp index in image 2, that links to features[im2_index]
    # position 3 is the index of image 1 within the sequence of images, i.e. im1_index
    # position 4 is the index of image 2 within the sequence of images, i.e. im2_index
    for col_idx in [2, 3]:
        predefined_stereo_matches[:, col_idx] = src_im_indices_to_target_im_indices[
            predefined_stereo_matches[:, col_idx]
        ]

    # the idx of the 4th row (2nd image) must be always larger than the idx of the 3rd row (1st image)
    # all the code follows this convention for encoding paris of image indices
    rows_where_wrong_pair_format = predefined_stereo_matches[:, 2] > predefined_stereo_matches[:, 3]
    tmp = predefined_stereo_matches.copy()
    predefined_stereo_matches[rows_where_wrong_pair_format, 2] = tmp[rows_where_wrong_pair_format, 3]
    predefined_stereo_matches[rows_where_wrong_pair_format, 3] = tmp[rows_where_wrong_pair_format, 2]
    predefined_stereo_matches[rows_where_wrong_pair_format, 0] = tmp[rows_where_wrong_pair_format, 1]
    predefined_stereo_matches[rows_where_wrong_pair_format, 1] = tmp[rows_where_wrong_pair_format, 0]
    del tmp
    print("Using {} predefined stereo matches !".format(predefined_stereo_matches.shape[0]))

    C, C_v2 = feature_tracks_from_pairwise_matches(features, predefined_stereo_matches, pairs_to_triangulate)
    # n_pts_fix = amount of columns with no observations in the new cameras to adjust
    # these columns have to be put at the beginning of C
    where_fix_pts = np.sum(1 * ~np.isnan(C[::2, :])[local_data["n_adj"] :], axis=0) == 0
    n_pts_fix = np.sum(1 * where_fix_pts)
    if n_pts_fix > 0:
        C = np.hstack([C[:, where_fix_pts], C[:, ~where_fix_pts]])
        C_v2 = np.hstack([C_v2[:, where_fix_pts], C_v2[:, ~where_fix_pts]])
    print("Found {} tracks in total".format(C.shape[1]))

    feature_tracks = {
        "C": C,
        "C_v2": C_v2,
        "features": features,
        "pairwise_matches": predefined_stereo_matches,
        "pairs_to_triangulate": pairs_to_triangulate,
        "pairs_to_match": pairs_to_match,
        "n_pts_fix": n_pts_fix,
    }

    loader.save_list_of_paths(output_dir + "/filenames.txt", local_data["fnames"])
    np.save(output_dir + "/matches.npy", predefined_stereo_matches)
    loader.save_list_of_pairs(output_dir + "/pairs_matching.npy", pairs_to_match)
    loader.save_list_of_pairs(output_dir + "/pairs_triangulation.npy", pairs_to_triangulate)

    stop = timeit.default_timer()
    print("\nFeature tracks computed in {}\n".format(loader.get_time_in_hours_mins_secs(stop - start)))

    return feature_tracks, stop - start


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
            l1_boundaries_ok = boundaries_ok(l1_x1, l1_y1) and boundaries_ok(l1_x2, l1_y2)
            l2_boundaries_ok = boundaries_ok(l2_x1, l2_y1) and boundaries_ok(l2_x2, l2_y2)
        else:
            l1_boundaries_ok = True
            l2_boundaries_ok = True

        if l1_boundaries_ok and l2_boundaries_ok:
            l1_args = [l1_x1, l1_y1, l1_x2, l1_y2, color]
            l2_args = [l2_x1, l2_y1, l2_x2, l2_y2, color]
            svg_pt_str = '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="5" />\n'.format(*l1_args)
            svg_pt_str += '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="5" />\n'.format(*l2_args)
        else:
            svg_pt_str = ""
        return svg_pt_str

    # write the svg
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    f_svg = open(output_filename, "w+")
    f_svg.write(svg_header(w, h))
    for p_idx in range(pts2d.shape[0]):
        f_svg.write(svg_pt(pts2d[p_idx, 0], pts2d[p_idx, 1], pt_r=r, color=c, im_w=w, im_h=h))
    f_svg.write("</svg>")