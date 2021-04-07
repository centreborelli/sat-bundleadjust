"""
A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery
author: Roger Mari <roger.mari@ens-paris-saclay.fr>
year: 2021

This script implements functions for feature tracks construction from pairwise matches
and other secondary tasks such as verifying that all cameras are properly connected
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from bundle_adjust import loader

from . import ft_match


def plot_connectivity_graph(C, min_matches):
    """
    Plot a figure of the connectivity graph
    Nodes of the graph represent cameras
    Edges between nodes indicate that a certain amount of matches exists between the two cameras

    Args:
        C: correspondence matrix describing a list of feature tracks connecting a set of cameras
        min_matches: integer, minimum number of matches in each edge of the connectivity graph
    """
    fig = plt.figure()
    G, _, _, _, _ = build_connectivity_graph(C, min_matches=min_matches)

    # compute node positions in a circular layout
    G_pos = nx.circular_layout(G)

    # draw nodes
    nx.draw_networkx_nodes(G, G_pos, node_size=600, node_color="#FFFFFF", edgecolors="#000000")

    # paint subgroup of nodes
    # subgroup = [41,42, 43, 44, 45]  # indices subgroup of nodes
    # nx.draw_networkx_nodes(G, G_pos, nodelist=subgroup, node_size=600, node_color="#FF6161", edgecolors="#000000")

    # draw edges and labels
    nx.draw_networkx_edges(G, G_pos)
    nx.draw_networkx_labels(G, G_pos, font_size=12, font_family="sans-serif")

    # show figure
    plt.axis("off")
    plt.show()


def build_connectivity_graph(C, min_matches, verbose=True):
    """
    Compute the connectivity graph
    Nodes of the graph represent cameras
    Edges between nodes indicate that a certain amount of matches exists between the two cameras

    Args:
        C: correspondence matrix describing a list of feature tracks connecting a set of cameras
        min_matches: integer, minimum number of matches in each edge of the connectivity graph

    Returns:
        G: networkx graph object encoding the connectivity graph
        n_cc: number of connected components in the connectivity graph
              if n_cc > 1 then there are subgroups of cameras disconnected from the rest
        edges: list of pairs, where each pair is a tuple of image indices
               edges contains all image pairs with more than min_matches
        matches_per_edge: list of integers, the number of matches in each edge
        missing_cams: integer, number of cameras that are not part of the biggest connected component of G
                      if n_cc is 1 then missing_cams is expected to be 0
    """

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
    edges = []
    matches_per_edge = []
    for i in range(len(tmp_pairs)):
        if n_correspondences_filt[i] >= min_matches:
            edges.append(tmp_pairs[i])
            matches_per_edge.append(n_correspondences_filt[i])

    # (3) Create networkx graph
    G = nx.Graph()
    # add edges
    for e in edges:
        G.add_edge(e[0], e[1])

    # get list of connected components (to see if there is any disconnected subgroup)
    G_cc = list(connected_component_subgraphs(G))
    n_cc = len(G_cc)
    missing_cams = list(set(np.arange(n_cam)) - set(G_cc[0].nodes))

    obs_per_cam = np.sum(1 * ~np.isnan(C), axis=1)[::2]

    if verbose:
        print("Connectivity graph: {} missing cameras: {}".format(len(missing_cams), missing_cams))
        print("                    {} connected components".format(n_cc))
        print("                    {} edges".format(len(edges)))
        print("                    {} min n_matches in an edge".format(min(matches_per_edge)))
        print("                    {} min obs per camera\n".format(min(obs_per_cam)))

    return G, n_cc, edges, matches_per_edge, missing_cams


def filter_C_using_pairs_to_triangulate(C, pairs_to_triangulate):
    """
    Filter a correspondence matrix using pairs_to_triangulate. The objective of this function
    is to detect tracks in C which do not contain at least 1 pair considered as suitable to triangulate.
    It is advisable to discard these tracks, which are made of matches with short baseline

    Args:
        C: correspondence matrix describing a list of feature tracks connecting a set of cameras
        pairs_to_triangulate: list of pairs, where each pair is a tuple of image indices

    Returns:
        columns_to_preserve: list of indices referring to the columns/tracks of C that can be discarded
    """

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
    Construct a set of feature tracks from a list of pairwise matches of image keypoints
    The set of feature tracks is represented using a "correspondence matrix", i.e. C

    C is a sparse matrix that uses the following format.
    Given M cameras connected by N feature tracks, C has shape 2MxN:

        x11 ... x1N
        y11 ... y1N
        x21 ... x2N
    C = y21 ... y2N
        ... ... ...
        xM1 ... xMN
        yM1 ... yMN

    where (x11, y11) is the observation of feature track 1 in camera 1
          (xm1, ym1) is the observation of feature track 1 in camera M
          (x1n, y1n) is the observation of feature track N in camera 1
          (xmn, ymn) is the observation of feature track N in camera M

    If the n-th feature track is not observed in the m-th camera, then the corresponding positions are NaN
    i.e. C[2*m, n] = np.nan
         C[2*m+1, n] = np.nan

    Args:
        features: a list of arrays with size Nx132, representing the keypoints in each image
        pairwise_matches: array of size Mx4, where each row represents a correspondence between keypoints
                          check ft_match.match_stereo_pairs for details about the format
        pairs_to_triangulate: a list of pairs, where each pair is represented by a tuple of image indices
                              the pairs in this list are considered as suitable for triangulation purposes

    Returns:
        C: correspondence matrix, with size 2MxN
        C_v2: keypoint id correspondence matrix, with size MxN
              variant of C, instead of storing the point coordinates we store the id of the keypoints
    """

    # create a unique id for each keypoint
    feature_ids = []
    id_count = 0
    for im_idx, features_i in enumerate(features):
        ids = np.arange(id_count, id_count + features_i.shape[0])
        feature_ids.append(ids)
        id_count += features_i.shape[0]
    feature_ids = np.array(feature_ids)

    # initialize an empty vector parents where each position corresponds to a keypoint id
    parents = [None] * (id_count)

    # define the union-find functions
    def find(parents, feature_id):
        p = parents[feature_id]
        return feature_id if not p else find(parents, p)

    def union(parents, feature_i_id, feature_j_id):
        p_1, p_2 = find(parents, feature_i_id), find(parents, feature_j_id)
        if p_1 != p_2:
            parents[p_1] = p_2

    # run union-find
    for i in range(pairwise_matches.shape[0]):
        kp_i, kp_j, im_i, im_j = pairwise_matches[i]
        feature_i_id, feature_j_id = feature_ids[im_i, kp_i], feature_ids[im_j, kp_j]
        union(parents, feature_i_id, feature_j_id)

    # handle parents for those keypoint ids that were not matched (so that they are not left as None)
    parents = [find(parents, feature_id) for feature_id, v in enumerate(parents)]

    # each track corresponds to the set of keypoints whose ids have the same value in parents
    # therefore the parents value can be understood as a track id
    # we are only interested in parents values that appear at least 2 times (a track must contain at least 2 points)
    _, parents_indices, parents_counts = np.unique(parents, return_inverse=True, return_counts=True)
    n_tracks = np.sum(1 * (parents_counts > 1))
    valid_parents = np.array(parents)[parents_counts[parents_indices] > 1]
    # create a substitute of parents, named track_indices
    # which considers only the valid parents and takes values between 0 and n_tracks
    # the advantage of track_indices is that it assigns a column of C according to the keypoint id
    _, track_idx_from_parent, _ = np.unique(valid_parents, return_inverse=True, return_counts=True)
    track_indices = np.zeros(len(parents))
    track_indices[:] = np.nan
    track_indices[parents_counts[parents_indices] > 1] = track_idx_from_parent

    # initialize correspondence matrix and keypoint id correspondence matrix
    n_cams = len(features)
    C = np.zeros((2 * n_cams, n_tracks))
    C[:] = np.nan

    C_v2 = np.zeros((n_cams, n_tracks))
    C_v2[:] = np.nan

    # fill both matrices
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

    # ensure each track contains at least one correspondence suitable to triangulate
    print("C.shape before baseline check {}".format(C.shape))
    tracks_to_preserve = filter_C_using_pairs_to_triangulate(C, pairs_to_triangulate)
    C = C[:, tracks_to_preserve]
    C_v2 = C_v2[:, tracks_to_preserve]
    print("C.shape after baseline check {}".format(C.shape))

    return C, C_v2


def check_pairs(camera_indices, pairs_to_match, pairs_to_triangulate):
    """
    Verifies if all cameras are part of pairs_to_match and pairs_to_traingulate

    Args:
        pairs_to_match: subset of pairs from init_pairs considered as well-posed for feature matching
        pairs_to_triangulate: subset of pairs from pairs_to_match considered as well-posed for triangulation

    Returns:
        fatal_error: boolean, True if more than half of the cameras are disconnected, False otherwise
        err_msg: string, error/warning message
        disconnected_cameras: list of camera indices pointing to disconnected cameras
    """
    fatal_error = False
    disconnected_cameras = []
    err_msg = ""
    camera_indices = set(camera_indices)

    camera_indices_in_pairs_to_match = set(np.unique(np.array(pairs_to_match).flatten()))
    if not len(camera_indices - camera_indices_in_pairs_to_match) == 0:
        disconnected_cameras = list(camera_indices - camera_indices_in_pairs_to_match)
        fatal_error = len(disconnected_cameras) > len(camera_indices) // 2
        to_print = [len(disconnected_cameras), len(camera_indices)]
        print("WARNING: Found {} cameras out of {} missing in pairs_to_match".format(*to_print))
        print("         The disconnected camera indices are: {}".format(disconnected_cameras))
        if fatal_error:
            err_msg = "More than 50% of the cameras are disconnected in terms of feature tracking"

    camera_indices_in_pairs_to_triangulate = set(np.unique(np.array(pairs_to_triangulate).flatten()))
    if not len(camera_indices - camera_indices_in_pairs_to_triangulate) == 0:
        disconnected_cameras = list(camera_indices - camera_indices_in_pairs_to_triangulate)
        fatal_error = len(disconnected_cameras) > len(camera_indices) // 2
        to_print = [len(disconnected_cameras), len(camera_indices)]
        print("WARNING: Found {} cameras out of {} missing in pairs_to_triangulate".format(*to_print))
        print("         The disconnected camera indices are: {}".format(disconnected_cameras))
        if fatal_error:
            err_msg = "More than 50% of the cameras are disconnected in terms of feature tracking"
    return fatal_error, err_msg, disconnected_cameras


def check_correspondence_matrix(C, min_obs_cam=10):
    """
    Verifies that there are enough feature tracks connecting all cameras according to C

    Args:
        C: correspondence matrix describing a list of feature tracks connecting a set of cameras
        min_obs_cam (optional): integer, minimum amount of feature track observations per camera

    Returns:
        fatal_error: boolean, True if more than half of the cameras are disconnected, False otherwise
        err_msg: string, error/warning message
        disconnected_cameras: list of camera indices pointing to disconnected cameras
    """
    fatal_error = False
    disconnected_cameras = []
    err_msg = ""
    if C is None:
        fatal_error = True
        err_msg = "Found less tracks than cameras"
        return fatal_error, err_msg, disconnected_cameras
    n_cam = C.shape[0] // 2
    if n_cam > C.shape[1]:
        fatal_error = True
        err_msg = "Found less tracks than cameras"
        return fatal_error, err_msg, disconnected_cameras
    obs_per_cam = np.sum(~np.isnan(C[::2]), axis=1)
    if np.sum(obs_per_cam < min_obs_cam) > 0:
        disconnected_cameras = np.arange(n_cam)[obs_per_cam < min_obs_cam].tolist()
        fatal_error = len(disconnected_cameras) > n_cam // 2
        if len(disconnected_cameras) > 0:
            to_print = [len(disconnected_cameras), n_cam, min_obs_cam]
            print("WARNING: Found {} cameras out of {} with less than {} tie point observations".format(*to_print))
            print("         The disconnected camera indices are: {}".format(disconnected_cameras))
            if fatal_error:
                err_msg = "More than 50% of the cameras are disconnected in terms of feature tracking"
    return fatal_error, err_msg, disconnected_cameras


def init_feature_tracks_config(config=None):
    """
    Initializes the feature tracking configuration with the default values
    The configuration is encoded using a dictionary

          KEY                  TYPE       DESCRIPTION
        - FT_preprocess        bool     - if True the image histograms are equalized to within 0-255
        - FT_preprocess_aoi    bool     - if True, the preprocessing considers pixels inside the aoi
        - FT_sift_detection    string   - 'opencv' or 's2p'
        - FT_sift_matching     string   - 'bruteforce', 'flann' or 'epipolar_based'
        - FT_rel_thr           float    - distance ratio threshold for matching
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

    Args:
        config (optional): dictionary specifying customized values for some of the keys above

    Returns:
        output_config: the output feature tracking configuration dictionary
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
            output_config[k] = config.get(k, v)
    else:
        output_config = dict(zip(keys, default_values))

    # opencv sift requires all images in uint within the range 0-255
    if output_config["FT_sift_detection"] == "opencv":
        output_config["FT_preprocess"] = True

    return output_config


def load_tracks_from_predefined_matches(input_dir, output_dir, local_data, tracks_config):
    """
    Equivalent to FeatureTracksPipeline, but using a set of predefined pairwise matches,
    therefore the feature detection and pairwise matching blocks are not required

    Args:
        same as FeatureTracksPipeline

    Returns:
        same as FeatureTracksPipeline
    """
    import timeit

    start = timeit.default_timer()

    predefined_matches_dir = input_dir
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

    matches = np.load(predefined_matches_dir + "/matches.npy")
    total_cams = len(src_im_paths)
    true_where_im_in_use = np.zeros(total_cams).astype(bool)
    true_where_im_in_use[target_im_indices] = True
    true_where_prev_match = true_where_im_in_use[matches[:, 2]] & true_where_im_in_use[matches[:, 3]]
    matches = matches[true_where_prev_match, :]

    src_im_indices_to_target_im_indices = np.array([np.nan] * total_cams)
    src_im_indices_to_target_im_indices[target_im_indices] = np.arange(len(target_im_indices))

    # regorganize all_predefined_matches
    # pairwise match format is a 1x4 vector with the format from ft_match.match_stereo_pairs
    for col_idx in [2, 3]:
        matches[:, col_idx] = src_im_indices_to_target_im_indices[matches[:, col_idx]]

    # the idx of the 4th row (2nd image) must be always larger than the idx of the 3rd row (1st image)
    # all the code follows this convention for encoding paris of image indices
    rows_where_wrong_pair_format = matches[:, 2] > matches[:, 3]
    tmp = matches.copy()
    matches[rows_where_wrong_pair_format, 2] = tmp[rows_where_wrong_pair_format, 3]
    matches[rows_where_wrong_pair_format, 3] = tmp[rows_where_wrong_pair_format, 2]
    matches[rows_where_wrong_pair_format, 0] = tmp[rows_where_wrong_pair_format, 1]
    matches[rows_where_wrong_pair_format, 1] = tmp[rows_where_wrong_pair_format, 0]
    del tmp
    print("Using {} predefined stereo matches !".format(matches.shape[0]))

    C, C_v2 = feature_tracks_from_pairwise_matches(features, matches, pairs_to_triangulate)
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
        "pairwise_matches": matches,
        "pairs_to_triangulate": pairs_to_triangulate,
        "pairs_to_match": pairs_to_match,
        "n_pts_fix": n_pts_fix,
    }

    loader.save_list_of_paths(output_dir + "/filenames.txt", local_data["fnames"])
    np.save(output_dir + "/matches.npy", matches)
    loader.save_list_of_pairs(output_dir + "/pairs_matching.npy", pairs_to_match)
    loader.save_list_of_pairs(output_dir + "/pairs_triangulation.npy", pairs_to_triangulate)

    stop = timeit.default_timer()
    print("\nFeature tracks computed in {}\n".format(loader.get_time_in_hours_mins_secs(stop - start)))

    return feature_tracks, stop - start


def save_pts2d_as_svg(output_filename, pts2d, c="yellow", r=5, w=None, h=None):
    """
    Write a svg file displaying a set of image points
    This file can be displayed upon a geotiff image

    Args:
        output_filename: path to output svg
        pts2d: array of size Nx2 with the (col, row) coordinates of a set of image points
        c (optional): matplotlib color of the points
        r (optional): integer, radius of the points
        w (optional): integer, width of the tif image
        h (optional): integer, height of the tif image
    """

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
