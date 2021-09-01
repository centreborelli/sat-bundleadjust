"""
A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery
author: Roger Mari <roger.mari@ens-paris-saclay.fr>
year: 2021

This script implements an algorithm dedicated to the selection of an optimal subset of feature tracks
to improve the efficiency of bundle adjustment. The selection algorithm was introduced in:
"Tracks selection for robust, efficient and scalable large-scale structure from motion" (Pattern Recognition, 2017)
"""

import numpy as np
import timeit

from bundle_adjust import ba_core
from bundle_adjust.loader import flush_print


def build_connectivity_matrix(C, min_matches=10):
    """
    Compute the connectivity matrix A,
    A is a matrix with size (n_cam)x(n_cam), where n_cam is the number of cameras
    the value at position (i,j) is equal to the amount of pairwise matches found between image i and image j
    """
    n_cam = C.shape[0] // 2
    A = np.zeros((n_cam, n_cam))
    C_not_nan = ~np.isnan(C[::2])
    for im1 in range(n_cam):
        for im2 in range(im1 + 1, n_cam):
            A[im1, im2] = np.sum(C_not_nan[im1, :] & C_not_nan[im2, :])
            A[im2, im1] = A[im1, im2]
    A[A < min_matches] = 0

    return A


def compute_C_scale(C_v2, features):
    """
    Compute C_scale, a variant of the correspondence matrix C
    instead of having shape (2*n_cam)x(n_tracks) it has shape (n_cam)x(n_tracks)
    where each position contains the scale of the associated keypoint, else nan
    """
    C_scale = C_v2.copy()
    n_cam = C_v2.shape[0]
    for cam_idx in range(n_cam):
        current_im_kp = np.load(features[cam_idx], mmap_mode='r')
        where_obs_current_cam = ~np.isnan(C_v2[cam_idx, :])
        kp_indices = C_v2[cam_idx, where_obs_current_cam]
        kp_scales = current_im_kp[kp_indices.astype(np.int32), 2]
        C_scale[cam_idx, where_obs_current_cam] = kp_scales
        del current_im_kp

    return C_scale


def compute_C_reproj(C, pts3d, cameras, cam_model, pairs_to_triangulate, camera_centers):
    """
    Compute C_reproj, a variant of the correspondence matrix C
    instead of having shape (2*n_cam)x(n_tracks) it has shape (n_cam)x(n_tracks)
    where each position contains the reprojection error of the associated keypoint, else nan
    """

    # set ba parameters
    from bundle_adjust.ba_params import BundleAdjustmentParameters

    args = [C, pts3d, cameras, cam_model, pairs_to_triangulate, camera_centers]
    d = {"reduce": False, "verbose": False}
    p = BundleAdjustmentParameters(*args, d)

    # compute reprojection error at the current parameters
    reprojection_err_per_obs = ba_core.compute_reprojection_error(ba_core.fun(p.params_opt.copy(), p))

    # create C_reproj
    n_cam, n_pts = C.shape[0] // 2, C.shape[1]
    C_reproj = np.zeros((n_cam, n_pts))
    C_reproj[:] = np.nan
    for i, err in enumerate(reprojection_err_per_obs):
        C_reproj[p.cam_ind[i], p.pts_ind[i]] = err

    return C_reproj


def compute_camera_weights(C, C_reproj, connectivity_matrix=None):
    """
    Compute camera weights. Each camera is assigned a float weight W(camera) which depends on 2 factors:
    (1) neighbors(camera), i.e. the number of cameras it is connected to according to the connectivity matrix
    (2) cost(camera), i.e. the reprojection error of the feature track observations that are seen in the camera
    In particular,
            W(camera) = neighbors(camera) + e^(-cost(camera))
    The tracks selection algorithm visits cameras in decreasing weight order
    """

    # initialize connectivity matrix
    n_cam, n_tracks = C.shape[0] // 2, C.shape[1]
    A = build_connectivity_matrix(C, min_matches=0) if connectivity_matrix is None else connectivity_matrix
    mask = ~np.isnan(C[::2])

    # compute vector of camera weights
    w_cam = []
    for i in range(n_cam):

        # compute neighbors(camera)
        nC_i = np.sum(A[i, :] > 0)

        # compute cost(camera)
        if nC_i > 0:
            # cost(camera) results from the mean/std of the average reprojection error per track seen in the camera
            indices_of_tracks_seen_in_current_cam = np.arange(n_tracks)[mask[i]]
            avg_reproj_err_tracks_seen = np.nanmean(C_reproj[:, indices_of_tracks_seen_in_current_cam], axis=0)
            avg_cost = np.mean(avg_reproj_err_tracks_seen)
            std_cost = np.std(avg_reproj_err_tracks_seen)
            costC_i = avg_cost + 3.0 * std_cost
        else:
            costC_i = 0.0

        w_cam.append(float(nC_i) + np.exp(-costC_i))

    return w_cam


def order_tracks(C, C_scale, C_reproj, priority=["length", "scale", "cost"]):
    """
    Compute a ranking of feature tracks, ordered in decreasing priority
    Tracks with highest priority will be selected first
    Output ranked_track_indices is a dict where
                                                key = index of track in C
                                                value = position in track ranking
    """
    n_tracks = C.shape[1]
    tracks_length = (np.sum(~np.isnan(C), axis=0) / 2).astype(np.int32)
    tracks_scale = np.round(np.nanmean(C_scale, axis=0), 2).astype(np.float64)
    tracks_cost = np.nanmean(C_reproj, axis=0).astype(np.float64)

    tracks_dtype = [("length", int), ("scale", float), ("cost", float)]
    track_values = np.array(list(zip(tracks_length, -tracks_scale, -tracks_cost)), dtype=tracks_dtype)
    ranked_track_indices = dict(list(zip(np.argsort(track_values, order=priority)[::-1], np.arange(n_tracks))))

    return ranked_track_indices


def get_inverted_track_list(C, ranked_track_indices):
    """
    Compute inverted list based on the feature tracks ranking
    The i-th position of the output inverted_track_list contains a list with the indices
    of the visible tracks in the i-th camera from from highest to lowest priority
    """
    inverted_track_list = []
    n_cam = C.shape[0] // 2
    mask = ~np.isnan(C[::2])
    f = lambda idx: ranked_track_indices[idx]
    for i in range(n_cam):
        indices_of_tracks_seen_in_current_cam = np.where(mask[i])[0]
        s = sorted(indices_of_tracks_seen_in_current_cam, key=f)
        inverted_track_list.append(s)

    return inverted_track_list


def get_cam_indices_per_track(C):
    """
    For each track in the connectivity matrix C, get the indices of the cameras where it is seen
    """
    mask = ~np.isnan(C[::2])
    cam_indices_per_track = []
    for track_idx in range(C.shape[1]):
        s = set(np.where(mask[:, track_idx])[0])
        cam_indices_per_track.append(s)
    return cam_indices_per_track


def get_cam_indices_per_cam(A):
    """
    For each camera, get the indices of the neighbor cameras according to the connectivity matrix A
    """
    cam_indices_per_cam = []
    for cam_idx in range(A.shape[1]):
        indices = np.nonzero(A[cam_idx])[0]
        cam_indices_per_cam.append(set(indices))
    return cam_indices_per_cam


def get_tracks_current_tree(A, V, cam_weights, cam_indices_per_track, inverted_track_list):
    """
    Constructs one spanning tree, that connects as much cameras as possible using the tracks with highest priority
    Output Sk is the group of tracks used to construct the tree
    """
    cam_indices_per_cam = get_cam_indices_per_cam(A)

    Croot = np.argmax(cam_weights)
    nodes_last_layer_Hk = [Croot]
    Sk = set()
    Ik = set(nodes_last_layer_Hk)

    while True:
        nodes_next_layer_Hk = []
        for cam_idx in nodes_last_layer_Hk:
            for track_idx in inverted_track_list[cam_idx]:
                if track_idx in Sk:
                    continue

                Wq = cam_indices_per_track[track_idx]
                Rq = cam_indices_per_cam[cam_idx]
                not_done = (Wq & Rq) - Ik
                if len(not_done) > 0:
                    nodes_next_layer_Hk.extend(not_done)
                    Sk.add(track_idx)
                    Ik |= not_done

        if len(V - Ik) == 0 or len(nodes_next_layer_Hk) == 0:
            break

        nodes_last_layer_Hk = sorted(nodes_next_layer_Hk, key=lambda a: -cam_weights[a])

    return Sk


def get_tracks(C, C_reproj, K, ranked_track_indices):
    """
    Runs the feature track selection
    Outputs an optimal subset S of feature tracks extracted from an input correspondence matrix C
    and ranked in decreasing priority using order_tracks (length, scale, cost criterion)
    The output set of feature tracks, S, results from constructing K spanning trees following the next criterion:
    cameras with highest weight are visited first, and for each camera, tracks with highest priority are visited first
    Each spanning tree selects some tracks, which are expected to connect as much cameras as possible
    """
    n_cam = C.shape[0] // 2
    remaining_T = set(np.arange(C.shape[1]))
    T = set(np.arange(C.shape[1]))  # all track indices
    V = set(np.arange(n_cam))  # all camera indices

    k = 0  # current spanning tree index
    S = []  # indices of the selected feature tracks (= indices of columns of C that are selected)
    cam_indices_per_track = get_cam_indices_per_track(C)
    updated_C = C.copy()
    while k < K and len(S) < len(T):

        # update connectivity matrix, inverted track list and camera weights with the new correspondence matrix C
        A = build_connectivity_matrix(updated_C, min_matches=0)
        inverted_track_list = get_inverted_track_list(updated_C, ranked_track_indices)
        cam_weights = np.array(compute_camera_weights(updated_C, C_reproj, connectivity_matrix=A))

        Sk = get_tracks_current_tree(A, V, cam_weights, cam_indices_per_track, inverted_track_list)
        k += 1
        remaining_T -= Sk
        S.extend(Sk)
        updated_C[:, list(Sk)] = np.nan

    return S


def select_best_tracks(C, C_scale, C_reproj, K=30, priority=["length", "scale", "cost"], verbose=False):
    """
    Main of the algorithm
    First ranks all input feature tracks using order_tracks
    Then runs the selection procedure using get_tracks
    """
    t0 = timeit.default_timer()
    if verbose:
        flush_print("\nRunning feature tracks selection algorithm...")

    ranked_track_indices = order_tracks(C, C_scale, C_reproj, priority=priority)
    S = get_tracks(C, C_reproj, K, ranked_track_indices)

    if verbose:
        flush_print("...done in {:.2f} seconds".format(timeit.default_timer() - t0))
        count_obs_per_cam = lambda C: np.sum(1 * ~np.isnan(C), axis=1)[::2]
        n_tracks_out, n_tracks_in = len(S), C.shape[1]
        to_print = [n_tracks_out, n_tracks_in, float(n_tracks_out) / n_tracks_in * 100.0]
        flush_print("Selected {} tracks out of {} ({:.2f}%)".format(*to_print))
        flush_print("     - priority: {}".format(priority))
        flush_print("     - obs per cam before: {}".format(count_obs_per_cam(C)))
        flush_print("     - obs per cam after:  {}\n".format(count_obs_per_cam(C[:, S])))

    return np.array(S)
