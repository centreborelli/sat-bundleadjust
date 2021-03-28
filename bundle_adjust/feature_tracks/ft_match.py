import os

import numpy as np

from bundle_adjust import geo_utils
from . import ft_opencv, ft_s2p
from bundle_adjust.loader import flush_print


def compute_pairs_to_match(init_pairs, footprints, optical_centers, no_filter=False, verbose=True):
    def set_pair(i, j):
        return (min(i, j), max(i, j))

    pairs_to_match, pairs_to_triangulate = [], []
    for (i, j) in init_pairs:
        i, j = int(i), int(j)

        # check there is enough overlap between the images (at least 10% w.r.t image 1)
        intersection_polygon = footprints[i]["poly"].intersection(footprints[j]["poly"])

        # check if the baseline between both cameras is large enough
        baseline = np.linalg.norm(optical_centers[i] - optical_centers[j])

        if no_filter:
            overlap_ok = True
            baseline_ok = True
        else:
            overlap_ok = intersection_polygon.area / footprints[i]["poly"].area >= 0.1
            baseline_ok = baseline / 500000.0 > 1 / 4

        if overlap_ok:
            pairs_to_match.append(set_pair(i, j))
            if baseline_ok:
                pairs_to_triangulate.append(set_pair(i, j))

    # total number of possible pairs given n_imgs is int((n_img*(n_img-1))/2)
    if verbose:
        print("     {} / {} pairs suitable to match".format(len(pairs_to_match), len(init_pairs)))
        print("     {} / {} pairs suitable to triangulate".format(len(pairs_to_triangulate), len(init_pairs)))
    return pairs_to_match, pairs_to_triangulate


def get_pt_indices_inside_utm_bbx(easts, norths, min_east, max_east, min_north, max_north):
    east_ok = (easts > min_east) & (easts < max_east)
    north_ok = (norths > min_north) & (norths < max_north)
    return np.where(east_ok & north_ok)[0]


def match_kp_within_utm_polygon(features_i, features_j, utm_i, utm_j, utm_polygon, d):

    method = d.get("method", "epipolar_based")
    rel_thr = d.get("rel_thr", 0.6)
    abs_thr = d.get("abs_thr", 250)
    ransac = d.get("ransac", 0.3)
    F = d.get("F", None)

    easts_i, norths_i = utm_i[:, 0], utm_i[:, 1]
    easts_j, norths_j = utm_j[:, 0], utm_j[:, 1]

    # get instersection polygon utm coords
    east_poly, north_poly = utm_polygon.exterior.coords.xy
    east_poly, north_poly = np.array(east_poly), np.array(north_poly)

    # use the rectangle containing the intersection polygon as AOI
    min_east, max_east = east_poly.min(), east_poly.max()
    min_north, max_north = north_poly.min(), north_poly.max()

    indices_i_inside = get_pt_indices_inside_utm_bbx(easts_i, norths_i, min_east, max_east, min_north, max_north)
    if len(indices_i_inside) == 0:
        return None
    indices_j_inside = get_pt_indices_inside_utm_bbx(easts_j, norths_j, min_east, max_east, min_north, max_north)
    if len(indices_j_inside) == 0:
        return None

    # pick kp in overlap area and the descriptors
    utm_i_inside, utm_j_inside = utm_i[indices_i_inside], utm_j[indices_j_inside]
    features_i_inside, features_j_inside = features_i[indices_i_inside], features_j[indices_j_inside]

    """
    import matplotlib.patches as patches
    fig, ax = plt.subplots(figsize=(10,6))
    plt.scatter(kp_i[:,0], kp_i[:,1], c='b')
    plt.scatter(kp_j[:,0], kp_j[:,1], c='r')
    plt.scatter(kp_i[indices_i_poly_int,0], kp_i[indices_i_poly_int,1], c='g')
    plt.scatter(kp_j[indices_j_poly_int,0], kp_j[indices_j_poly_int,1], c='g')
    rect = patches.Rectangle((min_east,min_north),max_east-min_east, max_north-min_north, facecolor='g', alpha=0.4)
    #plt.scatter(east_centroid, north_centroid, s=100, c='g')
    #plt.scatter(east_poly, north_poly, s=100, c='g')
    ax.add_patch(rect)
    plt.show()   
    """

    if method == "epipolar_based":
        matches_ij_poly, n = ft_s2p.s2p_match_SIFT(
            features_i_inside,
            features_j_inside,
            F,
            dst_thr=rel_thr,
            ransac_thr=ransac,
        )
    else:
        matches_ij_poly, n = ft_opencv.opencv_match_SIFT(
            features_i_inside,
            features_j_inside,
            dst_thr=rel_thr,
            ransac_thr=ransac,
            matcher=method,
        )

    # go back from the filtered indices inside the polygon to the original indices of all the kps in the image
    if matches_ij_poly is None:
        matches_ij = None
    else:
        indices_m_kp_i = indices_i_inside[matches_ij_poly[:, 0]]
        indices_m_kp_j = indices_j_inside[matches_ij_poly[:, 1]]
        matches_ij = np.vstack((indices_m_kp_i, indices_m_kp_j)).T

    n_matches_init = 0 if matches_ij is None else matches_ij.shape[0]
    if n_matches_init > 0:
        matches_ij = filter_matches_inconsistent_utm_coords(matches_ij, utm_i, utm_j)
        n_matches = 0 if matches_ij is None else matches_ij.shape[0]
        n.append(n_matches)
    else:
        n.append(0)

    return matches_ij, n


def keypoints_to_utm_coords(features, rpcs, footprints, offsets):

    utm = []
    for features_i, rpc_i, footprint_i, offset_i in zip(features, rpcs, footprints, offsets):

        # convert image coords to utm coords (remember to deal with the nan pad...)
        n_kp = np.sum(1 * ~np.isnan(features_i[:, 0]))
        cols = (features_i[:n_kp, 0] + offset_i["col0"]).tolist()
        rows = (features_i[:n_kp, 1] + offset_i["row0"]).tolist()
        alts = [footprint_i["z"]] * n_kp
        lon, lat = rpc_i.localization(cols, rows, alts)
        east, north = geo_utils.utm_from_lonlat(lon, lat)
        utm_coords = np.vstack((east, north)).T
        rest = features_i[n_kp:, :2].copy()
        utm.append(np.vstack((utm_coords, rest)))

    return utm


def filter_matches_inconsistent_utm_coords(matches_ij, utm_i, utm_j):

    pt_i_utm = utm_i[matches_ij[:, 0]]
    pt_j_utm = utm_j[matches_ij[:, 1]]

    all_utm_distances = np.linalg.norm(pt_i_utm - pt_j_utm, axis=1)
    from bundle_adjust.ba_outliers import get_elbow_value

    utm_thr, success = get_elbow_value(all_utm_distances, max_outliers_percent=20, verbose=False)
    utm_thr = utm_thr + 5 if success else np.max(all_utm_distances)
    matches_ij_filt = matches_ij[all_utm_distances <= utm_thr]

    return matches_ij_filt


def match_stereo_pairs(pairs_to_match, features, footprints, utm_coords, d):

    method = d.get("method", "epipolar_based")
    rel_thr = d.get("rel_thr", 0.6)
    abs_thr = d.get("abs_thr", 250)
    ransac = d.get("ransac", 0.3)
    F = d.get("F", None)
    thread_idx = d.get("thread_idx", None)

    pairwise_matches_kp_indices = []
    pairwise_matches_im_indices = []

    if F is None:
        F = [None] * len(pairs_to_match)

    n_pairs = len(pairs_to_match)
    for idx, pair in enumerate(pairs_to_match):
        i, j = pair[0], pair[1]

        utm_polygon = footprints[i]["poly"].intersection(footprints[j]["poly"])

        args = [features[i], features[j], utm_coords[i], utm_coords[j], utm_polygon]
        d = {"method": method, "rel_thr": rel_thr, "abs_thr": abs_thr, "ransac": ransac, "F": F[idx]}
        matches_ij, n = match_kp_within_utm_polygon(*args, d)

        n_matches = 0 if matches_ij is None else matches_ij.shape[0]
        tmp = ""
        if thread_idx is not None:
            tmp = " (thread {} -> {}/{})".format(thread_idx, idx + 1, n_pairs)

        if method == "epipolar_based":
            to_print = [n_matches, method, n[0], "utm", n[1], (i, j), tmp]
            flush_print("{:4} matches ({}: {:4}, {}: {:4}) in pair {}{}".format(*to_print))
        else:
            to_print = [n_matches, "test ratio", n[0], "ransac", n[1], "utm", n[2], (i, j), tmp]
            flush_print("{:4} matches ({}: {:4}, {}: {:4}, {}: {:4}) in pair {}{}".format(*to_print))

        if n_matches > 0:
            im_indices = np.vstack((np.array([i] * n_matches), np.array([j] * n_matches))).T
            pairwise_matches_kp_indices.extend(matches_ij.tolist())
            pairwise_matches_im_indices.extend(im_indices.tolist())

    # pairwise match format is a 1x4 vector
    # position 1 corresponds to the kp index in image 1, that links to features[im1_index]
    # position 2 corresponds to the kp index in image 2, that links to features[im2_index]
    # position 3 is the index of image 1 within the sequence of images, i.e. im1_index
    # position 4 is the index of image 2 within the sequence of images, i.e. im2_index
    pairwise_matches = np.hstack((np.array(pairwise_matches_kp_indices), np.array(pairwise_matches_im_indices)))
    return pairwise_matches


def match_stereo_pairs_multiprocessing(pairs_to_match, features, footprints, utm_coords, n_proc, d):

    method = d.get("method", "epipolar_based")
    rel_thr = d.get("rel_thr", 0.6)
    abs_thr = d.get("abs_thr", 250)
    ransac = d.get("ransac", 0.3)
    F = d.get("F", None)

    n_pairs = len(pairs_to_match)

    n = int(np.ceil(n_pairs / n_proc))
    args_d = {"method": method, "rel_thr": rel_thr, "abs_thr": abs_thr, "ransac": ransac}

    args = []
    for k, i in enumerate(np.arange(0, n_pairs, n)):
        args_d["F"] = F[i : i + n] if method == "epipolar_based" else None
        args_d["thread_idx"] = k
        args.append([pairs_to_match[i : i + n], features, footprints, utm_coords, args_d])

    from multiprocessing import Pool

    with Pool(len(args)) as p:
        matching_output = p.starmap(match_stereo_pairs, args)
    return np.vstack(matching_output)
