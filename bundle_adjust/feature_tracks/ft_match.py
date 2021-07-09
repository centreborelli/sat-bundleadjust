"""
A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery
author: Roger Mari <roger.mari@ens-paris-saclay.fr>
year: 2021

This script implements functions dedicated to matching keypoints between two satellite images
"""

import numpy as np

from . import ft_opencv, ft_s2p
from bundle_adjust import geo_utils
from bundle_adjust.loader import flush_print


def compute_pairs_to_match(init_pairs, footprints, optical_centers,
                           min_overlap=0.1, min_baseline=1/4, orbit_alt=500000, verbose=True):
    """
    Compute pairs_to_match and pairs_to_triangulate

    Args:
        init_pairs: a list of pairs, where each pair is represented by a tuple of image indices
        footprints: a list of footprints as defined by BundleAdjustmentPipeline.get_footprints
        optical_centers: a list of optical centers as defined by BundleAdjustmentPipeline.get_optical_centers
        verbose (optional): boolean, if true prints how many pairs to match were considered suitable

    Returns:
        pairs_to_match: subset of pairs from init_pairs considered as well-posed for feature matching
        pairs_to_triangulate: subset of pairs from pairs_to_match considered as well-posed for triangulation
    """

    def set_pair(i, j):
        return (min(i, j), max(i, j))

    pairs_to_match, pairs_to_triangulate = [], []
    for (i, j) in init_pairs:
        i, j = int(i), int(j)

        # check there is enough overlap between the images (at least 10% w.r.t image 1)
        shapely_i = geo_utils.geojson_to_shapely_polygon(footprints[i]["geojson"])
        shapely_j = geo_utils.geojson_to_shapely_polygon(footprints[j]["geojson"])
        intersection_polygon = shapely_i.intersection(shapely_j)
        overlap_ok = intersection_polygon.area / shapely_i.area > min_overlap

        if overlap_ok:
            pairs_to_match.append(set_pair(i, j))

            # check if the baseline between both cameras is large enough
            baseline = np.linalg.norm(optical_centers[i] - optical_centers[j])
            baseline_ok = baseline / orbit_alt > min_baseline

            if baseline_ok:
                pairs_to_triangulate.append(set_pair(i, j))

    # exception: some of the cameras may be ONLY matched with other cameras at insufficient distance
    # i.e. these are cameras that are part of pairs_to_match and not pairs_to_triangulate
    # to avoid dropping these cameras, we will exceptionally consider pairs involving them as good to triangulate
    camera_indices_in_pairs_to_match = set(np.unique(np.array(pairs_to_match).flatten()))
    camera_indices_in_pairs_to_triangulate = set(np.unique(np.array(pairs_to_triangulate).flatten()))
    cams_bad_baseline = list(camera_indices_in_pairs_to_match - camera_indices_in_pairs_to_triangulate)
    pairs_to_triangulate2 = [(i, j) for (i, j) in pairs_to_match if i in cams_bad_baseline or j in cams_bad_baseline]
    pairs_to_triangulate.extend(pairs_to_triangulate2)

    if verbose:
        print("     {} / {} pairs suitable to match".format(len(pairs_to_match), len(init_pairs)))
        print("     {} / {} pairs suitable to triangulate".format(len(pairs_to_triangulate), len(init_pairs)))
        n = len(cams_bad_baseline)
        if n > 0:
            print("     WARNING: Found {} cameras with insufficient baseline w.r.t. all neighbor cameras".format(n))
            print("              Concerned cameras are: {}".format(cams_bad_baseline))

    return pairs_to_match, pairs_to_triangulate


def get_pt_indices_inside_utm_bbx(easts, norths, min_east, max_east, min_north, max_north):
    """
    Compute which 2d points are inside a utm bounding box

    Args:
        easts, norths: 2 arrays of N values representing the utm coordinates of N 2d points
        min_east, max_east, min_north, max_north: float values delimiting a bounding box in utm coordinates

    Returns:
        indices_keypoints_inside: indices of the points which are located inside the utm bounding box
    """
    east_ok = (easts > min_east) & (easts < max_east)
    north_ok = (norths > min_north) & (norths < max_north)
    indices_keypoints_inside = np.where(east_ok & north_ok)[0]
    return indices_keypoints_inside


def match_kp_within_utm_polygon(features_i, features_j, utm_i, utm_j, utm_polygon, tracks_config, F=None):
    """
    Match two sets of image keypoints, but restrict the matching to those points inside a utm polygon

    Args:
        features_i: array of size Nx132, where each row represents an image keypoint detected in image i
                    row format is the following: (col, row, scale, orientation, sift descriptor)
        features_j: the equivalent to features_i for image j
        utm_i: the approximate geographic utm coordinates(east, north) of each keypoint in features_i
        utm_j: the approximate geographic utm coordinates(east, north) of each keypoint in features_j
        utm_polygon: geojson polygon in utm coordinates
        tracks_config: dictionary with the feature tracking configuration (ft_utils.init_feature_tracks_config)
        F (optional): array of size 3x3, the fundamental matrix between image i and image j

    Returns:
        matches_ij: array of shape Mx2 representing the output matches
                    column 0 corresponds to the keypoint index in features_i
                    column 1 corresponds to the keypoint index in features_j
        n: number of matches found inside the utm polygon
    """
    features_i, features_j = np.load(features_i, mmap_mode='r'), np.load(features_j, mmap_mode='r')
    utm_i, utm_j = np.load(utm_i, mmap_mode='r'), np.load(utm_j, mmap_mode='r')

    easts_i, norths_i = utm_i[:, 0], utm_i[:, 1]
    easts_j, norths_j = utm_j[:, 0], utm_j[:, 1]

    # get the bounding box of the instersection polygon utm coords
    east_poly, north_poly = utm_polygon.exterior.coords.xy
    east_poly, north_poly = np.array(east_poly), np.array(north_poly)

    # use the rectangle containing the intersection polygon as AOI
    min_east, max_east = east_poly.min(), east_poly.max()
    min_north, max_north = north_poly.min(), north_poly.max()

    indices_i_inside = get_pt_indices_inside_utm_bbx(easts_i, norths_i, min_east, max_east, min_north, max_north)
    indices_j_inside = get_pt_indices_inside_utm_bbx(easts_j, norths_j, min_east, max_east, min_north, max_north)
    if len(indices_i_inside) == 0 or len(indices_j_inside) == 0:
        return None, [0, 0, 0]

    # pick kp in overlap area and the descriptors
    features_i_inside, features_j_inside = features_i[indices_i_inside], features_j[indices_j_inside]

    if tracks_config["FT_sift_matching"] == "epipolar_based":
        matches_ij_poly, n = ft_s2p.s2p_match_SIFT(
            features_i_inside,
            features_j_inside,
            F,
            dst_thr=tracks_config["FT_rel_thr"],
            ransac_thr=tracks_config["FT_ransac"],
        )
        n = [n]
    elif tracks_config["FT_sift_matching"] == "local_window":
        matches_ij_poly, n_local, n_ransac = locally_match_SIFT_utm_coords(
            features_i_inside,
            features_j_inside,
            utm_i[indices_i_inside],
            utm_j[indices_j_inside],
            sift_thr=tracks_config["FT_abs_thr"],
            ransac_thr=tracks_config["FT_ransac"],
        )
        n = [n_local, n_ransac]
    else:
        matches_ij_poly, n_ratio_test, n_ransac = ft_opencv.opencv_match_SIFT(
            features_i_inside,
            features_j_inside,
            dst_thr=tracks_config["FT_rel_thr"],
            ransac_thr=tracks_config["FT_ransac"],
            matcher=tracks_config["FT_sift_matching"],
        )
        n = [n_ratio_test, n_ransac]

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


def keypoints_to_utm_coords(im_features, im_rpc, im_offset, alt):
    """
    Compute the approximate geographic utm coordinates of a list of image features

    Args:
        im_features: array of size Nx132, each row represents an image keypoint
                     row format is the following: (col, row, scale, orientation, sift descriptor)
        im_rpc: the RPC model associated to the image
        im_offset: crop offset associated to the keypoint image coordinates
        alt: the altitude value that will be used to localize the geographic coordinates of the points

    Returns:
        utm_coords: array of size Nx2, each row gives the approx. (east, north) coordinates of a keypoint
    """

    n_kp = np.sum(1 * ~np.isnan(im_features[:, 0]))
    cols = (im_features[:n_kp, 0] + im_offset["col0"]).tolist()
    rows = (im_features[:n_kp, 1] + im_offset["row0"]).tolist()
    alts = [alt] * n_kp
    lon, lat = im_rpc.localization(cols, rows, alts)
    east, north = geo_utils.utm_from_lonlat(lon, lat)
    utm_coords = np.vstack((east, north)).T

    # add nan padding, in the same way as it is done with im_features
    rest = im_features[n_kp:, :2].copy()
    utm_coords = np.vstack((utm_coords, rest))

    return utm_coords


def filter_matches_inconsistent_utm_coords(matches_ij, utm_i, utm_j):
    """
    Filter matches between 2 satellite images based on the distance between their geographic coordinates
    If the matches are good, then the distances between their geographic coordinates
    should be only affected by the RPC bias; if there are outliers they will cause spikes in the distribution

    Args:
        matches_ij: array of size Mx2, each row encodes a match between utm_i and utm_j
                    column 0 is a list of indices referring to utm_i,
                    column 1 is a list of indices referring to utm_j
        utm_i: array of size Nx2, each row gives the approx. (east, north) coordinates of a keypoint in image i
        utm_j: array of size Nx2, each row gives the approx. (east, north) coordinates of a keypoint in image j

    Returns:
        matches_ij_filt: filtered version of matches_ij (will contain same amount of rows or less)
    """

    pt_i_utm = utm_i[matches_ij[:, 0]]
    pt_j_utm = utm_j[matches_ij[:, 1]]

    all_utm_distances = np.linalg.norm(pt_i_utm - pt_j_utm, axis=1)
    from bundle_adjust.ba_outliers import get_elbow_value

    utm_thr, success = get_elbow_value(all_utm_distances, max_outliers_percent=20, verbose=False)
    utm_thr = utm_thr + 5 if success else np.max(all_utm_distances)
    matches_ij_filt = matches_ij[all_utm_distances <= utm_thr]

    return matches_ij_filt


def match_stereo_pairs(pairs_to_match, features, footprints, utm_coords, tracks_config, F=None, thread_idx=None):
    """
    Pairwise matching of image keypoints of a series of pairs of satellite images

    Args:
        pairs_to_match: a list of pairs, where each pair is represented by a tuple of image indices
        features: a list of arrays with size Nx132, representing the keypoints in each image
        footprints: a list of footprints as defined by BundleAdjustmentPipeline.get_footprints
        utm_coords: a list of arrays with size Nx2, representing the utm coordinates of the keypoints in each image
        tracks_config: dictionary with the feature tracking configuration (ft_utils.init_feature_tracks_config)
        F (optional): a list of arrays with size 3x3, the fundamental matrices of each pair of images
        thread_idx (optional): integer, the thread index, only interesting for verbose when multiprocessing is used

    Returns:
        pairwise_matches: array of size Mx4, where each row represents a correspondence between keypoints
                          column 0 corresponds to the kp index in image 1, that links to features[im1_index]
                          column 1 corresponds to the kp index in image 2, that links to features[im2_index]
                          column 2 is the index of image 1 within the sequence of images, i.e. im1_index
                          column 3 is the index of image 2 within the sequence of images, i.e. im2_index
    """

    # a match consists of two pairs of corresponding (1) keypoint indices and (2) image indices
    # to identify (1) the keypoints (at image level) and (2) the images where they are seen (at image sequence level)
    pairwise_matches_kp_indices = []
    pairwise_matches_im_indices = []

    F = [None] * len(pairs_to_match) if F is None else F
    n_pairs = len(pairs_to_match)
    for idx, pair in enumerate(pairs_to_match):
        i, j = pair[0], pair[1]

        shapely_i = geo_utils.geojson_to_shapely_polygon(footprints[i]["geojson"])
        shapely_j = geo_utils.geojson_to_shapely_polygon(footprints[j]["geojson"])
        utm_polygon = shapely_i.intersection(shapely_j)

        args = [features[i], features[j], utm_coords[i], utm_coords[j], utm_polygon, tracks_config, F[idx]]
        matches_ij, n = match_kp_within_utm_polygon(*args)

        n_matches = 0 if matches_ij is None else matches_ij.shape[0]
        tmp = ""
        if thread_idx is not None:
            tmp = " (thread {} -> {}/{})".format(thread_idx, idx + 1, n_pairs)

        if tracks_config["FT_sift_matching"] == "epipolar_based":
            to_print = [n_matches, "epipolar_based", n[0], "utm", n[1], (i, j), tmp]
            flush_print("{:4} matches ({}: {:4}, {}: {:4}) in pair {}{}".format(*to_print))
        elif tracks_config["FT_sift_matching"] == "local_window":
            to_print = [n_matches, "local_window", n[0], "ransac", n[1], "utm", n[2], (i, j), tmp]
            flush_print("{:4} matches ({}: {:4}, {}: {:4}, {}: {:4}) in pair {}{}".format(*to_print))
        else:
            to_print = [n_matches, "test ratio", n[0], "ransac", n[1], "utm", n[2], (i, j), tmp]
            flush_print("{:4} matches ({}: {:4}, {}: {:4}, {}: {:4}) in pair {}{}".format(*to_print))

        if n_matches > 0:
            im_indices = np.vstack((np.array([i] * n_matches), np.array([j] * n_matches))).T
            pairwise_matches_kp_indices.extend(matches_ij.tolist())
            pairwise_matches_im_indices.extend(im_indices.tolist())

    pairwise_matches = np.hstack((np.array(pairwise_matches_kp_indices), np.array(pairwise_matches_im_indices)))
    return pairwise_matches


def match_stereo_pairs_multiprocessing(pairs_to_match, features, footprints, utm_coords, tracks_config, F=None):
    """
    This function is just a wrapper to call match_stereo_pairs using multiprocessing
    The inputs and outputs are therefore the ones defined in match_stereo_pairs
    The number of independent threads is given by tracks_config["FT_n_proc"]
    """
    n_proc = tracks_config["FT_n_proc"]  # number of threads
    n_pairs = len(pairs_to_match)  # number of pairs to match
    n = int(np.ceil(n_pairs / n_proc))  # number of pairs per thread

    parallel_lib = "pool"
    if parallel_lib == "ray":

        print("Using ray parallel computing")
        args = []
        for k, i in enumerate(np.arange(0, n_pairs, n)):
            F_k = F[i : i + n] if tracks_config["FT_sift_matching"] == "epipolar_based" else None
            thread_idx = k
            args.append([pairs_to_match[i : i + n], F_k, thread_idx])

        import ray

        ray.init(include_dashboard=False, configure_logging=True, logging_format="%(message)s")

        @ray.remote
        def func(pairs, Fs, t, features_, footprints_, utm_coords_, tracks_config_):
            return match_stereo_pairs(pairs, features_, footprints_, utm_coords_, tracks_config_, F=Fs, thread_idx=t)

        features_ = ray.put(features)
        footprints_ = ray.put(footprints)
        utm_coords_ = ray.put(utm_coords)
        tracks_config_ = ray.put(tracks_config)

        result_ids = [func.remote(a[0], a[1], a[2], features_, footprints_, utm_coords_, tracks_config_) for a in args]
        matching_output = ray.get(result_ids)
    else:
        args = []
        for k, i in enumerate(np.arange(0, n_pairs, n)):
            F_k = F[i : i + n] if tracks_config["FT_sift_matching"] == "epipolar_based" else None
            thread_idx = k
            args.append([pairs_to_match[i : i + n], features, footprints, utm_coords, tracks_config, F_k, thread_idx])

        if parallel_lib == "joblib":
            from joblib import Parallel, delayed, parallel_backend
            with parallel_backend('multiprocessing', n_jobs=n_proc):
                matching_output = Parallel()(delayed(match_stereo_pairs)(*a) for a in args)
        else:
            from multiprocessing import Pool
            with Pool(len(args)) as p:
                matching_output = p.starmap(match_stereo_pairs, args)
    pairwise_matches = np.vstack(matching_output)
    return pairwise_matches


def locally_match_SIFT_utm_coords(features_i, features_j, utm_i, utm_j, radius=30, sift_thr=250, ransac_thr=0.3):

    import os
    import ctypes
    from ctypes import c_float, c_int

    from numpy.ctypeslib import ndpointer

    from .ft_opencv import geometric_filtering

    # to create siftu.so use the command below in the imscript/src
    # gcc -shared -o siftu.so -fPIC siftu.c siftie.c iio.c -lpng -ljpeg -ltiff
    here = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(os.path.dirname(here), "bin", "siftu.so")
    lib = ctypes.CDLL(lib_path)

    # keep pixel coordinates in memory
    pix_i = features_i[:, :2].copy()
    pix_j = features_j[:, :2].copy()

    # the imscript function for local matching requires the bbx
    # containing all point coords -registered- to start at (0,0)
    # we employ the utm coords as coarsely registered coordinates
    utm_i[:, 1][utm_i[:, 1] < 0] += 10e6
    utm_j[:, 1][utm_j[:, 1] < 0] += 10e6
    min_east = min(utm_i[:, 0].min(), utm_j[:, 0].min())
    min_north = min(utm_i[:, 1].min(), utm_j[:, 1].min())
    offset = np.array([min_east, min_north])
    features_i[:, :2] = utm_i - np.tile(offset, (features_i.shape[0], 1))
    features_j[:, :2] = utm_j - np.tile(offset, (features_j.shape[0], 1))

    n_i = features_i.shape[0]
    n_j = features_j.shape[0]
    max_n = max(n_i, n_j)

    # define the argument types of the stereo_corresp_to_lonlatalt function from disp_to_h.so
    lib.main_siftcpairsg_v2.argtypes = (
        c_float,
        c_float,
        c_float,
        c_int,
        c_int,
        ndpointer(dtype=c_float, shape=(n_i, 132)),
        ndpointer(dtype=c_float, shape=(n_j, 132)),
        ndpointer(dtype=c_int, shape=(max_n, 2)),
    )

    matches = -1 * np.ones((max_n, 2), dtype="int32")
    lib.main_siftcpairsg_v2(
        sift_thr,
        radius,
        radius,
        n_i,
        n_j,
        features_i.astype("float32"),
        features_j.astype("float32"),
        matches,
    )

    n_matches = min(np.arange(max_n)[matches[:, 0] == -1])
    if n_matches > 0:
        matches_ij = matches[:n_matches, :]
        matches_ij = geometric_filtering(pix_i, pix_j, matches_ij, ransac_thr)
    else:
        matches_ij = None
    n_matches_after_geofilt = 0 if matches_ij is None else matches_ij.shape[0]

    return matches_ij, n_matches, n_matches_after_geofilt
