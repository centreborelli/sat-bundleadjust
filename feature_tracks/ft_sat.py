import numpy as np
import os

from bundle_adjust import geotools
from bundle_adjust import ba_core
from feature_tracks import ft_utils as fd

from feature_tracks import ft_s2p
from feature_tracks import ft_opencv

def keypoints_to_utm_coords(features, rpcs, footprints, offsets):

    utm = []
    for features_i, rpc_i, footprint_i, offset_i in zip(features, rpcs, footprints, offsets):

        # convert image coords to utm coords (remember to deal with the nan pad...)
        n_kp = np.sum(1*~np.isnan(features_i[:,0])) 
        cols = (features_i[:n_kp,0] + offset_i['col0']).tolist()
        rows = (features_i[:n_kp,1] + offset_i['row0']).tolist()
        alts = [footprint_i['z']] * n_kp
        lon, lat = rpc_i.localization(cols, rows, alts)
        east, north = geotools.utm_from_lonlat(lon, lat)
        utm_coords = np.vstack((east, north)).T
        rest = features_i[n_kp:, :2].copy()
        utm.append(np.vstack((utm_coords, rest)))

    return utm


def compute_pairs_to_match(init_pairs, footprints, optical_centers, no_filter=False, verbose=True):

    def set_pair(i, j):
        return (min(i, j), max(i, j))

    pairs_to_match, pairs_to_triangulate = [], []
    for (i, j) in init_pairs:
            i, j = int(i), int(j)

            # check there is enough overlap between the images (at least 10% w.r.t image 1)
            intersection_polygon = footprints[i]['poly'].intersection(footprints[j]['poly'])
            
            # check if the baseline between both cameras is large enough
            baseline = np.linalg.norm(optical_centers[i] - optical_centers[j])

            if no_filter:
                overlap_ok = True
                baseline_ok = True
            else:
                overlap_ok = intersection_polygon.area/footprints[i]['poly'].area >= 0.1
                baseline_ok = baseline > 125000 #150000

            if overlap_ok:    
                pairs_to_match.append(set_pair(i, j))
                if baseline_ok:
                    pairs_to_triangulate.append(set_pair(i, j))

    # total number of possible pairs given n_imgs is int((n_img*(n_img-1))/2)
    if verbose:
        print('     {} / {} pairs suitable to match'.format(len(pairs_to_match), len(init_pairs)))
        print('     {} / {} pairs suitable to triangulate'.format(len(pairs_to_triangulate), len(init_pairs)))
    return pairs_to_match, pairs_to_triangulate


def match_kp_within_utm_polygon(features_i, features_j, utm_i, utm_j, utm_polygon,
                                method='local', rel_thr=0.6, abs_thr=250, ransac=0.3, F=None):

    east_i, north_i, east_j, north_j = utm_i[:, 0], utm_i[:, 1], utm_j[:, 0], utm_j[:, 1]

    # get instersection polygon utm coords
    east_poly, north_poly = utm_polygon.exterior.coords.xy
    east_poly, north_poly = np.array(east_poly), np.array(north_poly)

    # use the rectangle containing the intersection polygon as AOI 
    min_east, max_east, min_north, max_north = min(east_poly), max(east_poly), min(north_poly), max(north_poly)

    east_ok_i = np.logical_and(east_i > min_east, east_i < max_east)
    north_ok_i = np.logical_and(north_i > min_north, north_i < max_north)
    indices_i_poly_bool, all_indices_i = np.logical_and(east_ok_i, north_ok_i), np.arange(utm_i.shape[0])
    indices_i_poly_int = all_indices_i[indices_i_poly_bool]
    if not any(indices_i_poly_bool):
        return None

    east_ok_j = np.logical_and(east_j > min_east, east_j < max_east)
    north_ok_j = np.logical_and(north_j > min_north, north_j < max_north)
    indices_j_poly_bool, all_indices_j = np.logical_and(east_ok_j, north_ok_j), np.arange(utm_j.shape[0])
    indices_j_poly_int = all_indices_j[indices_j_poly_bool]
    if not any(indices_j_poly_bool):
        return None

    # pick kp in overlap area and the descriptors
    utm_i_poly, utm_j_poly = utm_i[indices_i_poly_bool], utm_j[indices_j_poly_bool]
    features_i_poly, features_j_poly = features_i[indices_i_poly_bool], features_j[indices_j_poly_bool]

    '''
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
    '''

    if method == 'epipolar_based':
        matches_ij_poly, n = ft_s2p.s2p_match_SIFT(features_i_poly, features_j_poly, F,
                                                   dst_thr=rel_thr, ransac_thr=ransac)
    elif method == 'local_window':
        matches_ij_poly, n = locally_match_SIFT_utm_coords(features_i_poly, features_j_poly,
                                                           utm_i_poly, utm_j_poly,
                                                           sift_thr=abs_thr, ransac_thr=ransac)
    else:
        matches_ij_poly, n = ft_opencv.opencv_match_SIFT(features_i_poly, features_j_poly,
                                                         dst_thr=rel_thr, ransac_thr=ransac, matcher=method)

    # go back from the filtered indices inside the polygon to the original indices of all the kps in the image
    if matches_ij_poly is None:
        matches_ij = None
    else:
        indices_m_kp_i = indices_i_poly_int[matches_ij_poly[:, 0]]
        indices_m_kp_j = indices_j_poly_int[matches_ij_poly[:, 1]]
        matches_ij = np.vstack((indices_m_kp_i, indices_m_kp_j)).T

    n_matches_init = 0 if matches_ij is None else matches_ij.shape[0]
    if n_matches_init > 0:
        matches_ij = filter_matches_inconsistent_utm_coords(matches_ij, utm_i, utm_j)
        n_matches = 0 if matches_ij is None else matches_ij.shape[0]
        n.append(n_matches)
    else:
        n.append(0)

    return matches_ij, n


def filter_matches_inconsistent_utm_coords(matches_ij, utm_i, utm_j):

    pt_i_utm = utm_i[matches_ij[:,0]]
    pt_j_utm = utm_j[matches_ij[:,1]]

    all_utm_distances = np.linalg.norm(pt_i_utm - pt_j_utm, axis=1)
    from bundle_adjust.ba_outliers import get_elbow_value
    utm_thr, success = get_elbow_value(all_utm_distances, max_outliers_percent=20, verbose=False)
    utm_thr = utm_thr + 5 if success else np.max(all_utm_distances)
    matches_ij_filt = matches_ij[all_utm_distances <= utm_thr]

    '''
    print('UTM consistency distance threshold set to {:.2f} m'.format(utm_thr))
    n_init = matches_ij.shape[0]
    n_filt = matches_ij_filt.shape[0]
    removed = n_init - n_filt
    percent = (float(removed)/n_init) * 100.
    args = [removed, percent, n_filt]
    print('Removed {} pairwise matches ({:.2f}%) due to inconsistent UTM coords ({} left)'.format(*args))
    '''
    return matches_ij_filt


def locally_match_SIFT_utm_coords(features_i, features_j, utm_i, utm_j,
                                  radius=30, sift_thr=250, ransac_thr=0.3):

    import ctypes
    from ctypes import c_int, c_float
    from numpy.ctypeslib import ndpointer
    from feature_tracks.ft_opencv import geometric_filtering

    # to create siftu.so use the command below in the imscript/src
    # gcc -shared -o siftu.so -fPIC siftu.c siftie.c iio.c -lpng -ljpeg -ltiff
    here = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(os.path.dirname(here), 'bin', 'siftu.so')
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
    lib.main_siftcpairsg_v2.argtypes = (c_float, c_float, c_float, c_int, c_int,
                                        ndpointer(dtype=c_float, shape=(n_i, 132)),
                                        ndpointer(dtype=c_float, shape=(n_j, 132)),
                                        ndpointer(dtype=c_int, shape=(max_n, 2)))

    matches =  -1*np.ones((max_n, 2), dtype='int32')
    lib.main_siftcpairsg_v2(sift_thr, radius, radius, n_i, n_j, features_i.astype('float32'),
                            features_j.astype('float32'), matches)

    n_matches = min(np.arange(max_n)[matches[:, 0] == -1])
    if n_matches > 0:
        matches_ij = matches[:n_matches, :]
        matches_ij = geometric_filtering(pix_i, pix_j, matches_ij, ransac_thr)
    n_matches_after_geofilt = 0 if matches_ij is None else matches_ij.shape[0]

    return matches_ij, [n_matches, n_matches_after_geofilt]


def match_stereo_pairs(pairs_to_match, features, footprints, utm_coords,
                       method='local_window', rel_thr=0.6, abs_thr=250, ransac=0.3, F=None, thread_idx=None):

    pairwise_matches_kp_indices = []
    pairwise_matches_im_indices = []

    if F is None:
        F = [None]*len(pairs_to_match)

    n_pairs = len(pairs_to_match)
    for idx, pair in enumerate(pairs_to_match):
        i, j = pair[0], pair[1]

        utm_polygon = footprints[i]['poly'].intersection(footprints[j]['poly'])

        matching_args = [features[i], features[j], utm_coords[i], utm_coords[j], utm_polygon,
                         method, rel_thr, abs_thr, ransac, F[idx]]
        matches_ij, n = match_kp_within_utm_polygon(*matching_args)

        n_matches = 0 if matches_ij is None else matches_ij.shape[0]
        tmp = ''
        if thread_idx is not None:
            tmp = ' (thread {} -> {}/{})'.format(thread_idx, idx + 1, n_pairs)

        if method == 'epipolar_based':
            args = [n_matches, 's2p epipolar based', n[0], 'utm', n[1], (i, j), tmp]
            print('{:4} matches ({}: {:4}, {}: {:4}) in pair {}{}'.format(*args), flush=True)
        elif method == 'local_window':
            args = [n_matches, 'imscript local window', n[0], 'ransac', n[1], 'utm', n[2], (i, j), tmp]
            print('{:4} matches ({}: {:4}, {}: {:4}, {}: {:4}) in pair {}{}'.format(*args), flush=True)
        else:
            args = [n_matches, 'test ratio', n[0], 'ransac', n[1], 'utm', n[2], (i, j), tmp]
            print('{:4} matches ({}: {:4}, {}: {:4}, {}: {:4}) in pair {}{}'.format(*args), flush=True)

        if n_matches > 0:
            im_indices = np.vstack((np.array([i]*n_matches), np.array([j]*n_matches))).T
            pairwise_matches_kp_indices.extend(matches_ij.tolist())
            pairwise_matches_im_indices.extend(im_indices.tolist())

    # pairwise match format is a 1x4 vector
    # position 1 corresponds to the kp index in image 1, that links to features[im1_index]
    # position 2 corresponds to the kp index in image 2, that links to features[im2_index]
    # position 3 is the index of image 1 within the sequence of images, i.e. im1_index
    # position 4 is the index of image 2 within the sequence of images, i.e. im2_index
    pairwise_matches = np.hstack((np.array(pairwise_matches_kp_indices), np.array(pairwise_matches_im_indices)))
    return pairwise_matches


def match_stereo_pairs_multiprocessing(pairs_to_match, features, footprints, utm_coords,
                                       method='local_window', rel_thr=0.6, abs_thr=250,
                                       ransac=0.3, F=None, n_proc=5):

    n_pairs = len(pairs_to_match)

    n = int(np.ceil(n_pairs / n_proc))

    if method == 'epipolar_based':
        args = [(pairs_to_match[i:i + n], features, footprints, utm_coords, method,
                 rel_thr, abs_thr, ransac, F[i:i + n], k) for k, i in enumerate(np.arange(0, n_pairs, n))]
    else:
        args = [(pairs_to_match[i:i + n], features, footprints, utm_coords, method,
                 rel_thr, abs_thr, ransac, None, k) for k, i in enumerate(np.arange(0, n_pairs, n))]

    from multiprocessing import Pool
    with Pool(len(args)) as p:
        matching_output = p.starmap(match_stereo_pairs, args)
    return np.vstack(matching_output)
