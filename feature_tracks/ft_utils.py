import numpy as np

import cv2
from IS18 import utils
from bundle_adjust import ba_utils
import os

from feature_tracks import ft_sat
import matplotlib.pyplot as plt
import pickle

def get_fname_id(fname):
    return os.path.splitext(os.path.basename(fname))[0]
    

def plot_features_stereo_pair(i, j, features, input_seq):

    # i, j : indices of the images
    
    pts1, pts2 = features[i][:,:2], features[j][:,:2]  
    print('Found {} keypoints in image {} and {} keypoints in image {}'.format(pts1.shape[0], i,
                                                                               pts2.shape[0], j))
    
    fig = plt.figure(figsize=(20,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow((input_seq[i]), cmap="gray")
    ax2.imshow((input_seq[j]), cmap="gray")
    if pts1.shape[0] > 0:
        ax1.scatter(x=pts1[:,0], y=pts1[:,1], c='r', s=40)
    if pts2.shape[0] > 0:
        ax2.scatter(x=pts2[:,0], y=pts2[:,1], c='r', s=40)
    plt.show()

def plot_track_observations_stereo_pair(i, j, C, input_seq):
    
    # i, j : indices of the images
    
    visible_idx = np.logical_and(~np.isnan(C[i*2,:]), ~np.isnan(C[j*2,:])) 
    pts1, pts2 = C[(i*2):(i*2+2), visible_idx], C[(j*2):(j*2+2), visible_idx]
    pts1, pts2 = pts1.T, pts2.T
    
    print('{} track observations to display for pair ({},{})'.format(pts1.shape[0], i, j))
    
    print('List of track indices: {}'.format(np.arange(C.shape[1])[visible_idx]))
    
    fig = plt.figure(figsize=(20,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow((input_seq[i]), cmap="gray")
    ax2.imshow((input_seq[j]), cmap="gray")
    if pts1.shape[0] > 0:
        ax1.scatter(x=pts1[:,0], y=pts1[:,1], c='r', s=40)
        ax2.scatter(x=pts2[:,0], y=pts2[:,1], c='r', s=40)
    plt.show()
    
def plot_pairwise_matches_stereo_pair(i, j, features, pairwise_matches, input_seq):
    
    # i, j : indices of the images
    
    pairwise_matches_kp_indices = pairwise_matches[:, :2]
    pairwise_matches_im_indices = pairwise_matches[:, 2:]
    
    true_where_matches = np.all(pairwise_matches_im_indices == np.array([i, j]), axis=1)
    matched_kps_i = features[i][pairwise_matches_kp_indices[true_where_matches, 0]]
    matched_kps_j = features[j][pairwise_matches_kp_indices[true_where_matches, 1]]
    
    print('{} pairwise matches to display for pair ({},{})'.format(matched_kps_i.shape[0], i, j))
    
    
    h, w = input_seq[i].shape
    max_v = max(input_seq[i].max(), input_seq[j].max())
    margin = 100
    fig = plt.figure(figsize=(42,6))
    complete_im = np.hstack([input_seq[i], np.ones((h, margin))*max_v, input_seq[j]])
    ax = plt.gca()
    ax.imshow((complete_im), cmap="gray")
    if matched_kps_i.shape[0] > 0:
        ax.scatter(x=matched_kps_i[:, 0], y=matched_kps_i[:, 1], c='r', s=30)
        ax.scatter(x=w + margin + matched_kps_j[:, 0], y=matched_kps_j[:, 1], c='r', s=30)
        for k in range(matched_kps_i.shape[0]):
            ax.plot([matched_kps_i[k, 0], w + margin + matched_kps_j[k, 0] ],
                    [matched_kps_i[k, 1], matched_kps_j[k, 1] ], 'y--', lw=3)
    plt.show()
    
    fig = plt.figure(figsize=(20,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow((input_seq[i]), cmap="gray")
    ax2.imshow((input_seq[j]), cmap="gray")
    if matched_kps_i.shape[0] > 0:
        ax1.scatter(x=matched_kps_i[:, 0], y=matched_kps_i[:, 1], c='r', s=10)
        ax2.scatter(x=matched_kps_j[:, 0], y=matched_kps_j[:, 1], c='r', s=10)
    plt.show()

def filter_C_using_pairs_to_triangulate(C, pairs_to_triangulate):
    # remove matches found in pairs with short baseline that were not extended to more images
    # since these columns of C will not be triangulated
    # ATTENTION: this is very slow in comparison to the rest of the function
    # it can take various seconds while the rest is instantaneous, optimize it in the future
    columns_to_preserve = []
    n_cams, n_tracks = int(C.shape[0]/2), C.shape[1]
    for i in range(n_tracks):
        im_ind = [k for k, j in enumerate(range(n_cams)) if not np.isnan(C[j*2, i])]
        all_pairs = [(im_i, im_j) for im_i in im_ind for im_j in im_ind if im_i != im_j and im_i<im_j]
        good_pairs = [pair for pair in all_pairs if pair in pairs_to_triangulate]
        columns_to_preserve.append(len(good_pairs) > 0)
    return columns_to_preserve

def feature_tracks_from_pairwise_matches(features, pairwise_matches, pairs_to_triangulate):

    '''
    TO DO:
    This function has a drawback: everytime we build the feature tracks we load ALL features of the scene
    and ALL pairwise matches. When a reasonable amount of images is used this will be fine but for 1000 
    images the computation time may increase in a relevant way.
    The solution would be to save matches separately per image and directly load those of interest 
    (instead of filtering them from all_pairwise_matches).
    '''
    
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
    
    parents = [None]*(id_count)
    for kp_i, kp_j, im_i, im_j in pairwise_matches_of_interest:
        feature_i_id, feature_j_id = feature_ids[im_i, kp_i], feature_ids[im_j, kp_j]
        union(parents, feature_i_id, feature_j_id)
        
    # handle parents without None
    parents = [find(parents, feature_id) for feature_id, v in enumerate(parents)]
    
    # parents = track_id
    _, parents_indices, parents_counts = np.unique(parents, return_inverse=True, return_counts=True)
    n_tracks = np.sum(1*(parents_counts>1))
    track_parents = np.array(parents)[parents_counts[parents_indices] > 1]
    _, track_idx_from_parent, _ = np.unique(track_parents, return_inverse=True, return_counts=True)
    
    # t_idx, parent_id
    track_indices = np.zeros(len(parents))
    track_indices[:] = np.nan
    track_indices[parents_counts[parents_indices] > 1] = track_idx_from_parent
        
    '''
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
    '''  
    
    # build correspondence matrix
    C = np.zeros((2*n_cams, n_tracks))
    C[:] = np.nan
    
    C_v2 = np.zeros((n_cams, n_tracks))
    C_v2[:] = np.nan
    
    kp_i, kp_j = pairwise_matches[:, 0], pairwise_matches[:, 1]
    im_i, im_j = pairwise_matches[:, 2], pairwise_matches[:, 3]
    feature_i_id, feature_j_id = feature_ids[im_i, kp_i], feature_ids[im_j, kp_j]
    t_idx = track_indices[feature_i_id].astype(int)
    features_tmp = np.moveaxis(np.dstack(features), 2, 0)
    C[2*im_i  , t_idx] = features_tmp[im_i, kp_i, 0]
    C[2*im_i+1, t_idx] = features_tmp[im_i, kp_i, 1]
    C[2*im_j  , t_idx] = features_tmp[im_j, kp_j, 0]
    C[2*im_j+1, t_idx] = features_tmp[im_j, kp_j, 1]
    C_v2[im_i, t_idx] = kp_i
    C_v2[im_j, t_idx] = kp_j
    
    print('C.shape before baseline check {}'.format(C.shape))

    tracks_to_preserve = filter_C_using_pairs_to_triangulate(C, pairs_to_triangulate)
    C = C[:, tracks_to_preserve]
    C_v2 = C_v2[:, tracks_to_preserve]
    
    print('C.shape after baseline check {}'.format(C.shape))
    
    return C, C_v2

def corresp_matrix_from_tracks(feature_tracks, r):
    '''
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
    '''
    
    N = feature_tracks.shape[0]
    M = r.shape[1]
    C = np.zeros((2*M,N))
    C[:] = np.nan
    for i in range(N):
        im_ind = [k for k, j in enumerate(range(M)) if r[i,j]!=0]
        for ind in im_ind:
            C[ind*2:(ind*2)+2,i] = feature_tracks[i,:,ind]        
    return C  



def save_pts2d_as_svg(output_filename, pts2d, c, r=5, w=None, h=None):

    def boundaries_ok(col, row):
        return (col > 0 and col < w-1 and row > 0 and row < h-1)

    def svg_header(w,h):
        svg_header = '<?xml version="1.0" standalone="no"?>\n' + \
                     '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n' + \
                     ' "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n' + \
                     '<svg width="{}px" height="{}px" version="1.1"\n'.format(w,h) + \
                     ' xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">\n'
        return svg_header
    
    def svg_pt(col, row, color, pt_r, im_w=None, im_h=None):

        col, row = int(col), int(row)

        l1_x1, l1_y1, l1_x2, l1_y2 = col-pt_r, row-pt_r, col+pt_r, row+pt_r
        l2_x1, l2_y1, l2_x2, l2_y2 = col+pt_r, row-pt_r, col-pt_r, row+pt_r

        if (im_w is not None) and (im_h is not None):
            l1_boundaries_ok = boundaries_ok(l1_x1, l1_y1) and boundaries_ok(l1_x2, l1_y2)
            l2_boundaries_ok = boundaries_ok(l2_x1, l2_y1) and boundaries_ok(l2_x2, l2_y2)
        else:
            l1_boundaries_ok = True
            l2_boundaries_ok = True

        if l1_boundaries_ok and l2_boundaries_ok:
            l1_args = [l1_x1, l1_y1, l1_x2, l1_y2, color]
            l2_args = [l2_x1, l2_y1, l2_x2, l2_y2, color]
            svg_pt_str = '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="5" />\n'.format(*l1_args) + \
                         '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="5" />\n'.format(*l2_args)
        else:
            svg_pt_str = ''
        return svg_pt_str
    
    #write the svg
    f_svg= open(output_filename,"w+")
    f_svg.write(svg_header(w,h))
    for p_idx in range(pts2d.shape[0]):
        f_svg.write(svg_pt(pts2d[p_idx,0], pts2d[p_idx,1], pt_r=r, color=c, im_w=w, im_h=h))
    f_svg.write('</svg>')


def save_sequence_features_svg(output_dir, seq_fnames, seq_features):
    os.makedirs(output_dir, exist_ok=True)
    for fname, features in zip(seq_fnames, seq_features):
        f_id = get_fname_id(fname)
        save_pts2d_as_svg(os.path.join(output_dir,  f_id + '.svg'), features[:,:2], c='yellow')


def save_sequence_features_txt(output_dir, seq_fnames, seq_features, seq_features_utm=None):
    
    do_utm = seq_features_utm is not None
    n_img = len(seq_fnames)
    
    kps_dir = os.path.join(output_dir, 'kps')
    des_dir = os.path.join(output_dir, 'des')
    os.makedirs(kps_dir, exist_ok=True)
    os.makedirs(des_dir, exist_ok=True)
    
    if do_utm:
        utm_dir = os.path.join(output_dir, 'utm')
        os.makedirs(utm_dir, exist_ok=True)
                              
    for i in range(n_img):
        f_id = get_fname_id(seq_fnames[i])       
        np.savetxt(os.path.join(kps_dir,  f_id + '.txt'), seq_features[i][:,:4], fmt='%.6f')
        np.savetxt(os.path.join(des_dir,  f_id + '.txt'), seq_features[i][:,4:], fmt='%d')
        if do_utm:
            np.savetxt(os.path.join(utm_dir,  f_id + '.txt'), seq_features_utm[i], fmt='%.6f')


def init_feature_tracks_config(config=None):

    # initialize parameters
    keys = ['sift', 'relative_thr', 'absolute_thr', 'max_kp', 'K', 'K_priority',
            'use_masks', 'predefined_pairs', 'filter_pairs', 'continue', 'n_proc', 'compress']
    default_values = ['local', 0.6, 250, 60000, 0, ['length', 'scale', 'cost'],
                      False, None, True, False, 5, False]
    output_config = {}
    if config is not None:
        for v, k in zip(default_values, keys):
            output_config[k] = config[k] if k in config.keys() else v
    else:
        output_config = dict(zip(keys, default_values))

    return output_config
