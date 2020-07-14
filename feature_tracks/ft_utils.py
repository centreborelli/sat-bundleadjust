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

def plot_track_observations_stereo_pair(i, j, C, input_seq):
    
    # i, j : indices of the images
    
    visible_idx = np.logical_and(~np.isnan(C[i*2,:]), ~np.isnan(C[j*2,:])) 
    pts1, pts2 = C[(i*2):(i*2+2), visible_idx], C[(j*2):(j*2+2), visible_idx]
    pts1, pts2 = pts1.T, pts2.T
    
    print('{} track observations to display for pair ({},{})'.format(pts1.shape, i, j))
    
    print('List of track indices: {}'.format(np.arange(C.shape[1])[visible_idx]))
    
    if pts1.shape[0] > 0:
    
        fig = plt.figure(figsize=(20,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow((input_seq[i]), cmap="gray")
        ax2.imshow((input_seq[j]), cmap="gray")

        ax1.scatter(x=pts1[:,0], y=pts1[:,1], c='r', s=40)
        ax2.scatter(x=pts2[:,0], y=pts2[:,1], c='r', s=40)
        plt.show()
   

        
def plot_pairwise_matches_stereo_pair(i, j, features, pairwise_matches, input_seq):
    
    # i, j : indices of the images
    
    pairwise_matches_kp_indices = pairwise_matches[:,:2]
    pairwise_matches_im_indices = pairwise_matches[:,2:]
    
    true_where_matches = np.all(pairwise_matches_im_indices == np.array([i, j]), axis=1)
    matched_kps_i = features[i][pairwise_matches_kp_indices[true_where_matches,0]]
    matched_kps_j = features[j][pairwise_matches_kp_indices[true_where_matches,1]]
    
    print('{} pairwise matches to display for pair ({},{})'.format(matched_kps_i.shape[0], i, j))
    
    if matched_kps_i.shape[0] > 0:
    
        fig = plt.figure(figsize=(20,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow((input_seq[i]), cmap="gray")
        ax2.imshow((input_seq[j]), cmap="gray")

        ax1.scatter(x=matched_kps_i[:,0], y=matched_kps_i[:,1], c='r', s=40)
        ax2.scatter(x=matched_kps_j[:,0], y=matched_kps_j[:,1], c='r', s=40)
        plt.show()

    
def feature_tracks_from_pairwise_matches(features, pairwise_matches, pairs_to_triangulate, indices_img_global):

    '''
    TO DO:
    This function has a drawback: everytime we build the feature tracks we load ALL features of the scene
    and ALL pairwise matches. When a reasonable amount of images is used this will be fine but for 1000 
    images the computation time may increase in a relevant way.
    The solution would be to save matches separately per image and directly load those of interest 
    (instead of filtering them from all_pairwise_matches).
    '''
    
    n_cams_in_use = len(indices_img_global)
    total_cams = len(features)

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
    true_where_im_in_use = np.zeros(total_cams).astype(bool)
    true_where_im_in_use[indices_img_global] = True
    true_where_match_in_use = np.logical_or(true_where_im_in_use[pairwise_matches[:,2]],
                                             true_where_im_in_use[pairwise_matches[:,3]])
    pairwise_matches_of_interest = pairwise_matches[true_where_match_in_use]

    matched_features_kp = np.hstack((pairwise_matches_of_interest[:,0], pairwise_matches_of_interest[:,1]))
    matched_features_im = np.hstack((pairwise_matches_of_interest[:,2], pairwise_matches_of_interest[:,3]))
    feature_ids_in_use = np.unique(feature_ids[matched_features_im, matched_features_kp]).tolist()
    pairwise_matches_of_interest = pairwise_matches_of_interest.tolist()
    
    # associate a track index to each feature id
    feature_ids_to_t_idx  = -1 * np.ones(np.prod(feature_ids.shape))
    feature_ids_to_t_idx[feature_ids_in_use] = np.arange(len(feature_ids_in_use)).astype(int)
    feature_ids_to_t_idx = feature_ids_to_t_idx.astype(int)
    
    parents = [None]*(len(feature_ids_in_use))
    for kp_i, kp_j, im_i, im_j in pairwise_matches_of_interest:
        feature_i_id, feature_j_id = feature_ids[im_i, kp_i], feature_ids[im_j, kp_j]
        union(parents, feature_ids_to_t_idx[feature_i_id], feature_ids_to_t_idx[feature_j_id])
        
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
    C = np.zeros((2*n_cams_in_use, n_tracks))
    C[:] = np.nan
    
    global_idx_to_local_idx = -1 * np.ones(total_cams)
    global_idx_to_local_idx[indices_img_global] = np.arange(n_cams_in_use)
    global_idx_to_local_idx = global_idx_to_local_idx.astype(int)
            
    pairwise_matches_for_C = pairwise_matches[np.logical_and(true_where_im_in_use[pairwise_matches[:,2]],
                                                             true_where_im_in_use[pairwise_matches[:,3]])]
    kp_i, kp_j = pairwise_matches_for_C[:,0], pairwise_matches_for_C[:,1]
    im_i, im_j = pairwise_matches_for_C[:,2], pairwise_matches_for_C[:,3]
    feature_i_id, feature_j_id = feature_ids[im_i, kp_i], feature_ids[im_j, kp_j]
    t_idx = track_indices[feature_ids_to_t_idx[feature_i_id]].astype(int)
    im_idx_i = global_idx_to_local_idx[im_i]
    im_idx_j = global_idx_to_local_idx[im_j]
    features_tmp = np.moveaxis(np.dstack(features), 2, 0)
    C[2*im_idx_i  , t_idx] = features_tmp[im_i,kp_i, 0]
    C[2*im_idx_i+1, t_idx] = features_tmp[im_i,kp_i, 1]
    C[2*im_idx_j  , t_idx] = features_tmp[im_j,kp_j, 0]
    C[2*im_idx_j+1, t_idx] = features_tmp[im_j,kp_j, 1]
    
    # remove matches found in pairs with short baseline that were not extended to more images
    # since these columns of C will not be triangulated
    # ATTENTION: this is very slow in comparison to the rest of the function 
    # it can take various seconds while the rest is instantaneous, optimize it in the future
    columns_to_preserve = []
    for i in range(C.shape[1]):
        im_ind = [k for k, j in enumerate(range(n_cams_in_use)) if not np.isnan(C[j*2,i])]
        all_pairs = [(im_i, im_j) for im_i in im_ind for im_j in im_ind if im_i != im_j and im_i<im_j]
        good_pairs = [pair for pair in all_pairs if pair in pairs_to_triangulate]
        columns_to_preserve.append( len(good_pairs) > 0 )
    C = C[:, columns_to_preserve]  
    
    return C

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


def save_sequence_features_svg(output_dir, seq_fnames, seq_features, seq_id=None):
    
    subdir = 'sift/{}'.format(seq_id) if seq_id is not None else 'sift'
    svg_dir = os.path.join(output_dir, os.path.join(subdir, 'svg'))
    img_dir = os.path.join(output_dir, os.path.join(subdir, 'img'))
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    for fname, features in zip(seq_fnames, seq_features):
        f_id = get_fname_id(fname)
        save_pts2d_as_svg(os.path.join(svg_dir,  f_id + '.svg'), features[:,:2], c='yellow')
        os.system('cp {} {}'.format(fname, os.path.join(img_dir, os.path.basename(fname))))
    
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
    
    
