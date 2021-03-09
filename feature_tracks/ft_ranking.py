import numpy as np
from bundle_adjust import ba_core
import timeit

def build_connectivity_matrix(C, min_matches=10):

    '''
    the connectivity matrix A is a matrix with size NxN, where N is the numbe of cameras
    the value at posiition (i,j) is equal to the amount of matches found between image i and image j
    '''
    n_cam = int(C.shape[0]/2)
    A = np.zeros((n_cam, n_cam))
    for im1 in range(n_cam):
        for im2 in range(im1+1, n_cam):
            A[im1, im2] = np.sum(1*np.logical_and(1*~np.isnan(C[2*im1, :]), 1*~np.isnan(C[2*im2, :])))
            A[im2, im1] = A[im1, im2]
    A[A < min_matches] = 0
    return A


def compute_C_scale(C_v2, features):

    # C_scale is similar to C, but instead of having shape (2*n_cam)x(n_tracks) it has shape (n_cam)x(n_tracks)
    # where each slot contains the scale of the track observation associated, else nan

    C_scale = C_v2.copy()
    n_cam = C_v2.shape[0]
    for cam_idx in range(n_cam):
        where_obs_current_cam = ~np.isnan(C_v2[cam_idx, :])
        kp_indices = C_v2[cam_idx, where_obs_current_cam]
        kp_scales = features[cam_idx][kp_indices.astype(int), 2]
        C_scale[cam_idx, where_obs_current_cam] = kp_scales

    return C_scale


def compute_C_reproj(C, pts3d, cameras, cam_model, pairs_to_triangulate, camera_centers):

    # C_reproj is similar to C, but instead of having shape (2*n_cam)x(n_tracks) it has shape (n_cam)x(n_tracks)
    # where each slot contains the reprojection error of the track observation associated, else nan

    # set ba parameters
    from bundle_adjust.ba_params import BundleAdjustmentParameters
    p = BundleAdjustmentParameters(C, pts3d, cameras, cam_model, pairs_to_triangulate, camera_centers,
                                   n_cam_fix=0, n_pts_fix=0, reduce=False, verbose=False)

    # compute reprojection error at the initial parameters
    reprojection_err_per_obs = ba_core.compute_reprojection_error(ba_core.fun(p.params_opt.copy(), p))
    
    # create the equivalent of C but fill the slot of each observation with the corresponding reprojection error
    n_cam, n_pts = int(C.shape[0] / 2), int(C.shape[1])
    C_reproj = np.zeros((n_cam, n_pts))
    C_reproj[:] = np.nan
    for i, err in enumerate(reprojection_err_per_obs):
        C_reproj[p.cam_ind[i], p.pts_ind[i]] = err
    
    return C_reproj


def compute_camera_weights(C, C_reproj, connectivity_matrix=None):
    
    n_cam = int(C.shape[0]/2)
    
    if connectivity_matrix is None:
        A = build_connectivity_matrix(C)
    else:
        A = connectivity_matrix
    
    w_cam = []
    for i in range(n_cam):
    
        nC_i = np.sum(A[i, :]>0) 
        
        if nC_i > 0:
            indices_of_tracks_seen_in_current_cam = np.arange(C.shape[1])[~np.isnan(C[i*2,:])]
            
            # reprojection error of all tracks in the current cam
            #reproj_err_current_cam = C_reproj[i, indices_of_tracks_seen_in_current_cam]
            #avg_cost = np.mean(reproj_err_current_cam)
            #std_cost = np.std(reproj_err_current_cam)
            
            #mean and std of the average reprojection error of the tracks seen in the current camera
            avg_reproj_err_tracks_seen = np.nanmean(C_reproj[:, indices_of_tracks_seen_in_current_cam], axis=0)
            avg_cost = np.mean(avg_reproj_err_tracks_seen)
            std_cost = np.std(avg_reproj_err_tracks_seen)
            
            costC_i = avg_cost + 3. * std_cost
            
        else:
            costC_i = 0.
    
        w_cam.append( float(nC_i) + np.exp( - costC_i ) )
   
    
    return w_cam


def order_tracks(C, C_scale, C_reproj, priority=['length', 'scale', 'cost'], verbose=False):

    n_tracks = C.shape[1]
    tracks_length = (np.sum(~np.isnan(C), axis=0)/2).astype(int)
    tracks_scale = np.round(np.nanmean(C_scale, axis=0), 2).astype(float)
    tracks_cost = np.nanmean(C_reproj, axis=0).astype(float)

    tracks_dtype = [('length', int), ('scale', float), ('cost', float)]
    track_values = np.array(list(zip(tracks_length, -tracks_scale, -tracks_cost)), dtype=tracks_dtype)
    ranked_track_indices = dict(list(zip(np.argsort(track_values, order=priority)[::-1], np.arange(n_tracks))))
    '''
    ranked_track_indices is a dict
    key = index of track in C
    value = position in track ranking
    '''

    if verbose:
        print('\nTRACK RANKING (first 20):')
        sorted_indices = sorted(np.arange(n_tracks), key=lambda idx: ranked_track_indices[idx])
        for i in range(20):
            track_idx = sorted_indices[i]
            length, scale, cost = tracks_length[track_idx], tracks_scale[track_idx], tracks_cost[track_idx]
            print('rank position: {:3}, length: {:2}, scale: {:.3f}, cost: {:.3f}'.format(i, length, scale, cost))
        print('\n')

    return ranked_track_indices


def get_inverted_track_list(C, ranked_track_indices):

    inverted_track_list = {}
    n_cam = int(C.shape[0]/2)
    for i in range(n_cam):
        indices_of_tracks_seen_in_current_cam = np.arange(C.shape[1])[~np.isnan(C[i*2,:])]
        #print('cam:', i, ', tracks:', len(indices_of_tracks_seen_in_current_cam))
        inverted_track_list[i] = sorted(indices_of_tracks_seen_in_current_cam, key=lambda idx: ranked_track_indices[idx])
        
    return inverted_track_list


def select_best_tracks(C, C_scale, C_reproj, K=30, priority=['length', 'scale', 'cost'], verbose=False):

    """
    Tracks selection for robust, efficient and scalable large-scale structure from motion
    H Cui, Pattern Recognition (2017)
    """

    super_verbose = False
    start = timeit.default_timer()
    
    n_cam = int(C.shape[0]/2)
    V = set(np.arange(n_cam).tolist())  # all cam nodes
    cam_indices = np.arange(n_cam)

    ranked_track_indices = order_tracks(C, C_scale, C_reproj, priority=priority)
    remaining_T = np.arange(C.shape[1])
    T = np.arange(C.shape[1])
    
    k = 0
    S = []  # subset of track indices selected
    
    updated_C = C.copy()

    while k < K and len(S) < len(T):

        tracks_already_selected = list(set(T) - set(remaining_T))
        updated_C[:, tracks_already_selected] = np.nan
        
        if super_verbose and k > 0:
            tracks_cost = np.nanmean(C_reproj[:, tracks_already_selected], axis=0)
            avg_reproj_err = np.mean(tracks_cost)
            args = [k, len(tracks_already_selected), avg_reproj_err]
            print('k: {:02}   |   selected tracks: {:03}   |   avg reproj err: {:.3f}:'.format(*args))

        A = build_connectivity_matrix(updated_C, min_matches=0)
        inverted_track_list = get_inverted_track_list(updated_C, ranked_track_indices)
        
        cam_weights = np.array(compute_camera_weights(updated_C, C_reproj, connectivity_matrix=A))
        Croot = np.argmax(cam_weights)
        l, nodes_last_layer_Hk = 0, [Croot]
        Sk, Ik = [], set(nodes_last_layer_Hk)
        if super_verbose:
            print('      l: {}, nodes: {}, Ik: {}'.format(l, nodes_last_layer_Hk, Ik))

        iterate_current_tree = True
        while iterate_current_tree:
            nodes_next_layer_Hk = []
            for cam_idx in nodes_last_layer_Hk:
                for track_idx in inverted_track_list[cam_idx]:
                    if track_idx not in Sk:
                        Wq = set(cam_indices[~np.isnan(C[::2, track_idx])].tolist())  # visible_cams_track_idx
                        Rq = set(cam_indices[A[cam_idx, :] > 0].tolist())  # neighbor_cams_cam_idx
                        Zq = Wq.intersection(Rq)
                        if not Zq.issubset(Ik) and len(Zq) > 0:
                            nodes_next_layer_Hk.extend(list(Zq - Zq.intersection(Ik)))
                            Sk.extend([track_idx])
                            Ik = Ik.union(Zq)
            l += 1
            if len(V - Ik) == 0 or len(nodes_next_layer_Hk) == 0:
                iterate_current_tree = False
            else:
                nodes_next_layer_Hk = np.array(nodes_next_layer_Hk)
                sorted_node_indices = np.argsort(cam_weights[nodes_next_layer_Hk])[::-1]
                nodes_last_layer_Hk = nodes_next_layer_Hk[sorted_node_indices].tolist()
            if super_verbose:
                print('      l: {}, nodes: {}, Ik: {}'.format(l, nodes_last_layer_Hk, Ik))
        k += 1
        remaining_T = list(set(remaining_T) - set(Sk))
        S.extend(Sk)

    if verbose:
        count_obs_per_cam = lambda C: np.sum(1 * ~np.isnan(C), axis=1)[::2]
        n_tracks_out, n_tracks_in = len(S), C.shape[1]
        args = [n_tracks_out, n_tracks_in, (float(n_tracks_out)/n_tracks_in)*100., timeit.default_timer() - start]
        print('\nSelected {} tracks out of {} ({:.2f}%) in {:.2f} seconds'.format(*args))
        print('     - priority: {}'.format(priority))
        print('     - obs per cam before: {}'.format(count_obs_per_cam(C)))
        print('     - obs per cam after:  {}\n'.format(count_obs_per_cam(C[:, S])))
    return np.array(S)
