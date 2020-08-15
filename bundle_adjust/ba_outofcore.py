import numpy as np
import os

from bundle_adjust import ba_timeseries as ba_t
from bundle_adjust import ba_utils
from feature_tracks import ft_sat

def get_base_pair_current_date(timeline, ba_dir, idx_current_date, idx_prev_date, prev_base_pair):
    
    '''
    given the base node of a precedent date, this function chooses the most suitable base node of the current date
    considering triangulation and connectivity criteria
    '''
    
    verbose=False
    P_dir = os.path.join(ba_dir, 'P_adj')
    rpc_dir = os.path.join(ba_dir, 'RPC_adj')

    #
    # decide the base_node of the current base_pair
    #
    
    fnames_prev_base_pair = (np.array(timeline[idx_prev_date]['fnames'])[np.array(prev_base_pair)]).tolist()
    P_prev_base_pair = ba_t.load_matrices_from_dir(fnames_prev_base_pair, P_dir, suffix='pinhole_adj', verbose=verbose)
    rpcs_prev_base_pair = ba_t.load_rpcs_from_dir(fnames_prev_base_pair, rpc_dir, suffix='RPC_adj', verbose=verbose)
    offsets_prev_base_pair = ba_t.load_offsets_from_dir(fnames_prev_base_pair, P_dir, suffix='pinhole_adj', verbose=verbose)
    footprints_prev_base_pair = ba_utils.get_image_footprints(rpcs_prev_base_pair, offsets_prev_base_pair)
    
    # choose the best base node from date 2 according to the choice made for date 1
    
    fnames_current_date = timeline[idx_current_date]['fnames']
    P_current_date = ba_t.load_matrices_from_dir(fnames_current_date, P_dir, suffix='pinhole_adj', verbose=verbose)
    rpcs_current_date = ba_t.load_rpcs_from_dir(fnames_current_date, rpc_dir, suffix='RPC_adj', verbose=verbose)
    offsets_current_date = ba_t.load_offsets_from_dir(fnames_current_date, P_dir, suffix='pinhole_adj', verbose=verbose)
    footprints_current_date = ba_utils.get_image_footprints(rpcs_current_date, offsets_current_date)
    
    
    projection_matrices = P_prev_base_pair + P_current_date
    rpcs = rpcs_prev_base_pair + rpcs_current_date
    footprints = footprints_prev_base_pair + footprints_current_date

    n_base_nodes = len(fnames_prev_base_pair)
    n_img_current_date = len(P_current_date)
    #init pairs can be each of the nodes in the base pair with prev date nodes
    init_pairs = []
    for i in np.arange(n_base_nodes):
        for j in np.arange(n_base_nodes, n_base_nodes + n_img_current_date):
            init_pairs.append((i, j))
            
    _, T = ft_sat.compute_pairs_to_match(init_pairs, footprints, projection_matrices, verbose=verbose)
    # where T are the pairs to triangulate between the previous base pair and the current date

    T = np.array(T)
    current_base_node_candidates = np.unique(T[T[:,0]==0,1]) - 2
    
    # how to choose a candidate?
    #  - take the candidate with the highest weight
    current_weights = np.array(timeline[idx_current_date]['image_weights'])
    idx_current_base_node = current_base_node_candidates[np.argmax(current_weights[current_base_node_candidates])]
    current_auxiliary_node_candidates = current_base_node_candidates.tolist()
    current_auxiliary_node_candidates.remove(idx_current_base_node)
    
    #
    # decide the auxiliary node of the current base_pair
    #
    base_pair_candidates = [p for p in timeline[idx_current_date]['pairs_to_triangulate'] if idx_current_base_node in p]
    auxiliary_candidates_array = np.unique(np.array(base_pair_candidates).flatten())
    auxiliary_candidates = auxiliary_candidates_array.copy().tolist()
    auxiliary_candidates.remove(idx_current_base_node)
    
    auxiliary_node_found = False
    while len(auxiliary_candidates) > 0 and not auxiliary_node_found:
        
        auxiliary_node = auxiliary_candidates[np.argmax(current_weights[auxiliary_candidates])]
        if auxiliary_node in current_auxiliary_node_candidates:
            base_pair = (idx_current_base_node, auxiliary_node)
            auxiliary_node_found = True
        else:
            auxiliary_candidates.remove(auxiliary_node)
        
    if not auxiliary_node_found:
        auxiliary_candidates = auxiliary_candidates_array.copy().tolist()
        auxiliary_candidates.remove(idx_current_base_node)
        auxiliary_node = auxiliary_candidates[np.argmax(current_weights[auxiliary_candidates])]
        base_pair = (idx_current_base_node, auxiliary_node)
        
    return base_pair


def get_base_pairs_complete_sequence(timeline, timeline_indices, ba_dir):
    
    n_dates = len(timeline_indices)
    base_pairs = []
    
    # initial base pair (base node is given by the max image weight,
    # auxiliary node is given by max weight out of all those image that are well posed for triangulation with the base node)
    image_weights = timeline[timeline_indices[0]]['image_weights']
    base_node_idx = np.argmax(image_weights)
    base_pair_candidates = [p for p in timeline[timeline_indices[0]]['pairs_to_triangulate'] if base_node_idx in p]
    base_pair_idx = np.argmax([image_weights[p[0]] + image_weights[p[1]] for p in base_pair_candidates])
    base_pair = np.array(base_pair_candidates[base_pair_idx])
    base_pair = (base_node_idx, base_pair[base_pair != base_node_idx][0]) 
    
    base_pairs.append(base_pair)
    
    for i in np.arange(1, n_dates):
        idx_current_date = timeline_indices[i]
        idx_prev_date = timeline_indices[i-1]
        x = get_base_pair_current_date(timeline, ba_dir, idx_current_date, idx_prev_date, base_pairs[-1])
        base_pairs.append(x)
    
    return base_pairs