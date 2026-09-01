import numpy as np
import networkx as nx
from collections import defaultdict
from .ft_pair_classifier import extract_DINO_embeddings, cosine_similarity_np

def build_top_m_neighbor_graph(n_images, pairs, scores, M):
    """
    Keep only top-M neighbors per node.
    """
    neigh = defaultdict(list)
    for (i, j), s in zip(pairs, scores):
        neigh[i].append((j, s))
        neigh[j].append((i, s))

    kept = set()
    for i in range(n_images):
        best = sorted(neigh[i], key=lambda x: x[1], reverse=True)[:M]
        for j, s in best:
            a, b = min(i, j), max(i, j)
            kept.add((a, b))

    kept_pairs = []
    kept_scores = []
    score_dict = {pair: s for pair, s in zip(pairs, scores)}
    for pair in kept:
        kept_pairs.append(pair)
        kept_scores.append(score_dict[pair])

    return kept_pairs, kept_scores

def select_k_spanning_trees(n_images, pairs, scores, K):
    """
    Greedy extraction of K edge-disjoint maximum spanning trees.

    Parameters
    ----------
    n_images : int
        Number of nodes/images.
    pairs : list[tuple[int, int]]
        Candidate edges.
    scores : list[float]
        Edge weights (larger = better).
    K : int
        Number of spanning trees to extract.

    Returns
    -------
    selected_pairs : list[tuple[int, int]]
        Union of edges across all extracted trees.
    trees : list[list[tuple[int, int]]]
        One list of edges per tree.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_images))
    for (i, j), s in zip(pairs, scores):
        G.add_edge(i, j, weight=float(s))

    selected_pairs = []
    trees = []

    for _ in range(K):
        if not nx.is_connected(G):
            break

        T = nx.maximum_spanning_tree(G, weight="weight")
        tree_edges = list(T.edges())

        # Safety: ensure it is a full spanning tree
        if len(tree_edges) != n_images - 1:
            break

        trees.append(tree_edges)
        selected_pairs.extend(tree_edges)
        G.remove_edges_from(tree_edges)



    return selected_pairs, trees


def select_optimal_pairs_to_match(pairs_to_match, images, K=10, M=10, sparsify=False):
    '''
    This algorithim selects the optimal pairs to match based on pair similarity score

    For each pair, the similarity score is computed as the cosine similarity between DINO embeddings
    
    pairs_to_match: list of tuples with the image indexes of each pair e.g. [(0, 1), (2, 3), (1, 3)]
    images:         list of cam_utils.SatelliteImage instances
    K:              int, max number of spanning trees connecting all cameras.
    M:              int, for each image, preserve only pairs corresponding to the M top-similar neighbors
    sparsify:       bool, indicating if M will be used. If False, all neighbors per image are considered
    '''

    # Part 1 - Precompte image embeddings just once
    n_images = len(images)
    embeddings = extract_DINO_embeddings(images)

    # Part 2 - Extract similairty scores for each input pair
    scores = []
    for pair in pairs_to_match:
        img_idx_i, img_idx_j = pair
        score = cosine_similarity_np(embeddings[img_idx_i], embeddings[img_idx_j])
        scores.append(score)

    # Part 3 - Pick top-similar pairs connecting all cameras K times
    if sparsify:
        M = max(K, M) # to prevent sparsifying too much
        sparse_pairs, sparse_scores = build_top_m_neighbor_graph(
            n_images, pairs_to_match, scores, M
        )
    else:
        sparse_pairs = pairs_to_match
        sparse_scores = scores

    selected_pairs, trees = select_k_spanning_trees(
        n_images=n_images,
        pairs=sparse_pairs,
        scores=sparse_scores,
        K=K,
    )

    print(f"Found {len(trees)} trees of pairs connecting all cameras")
    print(f"Selected {len(selected_pairs)} pairs")
    #for k, tree in enumerate(trees):
    #    print(f"Tree {k+1}: {len(tree)} edges")

    ratio = len(selected_pairs)/len(pairs_to_match)
    print(f"Original number of pairs_to_match: {len(pairs_to_match)}")
    print(f"Number of pairs_to_match was reduced to {ratio*100:.2f}%")

    return selected_pairs
