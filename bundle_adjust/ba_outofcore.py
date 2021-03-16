import os

import numpy as np

from bundle_adjust import ba_timeseries as ba_t
from bundle_adjust import ba_utils

from .feature_tracks import ft_sat


def get_base_pair_current_date(
    timeline, ba_dir, idx_current_date, idx_prev_date, prev_base_pair
):

    """
    given the base node of a precedent date, this function chooses the most suitable base node of the current date
    considering triangulation and connectivity criteria
    """

    verbose = False
    P_dir = os.path.join(ba_dir, "P_adj")
    rpc_dir = os.path.join(ba_dir, "RPC_adj")

    #
    # decide the base_node of the current base_pair
    #

    fnames_prev_base_pair = (
        np.array(timeline[idx_prev_date]["fnames"])[np.array(prev_base_pair)]
    ).tolist()
    P_prev_base_pair = ba_t.load_matrices_from_dir(
        fnames_prev_base_pair, P_dir, suffix="pinhole_adj", verbose=verbose
    )
    rpcs_prev_base_pair = ba_t.load_rpcs_from_dir(
        fnames_prev_base_pair, rpc_dir, suffix="RPC_adj", verbose=verbose
    )
    offsets_prev_base_pair = ba_t.load_offsets_from_dir(
        fnames_prev_base_pair, P_dir, suffix="pinhole_adj", verbose=verbose
    )
    footprints_prev_base_pair = ba_utils.get_image_footprints(
        rpcs_prev_base_pair, offsets_prev_base_pair
    )

    # choose the best base node from date 2 according to the choice made for date 1

    fnames_current_date = timeline[idx_current_date]["fnames"]
    P_current_date = ba_t.load_matrices_from_dir(
        fnames_current_date, P_dir, suffix="pinhole_adj", verbose=verbose
    )
    rpcs_current_date = ba_t.load_rpcs_from_dir(
        fnames_current_date, rpc_dir, suffix="RPC_adj", verbose=verbose
    )
    offsets_current_date = ba_t.load_offsets_from_dir(
        fnames_current_date, P_dir, suffix="pinhole_adj", verbose=verbose
    )
    footprints_current_date = ba_utils.get_image_footprints(
        rpcs_current_date, offsets_current_date
    )

    projection_matrices = P_prev_base_pair + P_current_date
    rpcs = rpcs_prev_base_pair + rpcs_current_date
    footprints = footprints_prev_base_pair + footprints_current_date

    n_base_nodes = len(fnames_prev_base_pair)
    n_img_current_date = len(P_current_date)
    # init pairs can be each of the nodes in the base pair with prev date nodes
    init_pairs = []
    for i in np.arange(n_base_nodes):
        for j in np.arange(n_base_nodes, n_base_nodes + n_img_current_date):
            init_pairs.append((i, j))

    _, T = ft_sat.compute_pairs_to_match(
        init_pairs, footprints, projection_matrices, verbose=verbose
    )
    # where T are the pairs to triangulate between the previous base pair and the current date

    T = np.array(T)
    current_base_node_candidates = np.unique(T[T[:, 0] == 0, 1]) - 2

    # how to choose a candidate?
    #  - take the candidate with the highest weight
    current_weights = np.array(timeline[idx_current_date]["image_weights"])
    idx_current_base_node = current_base_node_candidates[
        np.argmax(current_weights[current_base_node_candidates])
    ]
    current_auxiliary_node_candidates = current_base_node_candidates.tolist()
    current_auxiliary_node_candidates.remove(idx_current_base_node)

    #
    # decide the auxiliary node of the current base_pair
    #
    base_pair_candidates = [
        p
        for p in timeline[idx_current_date]["pairs_to_triangulate"]
        if idx_current_base_node in p
    ]
    auxiliary_candidates_array = np.unique(np.array(base_pair_candidates).flatten())
    auxiliary_candidates = auxiliary_candidates_array.copy().tolist()
    auxiliary_candidates.remove(idx_current_base_node)

    auxiliary_node_found = False
    while len(auxiliary_candidates) > 0 and not auxiliary_node_found:

        auxiliary_node = auxiliary_candidates[
            np.argmax(current_weights[auxiliary_candidates])
        ]
        if auxiliary_node in current_auxiliary_node_candidates:
            base_pair = (idx_current_base_node, auxiliary_node)
            auxiliary_node_found = True
        else:
            auxiliary_candidates.remove(auxiliary_node)

    if not auxiliary_node_found:
        auxiliary_candidates = auxiliary_candidates_array.copy().tolist()
        auxiliary_candidates.remove(idx_current_base_node)
        auxiliary_node = auxiliary_candidates[
            np.argmax(current_weights[auxiliary_candidates])
        ]
        base_pair = (idx_current_base_node, auxiliary_node)

    return base_pair


def get_base_pairs_complete_sequence(timeline, timeline_indices, ba_dir):

    n_dates = len(timeline_indices)
    base_pairs = []

    # initial base pair (base node is given by the max image weight,
    # auxiliary node is given by max weight out of all those image that are well posed for triangulation with the base node)
    image_weights = timeline[timeline_indices[0]]["image_weights"]
    base_node_idx = np.argmax(image_weights)
    base_pair_candidates = [
        p
        for p in timeline[timeline_indices[0]]["pairs_to_triangulate"]
        if base_node_idx in p
    ]
    base_pair_idx = np.argmax(
        [image_weights[p[0]] + image_weights[p[1]] for p in base_pair_candidates]
    )
    base_pair = np.array(base_pair_candidates[base_pair_idx])
    base_pair = (base_node_idx, base_pair[base_pair != base_node_idx][0])

    base_pairs.append(base_pair)

    for i in np.arange(1, n_dates):
        idx_current_date = timeline_indices[i]
        idx_prev_date = timeline_indices[i - 1]
        x = get_base_pair_current_date(
            timeline, ba_dir, idx_current_date, idx_prev_date, base_pairs[-1]
        )
        base_pairs.append(x)

    return base_pairs


def run_out_of_core_bundle_adjustment(
    self,
    timeline_indices,
    reset=False,
    verbose=True,
    parallelize=True,
    tie_points=False,
):

    if parallelize:
        verbose = False

    import pickle
    import timeit
    from multiprocessing import Pool

    if reset:
        self.reset_ba_params("out-of-core")
    self.print_ba_headline(timeline_indices)

    print(
        "########################\n Running out of core BA \n########################\n"
    )
    abs_start = timeit.default_timer()

    ###############
    # local sweep
    ###############

    ba_dir = os.path.join(self.dst_dir, "ba_out-of-core")
    os.makedirs(ba_dir, exist_ok=True)

    all_filenames = []
    for t_idx in timeline_indices:
        all_filenames.extend(self.timeline[t_idx]["fnames"])
    pickle_out = open(ba_dir + "/filenames.pickle", "wb")
    pickle.dump(all_filenames, pickle_out)
    pickle_out.close()

    local_sweep_args = [
        (
            [t_idx],
            ba_dir,
            ba_dir,
            0,
            None,
            True,
            self.tracks_config,
            self.tracks_config["verbose local"],
            True,
        )
        for t_idx in timeline_indices
    ]

    time_per_date = []

    # base_local = []
    local_output = []
    start = timeit.default_timer()
    if parallelize:
        # with Pool(processes=2, maxtasksperchild=1000) as p:
        with Pool() as p:
            local_output = p.starmap(self.bundle_adjust, local_sweep_args)
    else:
        for idx, t_idx in enumerate(timeline_indices):
            (
                running_time,
                n_tracks,
                ba_e,
                init_e,
                im_w,
                p_triangulate,
            ) = self.bundle_adjust(*local_sweep_args[idx])
            local_output.append(
                [running_time, n_tracks, ba_e, init_e, im_w, p_triangulate]
            )

    stop = timeit.default_timer()
    total_time = int(stop - start)

    for idx, t_idx in enumerate(timeline_indices):
        ba_e, init_e, n_tracks = (
            local_output[idx][2],
            local_output[idx][3],
            local_output[idx][1],
        )
        print(
            "({}) {}, {}, ({}, {}) ".format(
                idx + 1, self.timeline[t_idx]["datetime"], n_tracks, init_e, ba_e
            )
        )

        self.timeline[t_idx]["image_weights"] = local_output[idx][4]
        self.timeline[t_idx]["pairs_to_triangulate"] = local_output[idx][5]

        # self.timeline[t_idx]['base_node'] = self.timeline[t_idx]['fnames'][np.argmax(iw)]
        # self.timeline[t_idx]['base_pair'] = bp

        # im_dir = '{}/images/{}'.format(self.dst_dir, self.timeline[t_idx]['id'])
        # os.makedirs(im_dir, exist_ok=True)
        # for fn in self.timeline[t_idx]['fnames']:
        #    os.system('cp {} {}'.format(fn, os.path.join(im_dir, os.path.basename(fn))))

        # print('image weights', self.timeline[t_idx]['image_weights'])
        # print('base node', self.timeline[t_idx]['base_node'])
        # print('base pair', self.timeline[t_idx]['base_pair'])

        # base_pair_fnames = [self.timeline[t_idx]['fnames'][bp[0]]] + [self.timeline[t_idx]['fnames'][bp[1]]]
        # P_dir = os.path.join(ba_dir, 'P_adj')
        # base_local.extend(load_matrices_from_dir(base_pair_fnames, P_dir, suffix='pinhole_adj', verbose=verbose))

    print("\n##############################################################")
    print(
        "- Local sweep done in {} seconds (avg per date: {:.2f} s)".format(
            total_time, total_time / len(timeline_indices)
        )
    )
    print("##############################################################\n")

    ##### get base nodes for the current consecutive dates

    base_pair_per_date = ba_outofcore.get_base_pairs_complete_sequence(
        self.timeline, timeline_indices, ba_dir
    )

    ##### express camera positions in local submaps in terms of the corresponding base node
    relative_poses_dir = "{}/relative_local_poses".format(ba_dir)
    os.makedirs(relative_poses_dir, exist_ok=True)
    for t_idx, bp in zip(timeline_indices, base_pair_per_date):
        P_dir = os.path.join(ba_dir, "P_adj")
        P_crop_ba = loader.load_matrices_from_dir(
            self.timeline[t_idx]["fnames"], P_dir, suffix="pinhole_adj", verbose=verbose
        )
        for P1, fn in zip(P_crop_ba, self.timeline[t_idx]["fnames"]):
            P_relative = (
                camera_utils.compute_relative_motion_between_projection_matrices(
                    P1, P_crop_ba[bp[0]]
                )
            )
            f_id = os.path.splitext(os.path.basename(fn))[0]
            np.savetxt(
                os.path.join(relative_poses_dir, f_id + ".txt"), P_relative, fmt="%.6f"
            )

    ###############
    # global sweep
    ###############

    start = timeit.default_timer()

    self.n_adj = 0
    self.myimages_adj = []
    self.myrpcs_adj = []
    self.mycrops_adj = []

    self.myimages_new = []
    self.myrpcs_new = []
    rpc_dir = os.path.join(ba_dir, "RPC_adj")
    for t_idx, bp in zip(timeline_indices, base_pair_per_date):
        base_fnames = (np.array(self.timeline[t_idx]["fnames"])[np.array(bp)]).tolist()
        self.myimages_new.extend(base_fnames)
        self.myrpcs_new.extend(
            loader.load_rpcs_from_dir(
                base_fnames, rpc_dir, suffix="RPC_adj", verbose=verbose
            )
        )
    self.n_new = len(self.myimages_new)
    self.mycrops_new = loader.load_image_crops(
        self.myimages_new,
        get_aoi_mask=self.compute_aoi_masks,
        rpcs=self.myrpcs_new,
        aoi=self.aoi_lonlat,
        use_mask_for_equalization=self.use_aoi_equalization,
        verbose=verbose,
    )

    self.ba_input_data = {}
    self.ba_input_data["input_dir"] = ba_dir
    self.ba_input_data["output_dir"] = ba_dir
    self.ba_input_data["n_new"] = self.n_new
    self.ba_input_data["n_adj"] = self.n_adj
    self.ba_input_data["image_fnames"] = self.myimages_adj + self.myimages_new
    self.ba_input_data["crops"] = self.mycrops_adj + self.mycrops_new
    self.ba_input_data["rpcs"] = self.myrpcs_adj + self.myrpcs_new
    self.ba_input_data["cam_model"] = self.cam_model
    self.ba_input_data["aoi"] = self.aoi_lonlat

    if self.compute_aoi_masks:
        self.ba_input_data["masks"] = [f["mask"] for f in self.mycrops_adj] + [
            f["mask"] for f in self.mycrops_new
        ]
    else:
        self.ba_input_data["masks"] = None

    running_time, n_tracks, ba_e, init_e = self.bundle_adjust(
        timeline_indices,
        ba_dir,
        ba_dir,
        0,
        self.ba_input_data,
        True,
        self.tracks_config,
        verbose,
        False,
    )

    # base_global = [P for P in self.ba_pipeline.P_crop_ba]
    # print('from local to global. Did base Ps change?', not np.allclose(np.array(base_local), np.array(base_global)))

    if tie_points:
        #### save coordinates of the base 3D points
        os.makedirs(ba_dir + "/tie_points", exist_ok=True)
        for im_idx, fn in enumerate(self.ba_pipeline.myimages):
            f_id = os.path.splitext(os.path.basename(fn))[0]

            true_if_track_seen_in_current_cam = ~np.isnan(
                self.ba_pipeline.C_v2[im_idx, :]
            )
            kp_index_per_track_observation = np.array(
                [self.ba_pipeline.C_v2[im_idx, true_if_track_seen_in_current_cam]]
            )
            pts_3d_per_track_observation = self.ba_pipeline.pts_3d_ba[
                true_if_track_seen_in_current_cam, :
            ].T

            kp_idx_to_3d_coords = np.vstack(
                (kp_index_per_track_observation, pts_3d_per_track_observation)
            ).T

            np.savetxt(
                os.path.join(ba_dir + "/tie_points", f_id + ".txt"),
                kp_idx_to_3d_coords,
                fmt="%.6f",
            )

            pickle_out = open(
                os.path.join(ba_dir + "/tie_points", f_id + ".pickle"), "wb"
            )
            pickle.dump(kp_idx_to_3d_coords, pickle_out)
            pickle_out.close()

    print(
        "All dates adjusted in {} seconds, {} ({}, {})".format(
            running_time, n_tracks, init_e, ba_e
        )
    )

    stop = timeit.default_timer()
    total_time = int(stop - start)

    print("\n##############################################################")
    print("- Global sweep completed in {} seconds".format(total_time))
    print("##############################################################\n")

    ###############
    # freeze base variables and update local systems
    ###############

    os.makedirs(ba_dir, exist_ok=True)

    update_args = [
        (t_idx, bp, verbose) for t_idx, bp in zip(timeline_indices, base_pair_per_date)
    ]

    start = timeit.default_timer()

    if tie_points:
        self.tracks_config["tie_points"] = True

    # base_update = []

    update_output = []

    # parallelize = False

    if parallelize:
        with Pool() as p:
            update_output = p.starmap(self.out_of_core_update_local_system, update_args)
    else:
        for idx, t_idx in enumerate(timeline_indices):
            running_time, n_tracks, ba_e, init_e = self.out_of_core_update_local_system(
                *update_args[idx]
            )
            update_output.append([running_time, n_tracks, ba_e, init_e])

            # base_update.append(self.ba_pipeline.P_crop_ba[0])
            # base_update.append(self.ba_pipeline.P_crop_ba[1])

    self.tracks_config["tie_points"] = False

    # print('from global to update. Did base Ps change?', not np.allclose(np.array(base_update), np.array(base_global)))

    for idx, t_idx in enumerate(timeline_indices):
        ba_e, init_e, n_tracks = (
            update_output[idx][2],
            update_output[idx][3],
            update_output[idx][1],
        )
        print(
            "({}) {}, {}, ({}, {}) ".format(
                idx + 1, self.timeline[t_idx]["datetime"], n_tracks, init_e, ba_e
            )
        )

    stop = timeit.default_timer()
    total_time = int(stop - start)

    print("\n##############################################################")
    print(
        "- Local update done in {} seconds (avg per date: {:.2} s)".format(
            total_time, total_time / len(timeline_indices)
        )
    )
    print("##############################################################\n")

    abs_stop = timeit.default_timer()
    self.print_running_time(int(abs_stop - abs_start))


def out_of_core_update_local_system(self, t_idx, base_pair, verbose):

    ba_dir = os.path.join(self.dst_dir, "ba_out-of-core")

    myimages_adj = (
        np.array(self.timeline[t_idx]["fnames"])[np.array(base_pair)]
    ).tolist()
    n_adj = len(myimages_adj)
    rpc_dir = os.path.join(ba_dir, "RPC_adj")
    myrpcs_adj = loader.load_rpcs_from_dir(
        myimages_adj, rpc_dir, suffix="RPC_adj", verbose=verbose
    )
    mycrops_adj = loader.load_image_crops(
        myimages_adj,
        get_aoi_mask=self.compute_aoi_masks,
        rpcs=myrpcs_adj,
        aoi=self.aoi_lonlat,
        use_mask_for_equalization=self.use_aoi_equalization,
        verbose=verbose,
    )

    myimages_new = list(set(self.timeline[t_idx]["fnames"]) - set(myimages_adj))
    n_new = len(myimages_new)
    myrpcs_new = loader.load_rpcs_from_dir(
        myimages_new, rpc_dir, suffix="RPC_adj", verbose=verbose
    )
    mycrops_new = loader.load_image_crops(
        myimages_new,
        get_aoi_mask=self.compute_aoi_masks,
        rpcs=myrpcs_new,
        aoi=self.aoi_lonlat,
        use_mask_for_equalization=self.use_aoi_equalization,
        verbose=verbose,
    )

    ba_input_data = {}
    ba_input_data["input_dir"] = ba_dir
    ba_input_data["output_dir"] = ba_dir
    ba_input_data["n_new"] = n_new
    ba_input_data["n_adj"] = n_adj
    ba_input_data["image_fnames"] = myimages_adj + myimages_new
    ba_input_data["crops"] = mycrops_adj + mycrops_new
    ba_input_data["rpcs"] = myrpcs_adj + myrpcs_new
    ba_input_data["cam_model"] = self.cam_model
    ba_input_data["aoi"] = self.aoi_lonlat

    # print('BASE NODE', self.timeline[t_idx]['base_node'])
    # print('FNAMES', ba_input_data['image_fnames'])

    relative_poses_dir = os.path.join(ba_dir, "relative_local_poses")
    base_node_fn = self.timeline[t_idx]["fnames"][base_pair[0]]
    P_base = loader.load_matrices_from_dir(
        [base_node_fn], ba_dir + "/P_adj", suffix="pinhole_adj", verbose=verbose
    )
    k_b, r_b, t_b, o_b = ba_core.decompose_perspective_camera(P_base[0])
    ext_b = np.hstack((r_b, t_b[:, np.newaxis]))
    P_init = loader.load_matrices_from_dir(
        ba_input_data["image_fnames"],
        ba_dir + "/P_adj",
        suffix="pinhole_adj",
        verbose=verbose,
    )
    ba_input_data["input_P"] = []
    for cam_idx, fname in enumerate(myimages_adj):
        ba_input_data["input_P"].append(P_init[cam_idx])
    for cam_idx, fname in enumerate(myimages_new):
        f_id = os.path.splitext(os.path.basename(fname))[0]
        k_cam, r_cam, t_cam, o_cam = ba_core.decompose_perspective_camera(
            P_init[n_adj + cam_idx]
        )
        ba_input_data["input_P"].append(
            k_cam @ ext_b @ np.loadtxt("{}/{}.txt".format(relative_poses_dir, f_id))
        )

    # print('P_base:', P_init[0])
    # print(self.ba_input_data['input_P'])

    if self.compute_aoi_masks:
        ba_input_data["masks"] = [f["mask"] for f in mycrops_adj] + [
            f["mask"] for f in mycrops_new
        ]
    else:
        ba_input_data["masks"] = None

    running_time, n_tracks, ba_e, init_e = self.bundle_adjust(
        [t_idx],
        ba_dir,
        ba_dir,
        0,
        ba_input_data,
        False,
        self.tracks_config,
        verbose,
        False,
    )

    return running_time, n_tracks, ba_e, init_e
