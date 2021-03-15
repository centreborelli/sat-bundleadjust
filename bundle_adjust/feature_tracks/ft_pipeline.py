import os
import timeit

import numpy as np

from bundle_adjust import data_loader as loader
from feature_tracks import ft_opencv, ft_ranking, ft_s2p, ft_sat, ft_utils


class FeatureTracksPipeline:
    def __init__(
        self,
        input_dir,
        output_dir,
        local_data,
        config=None,
        satellite=True,
        verbose=False,
    ):

        self.config = config
        self.satellite = satellite
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.local_data = local_data
        self.global_data = {}

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # initialize parameters
        self.config = ft_utils.init_feature_tracks_config(config)

    def save_feature_detection_results(self):

        features_dir = os.path.join(self.output_dir, "features")
        os.makedirs(features_dir, exist_ok=True)
        if self.satellite:
            features_utm_dir = os.path.join(self.output_dir, "features_utm")
            os.makedirs(features_utm_dir, exist_ok=True)

        loader.save_list_of_paths(
            self.output_dir + "/filenames.txt", self.global_data["fnames"]
        )
        t0 = timeit.default_timer()
        for idx in self.new_images_idx:
            f_id = os.path.splitext(os.path.basename(self.local_data["fnames"][idx]))[0]
            np.save(
                features_dir + "/" + f_id + ".npy", self.local_data["features"][idx]
            )
            if self.satellite:
                np.save(
                    features_utm_dir + "/" + f_id + ".npy",
                    self.local_data["features_utm"][idx],
                )
        print(
            "All keypoints saved in {:.2f} seconds (.npy format)".format(
                timeit.default_timer() - t0
            )
        )

    def save_feature_matching_results(self):

        np.save(self.output_dir + "/matches.npy", self.global_data["pairwise_matches"])
        loader.save_list_of_pairs(
            self.output_dir + "/pairs_matching.npy", self.global_data["pairs_to_match"]
        )
        loader.save_list_of_pairs(
            self.output_dir + "/pairs_triangulation.npy",
            self.global_data["pairs_to_triangulate"],
        )

    def init_feature_matching(self):

        # load previous matches and list of paris to be matched/triangulate if existent
        self.local_data["pairwise_matches"] = []
        self.local_data["pairs_to_triangulate"] = []
        self.local_data["pairs_to_match"] = []

        self.global_data["pairwise_matches"] = []
        self.global_data["pairs_to_match"] = []
        self.global_data["pairs_to_triangulate"] = []

        found_prev_matches = os.path.exists(self.input_dir + "/matches.npy")
        found_prev_m_pairs = os.path.exists(self.input_dir + "/pairs_matching.npy")
        found_prev_t_pairs = os.path.exists(self.input_dir + "/pairs_triangulation.npy")

        if (
            np.sum(1 * self.true_if_seen) > 0
            and found_prev_matches
            and found_prev_m_pairs
            and found_prev_t_pairs
        ):

            self.global_data["pairwise_matches"].append(
                np.load(self.input_dir + "/matches.npy")
            )
            path_to_npy = self.input_dir + "/pairs_matching.npy"
            self.global_data["pairs_to_match"].extend(
                loader.load_list_of_pairs(path_to_npy)
            )
            path_to_npy = self.input_dir + "/pairs_triangulation.npy"
            self.global_data["pairs_to_triangulate"].extend(
                loader.load_list_of_pairs(path_to_npy)
            )

            # load pairwise matches (if existent) within the images in use
            total_cams = len(self.global_data["fnames"])
            true_where_im_in_use = np.zeros(total_cams).astype(bool)
            true_where_im_in_use[self.local_idx_to_global_idx] = True

            true_where_prev_match = np.logical_and(
                true_where_im_in_use[self.global_data["pairwise_matches"][0][:, 2]],
                true_where_im_in_use[self.global_data["pairwise_matches"][0][:, 3]],
            )
            prev_pairwise_matches_in_use_global = self.global_data["pairwise_matches"][
                0
            ][true_where_prev_match]
            prev_pairwise_matches_in_use_local = (
                prev_pairwise_matches_in_use_global.copy()
            )

            prev_pairwise_matches_in_use_local[:, 2] = self.global_idx_to_local_idx[
                prev_pairwise_matches_in_use_local[:, 2]
            ]
            prev_pairwise_matches_in_use_local[:, 3] = self.global_idx_to_local_idx[
                prev_pairwise_matches_in_use_local[:, 3]
            ]
            self.local_data["pairwise_matches"].append(
                prev_pairwise_matches_in_use_local
            )

            # incorporate triangulation pairs composed by pairs of previously seen images now in use
            for pair in self.global_data["pairs_to_triangulate"]:
                local_im_idx_i = self.global_idx_to_local_idx[pair[0]]
                local_im_idx_j = self.global_idx_to_local_idx[pair[1]]
                if local_im_idx_i > -1 and local_im_idx_j > -1:
                    if (
                        self.true_if_seen[local_im_idx_i]
                        and self.true_if_seen[local_im_idx_j]
                    ):
                        self.local_data["pairs_to_triangulate"].append(
                            (
                                min(local_im_idx_i, local_im_idx_j),
                                max(local_im_idx_i, local_im_idx_j),
                            )
                        )

            # incorporate matching pairs composed by pairs of previously seen images now in use
            for pair in self.global_data["pairs_to_match"]:
                local_im_idx_i = self.global_idx_to_local_idx[pair[0]]
                local_im_idx_j = self.global_idx_to_local_idx[pair[1]]
                if local_im_idx_i > -1 and local_im_idx_j > -1:
                    if (
                        self.true_if_seen[local_im_idx_i]
                        and self.true_if_seen[local_im_idx_j]
                    ):
                        self.local_data["pairs_to_match"].append(
                            (
                                min(local_im_idx_i, local_im_idx_j),
                                max(local_im_idx_i, local_im_idx_j),
                            )
                        )

    def init_feature_detection(self):

        self.global_data["fnames"] = []

        # load previous features if existent and list of previously adjusted filenames
        feats_dir = os.path.join(self.input_dir, "features")
        feats_utm_dir = os.path.join(self.input_dir, "features_utm")

        self.local_data["features"] = []
        if self.satellite:
            self.local_data["features_utm"] = []

        if not self.config["FT_reset"] and os.path.exists(
            self.input_dir + "/filenames.txt"
        ):
            seen_fn = loader.load_list_of_paths(
                self.input_dir + "/filenames.txt"
            )  # previously seen filenames
            self.global_data["fnames"] = seen_fn
            # print('LOADED PREVIOUS FILENAMES')
        else:
            seen_fn = []
            # print('STARTING FROM ZERO')

        n_cams_so_far = len(seen_fn)
        n_cams_never_seen_before = 0

        # check if files in use have been previously seen or not
        self.true_if_seen = np.array(
            [fn in seen_fn for fn in self.local_data["fnames"]]
        )

        self.new_images_idx = []

        # global indices
        global_indices = []
        for k, fn in enumerate(self.local_data["fnames"]):
            if self.true_if_seen[k]:
                g_idx = seen_fn.index(fn)
                global_indices.append(g_idx)
                f_id = loader.get_id(seen_fn[g_idx])
                self.local_data["features"].append(
                    np.load(feats_dir + "/" + f_id + ".npy")
                )
                self.local_data["features_utm"].append(
                    np.load(feats_utm_dir + "/" + f_id + ".npy")
                )
            else:
                n_cams_never_seen_before += 1
                global_indices.append(n_cams_so_far + n_cams_never_seen_before - 1)
                self.local_data["features"].append(np.array([np.nan]))
                self.local_data["features_utm"].append(np.array([np.nan]))
                self.global_data["fnames"].append(fn)
                self.new_images_idx.append(k)

        self.local_idx_to_global_idx = global_indices

        n_cams_in_use = len(self.local_idx_to_global_idx)

        self.global_idx_to_local_idx = -1 * np.ones(len(self.global_data["fnames"]))
        self.global_idx_to_local_idx[self.local_idx_to_global_idx] = np.arange(
            n_cams_in_use
        )
        self.global_idx_to_local_idx = self.global_idx_to_local_idx.astype(int)

    def run_feature_detection(self):

        # load images where it is necessary to extract keypoints
        new_images = [self.local_data["images"][idx] for idx in self.new_images_idx]

        # do we want to find keypoints all over each image or only within the region of the aoi?
        if self.local_data["masks"] is not None and self.config["FT_kp_aoi"]:
            new_masks = [self.local_data["masks"][idx] for idx in self.new_images_idx]
        else:
            new_masks = None

        # preprocess images if specified. ATTENTION: this is mandatory for opencv sift
        if self.config["FT_preprocess"]:
            if (
                self.config["FT_preprocess_aoi"]
                and self.local_data["masks"] is not None
            ):
                new_images = [
                    loader.custom_equalization(im, mask=self.local_data["mask"][idx])
                    for im, idx in zip(new_images, self.new_images_idx)
                ]
            else:
                new_images = [
                    loader.custom_equalization(im, mask=None) for im in new_images
                ]

        if self.config["FT_sift_detection"] == "s2p":
            if self.config["FT_n_proc"] > 1:
                args = [
                    new_images,
                    new_masks,
                    self.config["FT_kp_max"],
                    self.config["FT_n_proc"],
                ]
                new_features = ft_s2p.detect_features_image_sequence_multiprocessing(
                    *args
                )
            else:
                args = [
                    new_images,
                    new_masks,
                    self.config["FT_kp_max"],
                    np.arange(len(new_images)),
                    None,
                ]
                new_features = ft_s2p.detect_features_image_sequence(*args)
        else:
            args = [new_images, new_masks, self.config["FT_kp_max"]]
            new_features = ft_opencv.detect_features_image_sequence(*args)

        if self.satellite:
            new_rpcs = [self.local_data["rpcs"][idx] for idx in self.new_images_idx]
            new_footprints = [
                self.local_data["footprints"][idx] for idx in self.new_images_idx
            ]
            new_offsets = [
                self.local_data["offsets"][idx] for idx in self.new_images_idx
            ]
            new_features_utm = ft_sat.keypoints_to_utm_coords(
                new_features, new_rpcs, new_footprints, new_offsets
            )

        for k, idx in enumerate(self.new_images_idx):
            self.local_data["features"][idx] = new_features[k]
            self.local_data["features_utm"][idx] = new_features_utm[k]

    def get_stereo_pairs_to_match(self):

        n_adj = self.local_data["n_adj"]
        n_new = self.local_data["n_new"]

        if len(self.config["FT_predefined_pairs"]) == 0:
            init_pairs = []
            # possible new pairs to match are composed by 1 + 2
            # 1. each of the previously adjusted images with the new ones
            for i in np.arange(n_adj):
                for j in np.arange(n_adj, n_adj + n_new):
                    init_pairs.append((i, j))
            # 2. each of the new images with the rest of the new images
            for i in np.arange(n_adj, n_adj + n_new):
                for j in np.arange(i + 1, n_adj + n_new):
                    init_pairs.append((i, j))
        else:
            init_pairs = self.config["FT_predefined_pairs"]

        # filter stereo pairs that are not overlaped
        # stereo pairs with small baseline should not be used to triangulate
        new_pairs_to_match, new_pairs_to_triangulate = ft_sat.compute_pairs_to_match(
            init_pairs,
            self.local_data["footprints"],
            self.local_data["optical_centers"],
        )

        # remove pairs to match or to triangulate already in local_data
        new_pairs_to_triangulate = list(
            set(new_pairs_to_triangulate) - set(self.local_data["pairs_to_triangulate"])
        )
        new_pairs_to_match = list(
            set(new_pairs_to_match) - set(self.local_data["pairs_to_match"])
        )

        print("{} new pairs to be matched".format(len(new_pairs_to_match)))

        # convert image indices from local to global (global indices consider all images, not only the ones in use)
        # and update all_pairs_to_match and all_pairs_to_triangulate
        for pair in new_pairs_to_triangulate:
            global_idx_i = self.local_idx_to_global_idx[pair[0]]
            global_idx_j = self.local_idx_to_global_idx[pair[1]]
            self.global_data["pairs_to_triangulate"].append(
                (min(global_idx_i, global_idx_j), max(global_idx_i, global_idx_j))
            )

        for pair in new_pairs_to_match:
            global_idx_i = self.local_idx_to_global_idx[pair[0]]
            global_idx_j = self.local_idx_to_global_idx[pair[1]]
            self.global_data["pairs_to_match"].append(
                (min(global_idx_i, global_idx_j), max(global_idx_i, global_idx_j))
            )

        self.local_data["pairs_to_match"] = new_pairs_to_match
        self.local_data["pairs_to_triangulate"].extend(new_pairs_to_triangulate)

        # print('PAIRS TO TRIANGULATE ', self.local_data['pairs_to_triangulate'])
        # print('PAIRS TO MATCH ', self.local_data['pairs_to_match'])

    def run_feature_matching(self):
        def init_F_pair_to_match(h, w, rpc_i, rpc_j):
            import s2p

            rpc_matches = s2p.rpc_utils.matches_from_rpc(rpc_i, rpc_j, 0, 0, w, h, 5)
            Fij = s2p.estimation.affine_fundamental_matrix(rpc_matches)
            return Fij

        if self.config["FT_sift_matching"] == "epipolar_based":
            F = []
            for pair in self.local_data["pairs_to_match"]:
                i, j = pair[0], pair[1]
                h, w = self.local_data["images"][i].shape
                F.append(
                    init_F_pair_to_match(
                        h, w, self.local_data["rpcs"][i], self.local_data["rpcs"][j]
                    )
                )
        else:
            F = None

        features_utm = self.local_data["features_utm"] if self.satellite else None
        footprints_utm = self.local_data["footprints"] if self.satellite else None

        args = [
            self.local_data["pairs_to_match"],
            self.local_data["features"],
            footprints_utm,
            features_utm,
            self.config["FT_sift_matching"],
            self.config["FT_rel_thr"],
            self.config["FT_abs_thr"],
            self.config["FT_ransac"],
            F,
        ]

        if (
            self.config["FT_sift_matching"] in ["epipolar_based", "local_window"]
            and self.config["FT_n_proc"] > 1
        ):
            args.append(self.config["FT_n_proc"])
            new_pairwise_matches = ft_sat.match_stereo_pairs_multiprocessing(*args)
        else:
            new_pairwise_matches = ft_sat.match_stereo_pairs(*args)

        print("Found {} new pairwise matches".format(new_pairwise_matches.shape[0]))

        # add the newly found pairwise matches to local and global data
        self.local_data["pairwise_matches"].append(new_pairwise_matches)
        self.local_data["pairwise_matches"] = np.vstack(
            self.local_data["pairwise_matches"]
        )

        if len(new_pairwise_matches) > 0:
            new_pairwise_matches[:, 2] = np.array(self.local_idx_to_global_idx)[
                new_pairwise_matches[:, 2]
            ]
            new_pairwise_matches[:, 3] = np.array(self.local_idx_to_global_idx)[
                new_pairwise_matches[:, 3]
            ]
            self.global_data["pairwise_matches"].append(new_pairwise_matches)
            self.global_data["pairwise_matches"] = np.vstack(
                self.global_data["pairwise_matches"]
            )

    def get_feature_tracks(self):
        if self.local_data["pairwise_matches"].shape[1] > 0:
            C, C_v2 = ft_utils.feature_tracks_from_pairwise_matches(
                self.local_data["features"],
                self.local_data["pairwise_matches"],
                self.local_data["pairs_to_triangulate"],
            )
            # n_pts_fix = amount of columns with no observations in the new cameras to adjust
            # these columns have to be put at the beginning of C
            where_fix_pts = (
                np.sum(1 * ~np.isnan(C[::2, :])[-self.local_data["n_new"] :], axis=0)
                == 0
            )
            n_pts_fix = np.sum(1 * where_fix_pts)
            if n_pts_fix > 0:
                C = np.hstack([C[:, where_fix_pts], C[:, ~where_fix_pts]])
                C_v2 = np.hstack([C_v2[:, where_fix_pts], C_v2[:, ~where_fix_pts]])
            print("Found {} tracks in total".format(C.shape[1]), flush=True)
        else:
            C, C_v2, n_pts_fix = None, None, 0
            print("Found 0 tracks in total", flush=True)

        feature_tracks = {
            "C": C,
            "C_v2": C_v2,
            "features": self.local_data["features"],
            "pairwise_matches": self.local_data["pairwise_matches"],
            "pairs_to_triangulate": self.local_data["pairs_to_triangulate"],
            "pairs_to_match": self.local_data["pairs_to_match"],
            "n_pts_fix": n_pts_fix,
        }

        return feature_tracks

    def build_feature_tracks(self):

        # FEATURE DETECTION + MATCHING ON THE NEW IMAGES

        print(
            "Building feature tracks - {} scenario\n".format(
                "satellite" if self.satellite else "generic"
            )
        )
        print("Parameters:")
        loader.display_dict(self.config)
        if self.config["FT_kp_aoi"] and self.local_data["masks"] is None:
            print(
                '\n"FT_kp_aoi" is enabled to restrict the search of keypoints, but no masks were found !'
            )

        start = timeit.default_timer()
        last_stop = start

        ###############
        # feature detection
        ##############

        self.init_feature_detection()

        if self.local_data["n_new"] > 0:
            print("\nRunning feature detection...\n", flush=True)
            self.run_feature_detection()
            self.save_feature_detection_results()

            stop = timeit.default_timer()
            print("\n...done in {:.2f} seconds".format(stop - last_stop), flush=True)
            last_stop = stop
        else:
            print("\nSkipping feature detection (no new images)", flush=True)

        ###############
        # compute stereo pairs to match
        ##############

        print("\nComputing pairs to match...\n", flush=True)
        self.init_feature_matching()
        self.get_stereo_pairs_to_match()

        stop = timeit.default_timer()
        print("\n...done in {:.2f} seconds".format(stop - last_stop), flush=True)
        last_stop = stop

        ###############
        # feature matching
        ##############

        if len(self.local_data["pairs_to_match"]) > 0:
            print("\nMatching...\n", flush=True)
            self.run_feature_matching()
            self.save_feature_matching_results()
            stop = timeit.default_timer()
            print("\n...done in {:.2f} seconds".format(stop - last_stop), flush=True)
            last_stop = stop
        else:
            self.local_data["pairwise_matches"] = np.vstack(
                self.local_data["pairwise_matches"]
            )
            self.global_data["pairwise_matches"] = np.vstack(
                self.global_data["pairwise_matches"]
            )
            print("\nSkipping matching (no pairs to match)", flush=True)

        nodes_in_pairs_to_triangulate = np.unique(
            np.array(self.local_data["pairs_to_triangulate"]).flatten()
        ).tolist()
        new_nodes = np.arange(
            self.local_data["n_adj"],
            self.local_data["n_adj"] + self.local_data["n_new"],
        ).tolist()
        sanity_check = (
            len(list(set(new_nodes) - set(nodes_in_pairs_to_triangulate))) == 0
        )
        print(
            "\nDo all new nodes appear at least once in pairs to triangulate?",
            sanity_check,
        )

        ###############
        # construct tracks
        ##############
        print("\nExtracting feature tracks...\n", flush=True)
        feature_tracks = self.get_feature_tracks()
        stop = timeit.default_timer()
        print("\n...done in {:.2f} seconds".format(stop - last_stop), flush=True)

        print(
            "\nFeature tracks computed in {}\n".format(
                loader.get_time_in_hours_mins_secs(stop - start)
            ),
            flush=True,
        )

        return feature_tracks, stop - start
