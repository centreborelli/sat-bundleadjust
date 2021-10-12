"""
A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery
author: Roger Mari <roger.mari@ens-paris-saclay.fr>
year: 2021

This script implements the FeatureTracksPipeline
Given an input group of satellite images this class is able to find a set of feature tracks connecting them
It is also able to extend the feature tracks found for a previous group if new images are added
The blocks that it covers to find a set of feature tracks are the following ones:
(1) feature detection
(2) stereo pairs selection
(3) pairwise matching
(4) feature tracks construction
"""

import os
import timeit

import numpy as np

from bundle_adjust import loader, geo_utils
from . import ft_opencv, ft_s2p, ft_match, ft_utils

from bundle_adjust.loader import flush_print


class FeatureTracksPipeline:
    def __init__(self, input_dir, output_dir, local_data, tracks_config=None):
        """
        Initialize the feature tracks detection pipeline

        Args:
            input_dir: string, input directory containing precomputed tracks (if available)
            output_dir: string, output directory where all output files will be written
            local_data: dictionary containing all data about the input satellite images
                        "images": list of instances of the class SatelliteImage
                        "n_adj": integer, number of input images already adjusted (if any)
                                 these image positions have to be on top of all previous lists in local_data
                                 and their feature tracks are expected to be available in the input_dir
                        "aoi" (optional): GeoJSON polygon with lon-lat coordinates
                                          this is used if we want to restrict the search of tie points to
                                          a specific area of interest (requires tracks_config["FT_kp_aoi"] = True)
            tracks_config (optional): dictionary specifying the configuration for the feature tracking
                                      see feature_tracks.ft_utils.init_feature_tracks_config to check
                                      the considered feature tracking parameters
        """

        # initialize parameters
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.images = local_data["images"]
        self.n_adj = local_data["n_adj"]
        self.aoi = local_data["aoi"]
        self.config = ft_utils.init_feature_tracks_config(tracks_config)
        self.config["in_dir"] = self.input_dir
        self.config["out_dir"] = self.output_dir

        # if specified, compute a mask per image to restrict the search of keypoints in an area of interest
        if self.config["FT_kp_aoi"]:
            self.mask_paths = []
            for im in self.images:
                y0, x0, h, w = im.offset["row0"], im.offset["col0"], im.offset["height"], im.offset["width"]
                mask = loader.get_binary_mask_from_aoi_lonlat_within_image(h, w, im.rpc, self.aoi)
                masks_dir = os.path.join(self.output_dir, "masks")
                os.makedirs(masks_dir, exist_ok=True)
                mask_path = masks_dir + "/" + loader.get_id(im.geotiff_path) + ".npy"
                np.save(mask_path, mask[y0 : y0 + h, x0 : x0 + w])
                self.mask_paths.append(mask_path)
        else:
            self.mask_paths = None

    def run_feature_detection(self):
        """
        Detect keypoints in all images that have not been visited yet
        """
        # get image filenames where it is necessary to extract keypoints
        image_paths = [im.geotiff_path for im in self.images]
        offsets = [im.offset for im in self.images]
        self.features = ["{}/features/{}.npy".format(self.output_dir, loader.get_id(p)) for p in image_paths]
        self.features_utm = ["{}/features_utm/{}.npy".format(self.output_dir, loader.get_id(p)) for p in image_paths]

        # do we want to find keypoints all over each image or only within the region of the aoi?
        masks = self.mask_paths if self.config["FT_kp_aoi"] else None

        # detect using s2p or opencv sift
        if self.config["FT_sift_detection"] == "s2p":
            if self.config["FT_n_proc"] > 1:
                args = [image_paths, masks, offsets, self.config, self.config["FT_n_proc"]]
                ft_s2p.detect_features_image_sequence_multiprocessing(*args)
            else:
                args = [image_paths, masks, offsets, self.config, None, None]
                ft_s2p.detect_features_image_sequence(*args)
        else:
            args = [image_paths, masks, offsets, self.config]
            ft_opencv.detect_features_image_sequence(*args)

        for i, (npy, npy_utm, im) in enumerate(zip(self.features, self.features_utm, self.images)):
            if not self.config["FT_reset"] and os.path.exists(npy_utm):
                continue
            else:
                features = np.load(npy, mmap_mode='r')
                features_utm = ft_match.keypoints_to_utm_coords(features, im.rpc, im.offset, im.alt)
                os.makedirs(os.path.dirname(npy_utm), exist_ok=True)
                np.save(npy_utm, features_utm)

    def get_stereo_pairs_to_match(self):
        """
        Compute which pairs of images have to be matched and which are suitable to triangulate
        """
        self.n_new = len(self.images) - self.n_adj

        if len(self.config["FT_predefined_pairs"]) == 0:
            init_pairs = []
            for i in np.arange(self.n_adj + self.n_new):
                for j in np.arange(i + 1, self.n_adj + self.n_new):
                    init_pairs.append((i, j))
        else:
            init_pairs = self.config["FT_predefined_pairs"]

        # filter stereo pairs that are not overlaped
        # stereo pairs with small baseline should not be used to triangulate
        utm_poly = lambda im: {"geojson": geo_utils.utm_geojson_from_lonlat_geojson(im.lonlat_geojson), "z": im.alt}
        self.footprints = [utm_poly(im) for im in self.images]
        self.optical_centers = [im.center for im in self.images]
        args = [init_pairs, self.footprints, self.optical_centers]
        if self.config["FT_filter_pairs"]:
            self.pairs_to_match, self.pairs_to_triangulate = ft_match.compute_pairs_to_match(*args)
        else:
            self.pairs_to_match, self.pairs_to_triangulate = ft_match.compute_pairs_to_match(*args, min_overlap=0, min_baseline=0)

        print("{} pairs to match".format(len(self.pairs_to_match)))

    def run_feature_matching(self):
        """
        Compute pairwise matching between all pairs to match not matched yet
        """

        def init_F_pair_to_match(h, w, rpc_i, rpc_j):
            from bundle_adjust.s2p.estimation import affine_fundamental_matrix
            from bundle_adjust.s2p.rpc_utils import matches_from_rpc

            rpc_matches = matches_from_rpc(rpc_i, rpc_j, 0, 0, w, h, 5)
            Fij = affine_fundamental_matrix(rpc_matches)
            return Fij

        if self.config["FT_sift_matching"] == "epipolar_based":
            F = []
            for pair in self.pairs_to_match:
                i, j = pair[0], pair[1]
                h, w = self.images[i].offset["height"], self.images[i].offset["width"]
                F.append(init_F_pair_to_match(h, w, self.images[i].rpc, self.images[j].rpc))
        else:
            F = None

        args = [self.pairs_to_match, self.features, self.footprints, self.features_utm]

        if self.config["FT_sift_matching"] == "epipolar_based" and self.config["FT_n_proc"] > 1:
            self.pairwise_matches = ft_match.match_stereo_pairs_multiprocessing(*args, self.config, F)
        else:
            self.pairwise_matches = ft_match.match_stereo_pairs(*args, self.config, F)

        print("Found {} new pairwise matches".format(self.pairwise_matches.shape[0]))


    def get_feature_tracks(self):
        """
        Construct feature tracks from all pairwise matches
        """
        if self.pairwise_matches.shape[1] > 0:
            args = [self.features, self.pairwise_matches, self.pairs_to_triangulate]
            C, C_v2 = ft_utils.feature_tracks_from_pairwise_matches(*args)
            # n_pts_fix = amount of columns with no observations in the new cameras to adjust
            # these columns have to be put at the beginning of C
            where_fix_pts = np.sum(1 * ~np.isnan(C[::2, :])[self.n_adj:], axis=0) == 0
            n_pts_fix = np.sum(1 * where_fix_pts)
            if n_pts_fix > 0:
                C = np.hstack([C[:, where_fix_pts], C[:, ~where_fix_pts]])
                C_v2 = np.hstack([C_v2[:, where_fix_pts], C_v2[:, ~where_fix_pts]])
            flush_print("Found {} tracks in total".format(C.shape[1]))
        else:
            C, C_v2, n_pts_fix = None, None, 0
            flush_print("Found 0 tracks in total")

        feature_tracks = {
            "C": C,
            "C_v2": C_v2,
            "features": self.features,
            "pairwise_matches": self.pairwise_matches,
            "pairs_to_triangulate": self.pairs_to_triangulate,
            "pairs_to_match": self.pairs_to_match,
            "n_pts_fix": n_pts_fix,
        }

        return feature_tracks

    def build_feature_tracks(self):
        """
        Run the complete feature tracking pipeline
        Builds feature tracks of arbitrary length of corresponding sift keypoints
        """

        print("Building feature tracks\n")
        print("Parameters:")
        loader.display_dict(self.config)
        if self.config["FT_kp_aoi"] and self.masks is None:
            print('\n"FT_kp_aoi" is enabled to restrict the search of keypoints, but no masks were found !')

        start = timeit.default_timer()
        last_stop = start

        ###############
        # feature detection
        ##############

        if self.n_adj == len(self.images):
            flush_print("\nSkipping feature detection (no new images)")
        else:
            flush_print("\nRunning feature detection...\n")
            self.run_feature_detection()
            stop = timeit.default_timer()
            flush_print("\n...done in {:.2f} seconds".format(stop - last_stop))
            last_stop = stop

        ###############
        # compute stereo pairs to match
        ##############

        flush_print("\nComputing pairs to match...\n")
        self.get_stereo_pairs_to_match()

        stop = timeit.default_timer()
        flush_print("\n...done in {:.2f} seconds".format(stop - last_stop))
        last_stop = stop

        ###############
        # feature matching
        ##############

        if len(self.pairs_to_match) > 0:
            flush_print("\nMatching...\n")
            self.run_feature_matching()
            stop = timeit.default_timer()
            flush_print("\n...done in {:.2f} seconds".format(stop - last_stop))
            last_stop = stop
        else:
            flush_print("\nSkipping matching (no pairs to match)")

        ###############
        # construct tracks
        ##############

        flush_print("\nExtracting feature tracks...\n")
        feature_tracks = self.get_feature_tracks()
        stop = timeit.default_timer()
        flush_print("\n...done in {:.2f} seconds".format(stop - last_stop))

        flush_print("\nFeature tracks computed in {}\n".format(loader.get_time_in_hours_mins_secs(stop - start)))

        return feature_tracks, stop - start
