"""
A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery
code for Image Processing On Line https://www.ipol.im/

author: Roger Mari <roger.mari@ens-paris-saclay.fr>

This script implements the BundleAdjustmentPipeline class
This class takes all the input data for the problem and solves it following the next blockchain
(1) feature detection
(2) stereo pairs selection
(3) pairwise matching
(4) feature tracks construction
(5) feature tracks triangulation
(6) feature tracks selection (optional)
(7) Define bundle adjustment parameters
(8) Run some initial bundle adjustment iterations with a softL1 cost function (optional)
(9) Remove outlier feature track observations based on reprojection errors (optional)
(10) Re-run bundle adjustment with a classic L2 cost function
(11) Fit corrected RPC models from the output of the bundle adjustment solution
"""

import numpy as np
import os
import timeit

from bundle_adjust import ba_core, ba_outliers, ba_params, camera_utils, geotools
from bundle_adjust import loader
from bundle_adjust import rpc_fit
from bundle_adjust.loader import flush_print


class Error(Exception):
    pass


class BundleAdjustmentPipeline:
    def __init__(self, ba_data, tracks_config={}, extra_ba_config={}):
        """
        Args:
            ba_data: dictionary specifying the bundle adjustment input data
            tracks_config (optional): dictionary specifying the configuration for the feature detection step
            extra_ba_config (optional): dictionary specifying any extra parameters to customize the BA configuration
        """
        from .feature_tracks.ft_utils import init_feature_tracks_config

        self.tracks_config = init_feature_tracks_config(tracks_config)

        # read extra bundle adjustment configuration parameters
        self.feature_detection = extra_ba_config.get("feature_detection", True)
        self.fix_ref_cam = extra_ba_config.get("fix_ref_cam", True)
        self.ref_cam_weight = extra_ba_config.get("ref_cam_weight", 1.0) if self.fix_ref_cam else 1.0
        self.clean_outliers = extra_ba_config.get("clean_outliers", True)
        self.max_init_reproj_error = extra_ba_config.get("max_init_reproj_error", None)
        self.verbose = extra_ba_config.get("verbose", True)
        self.display_plots = False

        # read bundle adjustment input data
        self.in_dir = ba_data["in_dir"]
        self.out_dir = ba_data["out_dir"]
        os.makedirs(self.out_dir, exist_ok=True)
        self.n_adj = ba_data["n_adj"]
        self.n_new = ba_data["n_new"]
        self.myimages = ba_data["image_fnames"].copy()
        self.crop_offsets = [{k: c[k] for k in ["col0", "row0", "width", "height"]} for c in ba_data["crops"]]
        self.input_seq = [f["crop"] for f in ba_data["crops"]]
        self.input_masks = ba_data["masks"].copy() if ba_data["masks"] is not None else None
        self.input_rpcs = ba_data["rpcs"].copy()
        self.cam_model = ba_data["cam_model"]
        self.aoi = ba_data["aoi"]
        self.correction_params = ba_data.get("correction_params", ["R"])
        self.predefined_matches_dir = ba_data.get("predefined_matches_dir", None)

        flush_print("Bundle Adjustment Pipeline created")
        flush_print("-------------------------------------------------------------")
        flush_print("Configuration:")
        flush_print("    - in_dir:       {}".format(self.in_dir))
        flush_print("    - out_dir:      {}".format(self.out_dir))
        flush_print("    - n_new:        {}".format(self.n_new))
        flush_print("    - n_adj:        {}".format(self.n_adj))
        flush_print("    - cam_model:    {}".format(self.cam_model))
        flush_print("    - fix_ref_cam:  {}".format(self.fix_ref_cam))
        flush_print("-------------------------------------------------------------\n")

        # stuff to be filled by 'run_feature_detection'
        self.features = []
        self.pairs_to_triangulate = []
        self.C = None
        self.n_pts_fix = 0

        # stuff to be filled by 'initialize_pts3d'
        self.pts3d = None

        # stuff to be filled by 'define_ba_parameters'
        self.ba_params = None

        # stuff to be filled by 'run_ba_softL1'-'clean_outlier_observations'-'run_ba_L2'
        self.ba_e = None
        self.init_e = None
        self.ba_iters = 0
        self.corrected_cameras = None
        self.corrected_pts3d = None

        # set initial cameras and image footprints
        ba_params.check_valid_cam_model(self.cam_model)
        if "cameras" in ba_data.keys():
            self.cameras = ba_data["cameras"].copy()
        else:
            self.set_cameras()
        self.cam_centers = self.get_optical_centers()
        self.footprints = self.get_footprints()
        print("\n")

    def get_footprints(self):
        """
        this function outputs a list containing the footprint associated to each input satellite image
        each footprint contains a polygon in utm coordinates covering the geographic area seen in the image
        and the srtm4 altitude value associated to the center of the polygon
        """
        from shapely.geometry import shape

        t0 = timeit.default_timer()
        flush_print("Getting image footprints...")
        args = [self.myimages, self.input_rpcs, self.crop_offsets]
        lonlat_footprints, alts = loader.load_geotiff_lonlat_footprints(*args)
        utm_footprints = []
        for x, z in zip(lonlat_footprints, alts):
            utm_footprints.append({"poly": shape(geotools.utm_geojson_from_lonlat_geojson(x)), "z": z})
        flush_print("...done in {:.2f} seconds".format(timeit.default_timer() - t0))
        return utm_footprints

    def check_projection_matrices(self, err, max_err=1.0):
        """
        this function is called if RPCs are approximated as projection matrices
        it checks that the average reprojection error associated to each approximation does not exceed a threshold
        """
        err_cams = np.arange(len(err))[np.array(err) > max_err]
        n_err_cams = len(err_cams)
        if n_err_cams > 0:
            to_print = [n_err_cams, " ".join(["\nCamera {}, error = {:.3f}".format(c, err[c]) for c in err_cams])]
            flush_print("WARNING: {} projection matrices with error larger than 1.0 px\n{}".format(*to_print))

    def get_optical_centers(self):
        """
        this function computes an approximation of an optical center for each camera model
        this is done by approximating the RPCs as perspective projection matrices and retrieving the camera center
        """
        t0 = timeit.default_timer()
        flush_print("Estimating camera positions...")
        if self.cam_model != "perspective":
            args = [self.input_rpcs, self.crop_offsets]
            tmp_perspective_cams, err = loader.approx_perspective_projection_matrices(*args, verbose=False)
            # self.check_projection_matrices(err)
            optical_centers = [camera_utils.get_perspective_optical_center(P) for P in tmp_perspective_cams]
        else:
            optical_centers = [camera_utils.get_perspective_optical_center(P) for P in self.cameras]
        flush_print("...done in {:.2f} seconds".format(timeit.default_timer() - t0))
        return optical_centers

    def set_cameras(self):
        """
        this function sets which camera model representation has to be used at internal level
        the input camera models are RPCs, but the BA pipeline can either use those or approximations
        such approximations can take the form of perspective or affine projection matrices
        """
        if self.cam_model == "affine":
            args = [self.input_rpcs, self.crop_offsets, self.aoi]
            self.cameras, err = loader.approx_affine_projection_matrices(*args, verbose=True)
            self.check_projection_matrices(err)
        elif self.cam_model == "perspective":
            args = [self.input_rpcs, self.crop_offsets]
            self.cameras, err = loader.approx_perspective_projection_matrices(*args, verbose=True)
            self.check_projection_matrices(err)
        else:
            self.cameras = self.input_rpcs.copy()

    def compute_feature_tracks(self):
        """
        this function launches the feature tracking pipeline, which covers the following steps
            - feature detection
            - stereo pairs selection
            - pairwise matching
            - feature tracks construction
            - feature tracks selection (optional)
        """
        if self.tracks_config["FT_sift_detection"] == "s2p" and os.path.exists(self.in_dir + "/../RPC_init"):
            args = [self.myimages, self.in_dir + "/../RPC_init", "_RPC", "txt", False]
            ft_rpcs = loader.load_rpcs_from_dir(*args)
        else:
            ft_rpcs = self.input_rpcs
        local_data = {
            "n_adj": self.n_adj,
            "n_new": self.n_new,
            "fnames": self.myimages,
            "images": self.input_seq,
            "rpcs": ft_rpcs,
            "offsets": self.crop_offsets,
            "footprints": self.footprints,
            "optical_centers": self.cam_centers,
            "masks": self.input_masks,
        }
        if not self.feature_detection:
            local_data["n_adj"], local_data["n_new"] = self.n_adj + self.n_new, 0

        if self.predefined_matches_dir is None:
            from bundle_adjust.feature_tracks.ft_pipeline import FeatureTracksPipeline

            ft_pipeline = FeatureTracksPipeline(self.in_dir, self.out_dir, local_data, config=self.tracks_config)
            feature_tracks, self.feature_tracks_running_time = ft_pipeline.build_feature_tracks()
        else:
            from .feature_tracks.ft_utils import load_tracks_from_predefined_matches

            args = [local_data, self.tracks_config, self.predefined_matches_dir, self.out_dir]
            feature_tracks, self.feature_tracks_running_time = load_tracks_from_predefined_matches(*args)

        self.features = feature_tracks["features"]
        self.pairs_to_triangulate = feature_tracks["pairs_to_triangulate"]
        self.C = feature_tracks["C"]
        if self.cam_model == "rpc":
            for i in range(self.C.shape[0] // 2):
                self.C[2 * i, :] += self.crop_offsets[i]["col0"]
                self.C[2 * i + 1, :] += self.crop_offsets[i]["row0"]
        self.C_v2 = feature_tracks["C_v2"]
        self.n_pts_fix = feature_tracks["n_pts_fix"]

        # sanity checks to verify if C looks good
        err_msg = "Insufficient SIFT matches"
        n_cam = int(self.C.shape[0] / 2)
        if n_cam > self.C.shape[1]:
            raise Error("{}: Found less tracks than cameras".format(err_msg))
        obs_per_cam = np.sum(1 * ~np.isnan(self.C), axis=1)[::2]
        min_obs_cam = 10
        if np.sum(obs_per_cam < min_obs_cam) > 0:
            n_cams_insufficient_obs = np.arange(n_cam)[obs_per_cam < min_obs_cam]
            to_print = [err_msg, np.sum(obs_per_cam < min_obs_cam), min_obs_cam, n_cams_insufficient_obs]
            raise Error("{}: Found {} cameras with less than {} tie point observations (nodes: {})".format(*to_print))

        del feature_tracks

    def initialize_pts3d(self):
        """
        this function initializes the ECEF coordinates of the 3d points that project into the feature tracks
        """
        from bundle_adjust.ba_triangulate import init_pts3d

        self.pts3d = np.zeros((self.C.shape[1], 3), dtype=np.float32)
        n_pts_opt = self.C.shape[1] - self.n_pts_fix
        verbose = True
        if self.n_pts_fix > 0:
            if verbose:
                flush_print("Initializing {} fixed 3d point coords !".format(self.n_pts_fix))
            C_fixed_pts3d = self.C[: self.n_adj * 2, : self.n_pts_fix]
            args = [C_fixed_pts3d, self.cameras, self.cam_model, self.pairs_to_triangulate, verbose]
            self.pts3d[: self.n_pts_fix, :] = init_pts3d(*args)
        if verbose:
            flush_print("Initializing {} 3d point coords to optimize !".format(n_pts_opt))
        C_opt_pts3d = self.C[:, -n_pts_opt:]
        args = [C_opt_pts3d, self.cameras, self.cam_model, self.pairs_to_triangulate, verbose]
        self.pts3d[-n_pts_opt:, :] = init_pts3d(*args)

    def define_ba_parameters(self, freeze_all_cams=False, verbose=True):
        """
        this function sets the feature tracks and the associated 3d points, as well as any other necessary data,
        in the necessary format to run the bundle adjustment optimization problem
        """
        args = [self.C, self.pts3d, self.cameras, self.cam_model, self.pairs_to_triangulate, self.cam_centers]
        d = {
            "n_cam_fix": self.C.shape[0] // 2 if freeze_all_cams else self.n_adj,
            "n_pts_fix": self.n_pts_fix,
            "ref_cam_weight": self.ref_cam_weight,
            "correction_params": self.correction_params,
            "verbose": verbose,
        }
        self.ba_params = ba_params.BundleAdjustmentParameters(*args, d)

    def run_ba_softL1(self):
        """
        this function runs the bundle adjustment optimization with a soft L1 norm for the reprojection errors
        """
        ls_params_L1 = {"loss": "soft_l1", "f_scale": 0.5, "max_iter": 50}
        args = [self.ba_params, ls_params_L1, True, self.display_plots]
        _, self.ba_sol, err, iters = ba_core.run_ba_optimization(*args)
        self.init_e, self.ba_e, self.init_e_cam, self.ba_e_cam = err
        self.ba_iters += iters

    def run_ba_L2(self):
        """
        this function runs the bundle adjustment optimization with a classic L2 norm for the reprojection errors
        """
        args = [self.ba_params, None, True, self.display_plots]
        _, self.ba_sol, err, iters = ba_core.run_ba_optimization(*args)
        self.init_e, self.ba_e, self.init_e_cam, self.ba_e_cam = err
        self.ba_iters += iters

    def save_corrected_cameras(self):
        """
        this function recovers the optimized 3d points and camera models from the bundle adjustment solution
        """
        args = [self.ba_sol, self.pts3d, self.cameras]
        self.corrected_pts3d, self.corrected_cameras = self.ba_params.reconstruct_vars(*args)
        if self.cam_model in ["perspective", "affine"]:
            self.save_corrected_matrices()
        self.save_corrected_rpcs()

    def clean_outlier_observations(self):
        """
        this function removes outliers from the available tracks according to their reprojection error
        """
        start = timeit.default_timer()
        self.ba_params = ba_outliers.rm_outliers(self.ba_e, self.ba_params, verbose=True)
        elapsed_time = timeit.default_timer() - start
        flush_print("Removal of outliers based on reprojection error took {:.2f} seconds".format(elapsed_time))

    def save_initial_matrices(self):
        """
        this function writes the initial projection matrices to json files
        """
        out_dir = os.path.join(self.out_dir, "P_init")
        fnames = [os.path.join(out_dir, loader.get_id(fn) + "_pinhole.json") for fn in self.myimages]
        loader.save_projection_matrices(fnames, self.cameras, self.crop_offsets)
        flush_print("\nInitial projection matrices written at {}\n".format(out_dir))

    def save_corrected_matrices(self):
        """
        this function writes the corrected projection matrices to json files
        """
        out_dir = os.path.join(self.out_dir, "P_adj")
        fnames = [os.path.join(out_dir, loader.get_id(fn) + "_pinhole_adj.json") for fn in self.myimages]
        loader.save_projection_matrices(fnames, self.corrected_cameras, self.crop_offsets)
        flush_print("Bundle adjusted projection matrices written at {}\n".format(out_dir))

    def save_corrected_rpcs(self):
        """
        this function writes the corrected RPC models to txt files
        """
        out_dir = os.path.join(self.out_dir, "RPC_adj")
        fnames = [os.path.join(out_dir, loader.get_id(fn) + "_RPC_adj.txt") for fn in self.myimages]
        for cam_idx, (fn, cam) in enumerate(zip(fnames, self.corrected_cameras)):
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            if self.cam_model in ["perspective", "affine"]:
                rpc_calib, err = rpc_fit.fit_rpc_from_projection_matrix(cam, self.ba_params.pts3d_ba)
                to_print = [cam_idx, 1e4 * err.max(), 1e4 * err.mean()]
                flush_print("cam {:2} - RPC fit error per obs [1e-4 px] max / avg: {:.2f} / {:.2f}".format(*to_print))
                rpc_calib.write_to_file(fn)
            else:
                cam.write_to_file(fn)
        flush_print("Bundle adjusted rpcs written at {}\n".format(out_dir))

    def save_corrected_points(self):
        """
        this function writes the 3d point locations optimized by the bundle adjustment pipeline into a ply file
        """
        pts3d_adj_ply_path = os.path.join(self.out_dir, "pts3d_adj.ply")
        loader.write_point_cloud_ply(pts3d_adj_ply_path, self.ba_params.pts3d_ba)
        flush_print("Bundle adjusted 3d points written at {}\n".format(pts3d_adj_ply_path))

    def select_best_tracks(self, K=60, priority=["length", "scale", "cost"]):
        """
        this function runs the feature track selection algorithm,
        which seeks to select an optimal subset of tie points to run the bundle adjustment optimization
        """
        if K > 0:
            from .feature_tracks import ft_ranking

            args_C_scale = [self.C_v2, self.features]
            C_scale = ft_ranking.compute_C_scale(*args_C_scale)
            if self.pts3d is not None:
                args = [self.C, self.pts3d, self.cameras, self.cam_model, self.pairs_to_triangulate, self.cam_centers]
                C_reproj = ft_ranking.compute_C_reproj(*args)
            else:
                C_reproj = np.zeros(C_scale.shape)

            true_if_new_track = np.sum(~np.isnan(self.C[::2, :])[-self.n_new :] * 1, axis=0).astype(bool)
            C_new = self.C[:, true_if_new_track]
            C_scale_new = C_scale[:, true_if_new_track]
            C_reproj_new = C_reproj[:, true_if_new_track]
            prev_track_indices = np.arange(len(true_if_new_track))[true_if_new_track]
            args = [C_new, C_scale_new, C_reproj_new, K, priority, self.verbose]
            selected_track_indices = ft_ranking.select_best_tracks(*args)
            selected_track_indices = prev_track_indices[np.array(selected_track_indices)]

            self.C = self.C[:, selected_track_indices]
            self.C_v2 = self.C_v2[:, selected_track_indices]
            self.n_pts_fix = len(selected_track_indices[selected_track_indices < self.n_pts_fix])
            if self.pts3d is not None:
                self.pts3d = self.pts3d[selected_track_indices, :]

    def check_connectivity_graph(self, min_matches=10):
        """
        this function checks that all views are matched to another view by a minimum amount of matches
        this is done to verify that no cameras are disconnected or unseen by the tie points being used
        """
        from feature_tracks.ft_utils import build_connectivity_graph

        _, n_cc, _, _, missing_cams = build_connectivity_graph(self.C, min_matches=min_matches, verbose=True)
        err_msg = "Insufficient SIFT matches"
        if n_cc > 1:
            args = [err_msg, n_cc, min_matches]
            raise Error("{}: Connectivity graph has {} connected components (min_matches = {})".format(*args))
        if len(missing_cams) > 0:
            args = [err_msg, len(missing_cams), missing_cams]
            raise Error("{}: Found {} cameras missing in the connectivity graph (nodes: {})".format(*args))

    def fix_reference_camera(self):
        """
        optinal step - this function fixes a reference camera from the set of input cameras
        this seeks to prevent large drifts in the 3d space in the solution of the bundle adjustment pipeline
        """

        # part 1: identify the camera connected to more cameras with highest number of observations

        from .feature_tracks import ft_ranking

        neighbor_nodes_per_cam = np.sum(ft_ranking.build_connectivity_matrix(self.C, 10) > 0, axis=1)
        obs_per_cam = np.sum(1 * ~np.isnan(self.C), axis=1)[::2]

        n_cam = self.C.shape[0] // 2
        cams_dtype = [("neighbor_nodes", int), ("obs", int)]
        cam_values = np.array(list(zip(neighbor_nodes_per_cam, obs_per_cam)), dtype=cams_dtype)
        ordered_cam_indices = np.argsort(cam_values)[::-1]
        ref_cam_idx = ordered_cam_indices[0]

        def rearange_corresp_matrix(C, ref_cam_idx):
            C = np.vstack([C[2 * ref_cam_idx : 2 * ref_cam_idx + 2, :], C])
            C = np.delete(C, 2 * (ref_cam_idx + 1), axis=0)
            C = np.delete(C, 2 * (ref_cam_idx + 1), axis=0)
            return C

        def rearange_list(input_list, new_indices):
            new_list = [input_list[idx] for idx in np.argsort(new_indices)]
            return new_list

        # part 2: the reference camera will be fix so it goes on top of all lists to work with the code
        #         rearange input rpcs, myimages, crops, footprints, C, C_v2, pairs_to_triangulate, etc.

        self.n_adj += 1
        self.n_new -= 1
        self.C = rearange_corresp_matrix(self.C, ref_cam_idx)
        C_v2 = np.vstack([self.C_v2[ref_cam_idx, :], self.C_v2])
        self.C_v2 = np.delete(C_v2, ref_cam_idx + 1, axis=0)
        new_cam_indices = np.arange(n_cam)
        new_cam_indices[new_cam_indices < ref_cam_idx] += 1
        new_cam_indices[ref_cam_idx] = 0
        new_pairs_to_triangulate = []
        for (cam_i, cam_j) in self.pairs_to_triangulate:
            new_cam_i, new_cam_j = new_cam_indices[cam_i], new_cam_indices[cam_j]
            new_pairs_to_triangulate.append((min(new_cam_i, new_cam_j), max(new_cam_i, new_cam_j)))
        self.pairs_to_triangulate = new_pairs_to_triangulate
        self.input_rpcs = rearange_list(self.input_rpcs, new_cam_indices)
        self.input_seq = rearange_list(self.input_seq, new_cam_indices)
        self.myimages = rearange_list(self.myimages, new_cam_indices)
        self.crop_offsets = rearange_list(self.crop_offsets, new_cam_indices)
        self.cam_centers = rearange_list(self.cam_centers, new_cam_indices)
        self.footprints = rearange_list(self.footprints, new_cam_indices)
        self.cameras = rearange_list(self.cameras, new_cam_indices)
        self.features = rearange_list(self.features, new_cam_indices)
        if self.input_masks is not None:
            self.input_masks = rearange_list(self.input_masks, new_cam_indices)

        flush_print("Using input image {} as reference image of the set".format(ref_cam_idx))
        flush_print("Reference geotiff: {}".format(self.myimages[0]))
        flush_print("Reference geotiff weight: {:.2f}".format(self.ref_cam_weight))
        flush_print("After this step the camera indices are modified to put the ref. camera at position 0,")
        flush_print("so they are not coincident anymore with the indices from the feature tracking step")

    def remove_all_obs_with_reprojection_error_higher_than(self, thr):
        """
        optinal step - this function filters the feature track observations output by compute_feature_tracks
        by removing those with a reprojection error higher than a certain threshold
        this may be useful to use under certain knowledge of the maximum reprojection error
        that should be expected for the input set of satellite images
        """

        t0 = timeit.default_timer()
        self.define_ba_parameters(verbose=False)
        args = [self.ba_params, {"max_iter": 1, "verbose": 0}, False, self.display_plots]
        _, _, err, _ = ba_core.run_ba_optimization(*args)
        _, ba_e, _, _ = err

        to_rm = ba_e > thr
        self.C_v2[self.ba_params.cam_ind[to_rm], self.ba_params.pts_ind[to_rm]] = np.nan
        pts_ind_to_rm, cam_ind_to_rm = self.ba_params.pts_ind[to_rm], self.ba_params.cam_ind[to_rm]
        args = [self.ba_params, pts_ind_to_rm, cam_ind_to_rm, thr, self.correction_params, False]
        p = ba_outliers.rm_outliers(*args)
        self.C = p.C
        self.pts3d = p.pts3d
        self.n_pts_fix = p.n_pts_fix
        t = timeit.default_timer() - t0
        to_print = [np.sum(to_rm), thr, t]
        flush_print("Removed {} obs with reprojection error above {} px ({:.2f} seconds)\n".format(*to_print))

    def save_estimated_params(self):
        """
        this function writes the camera parameters optimized by the bundle adjustment pipeline to txt files
        """
        for cam_idx, cam_prev_idx in enumerate(self.ba_params.cam_prev_indices):
            cam_id = loader.get_id(self.myimages[cam_prev_idx])
            params_fname = "{}/ba_params/{}.params".format(self.out_dir, cam_id)
            os.makedirs(os.path.dirname(params_fname), exist_ok=True)
            params_file = open(params_fname, "w")
            for k in self.ba_params.estimated_params[cam_idx].keys():
                params_file.write("{}\n".format(k))
                params_file.write(" ".join(["{:.16f}".format(v) for v in self.ba_params.estimated_params[cam_idx][k]]))
                params_file.write("\n")
            params_file.close()
        flush_print("All estimated camera parameters written at {}".format(os.path.dirname(params_fname)))

    def run(self):
        """
        this function runs the entire bundle adjustment pipeline
        """

        pipeline_start = timeit.default_timer()
        # feature tracking stage
        self.compute_feature_tracks()
        self.initialize_pts3d()

        if self.max_init_reproj_error is not None:
            self.remove_all_obs_with_reprojection_error_higher_than(thr=self.max_init_reproj_error)
        self.select_best_tracks(K=self.tracks_config["FT_K"], priority=self.tracks_config["FT_priority"])
        self.check_connectivity_graph(min_matches=5)

        # bundle adjustment stage
        if self.fix_ref_cam:
            self.fix_reference_camera()
        t0 = timeit.default_timer()
        self.define_ba_parameters(verbose=True)
        if self.clean_outliers:
            self.run_ba_softL1()
            self.clean_outlier_observations()
        self.run_ba_L2()
        optimization_time = loader.get_time_in_hours_mins_secs(timeit.default_timer() - t0)
        flush_print("Optimization problem solved in {} ({} iterations)\n".format(optimization_time, self.ba_iters))

        # create corrected camera models and save output files
        self.save_corrected_cameras()
        self.save_corrected_points()
        self.save_estimated_params()

        pipeline_time = loader.get_time_in_hours_mins_secs(timeit.default_timer() - pipeline_start)
        flush_print("BA pipeline completed in {}\n".format(pipeline_time))

