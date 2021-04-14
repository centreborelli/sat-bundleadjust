"""
A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery
author: Roger Mari <roger.mari@ens-paris-saclay.fr>
year: 2021

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

from bundle_adjust import ba_core, ba_outliers, ba_params, ba_rpcfit
from bundle_adjust import loader, cam_utils, geo_utils
from bundle_adjust.loader import flush_print
from .feature_tracks import ft_utils


class Error(Exception):
    pass


class BundleAdjustmentPipeline:
    def __init__(self, ba_data, tracks_config={}, extra_ba_config={}):
        """
        Args:
            ba_data: dictionary specifying the bundle adjustment input data, must contain
                     "in_dir": string, input directory containing precomputed tracks (if available)
                     "out_dir": string, output directory where the corrected RPC models will be written
                     "image_fnames": list of strings, contains all paths to input geotiffs
                     "rpcs": list of rpc models
                     "crops": a list of dictionaries, each dictionary with a field "crop" containing the matrix
                              corresponding to the image, and then the fields "col0", "row0", "width", "height"
                              which delimit the area of the geotiff file that is seen in "crop"
                              i.e. crop = entire_geotiff[row0: row0 + height, col0 : col0 + width]

                    "
            tracks_config (optional): dictionary specifying the configuration for the feature tracking
                                      see feature_tracks.ft_utils.init_feature_tracks_config to check
                                      the considered feature tracking parameters
            extra_config (optional): dictionary specifying any extra parameters to customize the configuration
                                     of the bundle adjustment pipeline. Considered extra parameters are
                                     "cam_model": string, the camera model that will be used at internal level for
                                                  the bundle adjustment. values: "affine", "perspective" or "rpc"
                                     "aoi": dict, area of interest where RPC have to be consistent in GeoJSON format
                                            and longitude and latitude coordinates. if this field is not specified
                                            the aoi is computed from the union of all geotiff footprints
                                     "n_adj": integer, number of input images already adjusted (if any)
                                              these images have to be on top of the lists in ba_data
                                              and they will be frozen during the numerical optimization
                                     "correction_params": list of strings, contains the parameters to optimize
                                                          values: 'R' (rotation), 'T' (translation), 'K' (calibration)
                                                          or 'COMMON_K' to fix K in all cams when 'K' is in the list
                                     "predefined_matches": boolean, set to True if predefined matches are
                                                           available in the input directory; else False
                                     "fix_ref_cam": boolean, set to True to fix a reference camera; else False
                                     "ref_cam_weight": float, weight of the importance of the reference camera
                                                       the higher the more weight it is given in the cost function
                                     "clean_outliers": boolean, set to True to apply outlier feature observations
                                                       filtering after some initial iterations using soft-L1 cost
                                     "verbose": boolean, set to False to avoid printing output on command line
        """

        # read bundle adjustment input data
        self.in_dir = ba_data["in_dir"]
        self.out_dir = ba_data["out_dir"]
        os.makedirs(self.out_dir, exist_ok=True)
        self.myimages = ba_data["image_fnames"].copy()
        self.input_rpcs = ba_data["rpcs"].copy()
        self.input_seq = [f["crop"] for f in ba_data["crops"]]
        self.crop_offsets = [{k: c[k] for k in ["col0", "row0", "width", "height"]} for c in ba_data["crops"]]

        # read the feature tracking configuration
        self.tracks_config = ft_utils.init_feature_tracks_config(tracks_config)

        # read extra bundle adjustment configuration parameters
        self.cam_model = extra_ba_config.get("cam_model", "rpc")
        if self.cam_model not in ["rpc", "affine", "perspective"]:
            raise Error("cam_model is not valid")
        aoi_args = [self.myimages, self.input_rpcs, self.crop_offsets]
        self.aoi = extra_ba_config.get("aoi", loader.load_aoi_from_multiple_geotiffs(*aoi_args))
        self.n_adj = extra_ba_config.get("n_adj", 0)
        self.n_new = len(self.myimages) - self.n_adj
        self.correction_params = extra_ba_config.get("correction_params", ["R"])
        self.predefined_matches = extra_ba_config.get("predefined_matches", False)
        self.fix_ref_cam = extra_ba_config.get("fix_ref_cam", True)
        self.ref_cam_weight = extra_ba_config.get("ref_cam_weight", 1.0) if self.fix_ref_cam else 1.0
        self.clean_outliers = extra_ba_config.get("clean_outliers", True)
        self.verbose = extra_ba_config.get("verbose", True)

        flush_print("Bundle Adjustment Pipeline created")
        flush_print("-------------------------------------------------------------")
        flush_print("    - input path:     {}".format(self.in_dir))
        flush_print("    - output path:    {}".format(self.out_dir))
        center_lat, center_lon = self.aoi["center"][1], self.aoi["center"][0]
        flush_print("    - aoi center:     ({:.4f}, {:.4f}) lat, lon".format(center_lat, center_lon))
        sq_km = geo_utils.measure_squared_km_from_lonlat_geojson(self.aoi)
        flush_print("    - aoi area:       {:.2f} squared km".format(sq_km))
        flush_print("    - input cameras:  {}".format(len(self.myimages)))
        flush_print("\nConfiguration:")
        flush_print("    - cam_model:           {}".format(self.cam_model))
        flush_print("    - n_new:               {}".format(self.n_new))
        flush_print("    - n_adj:               {}".format(self.n_adj))
        flush_print("    - correction_params:   {}".format(self.correction_params))
        flush_print("    - predefined_matches:  {}".format(self.predefined_matches))
        flush_print("    - fix_ref_cam:         {}".format(self.fix_ref_cam))
        flush_print("    - ref_cam_weight:      {}".format(self.ref_cam_weight))
        flush_print("    - clean_outliers:      {}".format(self.clean_outliers))
        flush_print("    - tracks_selection:    {}".format(self.tracks_config["FT_K"] > 0))
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
        if "cameras" in ba_data.keys():
            self.cameras = ba_data["cameras"].copy()
        else:
            self.set_cameras()
        self.cam_centers = self.get_optical_centers()
        self.footprints = self.get_footprints()
        flush_print("\n")

    def get_footprints(self):
        """
        this function outputs a list containing the footprint associated to each input satellite image
        each footprint contains a polygon in utm coordinates covering the geographic area seen in the image
        and the srtm4 altitude value associated to the center of the polygon
        """
        t0 = timeit.default_timer()
        flush_print("Getting image footprints...")
        args = [self.myimages, self.input_rpcs, self.crop_offsets]
        lonlat_footprints, alts = loader.load_geotiff_lonlat_footprints(*args)
        utm_footprints = []
        for x, z in zip(lonlat_footprints, alts):
            utm_footprints.append({"geojson": geo_utils.utm_geojson_from_lonlat_geojson(x), "z": z})
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
            optical_centers = [cam_utils.get_perspective_optical_center(P) for P in tmp_perspective_cams]
        else:
            optical_centers = [cam_utils.get_perspective_optical_center(P) for P in self.cameras]
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
        if self.tracks_config["FT_sift_detection"] == "s2p" and os.path.exists(self.in_dir + "/../rpcs_init"):
            args = [self.myimages, self.in_dir + "/../rpcs_init", "", "rpc", False]
            ft_rpcs = loader.load_rpcs_from_dir(*args)
        else:
            ft_rpcs = self.input_rpcs
        local_data = {
            "n_adj": self.n_adj,
            "fnames": self.myimages,
            "images": self.input_seq,
            "rpcs": ft_rpcs,
            "offsets": self.crop_offsets,
            "footprints": self.footprints,
            "optical_centers": self.cam_centers,
            "aoi": self.aoi,
        }

        if self.predefined_matches:
            args = [self.in_dir + "/predefined_matches", self.out_dir, local_data, self.tracks_config]
            feature_tracks, self.feature_tracks_running_time = ft_utils.load_tracks_from_predefined_matches(*args)
        else:
            from bundle_adjust.feature_tracks.ft_pipeline import FeatureTracksPipeline

            args = [self.in_dir, self.out_dir, local_data]
            ft_pipeline = FeatureTracksPipeline(*args, tracks_config=self.tracks_config)
            feature_tracks, self.feature_tracks_running_time = ft_pipeline.build_feature_tracks()

        # sanity checks to verify if the feature tracking output looks good
        new_camera_indices = np.arange(local_data["n_adj"], len(local_data["fnames"]))
        args = [new_camera_indices, feature_tracks["pairs_to_match"], feature_tracks["pairs_to_triangulate"]]
        fatal_error, err_msg, disconnected_cameras1 = ft_utils.check_pairs(*args)
        if fatal_error:
            raise Error("{}".format(err_msg))
        fatal_error, err_msg, disconnected_cameras2 = ft_utils.check_correspondence_matrix(feature_tracks["C"])
        if fatal_error:
            raise Error("{}".format(err_msg))
        disconnected_cameras = np.unique(disconnected_cameras1 + disconnected_cameras2).tolist()

        # the feature tracking output looks good, so the pipeline can continue
        self.features = feature_tracks["features"]
        self.pairs_to_triangulate = feature_tracks["pairs_to_triangulate"]
        self.C = feature_tracks["C"]
        if self.cam_model == "rpc":
            for i in range(self.C.shape[0] // 2):
                self.C[2 * i, :] += self.crop_offsets[i]["col0"]
                self.C[2 * i + 1, :] += self.crop_offsets[i]["row0"]
        self.C_v2 = feature_tracks["C_v2"]
        self.n_pts_fix = feature_tracks["n_pts_fix"]
        del feature_tracks

        if len(disconnected_cameras) > 0:
            # there was a small group of disconnected cameras which we will discard
            # the pipeline will continue and try to correct the cameras that are left
            self.drop_disconnected_cameras(disconnected_cameras)
            flush_print("Cameras {} were dropped due to insufficient feature tracks\n".format(disconnected_cameras))

    def initialize_pts3d(self):
        """
        this function initializes the ECEF coordinates of the 3d points that project into the feature tracks
        """
        from .feature_tracks.ft_triangulate import init_pts3d

        self.pts3d = np.zeros((self.C.shape[1], 3), dtype=np.float32)
        n_pts_opt = self.C.shape[1] - self.n_pts_fix
        if self.n_pts_fix > 0:
            t0 = timeit.default_timer()
            flush_print("Initializing {} fixed 3d point coords...".format(self.n_pts_fix))
            C_fixed_pts3d = self.C[: self.n_adj * 2, : self.n_pts_fix]
            args = [C_fixed_pts3d, self.cameras, self.cam_model, self.pairs_to_triangulate, False]
            self.pts3d[: self.n_pts_fix, :] = init_pts3d(*args)
            flush_print("...done in {:.2f} seconds".format(timeit.default_timer() - t0))
        t0 = timeit.default_timer()
        flush_print("Initializing {} 3d point coords to optimize...".format(n_pts_opt))
        C_opt_pts3d = self.C[:, -n_pts_opt:]
        args = [C_opt_pts3d, self.cameras, self.cam_model, self.pairs_to_triangulate, False]
        self.pts3d[-n_pts_opt:, :] = init_pts3d(*args)
        flush_print("...done in {:.2f} seconds".format(timeit.default_timer() - t0))

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
        args = [self.ba_params, ls_params_L1, True, False]
        _, self.ba_sol, self.init_e, self.ba_e, iters = ba_core.run_ba_optimization(*args)
        self.ba_iters += iters

    def run_ba_L2(self):
        """
        this function runs the bundle adjustment optimization with a classic L2 norm for the reprojection errors
        """
        args = [self.ba_params, None, True, False]
        _, self.ba_sol, self.init_e, self.ba_e, iters = ba_core.run_ba_optimization(*args)
        self.ba_iters += iters

    def save_corrected_cameras(self):
        """
        this function recovers the optimized 3d points and camera models from the bundle adjustment solution
        """
        args = [self.ba_sol, self.pts3d, self.cameras]
        self.corrected_pts3d, self.corrected_cameras = self.ba_params.reconstruct_vars(*args)
        if self.cam_model in ["perspective", "affine"]:
            self.save_corrected_matrices()
        flush_print("Fitting corrected RPC models...")
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
        out_dir = os.path.join(self.out_dir, "rpcs_adj")
        fnames = [os.path.join(out_dir, loader.get_id(fn) + ".rpc_adj") for fn in self.myimages]
        if self.cam_model in ["perspective", "affine"]:
            # cam_model is a projection matrix
            for cam_idx, (fn, cam) in enumerate(zip(fnames, self.corrected_cameras)):
                tracks_seen_current_camera = ~np.isnan(self.ba_params.C[2 * cam_idx])
                pts3d_seen_current_camera = self.ba_params.pts3d_ba[tracks_seen_current_camera]
                args = [cam, self.input_rpcs[cam_idx], self.crop_offsets[cam_idx], pts3d_seen_current_camera]
                rpc_calib, err, margin = ba_rpcfit.fit_rpc_from_projection_matrix(*args)
                errors = " [1e-4 px] max / med: {:.2f} / {:.2f}".format(1e4 * err.max(), 1e4 * np.median(err))
                flush_print("cam {:2} - RPC fit error per obs {} (margin {})".format(cam_idx, errors, margin))
                os.makedirs(os.path.dirname(fn), exist_ok=True)
                rpc_calib.write_to_file(fn)
        else:
            # cam_model is "rpc"
            for cam_idx in np.arange(self.n_adj):
                rpc_calib = self.cameras[cam_idx]
                os.makedirs(os.path.dirname(fnames[cam_idx]), exist_ok=True)
                rpc_calib.write_to_file(fnames[cam_idx])
            for cam_idx in np.arange(self.n_adj, self.n_adj + self.n_new):
                Rt_vec = self.corrected_cameras[cam_idx]
                original_rpc = self.cameras[cam_idx]
                tracks_seen_current_camera = ~np.isnan(self.ba_params.C[2 * cam_idx])
                pts3d_seen_current_camera = self.ba_params.pts3d_ba[tracks_seen_current_camera]
                args = [Rt_vec.reshape(1, 9), original_rpc, self.crop_offsets[cam_idx], pts3d_seen_current_camera]
                rpc_calib, err, margin = ba_rpcfit.fit_Rt_corrected_rpc(*args)
                errors = " [1e-4 px] max / med: {:.2f} / {:.2f}".format(1e4 * err.max(), 1e4 * np.median(err))
                flush_print("cam {:2} - RPC fit error per obs {} (margin {})".format(cam_idx, errors, margin))
                os.makedirs(os.path.dirname(fnames[cam_idx]), exist_ok=True)
                rpc_calib.write_to_file(fnames[cam_idx])
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
        _, n_cc, _, _, missing_cams = ft_utils.build_connectivity_graph(self.C, min_matches=min_matches, verbose=True)
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

        # part 2: the reference camera will be fixed so it goes on top of all lists to work with the code
        #         rearange input rpcs, myimages, crops, footprints, C, C_v2, pairs_to_triangulate, etc.

        self.n_adj += 1
        self.n_new -= 1
        old_cam_indices = np.arange(n_cam)
        new_cam_indices = np.arange(n_cam)
        new_cam_indices[new_cam_indices < ref_cam_idx] += 1
        new_cam_indices[ref_cam_idx] = 0
        cam_indices = np.vstack([new_cam_indices, old_cam_indices]).T
        self.permute_cameras(cam_indices)

        flush_print("Using input image {} as reference image of the set".format(ref_cam_idx))
        flush_print("Reference geotiff: {}".format(self.myimages[0]))
        flush_print("Reference geotiff weight: {:.2f}".format(self.ref_cam_weight))
        flush_print("After this step the camera indices are modified to put the ref. camera at position 0,")
        flush_print("so they are not coincident anymore with the indices from the feature tracking step")

    def permute_cameras(self, cam_indices):
        """
        given a new order of camera indices rearanges all data of the ba_pipeline
        cam_indices is an array of size Mx2 where M is the number of cameras
        column 0 of cam_indices = new camera indices
        column 1 of cam_indices = old camera indices
        """

        def rearange_list(input_list, indices):
            new_list = [input_list[old_i] for new_i, old_i in sorted(indices, key=lambda x: x[0])]
            return new_list

        # rearange C and C_v2
        new_C = [self.C[2 * old_i : 2 * old_i + 2] for new_i, old_i in sorted(cam_indices, key=lambda x: x[0])]
        self.C = np.vstack(new_C)
        new_C_v2 = [self.C_v2[old_i] for new_i, old_i in sorted(cam_indices, key=lambda x: x[0])]
        self.C_v2 = np.vstack(new_C_v2)

        # rearange pairs_to_triangulate
        new_pairs_to_triangulate = []
        new_cam_idx_from_old_cam_idx = dict(zip(cam_indices[:, 1], cam_indices[:, 0]))
        for (cam_i, cam_j) in self.pairs_to_triangulate:
            new_cam_i, new_cam_j = new_cam_idx_from_old_cam_idx[cam_i], new_cam_idx_from_old_cam_idx[cam_j]
            new_pairs_to_triangulate.append((min(new_cam_i, new_cam_j), max(new_cam_i, new_cam_j)))
        self.pairs_to_triangulate = new_pairs_to_triangulate

        # rearange the rest
        self.input_rpcs = rearange_list(self.input_rpcs, cam_indices)
        self.input_seq = rearange_list(self.input_seq, cam_indices)
        self.myimages = rearange_list(self.myimages, cam_indices)
        self.crop_offsets = rearange_list(self.crop_offsets, cam_indices)
        self.cam_centers = rearange_list(self.cam_centers, cam_indices)
        self.footprints = rearange_list(self.footprints, cam_indices)
        self.cameras = rearange_list(self.cameras, cam_indices)
        self.features = rearange_list(self.features, cam_indices)

    def drop_disconnected_cameras(self, camera_indices_to_drop):
        """
        it may be impossible to find enough tie points for all images
        in certain challenging scenarios (e.g. clouds, water).
        this function handles this drawback by removing disconnected cameras from the input
        then the bundle adjustment pipeline continues with the cameras that are left
        """
        n_cam_before_drop = len(self.myimages)
        camera_indices_left = np.sort(list(set(np.arange(n_cam_before_drop)) - set(camera_indices_to_drop)))
        n_cam_after_drop = len(camera_indices_left)
        camera_indices = np.vstack([np.arange(n_cam_after_drop), camera_indices_left]).T
        self.n_adj -= np.sum(np.array(camera_indices_to_drop) < self.n_adj)
        self.n_new -= np.sum(np.array(camera_indices_to_drop) >= self.n_adj)
        self.permute_cameras(camera_indices)

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

    def save_feature_tracks(self):
        """
        this function writes an output svg per each input geotiff containing the feature track observations
        that were employed to bundle adjust the associated camera model
        """
        mask = ~np.isnan(self.ba_params.C[::2])
        for cam_idx, cam_prev_idx in enumerate(self.ba_params.cam_prev_indices):
            cam_id = loader.get_id(self.myimages[cam_prev_idx])
            svg_fname = "{}/ba_tracks/{}.svg".format(self.out_dir, cam_id)
            pts2d = self.ba_params.C[2 * cam_idx : 2 * cam_idx + 2, mask[cam_idx]].T
            offset = self.crop_offsets[cam_prev_idx]
            if self.cam_model == "rpc":
                pts2d[:, 0] -= offset["col0"]
                pts2d[:, 1] -= offset["row0"]
            ft_utils.save_pts2d_as_svg(svg_fname, pts2d, c="yellow", w=offset["width"], h=offset["height"])

    def remove_feature_tracking_files(self):
        """
        this function removes all output files created by FeatureTracksPipeline
        """
        ft_dir = self.out_dir
        if os.path.exists("{}/features".format(ft_dir)):
            os.system("rm -r {}/features".format(ft_dir))
        if os.path.exists("{}/features_utm".format(ft_dir)):
            os.system("rm -r {}/features_utm".format(ft_dir))
        if os.path.exists("{}/matches.npy".format(ft_dir)):
            os.system("rm -r {}/matches.npy".format(ft_dir))
        if os.path.exists("{}/pairs_matching.npy".format(ft_dir)):
            os.system("rm -r {}/pairs_matching.npy".format(ft_dir))
        if os.path.exists("{}/pairs_triangulation.npy".format(ft_dir)):
            os.system("rm -r {}/pairs_triangulation.npy".format(ft_dir))

    def save_figures(self):
        """
        this function saves some images illustrating the performance of the bundle adjustment
        """
        import rasterio
        import matplotlib.pyplot as plt

        # (1) save png histogram of reprojection errors
        img_path = os.path.join(self.out_dir, "error_histograms.png")
        ba_core.save_histogram_of_errors(img_path, self.init_e, self.ba_e)

        # (2) save georeferenced rasters with the interpolated reprojection error over the aoi
        tif_path_before = os.path.join(self.out_dir, "error_before.tif")
        ba_core.save_heatmap_of_reprojection_error(tif_path_before, self.ba_params, self.init_e, self.aoi)
        tif_path_after = os.path.join(self.out_dir, "error_after.tif")
        ba_core.save_heatmap_of_reprojection_error(tif_path_after, self.ba_params, self.ba_e, self.aoi)

        # (3) save png showing the interpolated reprojection errors
        img_path = os.path.join(self.out_dir, "interpolated_errors.png")
        vmin, vmax = 0.0, 2.0
        fig, axes = plt.subplots(1, 2, figsize=(20, 20))
        for i, (tif_fn, title) in enumerate(zip([tif_path_before, tif_path_after], ["Before BA", "After BA"])):
            with rasterio.open(tif_fn) as src:
                raster = src.read()[0, :, :].astype(np.float)
            im = axes[i].imshow(raster, vmin=vmin, vmax=vmax)
            axes[i].axis("off")
            axes[i].set_title(title)
        plt.subplots_adjust(wspace=0.01)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.25, 0.02, 0.5])
        cb = fig.colorbar(im, cax=cbar_ax)
        # cb = fig.colorbar(im, ax=axes[i])
        cb.set_label("Reprojection error across AOI (pixel units)", rotation=270, labelpad=25)
        plt.savefig(img_path, bbox_inches="tight")

        # (4) save connectivity graph
        img_path = os.path.join(self.out_dir, "connectivity_graph.png")
        ft_utils.save_connectivity_graph(img_path, self.ba_params.C, min_matches=0)

    def run(self):
        """
        this function runs the entire bundle adjustment pipeline
        """

        pipeline_start = timeit.default_timer()
        # feature tracking stage
        self.compute_feature_tracks()
        self.initialize_pts3d()
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
        self.save_feature_tracks()
        self.save_corrected_points()
        self.save_estimated_params()
        self.save_figures()

        pipeline_time = loader.get_time_in_hours_mins_secs(timeit.default_timer() - pipeline_start)
        flush_print("\nBundle adjustment pipeline completed in {}\n".format(pipeline_time))
