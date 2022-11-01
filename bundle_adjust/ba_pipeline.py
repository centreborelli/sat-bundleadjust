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
import srtm4
import copy
import shutil

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
                     "images": list of instances of the class SatelliteImage
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
                                     "save_figures": boolean, set to False to avoid save illustration images
                                                     of the reprojection error before and after bundle adjustment
        """

        # read bundle adjustment input data
        self.in_dir = ba_data["in_dir"]
        self.out_dir = ba_data["out_dir"]
        os.makedirs(self.out_dir, exist_ok=True)
        self.images = ba_data["images"]

        # read the feature tracking configuration
        self.tracks_config = ft_utils.init_feature_tracks_config(tracks_config)

        # read extra bundle adjustment configuration parameters
        self.cam_model = extra_ba_config.get("cam_model", "rpc")
        if self.cam_model not in ["rpc", "affine", "perspective"]:
            raise Error("cam_model is not valid")
        self.aoi = extra_ba_config.get("aoi", None)
        self.n_adj = extra_ba_config.get("n_adj", 0)
        self.n_new = len(self.images) - self.n_adj
        self.correction_params = extra_ba_config.get("correction_params", ["R"])
        self.predefined_matches = extra_ba_config.get("predefined_matches", False)
        self.fix_ref_cam = extra_ba_config.get("fix_ref_cam", False)
        self.ref_cam_weight = extra_ba_config.get("ref_cam_weight", 1.0) if self.fix_ref_cam else 1.0
        self.clean_outliers = extra_ba_config.get("clean_outliers", True)
        self.max_init_reproj_error = extra_ba_config.get("max_init_reproj_error", None)
        self.save_figures = extra_ba_config.get("save_figures", True)

        # if aoi is not defined we take the union of all footprints
        self.set_footprints()
        if self.aoi is None:
            self.predefined_aoi = False
            self.aoi = loader.load_aoi_from_multiple_images(self.images)
        else:
            self.predefined_aoi = True

        # set initial cameras
        if "cameras" in ba_data.keys():
            self.cameras = ba_data["cameras"].copy()
        else:
            self.set_cameras()
        self.set_camera_centers()
        flush_print("\n")

        flush_print("Bundle Adjustment Pipeline created")
        flush_print("-------------------------------------------------------------")
        flush_print("    - input path:     {}".format(self.in_dir))
        flush_print("    - output path:    {}".format(self.out_dir))
        center_lat, center_lon = self.aoi["center"][1], self.aoi["center"][0]
        flush_print("    - aoi center:     ({:.4f}, {:.4f}) lat, lon".format(center_lat, center_lon))
        sq_km = geo_utils.measure_squared_km_from_lonlat_geojson(self.aoi)
        flush_print("    - aoi area:       {:.2f} squared km".format(sq_km))
        flush_print("    - input cameras:  {}".format(len(self.images)))
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

        # save initial rpcs to output directory
        init_rpc_dir = os.path.join(self.out_dir, "rpcs")
        init_rpc_paths = ["{}/{}.rpc".format(init_rpc_dir, loader.get_id(im.geotiff_path)) for im in self.images]
        loader.save_rpcs(init_rpc_paths, [im.rpc for im in self.images])

    def set_footprints(self):
        """
        this function outputs a list containing the footprint associated to each input satellite image
        each footprint contains a polygon in utm coordinates covering the geographic area seen in the image
        and the srtm4 altitude value associated to the center of the polygon
        """
        t0 = timeit.default_timer()
        flush_print("Getting image footprints...")
        lonslats = np.array([[im.rpc.lon_offset, im.rpc.lat_offset] for im in self.images])
        alts = srtm4.srtm4(lonslats[:, 0], lonslats[:, 1])
        import warnings
        warnings.filterwarnings("ignore")
        for im, h in zip(self.images, alts):
            im.set_footprint(alt=h)
        flush_print("...done in {:.2f} seconds".format(timeit.default_timer() - t0))

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

    def set_camera_centers(self):
        """
        this function computes an approximation of an optical center for each camera model
        this is done by approximating the RPCs as perspective projection matrices and retrieving the camera center
        """
        t0 = timeit.default_timer()
        flush_print("Estimating camera positions...")
        if self.cam_model != "perspective":
            for im in self.images:
                im.set_camera_center()
        else:
            for im, cam in zip(self.images, self.cameras):
                _, _, _, center = cam_utils.decompose_perspective_camera(cam)
                im.set_camera_center(center=center)
        flush_print("...done in {:.2f} seconds".format(timeit.default_timer() - t0))

    def set_cameras(self):
        """
        this function sets which camera model representation has to be used at internal level
        the input camera models are RPCs, but the BA pipeline can either use those or approximations
        such approximations can take the form of perspective or affine projection matrices
        """
        if self.cam_model == "affine":
            lon, lat = self.aoi["center"][0], self.aoi["center"][1]
            alt = srtm4.srtm4(lon, lat)
            x, y, z = geo_utils.latlon_to_ecef_custom(lat, lon, alt)
            self.cameras = [cam_utils.affine_rpc_approx(im.rpc, x, y, z, im.offset) for im in self.images]
            #self.check_projection_matrices(err)
        elif self.cam_model == "perspective":
            self.cameras = [cam_utils.perspective_rpc_approx(im.rpc, im.offset)[0] for im in self.images]
            #self.check_projection_matrices(err)
        else:
            self.cameras = [copy.copy(im.rpc) for im in self.images]

    def compute_feature_tracks(self):
        """
        this function launches the feature tracking pipeline, which covers the following steps
            - feature detection
            - stereo pairs selection
            - pairwise matching
            - feature tracks construction
            - feature tracks selection (optional)
        """
        import copy
        ft_images = [copy.copy(im) for im in self.images]
        if os.path.exists(os.path.join(self.in_dir, "../rpcs_init")):
            args = [[im.geotiff_path for im in ft_images], os.path.join(self.in_dir, "../rpcs_init"), "", "rpc", False]
            ft_rpcs = loader.load_rpcs_from_dir(*args)
            for im, rpc in zip(ft_images, ft_rpcs):
                im.rpc = rpc
            lonslats = np.array([[im.rpc.lon_offset, im.rpc.lat_offset] for im in ft_images])
            alts = srtm4.srtm4(lonslats[:, 0], lonslats[:, 1])
            for im, h in zip(ft_images, alts):
                im.set_footprint(alt=h)

        local_data = {
            "n_adj": self.n_adj,
            "images": ft_images,
            "aoi": self.aoi,
        }

        output_dir = os.path.join(self.out_dir, "matches")
        if self.predefined_matches:
            args = [os.path.join(self.in_dir, "predefined_matches"), output_dir, local_data, self.tracks_config]
            feature_tracks, self.feature_tracks_running_time = ft_utils.load_tracks_from_predefined_matches(*args)
        else:
            from bundle_adjust.feature_tracks.ft_pipeline import FeatureTracksPipeline

            args = [output_dir, output_dir, local_data] # output dir is changed !
            ft_pipeline = FeatureTracksPipeline(*args, tracks_config=self.tracks_config)
            feature_tracks, self.feature_tracks_running_time = ft_pipeline.build_feature_tracks()

        # sanity checks to verify if the feature tracking output looks good
        new_camera_indices = np.arange(local_data["n_adj"], len(local_data["images"]))
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
                self.C[2 * i, :] += self.images[i].offset["col0"]
                self.C[2 * i + 1, :] += self.images[i].offset["row0"]
        self.C_v2 = feature_tracks["C_v2"]
        self.n_pts_fix = feature_tracks["n_pts_fix"]
        del feature_tracks

        if len(disconnected_cameras) > 0:
            # there was a small group of disconnected cameras which we will discard
            # the pipeline will continue and try to correct the cameras that are left
            self.drop_disconnected_cameras(disconnected_cameras)
            affected_geotiff_fnames = [os.path.basename(self.images[idx].geotiff_path) for idx in disconnected_cameras]
            flush_print("Cameras {} were dropped due to insufficient feature tracks".format(disconnected_cameras))
            flush_print("The affected images are:\n{}\n".format("\n".join(affected_geotiff_fnames)))

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
        cam_centers = [im.center for im in self.images]
        args = [self.C, self.pts3d, self.cameras, self.cam_model, self.pairs_to_triangulate, cam_centers]
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
        ls_params_L1 = {"loss": "soft_l1", "f_scale": 1.0, "max_iter": 300}
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
        fnames = [os.path.join(out_dir, loader.get_id(im.geotiff_path) + "_pinhole.json") for im in self.images]
        loader.save_projection_matrices(fnames, self.cameras, [im.offset for im in self.images])
        flush_print("\nInitial projection matrices written at {}\n".format(out_dir))

    def save_corrected_matrices(self):
        """
        this function writes the corrected projection matrices to json files
        """
        out_dir = os.path.join(self.out_dir, "P_adj")
        fnames = [os.path.join(out_dir, loader.get_id(im.geotiff_path) + "_pinhole_adj.json") for im in self.images]
        loader.save_projection_matrices(fnames, self.corrected_cameras, [im.offset for im in self.images])
        flush_print("Bundle adjusted projection matrices written at {}\n".format(out_dir))

    def save_corrected_rpcs(self):
        """
        this function writes the corrected RPC models to txt files
        """
        out_dir = os.path.join(self.out_dir, "rpcs_adj")
        fnames = [os.path.join(out_dir, loader.get_id(im.geotiff_path) + ".rpc_adj") for im in self.images]
        if self.cam_model in ["perspective", "affine"]:
            # cam_model is a projection matrix
            for cam_idx, (fn, cam) in enumerate(zip(fnames, self.corrected_cameras)):
                tracks_seen_current_camera = ~np.isnan(self.ba_params.C[2 * cam_idx])
                pts3d_seen_current_camera = self.ba_params.pts3d_ba[tracks_seen_current_camera]
                args = [cam, self.global_transform,
                        self.images[cam_idx].rpc, self.images[cam_idx].offset, pts3d_seen_current_camera]
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
                cam_prev_indices = list(self.ba_params.cam_prev_indices)
                tracks_seen_current_camera = ~np.isnan(self.ba_params.C[2 * cam_prev_indices.index(cam_idx)])
                pts3d_seen_current_camera = self.ba_params.pts3d_ba[tracks_seen_current_camera]
                args = [Rt_vec.reshape(1, 9), self.global_transform,
                        original_rpc, self.images[cam_idx].offset, pts3d_seen_current_camera]
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
        pts3d_to_save = self.ba_params.pts3d_ba.copy()
        if self.global_transform is not None:
            pts3d_to_save -= self.global_transform
        loader.write_point_cloud_ply(pts3d_adj_ply_path, pts3d_to_save)
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
                cam_centers = [im.center for im in self.images]
                args = [self.C, self.pts3d, self.cameras, self.cam_model, self.pairs_to_triangulate, cam_centers]
                C_reproj = ft_ranking.compute_C_reproj(*args)
            else:
                C_reproj = np.zeros(C_scale.shape)

            true_if_new_track = np.sum(~np.isnan(self.C[::2, :])[-self.n_new :] * 1, axis=0).astype(bool)
            C_new = self.C[:, true_if_new_track]
            C_scale_new = C_scale[:, true_if_new_track]
            C_reproj_new = C_reproj[:, true_if_new_track]
            prev_track_indices = np.arange(len(true_if_new_track))[true_if_new_track]
            args = [C_new, C_scale_new, C_reproj_new, K, priority, True]

            if self.tracks_config["FT_skysat_sensor_aware"]:
                selected_track_indices = ft_ranking.select_best_tracks_sensor_aware(self.images, *args)
            else:
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
        _, _, _, n_cc, missing_cams = ft_utils.build_connectivity_graph(self.C, min_matches=min_matches, verbose=True)
        self.connectivity_graph_looks_good = True
        if n_cc > 1:
            self.connectivity_graph_looks_good = False
            to_print = [n_cc, min_matches]
            print("WARNING: Connectivity graph has {} connected components (min_matches = {})".format(*to_print))
            to_print = [len(missing_cams), missing_cams]
            print("         {} missing cameras from the largest connected component: {}\n".format(*to_print))

        #from bundle_adjust import ba_utils
        #ba_utils.display_lonlat_geojson_list_over_map([im.lonlat_geojson for im in self.images], 12, missing_cams)

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
        flush_print("Reference geotiff: {}".format(self.images[0].geotiff_path))
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
            try:
                new_cam_i, new_cam_j = new_cam_idx_from_old_cam_idx[cam_i], new_cam_idx_from_old_cam_idx[cam_j]
                new_pairs_to_triangulate.append((min(new_cam_i, new_cam_j), max(new_cam_i, new_cam_j)))
            except:
                # one of the cameras of the pair was dropped, so the pair has to be dropped as well
                continue
        self.pairs_to_triangulate = new_pairs_to_triangulate

        # rearange the rest
        self.images = rearange_list(self.images, cam_indices)
        self.cameras = rearange_list(self.cameras, cam_indices)
        self.features = rearange_list(self.features, cam_indices)

    def drop_disconnected_cameras(self, camera_indices_to_drop):
        """
        it may be impossible to find enough tie points for all images
        in certain challenging scenarios (e.g. clouds, water).
        this function handles this drawback by removing disconnected cameras from the input
        then the bundle adjustment pipeline continues with the cameras that are left
        """
        n_cam_before_drop = len(self.images)
        camera_indices_left = np.sort(list(set(np.arange(n_cam_before_drop)) - set(camera_indices_to_drop)))
        n_cam_after_drop = len(camera_indices_left)
        camera_indices = np.vstack([np.arange(n_cam_after_drop), camera_indices_left]).T
        self.n_adj -= np.sum(np.array(camera_indices_to_drop) < self.n_adj)
        self.n_new -= np.sum(np.array(camera_indices_to_drop) >= self.n_adj)
        self.permute_cameras(camera_indices)

    def remove_all_obs_with_reprojection_error_higher_than(self, thr):
        """
        optinal step - this function filters the feature track observations output by compute_feature_tracks
        by removing those with a reprojection error higher than a certain threshold
        this may be useful to use under certain knowledge of the maximum reprojection error
        that should be expected for the input set of satellite images
        """
        print("\nAll observations with initial reprojection error higher than {} will be rejected !".format(thr))
        t0 = timeit.default_timer()
        self.define_ba_parameters(verbose=False)
        args = [self.ba_params, {"max_iter": 1, "verbose": 0}, False, False]
        _, _, _, ba_e, _ = ba_core.run_ba_optimization(*args)
        p = ba_outliers.rm_outliers(ba_e, self.ba_params, predef_thr=thr, verbose=False)
        if p.C.shape[0] != self.C.shape[0]:
            raise Error("At least one camera was lost, there might be something wrong with the input images")
        n_obs = np.sum(1 * ~np.isnan(self.C), axis=1)[::2]
        print("     - obs per cam before: {}", n_obs)
        n_obs = np.sum(1 * ~np.isnan(p.C), axis=1)[::2]
        print("     - obs per cam after: {}", n_obs)
        rm_track_indices = set(np.arange(self.C.shape[1])) - set(p.pts_prev_indices)
        rm_track_indices = np.array(list(rm_track_indices))
        n_tracks_rm = len(rm_track_indices)
        n_obs_rm = sum(~np.isnan(self.C[:, p.pts_prev_indices]).ravel() ^ ~np.isnan(p.C).ravel())/2
        n_obs_rm += sum(~np.isnan(self.C[:, rm_track_indices]).ravel())/2
        n_obs_in = sum(~np.isnan(self.C).ravel())/2
        n_tracks_in = self.C.shape[1]
        args = [n_obs_rm, n_obs_rm / n_obs_in * 100, n_tracks_rm, n_tracks_rm / n_tracks_in * 100]
        flush_print("Deleted {} observations ({:.2f}%) and {} tracks ({:.2f}%)".format(*args))
        flush_print("Rejection step took ({:.2f} seconds)\n".format(timeit.default_timer() - t0))

        # update the affected parameters and go on
        self.C = p.C
        self.pts3d = p.pts3d
        self.n_pts_fix = p.n_pts_fix
        self.C_v2 = self.C_v2[:, p.pts_prev_indices]
        self.C_v2[np.isnan(self.C[::2])] = np.nan

    def save_estimated_params(self):
        """
        this function writes the camera parameters optimized by the bundle adjustment pipeline to txt files
        """
        for cam_idx, cam_prev_idx in enumerate(self.ba_params.cam_prev_indices):
            cam_id = loader.get_id(self.images[cam_prev_idx].geotiff_path)
            params_fname = "{}/cam_params/{}.params".format(self.out_dir, cam_id)
            os.makedirs(os.path.dirname(params_fname), exist_ok=True)
            params_file = open(params_fname, "w")
            for k in self.ba_params.estimated_params[cam_idx].keys():
                params_file.write("{}\n".format(k))
                params_file.write(" ".join(["{:.16f}".format(v) for v in self.ba_params.estimated_params[cam_idx][k]]))
                params_file.write("\n")
            params_file.close()
        flush_print("All estimated camera parameters written at {}\n".format(os.path.dirname(params_fname)))

    def save_feature_tracks(self):
        """
        this function writes an output svg per each input geotiff containing the feature track observations
        that were employed to bundle adjust the associated camera model
        """
        mask = ~np.isnan(self.ba_params.C[::2])
        for cam_idx, cam_prev_idx in enumerate(self.ba_params.cam_prev_indices):
            cam_id = loader.get_id(self.images[cam_prev_idx].geotiff_path)
            svg_fname = "{}/ba_figures/track_obs/{}.svg".format(self.out_dir, cam_id)
            pts2d = self.ba_params.C[2 * cam_idx : 2 * cam_idx + 2, mask[cam_idx]].T
            offset = self.images[cam_prev_idx].offset
            if self.cam_model == "rpc":
                pts2d[:, 0] -= offset["col0"]
                pts2d[:, 1] -= offset["row0"]
            ft_utils.save_pts2d_as_svg(svg_fname, pts2d, c="yellow", w=offset["width"], h=offset["height"])

    def save_debug_figures(self):
        """
        this function saves some images illustrating the performance of the bundle adjustment
        """

        # (1) save image footprints and aoi contours
        img_path = os.path.join(self.out_dir, "ba_figures/image_footprints_and_aoi.png")
        input_ims_footprints_lonlat = [im.lonlat_geojson for im in self.images]
        loader.draw_image_footprints(img_path, input_ims_footprints_lonlat, self.aoi)

        # (2) save connectivity graph
        img_path = os.path.join(self.out_dir, "ba_figures/connectivity_graph.png")
        ft_utils.save_connectivity_graph(img_path, self.ba_params.C, min_matches=0)

        # (3) save png histogram of reprojection errors
        img_path = os.path.join(self.out_dir, "ba_figures/error_histograms.png")
        ba_core.save_histogram_of_errors(img_path, self.init_e, self.ba_e)

        # (4) save the interpolated reprojection error over the aoi
        aoi_roi = self.aoi if self.predefined_aoi else None
        tif_path_before = os.path.join(self.out_dir, "ba_figures/error_before.png")
        ba_core.save_heatmap_of_reprojection_error(tif_path_before, self.ba_params, self.init_e,
                                                   input_ims_footprints_lonlat, aoi_roi,
                                                   global_transform=self.global_transform)
        tif_path_after = os.path.join(self.out_dir, "ba_figures/error_after.png")
        ba_core.save_heatmap_of_reprojection_error(tif_path_after, self.ba_params, self.ba_e,
                                                   input_ims_footprints_lonlat, aoi_roi,
                                                   global_transform=self.global_transform)

    def correct_drift_object_space(self):

        pts3d_after_ba = self.ba_params.pts3d_ba # Nx3
        pts3d_before_ba = self.ba_params.pts3d # Nx3

        self.global_transform = np.mean(pts3d_after_ba - pts3d_before_ba, axis=0)
        errs = (pts3d_before_ba + self.global_transform) - pts3d_after_ba
        avg_errs = np.mean(errs, axis=0)
        flush_print("Global transform to correct drift in object space successfully computed.")
        flush_print("Average errors per dimension: {}\n".format(avg_errs))

    def run(self):
        """
        this function runs the entire bundle adjustment pipeline
        """

        pipeline_start = timeit.default_timer()
        # feature tracking stage
        self.compute_feature_tracks()
        self.initialize_pts3d()

        if not self.tracks_config["FT_save"]:
            shutil.rmtree(os.path.join(self.out_dir, "matches"))

        if self.max_init_reproj_error is not None:
            self.remove_all_obs_with_reprojection_error_higher_than(thr=self.max_init_reproj_error)

        # feature track selection is expected to work only on consistent connectivity graphs
        self.check_connectivity_graph(min_matches=5)
        if self.connectivity_graph_looks_good:
            self.select_best_tracks(K=self.tracks_config["FT_K"], priority=self.tracks_config["FT_priority"])
            self.check_connectivity_graph(min_matches=5)
        from .feature_tracks.ft_ranking import print_quick_camera_weights
        print_quick_camera_weights([im.geotiff_path for im in self.images], self.C)

        # bundle adjustment stage
        if self.fix_ref_cam:
            self.fix_reference_camera()
        t0 = timeit.default_timer()
        self.define_ba_parameters(verbose=True)
        if self.clean_outliers:
            self.run_ba_softL1()
            self.clean_outlier_observations()
        self.run_ba_L2()
        args = [self.ba_sol, self.pts3d, self.cameras]
        self.corrected_pts3d, self.corrected_cameras = self.ba_params.reconstruct_vars(*args)
        optimization_time = loader.get_time_in_hours_mins_secs(timeit.default_timer() - t0)
        flush_print("Optimization problem solved in {} ({} iterations)\n".format(optimization_time, self.ba_iters))

        # create corrected camera models and save output files
        if self.n_adj == 0:
            self.correct_drift_object_space()
        else:
            self.global_transform = None
        self.save_corrected_points()
        self.save_estimated_params()
        self.save_corrected_cameras()

        if self.save_figures:
            loader.save_geojson(os.path.join(self.out_dir, "AOI.json"), self.aoi)
            self.save_feature_tracks()
            self.save_debug_figures()

        pipeline_time = loader.get_time_in_hours_mins_secs(timeit.default_timer() - pipeline_start)
        flush_print("\nBundle adjustment pipeline completed in {}\n".format(pipeline_time))
