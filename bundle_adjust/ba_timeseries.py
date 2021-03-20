import numpy as np
import os
import sys

import timeit
import glob
import rpcm

from bundle_adjust import loader, ba_utils, geotools, vistools
from bundle_adjust.ba_pipeline import BundleAdjustmentPipeline
from bundle_adjust.loader import flush_print


class Error(Exception):
    pass


class Scene:
    def __init__(self, scene_config):

        t0 = timeit.default_timer()
        args = loader.load_dict_from_json(scene_config)

        # read scene args
        self.geotiff_dir = args["geotiff_dir"]
        self.rpc_dir = args["rpc_dir"]
        self.rpc_src = args["rpc_src"]
        self.dst_dir = args["output_dir"]

        # optional arguments for bundle adjustment configuration
        self.cam_model = args.get("cam_model", "rpc")
        self.ba_method = args.get("ba_method", "ba_bruteforce")
        self.selected_timeline_indices = args.get("timeline_indices", None)
        self.predefined_matches_dir = args.get("predefined_matches_dir", None)
        self.compute_aoi_masks = args.get("compute_aoi_masks", False)
        self.geotiff_label = args.get("geotiff_label", None)
        self.correction_params = args.get("correction_params", ["R"])
        self.n_dates = int(args.get("n_dates", 1))
        self.fix_ref_cam = args.get("fix_ref_cam", True)
        self.ref_cam_weight = float(args.get("ref_cam_weight", 1))
        self.filter_outliers = args.get("filter_outliers", True)
        self.reset = args.get("reset", True)

        # check geotiff_dir and rpc_dir exists
        if not os.path.isdir(self.geotiff_dir):
            raise Error('geotiff_dir "{}" does not exist'.format(self.geotiff_dir))
        if not os.path.isdir(self.rpc_dir):
            raise Error('rpc_dir "{}" does not exist'.format(self.rpc_dir))
        for v in self.correction_params:
            if v not in ["R", "T", "K", "COMMON_K"]:
                raise Error("{} is not a valid camera parameter to optimize".format(v))

        # create output path
        os.makedirs(self.dst_dir, exist_ok=True)

        # needed to run bundle adjustment
        self.init_ba_input_data()

        # feature tracks configuration
        from .feature_tracks.ft_utils import init_feature_tracks_config

        self.tracks_config = init_feature_tracks_config()
        for k in self.tracks_config.keys():
            if k in args.keys():
                self.tracks_config[k] = args[k]

        print("\n###################################################################################")
        print("\nLoading scene from {}\n".format(scene_config))
        print("-------------------------------------------------------------")
        print("Configuration:")
        print("    - geotiff_dir:   {}".format(self.geotiff_dir))
        print("    - rpc_dir:       {}".format(self.rpc_dir))
        print("    - rpc_src:       {}".format(self.rpc_src))
        print("    - output_dir:    {}".format(self.dst_dir))
        print("    - cam_model:     {}".format(self.cam_model))
        flush_print("-------------------------------------------------------------\n")

        # construct scene timeline
        self.timeline, self.aoi_lonlat = loader.load_scene(
            self.geotiff_dir, self.dst_dir, self.rpc_dir, rpc_src=self.rpc_src, geotiff_label=self.geotiff_label
        )
        # if aoi_geojson is not defined in ba_config define aoi_lonlat as the union of all geotiff footprints
        if "aoi_geojson" in args.keys():
            self.aoi_lonlat = loader.load_geojson(args["aoi_geojson"])
            print("AOI geojson loaded from {}".format(args["aoi_geojson"]))
        else:
            print("AOI geojson defined from the union of all geotiff footprints")
        loader.save_geojson("{}/AOI_init.json".format(self.dst_dir), self.aoi_lonlat)

        start_date = self.timeline[0]["datetime"].date()
        end_date = self.timeline[-1]["datetime"].date()
        print("Number of acquisition dates: {} (from {} to {})".format(len(self.timeline), start_date, end_date))
        print("Number of images: {}".format(np.sum([d["n_images"] for d in self.timeline])))
        sq_km = geotools.measure_squared_km_from_lonlat_geojson(self.aoi_lonlat)
        print("The aoi covers a surface of {:.2f} squared km".format(sq_km))
        print("Scene loaded in {:.2f} seconds".format(timeit.default_timer() - t0))
        flush_print("\n###################################################################################\n\n")

    def get_timeline_attributes(self, timeline_indices, attributes):
        loader.get_timeline_attributes(self.timeline, timeline_indices, attributes)

    def display_aoi(self, zoom=14):
        geotools.display_lonlat_geojson_list_over_map([self.aoi_lonlat], zoom_factor=zoom)

    def display_crops(self):
        mycrops = self.mycrops_adj + self.mycrops_new
        if len(mycrops) > 0:
            vistools.display_gallery([loader.custom_equalization(f["crop"]) for f in mycrops])
        else:
            print("No crops have been loaded. Use load_data_from_date() to load them.")

    def display_image_masks(self):
        if not self.compute_aoi_masks:
            print("compute_aoi_masks is False")
        else:
            mycrops = self.mycrops_adj + self.mycrops_new
            if len(mycrops) > 0:
                vistools.display_gallery([255.0 * f["mask"] for f in mycrops])
            else:
                print("No crops have been loaded. Use load_data_from_date() to load them.")

    def check_adjusted_dates(self, input_dir):

        dir_adj_rpc = os.path.join(input_dir, "RPC_adj")
        if os.path.exists(input_dir + "/filenames.txt") and os.path.isdir(dir_adj_rpc):

            # read tiff images
            adj_fnames = loader.load_list_of_paths(input_dir + "/filenames.txt")
            print("Found {} previously adjusted images in {}\n".format(len(adj_fnames), self.dst_dir))

            datetimes_adj = [loader.get_acquisition_date(img_geotiff_path) for img_geotiff_path in adj_fnames]
            timeline_adj = loader.group_files_by_date(datetimes_adj, adj_fnames)
            for d in timeline_adj:
                adj_id = d["id"]
                for idx in range(len(self.timeline)):
                    if self.timeline[idx]["id"] == adj_id:
                        self.timeline[idx]["adjusted"] = True

            prev_adj_data_found = True
        else:
            print("No previously adjusted data was found in {}\n".format(self.dst_dir))
            prev_adj_data_found = False

        return prev_adj_data_found

    def load_data_from_dates(self, timeline_indices, input_dir, adjusted=False):

        im_fnames = []
        for t_idx in timeline_indices:
            im_fnames.extend(self.timeline[t_idx]["fnames"])
        n_cam = len(im_fnames)
        to_print = [n_cam, "adjusted" if adjusted else "new"]
        flush_print("{} images for bundle adjustment !".format(*to_print))

        if n_cam > 0:
            # get rpcs
            rpc_dir = os.path.join(input_dir, "RPC_adj") if adjusted else os.path.join(self.dst_dir, "RPC_init")
            rpc_suffix = "RPC_adj" if adjusted else "RPC"
            im_rpcs = loader.load_rpcs_from_dir(im_fnames, rpc_dir, suffix=rpc_suffix, verbose=True)

            # get image crops
            im_crops = loader.load_image_crops(
                im_fnames, rpcs=im_rpcs, aoi=self.aoi_lonlat, compute_aoi_mask=self.compute_aoi_masks
            )

        if adjusted:
            self.n_adj += n_cam
            self.myimages_adj.extend(im_fnames.copy())
            self.myrpcs_adj.extend(im_rpcs.copy())
            self.mycrops_adj.extend(im_crops.copy())
        else:
            self.n_new += n_cam
            self.myimages_new.extend(im_fnames.copy())
            self.myrpcs_new.extend(im_rpcs.copy())
            self.mycrops_new.extend(im_crops.copy())

    def load_prev_adjusted_dates(self, t_idx, input_dir, previous_dates=1):

        # t_idx = timeline index of the new date to adjust
        dt2str = lambda t: t.strftime("%Y-%m-%d %H:%M:%S")
        found_adj_dates = self.check_adjusted_dates(input_dir)
        if found_adj_dates:
            # load data from closest date in time
            all_prev_adj_t_indices = [idx for idx, d in enumerate(self.timeline) if d["adjusted"]]
            closest_adj_t_indices = sorted(all_prev_adj_t_indices, key=lambda x: abs(x - t_idx))
            adj_t_indices_to_use = closest_adj_t_indices[:previous_dates]
            adj_dates_to_use = ", ".join([dt2str(self.timeline[k]["datetime"]) for k in adj_t_indices_to_use])
            print("Using {} previously adjusted date(s): {}\n".format(len(adj_t_indices_to_use), adj_dates_to_use))
            self.load_data_from_dates(adj_t_indices_to_use, input_dir, adjusted=True)

    def init_ba_input_data(self):
        self.n_adj = 0
        self.myimages_adj = []
        self.mycrops_adj = []
        self.myrpcs_adj = []
        self.n_new = 0
        self.myimages_new = []
        self.mycrops_new = []
        self.myrpcs_new = []

    def set_ba_input_data(self, t_indices, input_dir, output_dir, previous_dates):

        print("\n\n\nSetting bundle adjustment input data...\n")
        # init
        self.init_ba_input_data()
        # load previously adjusted data (if existent) relevant for the current date
        if previous_dates > 0:
            self.load_prev_adjusted_dates(min(t_indices), input_dir, previous_dates=previous_dates)
        # load new data to adjust
        self.load_data_from_dates(t_indices, input_dir)

        self.ba_data = {}
        self.ba_data["in_dir"] = input_dir
        self.ba_data["out_dir"] = output_dir
        self.ba_data["n_new"] = self.n_new
        self.ba_data["n_adj"] = self.n_adj
        self.ba_data["image_fnames"] = self.myimages_adj + self.myimages_new
        self.ba_data["crops"] = self.mycrops_adj + self.mycrops_new
        self.ba_data["rpcs"] = self.myrpcs_adj + self.myrpcs_new
        self.ba_data["cam_model"] = self.cam_model
        self.ba_data["aoi"] = self.aoi_lonlat
        self.ba_data["correction_params"] = self.correction_params
        self.ba_data["predefined_matches_dir"] = self.predefined_matches_dir

        if self.compute_aoi_masks:
            self.ba_data["masks"] = [f["mask"] for f in self.mycrops_adj] + [f["mask"] for f in self.mycrops_new]
        else:
            self.ba_data["masks"] = None
        flush_print("\n...bundle adjustment input data is ready !\n\n")

    def bundle_adjust(self, feature_detection=True):

        import timeit

        t0 = timeit.default_timer()

        k = ["feature_detection", "tracks_config", "fix_ref_cam", "ref_cam_weight", "filter_outliers"]
        v = [feature_detection, self.tracks_config, self.fix_ref_cam, self.ref_cam_weight, self.filter_outliers]
        extra_ba_config = dict(zip(k, v))

        # run bundle adjustment
        self.ba_pipeline = BundleAdjustmentPipeline(self.ba_data, self.tracks_config, extra_ba_config)
        self.ba_pipeline.run()

        # retrieve some stuff for verbose
        n_tracks = self.ba_pipeline.ba_params.pts3d_ba.shape[0]
        elapsed_time = timeit.default_timer() - t0
        ba_e, init_e = np.mean(self.ba_pipeline.ba_e), np.mean(self.ba_pipeline.init_e)
        elapsed_time_FT = self.ba_pipeline.feature_tracks_running_time

        return elapsed_time, elapsed_time_FT, n_tracks, ba_e, init_e

    def rm_tmp_files_after_ba(self):
        features_dir = "{}/{}/{}".format(self.dst_dir, self.ba_method, "features")
        features_utm_dir = "{}/{}/{}".format(self.dst_dir, self.ba_method, "features_utm")
        if os.path.exists(features_dir):
            os.system("rm -r {}".format(features_dir))
        if os.path.exists(features_utm_dir):
            os.system("rm -r {}".format(features_utm_dir))
        os.system("rm {}/{}/{}".format(self.dst_dir, self.ba_method, "matches.npy"))
        os.system("rm {}/{}/{}".format(self.dst_dir, self.ba_method, "pairs_matching.npy"))
        os.system("rm {}/{}/{}".format(self.dst_dir, self.ba_method, "pairs_triangulation.npy"))

    def reset_ba_params(self):
        ba_dir = "{}/{}".format(self.dst_dir, self.ba_method)
        if os.path.exists(ba_dir):
            os.system("rm -r {}".format(ba_dir))
        for t_idx in range(len(self.timeline)):
            self.timeline[t_idx]["adjusted"] = False

    def run_sequential_bundle_adjustment(self):

        ba_dir = os.path.join(self.dst_dir, self.ba_method)
        os.makedirs(ba_dir, exist_ok=True)

        n_input_dates = len(self.selected_timeline_indices)
        self.tracks_config["FT_predefined_pairs"] = []

        time_per_date, time_per_date_FT, ba_iters_per_date = [], [], []
        tracks_per_date, init_e_per_date, ba_e_per_date = [], [], []
        for idx, t_idx in enumerate(self.selected_timeline_indices):
            self.set_ba_input_data([t_idx], ba_dir, ba_dir, self.n_dates)
            if (idx == 0 and self.fix_ref_cam) or (self.n_dates == 0 and self.fix_ref_cam):
                self.fix_ref_cam = True
            else:
                self.fix_ref_cam = False
            running_time, time_FT, n_tracks, _, _ = self.bundle_adjust()
            pts_out_fn = "{}/pts3d_adj/{}_pts3d_adj.ply".format(ba_dir, self.timeline[t_idx]["id"])
            os.makedirs(os.path.dirname(pts_out_fn), exist_ok=True)
            os.system("mv {} {}".format(ba_dir + "/pts3d_adj.ply", pts_out_fn))
            init_e, ba_e = self.compute_reprojection_error_after_bundle_adjust()
            time_per_date.append(running_time)
            time_per_date_FT.append(time_FT)
            tracks_per_date.append(n_tracks)
            init_e_per_date.append(init_e)
            ba_e_per_date.append(ba_e)
            ba_iters_per_date.append(self.ba_pipeline.ba_iters)
            current_dt = self.timeline[t_idx]["datetime"]
            to_print = [idx + 1, n_input_dates, current_dt, running_time, n_tracks, init_e, ba_e]
            flush_print("({}/{}) {} adjusted in {:.2f} seconds, {} ({:.3f}, {:.3f})".format(*to_print))

        self.update_aoi_after_bundle_adjustment(ba_dir)
        self.rm_tmp_files_after_ba()
        total_time = sum(time_per_date)
        avg_tracks_per_date = int(np.ceil(np.mean(tracks_per_date)))
        to_print = [total_time, avg_tracks_per_date, np.mean(init_e_per_date), np.mean(ba_e_per_date)]
        flush_print("All dates adjusted in {:.2f} seconds, {} ({:.3f}, {:.3f})".format(*to_print))
        time_FT = loader.get_time_in_hours_mins_secs(sum(time_per_date_FT))
        flush_print("\nAll feature tracks constructed in {}\n".format(time_FT))
        flush_print("Average BA iterations per date: {}".format(int(np.ceil(np.mean(ba_iters_per_date)))))
        flush_print("\nTOTAL TIME: {}\n".format(loader.get_time_in_hours_mins_secs(total_time)))

    def run_global_bundle_adjustment(self):

        ba_dir = os.path.join(self.dst_dir, self.ba_method)
        os.makedirs(ba_dir, exist_ok=True)

        # only pairs from the same date or consecutive dates are allowed
        args = [self.timeline, self.selected_timeline_indices, self.n_dates]
        self.tracks_config["FT_predefined_pairs"] = loader.load_pairs_from_same_date_and_next_dates(*args)

        # load bundle adjustment data and run bundle adjustment
        self.set_ba_input_data(self.selected_timeline_indices, ba_dir, ba_dir, 0)
        running_time, time_FT, n_tracks, ba_e, init_e = self.bundle_adjust()
        self.update_aoi_after_bundle_adjustment(ba_dir)
        self.rm_tmp_files_after_ba()

        args = [running_time, n_tracks, init_e, ba_e]
        flush_print("All dates adjusted in {:.2f} seconds, {} ({:.3f}, {:.3f})".format(*args))
        time_FT = loader.get_time_in_hours_mins_secs(time_FT)
        flush_print("\nAll feature tracks constructed in {}\n".format(time_FT))
        flush_print("Total BA iterations: {}".format(int(self.ba_pipeline.ba_iters)))
        flush_print("\nTOTAL TIME: {}\n".format(loader.get_time_in_hours_mins_secs(running_time)))

    def run_bruteforce_bundle_adjustment(self):

        ba_dir = os.path.join(self.dst_dir, self.ba_method)
        os.makedirs(ba_dir, exist_ok=True)

        self.tracks_config["FT_predefined_pairs"] = []
        self.set_ba_input_data(self.selected_timeline_indices, ba_dir, ba_dir, 0)
        running_time, time_FT, n_tracks, ba_e, init_e = self.bundle_adjust()
        self.update_aoi_after_bundle_adjustment(ba_dir)
        self.rm_tmp_files_after_ba()

        args = [running_time, n_tracks, init_e, ba_e]
        flush_print("All dates adjusted in {:.2f} seconds, {} ({:.3f}, {:.3f})".format(*args))
        time_FT = loader.get_time_in_hours_mins_secs(time_FT)
        flush_print("\nAll feature tracks constructed in {}\n".format(time_FT))
        flush_print("Total BA iterations: {}".format(int(self.ba_pipeline.ba_iters)))
        flush_print("\nTOTAL TIME: {}\n".format(loader.get_time_in_hours_mins_secs(running_time)))

    def is_ba_method_valid(self, ba_method):
        return ba_method in ["ba_global", "ba_sequential", "ba_bruteforce"]

    def update_aoi_after_bundle_adjustment(self, ba_dir):

        ba_rpc_fn = glob.glob("{}/RPC_adj/*".format(ba_dir))[0]
        init_rpc_fn = "{}/RPC_init/{}".format(self.dst_dir, os.path.basename(ba_rpc_fn).replace("_RPC_adj", "_RPC"))
        init_rpc = rpcm.rpc_from_rpc_file(init_rpc_fn)
        ba_rpc = rpcm.rpc_from_rpc_file(ba_rpc_fn)
        corrected_aoi = ba_utils.reestimate_lonlat_geojson_after_rpc_correction(init_rpc, ba_rpc, self.aoi_lonlat)
        loader.save_geojson("{}/AOI_adj.json".format(ba_dir), corrected_aoi)

    def compute_reprojection_error_after_bundle_adjust(self):

        im_fnames = self.ba_pipeline.myimages
        C = self.ba_pipeline.ba_params.C
        pairs_to_triangulate = self.ba_pipeline.ba_params.pairs_to_triangulate
        cam_model = "rpc"

        # get init and bundle adjusted rpcs
        rpcs_init_dir = os.path.join(self.dst_dir, "RPC_init")
        rpcs_init = loader.load_rpcs_from_dir(im_fnames, rpcs_init_dir, suffix="RPC", verbose=False)
        rpcs_ba_dir = os.path.join(self.dst_dir, self.ba_method + "/RPC_adj")
        rpcs_ba = loader.load_rpcs_from_dir(im_fnames, rpcs_ba_dir, suffix="RPC_adj", verbose=False)

        # triangulate
        from bundle_adjust.ba_triangulate import init_pts3d

        pts3d_before = init_pts3d(C, rpcs_init, cam_model, pairs_to_triangulate, verbose=False)
        # pts3d_after = init_pts3d(C, rpcs_ba, cam_model, pairs_to_triangulate, verbose=False)
        pts3d_after = self.ba_pipeline.ba_params.pts3d_ba

        # reproject
        n_pts, n_cam = C.shape[1], C.shape[0] // 2
        not_nan_C = ~np.isnan(C)
        err_before, err_after = [], []
        for cam_idx in range(n_cam):
            pt_indices = np.where(not_nan_C[2 * cam_idx])[0]
            obs2d = C[(cam_idx * 2) : (cam_idx * 2 + 2), pt_indices].T
            pts3d_init = pts3d_before[pt_indices, :]
            pts3d_ba = pts3d_after[pt_indices, :]
            args = [rpcs_init[cam_idx], rpcs_ba[cam_idx], cam_model, obs2d, pts3d_init, pts3d_ba]
            _, _, err_b, err_a, _ = ba_utils.reproject_pts3d(*args)
            err_before.extend(err_b.tolist())
            err_after.extend(err_a.tolist())
        return np.mean(err_before), np.mean(err_after)

    def run_bundle_adjustment_for_RPC_refinement(self):

        # read the indices of the selected dates and print some information
        if self.selected_timeline_indices is None:
            self.selected_timeline_indices = np.arange(len(self.timeline), dtype=np.int32).tolist()
            flush_print("All dates selected to bundle adjust!\n")
        else:
            to_print = [len(self.selected_timeline_indices), self.selected_timeline_indices]
            flush_print("Found {} selected dates to bundle adjust! timeline_indices: {}\n".format(*to_print))
            self.get_timeline_attributes(self.selected_timeline_indices, ["datetime", "n_images", "id"])
        for idx, t_idx in enumerate(self.selected_timeline_indices):
            args = [idx + 1, self.timeline[t_idx]["datetime"], self.timeline[t_idx]["n_images"]]
            flush_print("({}) {} --> {} views".format(*args))

        if self.reset:
            self.reset_ba_params()

        # run bundle adjustment
        if self.ba_method == "ba_sequential":
            print("\nRunning sequential bundle adjustment !")
            flush_print("Each date aligned with {} previous date(s)\n".format(self.n_dates))
            self.run_sequential_bundle_adjustment()
        elif self.ba_method == "ba_global":
            print("\nRunning global bundle ajustment !")
            print("All dates will be adjusted together at once")
            flush_print("Track pairs restricted to the same date and the next {} dates\n".format(self.n_dates))
            self.run_global_bundle_adjustment()
        elif self.ba_method == "ba_bruteforce":
            print("\nRunning bruteforce bundle ajustment !")
            flush_print("All dates will be adjusted together at once\n")
            self.run_bruteforce_bundle_adjustment()
        else:
            print("ba_method {} is not valid !".format(self.ba_method))
            print("accepted values are: [ba_sequential, ba_global, ba_bruteforce]")
            sys.exit()
