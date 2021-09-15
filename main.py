#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
import rpcm
import glob
import shutil
import numpy as np
from bundle_adjust import loader
from bundle_adjust.cam_utils import SatelliteImage


class Error(Exception):
    pass


def main():

    title = "A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery"
    parser = argparse.ArgumentParser(description=title)

    parser.add_argument(
        "config",
        metavar="config.json",
        help="path to a json file containing the configuration parameters",
    )

    # parse command line arguments
    args = parser.parse_args()

    d = loader.load_dict_from_json(args.config)

    input_dir = d["input_dir"]
    output_dir = d["output_dir"]
    predefined_matches = d.get("predefined_matches", False)
    tracks_selection = d.get("tracks_selection", True)
    outliers_filtering = d.get("clean_outliers", True)

    if predefined_matches:

        matches_dir = os.path.join(input_dir, "predefined_matches")

        # load geotiff paths
        geotiff_paths = loader.load_list_of_paths(os.path.join(matches_dir, "filenames.txt"))

        # load predefined matches
        matches_path = os.path.join(matches_dir, "matches.npy")
        if not os.path.exists(matches_path):
            raise Error("predefined matches file {} not found".format(matches_path))

        keypoints_dir = os.path.join(matches_dir, "keypoints")
        for p_geotiff in geotiff_paths:
            bn_geotiff = os.path.basename(p_geotiff)
            p_kp = "{}/{}.npy".format(keypoints_dir, loader.get_id(bn_geotiff))
            if not os.path.exists(p_kp):
                raise Error("keypoints file {} corresponding to geotif {} not found".format(p_kp, bn_geotiff))

    else:
        # load geotiff paths and crop offsets
        images_dir = os.path.join(input_dir, "images")
        if not os.path.exists(images_dir):
            raise Error("images directory {} not found".format(images_dir))
        geotiff_paths = glob.glob(os.path.join(input_dir, "images/*.tif"))
        if len(geotiff_paths) == 0:
            raise Error("found 0 images with .tif extension in {}".format(images_dir))

    # load rpcs
    rpcs_dir = os.path.join(input_dir, "rpcs")
    rpcs = []
    for p_geotiff in geotiff_paths:
        bn_geotiff = os.path.basename(p_geotiff)
        p_rpc = "{}/{}.rpc".format(rpcs_dir, loader.get_id(bn_geotiff))
        if not os.path.exists(p_rpc):
            raise Error("rpc file {} corresponding to geotif {} not found".format(p_rpc, bn_geotiff))
        rpcs.append(rpcm.rpc_from_rpc_file(p_rpc))

    # load crop offsets
    crops = []
    for p_geotiff in geotiff_paths:
        # the image size is necessary to load the crop information
        # if the image is available, simply read its size
        # if the image is not available, estimate its size using the keypoint coordinates
        tmp = "{}/images/{}.tif".format(input_dir, loader.get_id(p_geotiff))
        if os.path.exists(tmp):
            h, w = loader.read_image_size(tmp)
        else:
            if predefined_matches:
                kps = np.load("{}/{}.npy".format(keypoints_dir, loader.get_id(p_geotiff)))
                max_col, min_col = np.nanmax(kps[:, 0]), np.nanmin(kps[:, 0])
                max_row, min_row = np.nanmax(kps[:, 1]), np.nanmin(kps[:, 1])
                h, w = max_row - min_row, max_col - min_col
            else:
                raise Error("geotif file {} not found".format(tmp, bn_geotiff))
        crops.append({"crop": None, "col0": 0.0, "row0": 0.0, "height": h, "width": w})

    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.system("cp {} {}/config.json".format(args.config, output_dir))

    ba_data = {}
    ba_data["in_dir"] = input_dir
    ba_data["out_dir"] = output_dir
    ba_data["images"] = [SatelliteImage(p, r, o) for p, r, o in zip(geotiff_paths, rpcs, crops)]

    # costumize bundle adjustment configuration
    extra_ba_config = {}
    tracks_config = {"FT_K": 60, "FT_sift_matching": "flann"}
    if os.path.exists(os.path.join(input_dir, "AOI.json")):
        extra_ba_config["aoi"] = loader.load_geojson(os.path.join(input_dir, "AOI.json"))
    if predefined_matches:
        extra_ba_config["predefined_matches"] = True
        tracks_config["FT_save"] = False
    if not outliers_filtering:
        extra_ba_config["clean_outliers"] = False
    if not tracks_selection:
        tracks_config["FT_K"] = 0

    # redirect all prints to a bundle adjustment logfile inside the output directory
    path_to_log_file = "{}/ba.log".format(output_dir, loader.get_id(args.config))
    print("Running bundle adjustment for RPC model refinement ...")
    print("Path to log file: {}".format(path_to_log_file))
    log_file = open(path_to_log_file, "w+")
    sys.stdout = log_file
    sys.stderr = log_file

    from bundle_adjust.ba_pipeline import BundleAdjustmentPipeline

    pipeline = BundleAdjustmentPipeline(ba_data, tracks_config=tracks_config, extra_ba_config=extra_ba_config)
    pipeline.run()

    # close logfile
    sys.stderr = sys.__stderr__
    sys.stdout = sys.__stdout__
    log_file.close()
    print("... done !")
    print("Path to output RPC files: {}".format(os.path.join(output_dir, "rpcs_adj")))

    # remove temporal files
    if os.path.exists(os.path.join(pipeline.out_dir, "P_adj")):
        shutil.rmtree(os.path.join(pipeline.out_dir, "P_adj"))
    # save predefined matches
    if not predefined_matches:
        loader.save_predefined_matches(os.path.join(pipeline.out_dir, "matches"), pipeline.out_dir)
        shutil.rmtree(os.path.join(pipeline.out_dir, "matches"))

if __name__ == "__main__":
    sys.exit(main())
