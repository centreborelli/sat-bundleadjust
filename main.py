import argparse
import sys
import os
import rpcm
import glob
from bundle_adjust import loader


class Error(Exception):
    pass


def main():

    parser = argparse.ArgumentParser(description="IPOL: A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery")

    parser.add_argument(
        "config",
        metavar="config.json",
        help="path to a json file containing the configuration parameters",
    )

    parser.add_argument(
        "--no_outliers_filtering",
        action="store_true",
        help="deactivate the filtering of outlier feature track observations based on reprojection error",
    )

    parser.add_argument(
        "--no_tracks_selection",
        action="store_true",
        help="deactivate the selection of an optimal subset of feature tracks",
    )

    parser.add_argument(
        "--predefined_matches",
        action="store_true",
        help="use predefined matches (if available)",
    )

    # parse command line arguments
    args = parser.parse_args()

    d = loader.load_dict_from_json(args.config)

    input_dir = d["input_dir"]
    output_dir = d["output_dir"]
    image_height = d["height"] # 1349
    image_width = d["width"] # 3199

    if args.predefined_matches:
        # load geotiff paths
        geotiff_paths = loader.load_list_of_paths(os.path.join(input_dir, 'filenames.txt'))

        # load predefined matches
        matches_path = os.path.join(input_dir, "matches.npy")
        if not os.path.exists(matches_path):
            raise Error ('predefined matches file {} not found'.format(matches_path))

        keypoints_dir = os.path.join(input_dir, "keypoints")
        for p_geotiff in geotiff_paths:
            bn_geotiff = os.path.basename(p_geotiff)
            p_kp = "{}/{}.npy".format(keypoints_dir, loader.get_id(bn_geotiff))
            if not os.path.exists(p_kp):
                raise Error ('keypoints file {} corresponding to geotiff {} not found'.format(p_kp, bn_geotiff))

        def default_crop(h, w):
            return {"crop": None, "col0": 0.0, "row0": 0.0, "height": h, "width": w}

        crops = [default_crop(image_height, image_width)] * len(geotiff_paths)
    else:
        # load geotiff paths
        geotiff_paths = glob.glob(os.path.join(input_dir, "geotiffs/*.tif"))

        crops = loader.load_image_crops(geotiff_paths, verbose=False)

    # load rpcs
    rpcs_dir = os.path.join(input_dir, "rpcs")
    rpcs = []
    for p_geotiff in geotiff_paths:
        bn_geotiff = os.path.basename(p_geotiff)
        p_rpc = "{}/{}.rpc".format(rpcs_dir, loader.get_id(bn_geotiff))
        if not os.path.exists(p_rpc):
            raise Error ("rpc file {} corresponding to geotiff {} not found".format(p_rpc, bn_geotiff))
        rpcs.append(rpcm.rpc_from_rpc_file(p_rpc))

    # create output directory
    os.makedirs(output_dir, exist_ok=True)

    ba_data = {}
    ba_data["in_dir"] = input_dir
    ba_data["out_dir"] = output_dir
    ba_data["image_fnames"] = geotiff_paths
    ba_data["rpcs"] = rpcs
    ba_data["crops"] = crops
    if os.path.join(input_dir, "AOI.json"):
        ba_data["aoi"] = loader.load_geojson(os.path.join(input_dir, "AOI.json"))
    if args.predefined_matches:
        ba_data["predefined_matches"] = True

    # redirect all prints to a bundle adjustment logfile inside the output directory
    path_to_log_file = "{}/bundle_adjust.log".format(output_dir, loader.get_id(args.config))
    print("Running bundle adjustment for RPC model refinement ...")
    print("Path to log file: {}".format(path_to_log_file))
    #log_file = open(path_to_log_file, "w+")
    #sys.stdout = log_file
    #sys.stderr = log_file

    from bundle_adjust.ba_pipeline import BundleAdjustmentPipeline
    extra_ba_config = {"fix_ref_cam": True}
    tracks_config = {"FT_K": 60}
    if args.no_outliers_filtering:
        extra_ba_config["clean_outliers"] = False
    if args.no_tracks_selection:
        tracks_config = {"FT_K": 0}
    pipeline = BundleAdjustmentPipeline(ba_data, tracks_config=tracks_config, extra_ba_config=extra_ba_config)
    pipeline.run()

    # close logfile
    #sys.stderr = sys.__stderr__
    #sys.stdout = sys.__stdout__
    #log_file.close()
    print("... done !")
    print("Path to output RPC files: {}".format(os.path.join(output_dir, "rpcs_adj")))

    # remove temporal files
    if os.path.exists("{}/P_adj".format(output_dir)):
        os.system("rm -r {}/P_adj".format(output_dir))

if __name__ == "__main__":
    sys.exit(main())
