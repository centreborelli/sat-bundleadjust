import argparse
import os
import sys

import numpy as np

from bundle_adjust import ba_timeseries, data_loader


def main():

    parser = argparse.ArgumentParser(description="Bundle Adjustment for S2P")

    parser.add_argument(
        "config",
        metavar="config.json",
        help=("path to a json file containing the necessary " "parameters of the scene to be bundle adjusted."),
    )

    parser.add_argument(
        "--timeline",
        action="store_true",
        help=("just print the timeline of the scene described " "by the configuration json, do not run anything else."),
    )

    # parse command line arguments
    args = parser.parse_args()
    verbose = True
    reset = True

    # load options from config file and copy config file to output_dir
    opt = data_loader.load_dict_from_json(args.config)
    os.makedirs(opt["output_dir"], exist_ok=True)
    os.system("cp {} {}".format(args.config, os.path.join(opt["output_dir"], os.path.basename(args.config))))

    # redirect all prints to a bundle adjustment logfile inside the output directory
    log_file = open("{}/{}_BA.log".format(opt["output_dir"], data_loader.get_id(args.config)), "w+")
    sys.stdout = log_file
    sys.stderr = log_file

    # load scene
    scene = ba_timeseries.Scene(args.config)
    timeline_indices = np.arange(len(scene.timeline), dtype=int).tolist()

    if args.timeline:
        scene.get_timeline_attributes(timeline_indices, ["datetime", "n_images", "id"])
        sys.exit()

    # optional options
    opt["reconstruct"] = opt["reconstruct"] if "reconstruct" in opt.keys() else False
    opt["pc3dr"] = opt["pc3dr"] if "pc3dr" in opt.keys() else False
    opt["geotiff_label"] = opt["geotiff_label"] if "geotiff_label" in opt.keys() else None
    opt["postprocess"] = opt["postprocess"] if "postprocess" in opt.keys() else True
    opt["skip_ba"] = opt["skip_ba"] if "skip_ba" in opt.keys() else False
    opt["s2p_parallel"] = opt["s2p_parallel"] if "s2p_parallel" in opt.keys() else 5
    opt["n_dates"] = opt["n_dates"] if "n_dates" in opt.keys() else 1
    opt["fix_ref_cam"] = opt["fix_ref_cam"] if "fix_ref_cam" in opt.keys() else True
    opt["ref_cam_weight"] = float(opt["ref_cam_weight"]) if "ref_cam_weight" in opt.keys() else 1.0
    opt["filter_outliers"] = float(opt["filter_outliers"]) if "filter_outliers" in opt.keys() else True

    # which timeline indices are to bundle adjust
    if "timeline_indices" in opt.keys():
        timeline_indices = [int(idx) for idx in opt["timeline_indices"]]
        print_args = [len(timeline_indices), timeline_indices]
        print(
            "Found {} selected dates ! timeline_indices: {}\n".format(*print_args),
            flush=True,
        )
        scene.get_timeline_attributes(timeline_indices, ["datetime", "n_images", "id"])
    else:
        print("All dates selected !\n", flush=True)

    # bundle adjust
    if opt["ba_method"] is None or opt["skip_ba"]:
        print("\nSkipping bundle adjustment !\n")
    else:
        if opt["ba_method"] == "ba_sequential":
            scene.run_sequential_bundle_adjustment(
                timeline_indices,
                previous_dates=opt["n_dates"],
                fix_ref_cam=opt["fix_ref_cam"],
                ref_cam_weight=opt["ref_cam_weight"],
                filter_outliers=opt["filter_outliers"],
                reset=reset,
                verbose=verbose,
            )
        elif opt["ba_method"] == "ba_global":
            scene.run_global_bundle_adjustment(
                timeline_indices,
                next_dates=opt["n_dates"],
                fix_ref_cam=opt["fix_ref_cam"],
                ref_cam_weight=opt["ref_cam_weight"],
                filter_outliers=opt["filter_outliers"],
                reset=reset,
                verbose=verbose,
            )
        elif opt["ba_method"] == "ba_bruteforce":
            scene.run_bruteforce_bundle_adjustment(
                timeline_indices,
                fix_ref_cam=opt["fix_ref_cam"],
                ref_cam_weight=opt["ref_cam_weight"],
                filter_outliers=opt["filter_outliers"],
                reset=reset,
                verbose=verbose,
            )
        else:
            print("ba_method {} is not valid !".format(opt["ba_method"]))
            print("accepted values are: [ba_sequential, ba_global, ba_bruteforce]")
            sys.exit()

    # close logfile
    sys.stderr = sys.__stderr__
    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == "__main__":
    sys.exit(main())
