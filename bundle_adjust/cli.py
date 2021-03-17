import argparse
import os
import sys

import numpy as np

from bundle_adjust import ba_timeseries, loader


def main():

    parser = argparse.ArgumentParser(description="Bundle Adjustment for S2P")

    parser.add_argument(
        "config",
        metavar="config.json",
        help="path to a json file containing the configuration parameters of the scene to be bundle adjusted.",
    )

    parser.add_argument(
        "--timeline",
        action="store_true",
        help="just print the timeline of the scene described by config.json, do not run anything else.",
    )

    # parse command line arguments
    args = parser.parse_args()

    if args.timeline:
        scene = ba_timeseries.Scene(args.config)
        timeline_indices = np.arange(len(scene.timeline), dtype=np.int32).tolist()
        scene.get_timeline_attributes(timeline_indices, ["datetime", "n_images", "id"])
        sys.exit()

    # load options from config file and copy config file to output_dir
    opt = loader.load_dict_from_json(args.config)
    os.makedirs(opt["output_dir"], exist_ok=True)
    os.system("cp {} {}".format(args.config, os.path.join(opt["output_dir"], os.path.basename(args.config))))

    # redirect all prints to a bundle adjustment logfile inside the output directory
    log_file = open("{}/bundle_adjust.log".format(opt["output_dir"], loader.get_id(args.config)), "w+")
    sys.stdout = log_file
    sys.stderr = log_file

    # load scene and run BA
    scene = ba_timeseries.Scene(args.config)
    scene.run_bundle_adjustment_for_RPC_refinement()

    # close logfile
    sys.stderr = sys.__stderr__
    sys.stdout = sys.__stdout__
    log_file.close()


if __name__ == "__main__":
    sys.exit(main())
