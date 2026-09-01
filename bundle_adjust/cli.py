import argparse
import os
import sys
import shutil

import numpy as np

import bundle_adjust
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

    # load scene and run BA
    bundle_adjust.main(args.config)