import sys

import numpy as np

from bundle_adjust import ba_timeseries, loader

__version__ = "0.1.0dev"


def main(config_path):

    # load scene and run BA
    scene = ba_timeseries.Scene(config_path)
    scene.run_bundle_adjustment_for_RPC_refinement()
