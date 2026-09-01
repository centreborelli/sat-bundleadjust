import sys
import os
import numpy as np

from bundle_adjust import ba_timeseries, loader

__version__ = "0.1.0dev"


def main(config):

    # config: path to a json file or a python dict with the BA arguments

    # load config save it to output_dir
    if isinstance(config, dict):
        opt = config
    else:
        opt = loader.load_dict_from_json(config)
    os.makedirs(opt["output_dir"], exist_ok=True)
    loader.save_dict_to_json(opt, os.path.join(opt["output_dir"], "config.json"))

    # redirect all prints to a bundle adjustment logfile inside the output directory
    log_path = "{}/bundle_adjust.log".format(opt["output_dir"])
    print("Running bundle adjustment for RPC model refinement ...")
    print("Path to log file: {}".format(log_path))
    log_file = open(log_path, "w+")
    sys.stdout = log_file
    sys.stderr = log_file

    # load scene and run BA
    scene = ba_timeseries.Scene(config)
    scene.run_BA_for_RPC_refinement()

    # close logfile
    sys.stderr = sys.__stderr__
    sys.stdout = sys.__stdout__
    log_file.close()
    print("... done !")
    print("Path to output files: {}".format(opt["output_dir"]))

