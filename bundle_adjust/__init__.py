import sys

import numpy as np

from bundle_adjust import ba_timeseries, data_loader

__version__ = "0.1.0dev"


def main(config_path):
    # load scene
    config = data_loader.load_dict_from_json(config_path)
    scene = ba_timeseries.Scene(config_path)
    if "timeline_indices" in config.keys():
        timeline_indices = [int(idx) for idx in config["timeline_indices"]]
    else:
        timeline_indices = np.arange(len(scene.timeline), dtype=int).tolist()
        timeline_indices = [int(idx) for idx in timeline_indices]
    n_dates = config["n_dates"] if "n_dates" in config.keys() else 1
    reset = config["reset"] if "reset" in config.keys() else True
    verbose = config["verbose"] if "verbose" in config.keys() else False

    # which timeline indices are to bundle adjust
    print_args = [len(timeline_indices), timeline_indices]
    print(
        "Found {} selected dates ! timeline_indices: {}\n".format(*print_args),
        flush=True,
    )
    scene.get_timeline_attributes(timeline_indices, ["datetime", "n_images", "id"])

    # bundle adjust
    ba_method = config["ba_method"]
    skip_ba = False if "skip_ba" not in config.keys() else config["skip_ba"]
    if ba_method is None or skip_ba:
        print("\nSkipping bundle adjustment !\n")
    else:
        if ba_method == "ba_sequential":
            scene.run_sequential_bundle_adjustment(
                timeline_indices, previous_dates=n_dates, reset=reset, verbose=verbose
            )
        elif ba_method == "ba_global":
            scene.run_global_bundle_adjustment(
                timeline_indices, next_dates=n_dates, reset=reset, verbose=verbose
            )
        elif ba_method == "ba_bruteforce":
            scene.run_bruteforce_bundle_adjustment(
                timeline_indices, reset=reset, verbose=verbose
            )
        else:
            print("ba_method {} is not valid !".format(ba_method))
            print("accepted values are: [ba_sequential, ba_global, ba_bruteforce]")
            sys.exit()


if __name__ == "__main__":
    sys.exit(main())
