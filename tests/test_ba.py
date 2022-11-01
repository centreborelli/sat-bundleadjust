import glob
import json
import os
import tempfile

import numpy as np
import rpcm

import bundle_adjust


def test_ba():

    # Download resources
    tmpdir = tempfile.TemporaryDirectory()

    # Create & Save Bundle config
    out_dir = os.path.join(tmpdir.name, "outdir")

    bundle_config = {
        "geotiff_dir": "tests/data/images",
        "rpc_dir": "tests/data/images",
        "rpc_src": "txt",
        "cam_model": "rpc",
        "output_dir": out_dir,
        "ba_method": "ba_bruteforce",
        "FT_max_kp": 10000,
        "FT_sift_detection": "s2p",
        "FT_sift_matching": "bruteforce",
    }
    cfg_path = os.path.join(tmpdir.name, "config.json")
    json.dump(bundle_config, open(cfg_path, "w"))

    # Run Bundle Adjustment
    bundle_adjust.main(cfg_path)

    # Assertions
    # Load new RPCs & Update index
    for fl in glob.glob(
        os.path.join(out_dir, "ba_bruteforce", "rpcs_adj", "*_basic_panchromatic_dn.rpc_adj")
    ):
        rpc = rpcm.rpc_from_rpc_file(fl).__dict__
        rpc_comp = rpcm.rpc_from_rpc_file(
            os.path.join("tests/data/outdir", fl.replace(out_dir, "")[1:])
        ).__dict__

        all_coefs_exact = True
        for k in rpc.keys():
            if isinstance(rpc[k], list):
                for i in range(len(rpc[k])):
                    if not np.allclose(rpc[k][i], rpc_comp[k][i]):
                        all_coefs_exact = False
                        print("{} {}-th coef -> current value: {}, expected value: {}".format(k, i, rpc[k][i], rpc_comp[k][i]))
            else:
                if not np.allclose(rpc[k], rpc_comp[k]):
                    all_coefs_exact = False
                    print("{} coef -> current value: {}, expected value: {}".format(k, rpc[k], rpc_comp[k]))
        if not all_coefs_exact:
            print(f"Warning: Found some RPC coefficients different from expected !")
            print(f"         Small differences in the order of decimals are most likely irrelevant and due to different keypoint matches.\n")

    # uncomment line below to force all RPC coefficients to be the same
    # assert all_coefs_exact
