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
        "FT_sift_matching": "epipolar_based",
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

        for k in rpc.keys():
            assert np.allclose(rpc[k], rpc_comp[k])
