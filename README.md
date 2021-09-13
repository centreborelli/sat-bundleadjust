# A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery

Python implementation of *A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery* ([IPOL](https://www.ipol.im/), 2021). 

Authors: Roger Mari, Carlo de Franchis, Enric Meinhardt-Llopis, Jeremy Anger, Gabriele Facciolo.

## Installation

Install the `bundle_adjust` package:

```bash
git clone https://github.com/centreborelli/sat-bundleadjust.git
cd sat-bundleadjust
pip install -e .
```

Check that the installation was successful by running
```bash
pytest tests/test_ba.py
````

## Usage

To run the code:

```bash
bundle_adjust config.json
```
where `config.json` contains a Python dictionary specifying the paths to the input data and any additional configuration parameters.

## Default configuration

To run the default configuration use a `config.json` as follows:

```json
{
  "geotiff_dir": "your/path/to/the/input/geotiff/images",
  "rpc_dir": "your/path/to/the/input/RPC/models",
  "rpc_src": "txt",
  "output_dir": "your/output/path"
}
```
where:
- `geotiff_dir` points to the directory containing all the input geotiff image files, with extension `.tif`
- `rpc_dir` points to the directory containing all the input RPC camera models, in txt files with extension `.rpc`. The [rpcm](https://github.com/cmla/rpcm) package is used to represent RPC models, which can be written to txt files using `rpcm.RPCModel.write_to_file`.
- `rpc_src` is a string that can be either `"txt"`, `"json"` or `"geotiff"`. If `"geotiff"` is used, then the input RPC models are directly read from the input geotiff image files. 
- The output RPC models are written in a folder named `rpcs_adj`, which is created in the `output_dir`.

## Customized configuration

Check `feature_tracks/ft_utils.init_feature_tracks_config` for the list of parameters that can be added to `config.json` to customize the feature tracking stage of the pipeline.

Check `ba_pipeline.__init__` for the list of parameters that can be added to `config.json` to customize the bundle adjustment pipeline.

## Test data

Examples:

```bash
bundle_adjust tests/config1.json
```