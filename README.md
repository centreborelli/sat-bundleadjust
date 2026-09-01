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

It is possible to change the BA configuration by adding other parameters to `config.json`. Listed below are the more important ones:

## Global configuration parameters

The majority of these parameters are also commented in `ba_pipeline.BundleAdjustmentPipeline`

| Parameter | Type | Allowed values | Default | Description |
|---|---|---|---|---|
| `ba_method` | string | `"ba_bruteforce"`, `"ba_global"`, `"ba_sequential"` | `"ba_brutefoce"` | BA strategy. By using `"ba_bruteforce"` all cameras are adjusted at once without considering the acquisition date of the images. The other values are experimental, for more info check `ba_timeseries.run_BA_for_RPC_refinement` |
| `cam_model` | string | `"rpc"`, `"affine"`, `"perspective"` | `"rpc"` | Camera model used at internal level to run the BA. Attention, this is different from the output camera model format, which always follows the RPC standard. |
| `aoi` | dict | `"rpc"`, `"affine"`, `"perspective"` | `None` | Area of interest where RPC have to be consistent in GeoJSON format and longitude and latitude coordinates. If `None`, it is computed from the union of all image footprints. |
| `correction_params` | list of strings | `R`, `T`, `K`, `COMMON_K` | `["R"]` | Correction parameters: `R` (rotation), `T` (translation), `K` (calibration matrix) or `COMMON_K` to fix K in all cams if `K` is also in the list. You combine these but only increasingly. For instance, `T` cannot be used alone, you should use `["R", "T"]`.
| `clean_outliers` | boolean | True, False  | True | Set to True to filter potential outlier feature observations after some initial iterations.
| `save_figures` | boolean | True, False  | True | Set to True to save illustration png images of the reprojection errors before and after.

## Feature track configuration parameters

Check `feature_tracks/ft_utils.init_feature_tracks_config` for the list of parameters that can be added to `config.json` to customize the feature tracking stage of the pipeline.

Check `ba_pipeline.__init__` for the list of parameters that can be added to `config.json` to customize the bundle adjustment pipeline.

## Test data

Examples:

```bash
bundle_adjust tests/config1.json
```

## Comparison with other methods

The companion repository [rogermm14/eval_sat-bundleadjust](https://github.com/rogermm14/eval_sat-bundleadjust) provides scripts and notebooks for comparing RPC bundle adjustment pipelines on the DFC2019 WorldView-3 multi-date imagery. It can be used to evaluate AMES, SAT-BA v1, and SAT-BA v2 on the Omaha and Jacksonville areas of interest.
