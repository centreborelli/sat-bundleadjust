# A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery

Python implementation of *A Generic Bundle Adjustment Methodology for Indirect RPC Model Refinement of Satellite Imagery* ([IPOL](https://www.ipol.im/), 2021). 

Authors: Roger Mari, Carlo de Franchis, Enric Meinhardt-Llopis, Jeremy Anger, Gabriele Facciolo.

## Installation

Install all Python dependencies:

```bash
pip install -r requirements.txt --user
```

## Usage

To run the code:

```bash
python3 main.py config.json
```
where `config.json` contains a Python dictionary specifying the paths to the input data and any additional configuration parameters. 

To test the code using the sample data run the following command:

```bash
python3 main.py example_config.json
```

## Basic configuration

**Example 1:** `config.json` to run the entire bundle adjustment pipeline.

```json
{
  "input_dir": "your/input/path",
  "output_dir": "your/output/path"
}
```
where `input_dir` points to a directory containing an `images` folder and an `rpcs` folder: `images` is expected to contain the input satellite images with extension `.tif` and `rpcs` is expected to contain the input RPC models in txt files with extension `.rpc`. The [rpcm](https://github.com/cmla/rpcm) library is used to represent RPC models, which can be written to txt files using `rpcm.RPCModel.write_to_file`.

The output RPC models are written in a folder named `rpcs_adj`, which is created in the `output_dir`.

## Customized configuration

The pipeline can be customized by specifying additional fields in the `config.json` file, one at a time or combining several of them.

**Example 2:** `config.json` to run the bundle adjustment pipeline using predefined pairwise matches.

```json
{
  "input_dir": "your/input/path",
  "output_dir": "your/output/path",
  "predefined_matches": true
}
```
where a directory `predefined_matches` is required in your `input_dir`. If you do not have any predefined matches, you can generate them by using a basic configuration file as in **Example 1**. This will generate a `predefined_matches` folder in the `output_dir`.

**Example 3:** `config.json` to run the bundle adjustment pipeline without feature tracks selection.

```json
{
  "input_dir": "your/input/path",
  "output_dir": "your/output/path",
  "tracks_selection": false
}
```
**Example 4:** `config.json` to run the bundle adjustment pipeline without filtering any feature track observations based on the reprojection error.

```json
{
  "input_dir": "your/input/path",
  "output_dir": "your/output/path",
  "clean_outliers": false
}
```

## Test data

Examples:

```bash
python3 main.py test/richards_bay/config.json
```

```bash
python3 main.py test/miami_university/config.json
```

```bash
python3 main.py test/san_luis_obispo_mountains/config.json
```

```bash
python3 main.py test/morenci_mine/config.json
```

