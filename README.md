## Bundle Adjustment for 3D Reconstruction from Multi-Date Satellite Images

Original *3D Reconstruction from Multi-Date Satellite Images* online demo [here](https://gfacciol.github.io/IS18/).

This code extends the previous work by adding a Bundle Adjustment block to the pipeline.

To run the code, with the default configuration, use the `config_example.json` as template.

```bash
python3 cli.py config_example.json --verbose
```
Set the paths to your data in the `"geotiff_dir"`, `"s2p_config_dir"` and `"output_dir"` keys of config json.

The corrected RPCs are written in txt format in the `RPC_adj` folder of the output path specified in the config json.

All the information of the BA process will be written in `output_dir/config_example_BA.log`.

