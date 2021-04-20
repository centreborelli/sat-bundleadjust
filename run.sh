#!/bin/bash
#/*
# * Copyright (c) 2021, Thibaud Ehret <roger.mari@ens-paris-saclay.fr>
# * All rights reserved.
# *
# */

set -e

# get inputs
virtualenv=$1
input='input_0.zip'
rm_outliers=${2,,}
ft_selection=${3,,}

# create config.json with bundle adjustment configuration
unzip -o $input -d input_data
config='{"input_dir":"./input_data", "output_dir":"./output_data", "predefined_matches": true, "clean_outliers": '"$rm_outliers"', "tracks_selection": '"$ft_selection"'}'
echo $config > ./config.json

# activate virtual environment
if [ -d $virtualenv ]; then
  source $virtualenv/bin/activate
fi

# run bundle adjustment pipeline for RPC correction
main.py config.json
