#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh

conda activate vulcan

python /net/student33/data2/hendrix/git/MRP/src/vulcan_configs/make_vulcan_configs.py
