#!/bin/bash

source /net/student33/data2/hendrix/miniconda3/etc/profile.d/conda.sh

conda activate mrp

python /net/student33/data2/hendrix/git/MRP/src/vulcan_configs/run_vulcan_for_configs.py -w 64
