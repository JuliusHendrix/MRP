#!/bin/bash

source /net/student33/data2/hendrix/miniconda3/etc/profile.d/conda.sh

conda activate mrp

python /net/student33/data2/hendrix/git/MRP/src/neural_nets/generate_dataset.py
