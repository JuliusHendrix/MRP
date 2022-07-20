#!/bin/bash

source /home/s1825216/miniconda3/etc/profile.d/conda.sh

conda activate mrp

python /home/s1825216/git/MRP/src/neural_nets/individualAEs/MRAE/train_MRAE.py
