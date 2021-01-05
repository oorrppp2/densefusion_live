#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/train.py --dataset ycb\
  --dataset_root /media/user/433c5472-5bea-42d9-86c4-e0794e47477f/YCB_dataset
#  --dataset_root ./datasets/ycb/YCB_Video_Dataset
