#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/eval_ycb.py --dataset_root /home/vision/python_packages/YCB_Video_Models\
  --model trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth\
  --refine_model trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth\
  --score_threshold 0.02\
  --trained_model tools/yolact/weight/yolact_resnet50_204_90000.pth --config=yolact_ycb_config --dataset=ycb_dataset --cross_class_nms=True
#  --dataset_root ./datasets/ycb/YCB_Video_Dataset\

