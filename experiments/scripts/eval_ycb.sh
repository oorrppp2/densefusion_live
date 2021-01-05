#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

if [ ! -d YCB_Video_toolbox ];then
    echo 'Downloading the YCB_Video_toolbox...'
    git clone https://github.com/yuxng/YCB_Video_toolbox.git
    cd YCB_Video_toolbox
    unzip results_PoseCNN_RSS2018.zip
    cd ..
    cp replace_ycb_toolbox/*.m YCB_Video_toolbox/
fi

python ./tools/eval_ycb.py --dataset_root /media/user/ssd_1TB/YCB_dataset\
  --model trained_checkpoints/ycb/pose_model_26_0.012863246640872631.pth\
  --refine_model trained_checkpoints/ycb/pose_refine_model_69_0.009449292959118935.pth\
  --posecnn_model trained_checkpoints/ycb/checkpoint_50Epoch.pth.tar\
  --trained_model=tools/yolact/weights/yolact_resnet50_18_8000.pth --config=yolact_ycb_config --dataset=ycb_dataset --cross_class_nms=True
#  --dataset_root ./datasets/ycb/YCB_Video_Dataset\

