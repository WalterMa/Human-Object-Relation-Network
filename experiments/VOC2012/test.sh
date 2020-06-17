#!/usr/bin/env bash

work_path=$(dirname $0)
cd ./${work_path}

python ./eval_voca.py --network resnet50_v1d --dataset voca --gpus 0 \
 --pretrained horelation_resnet50_v1d_voc_2012.params --save-outputs