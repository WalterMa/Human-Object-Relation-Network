#!/usr/bin/env bash

work_path=$(dirname $0)
cd ./${work_path}

python ./eval_hico.py --network resnet50_v1d --dataset hico --gpus 0 \
 --pretrained horelation_resnet50_v1d_hico.params --save-outputs