#!/usr/bin/env bash

work_path=$(dirname $0)
cd ./${work_path}

python ./train_hico.py --network resnet50_v1d --dataset hico --gpus 0 --epochs 20 \
       --start-epoch 0 --max-lr 1.5e-4 --min-lr 1e-5 --cycle-len 600000 --seed 233 --verbose