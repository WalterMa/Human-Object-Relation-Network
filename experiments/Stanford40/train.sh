#!/usr/bin/env bash

work_path=$(dirname $0)
cd ./${work_path}

python ./train_st40.py --network resnet50_v1d --dataset st40 --gpus 0 --epochs 15 \
       --start-epoch 0 --max-lr 3e-5 --min-lr 1e-6 --cycle-len 60000 --seed 233 --verbose