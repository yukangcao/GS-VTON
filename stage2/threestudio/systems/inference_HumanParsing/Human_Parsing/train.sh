#!/bin/zsh
python3 train.py --num-classes 8 --batch-size 2 --gpu '0' --schp-start 1000 --data-dir './MHP' --eval-epochs 1 --imagenet-pretrain './pretrain_model/atr.pth' --project-name 'HP-ver1' --num-worker 8
