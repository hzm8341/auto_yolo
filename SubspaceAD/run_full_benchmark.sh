#!/bin/bash
cd /media/hzm/Data/auto_yolo/SubspaceAD
mkdir -p results_full

/usr/bin/python3 -u main.py \
    --dataset_name mvtec_ad \
    --dataset_path datasets/mvtec-ad \
    --model_ckpt facebook/dinov2-base \
    --image_res 672 \
    --k_shot 1 \
    --aug_count 30 \
    --pca_ev 0.99 \
    --agg_method mean \
    --layers=-1,-2,-3,-4 \
    --batch_size 1 \
    --outdir results_full \
    2>&1 | tee results_full/benchmark_full.log
