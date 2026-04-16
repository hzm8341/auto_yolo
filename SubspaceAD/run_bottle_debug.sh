#!/bin/bash
cd /media/hzm/Data/auto_yolo/SubspaceAD
mkdir -p results_debug

/usr/bin/python3 -u main.py \
    --dataset_name mvtec_ad \
    --dataset_path datasets/mvtec-ad \
    --categories bottle \
    --model_ckpt facebook/dinov2-with-registers-giant \
    --image_res 672 \
    --k_shot 1 \
    --aug_count 30 \
    --pca_ev 0.99 \
    --agg_method mean \
    --layers "-12,-13,-14,-15,-16,-17,-18" \
    --batch_size 1 \
    --debug_limit 5 \
    --outdir results_debug
