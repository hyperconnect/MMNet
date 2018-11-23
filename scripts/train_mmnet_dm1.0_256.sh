#!/bin/bash
set -eux

dataset_path=${1}
train_dir=${2:-mmnet-traindir}

python train.py \
    --num_classes 2 \
    --task_type matting \
    --output_name output/score \
    --output_type prob \
    --width 256 \
    --height 256 \
    --train_dir ${train_dir} \
    --batch_size 32 \
    --dataset_path ${dataset_path} \
    --dataset_split_name train \
    --learning_rate 1e-4 \
    --preprocess_method preprocess_normalize \
    --no-use_fused_batchnorm \
    --step_save_summaries 500 \
    --step_save_checkpoint 500 \
    --max_to_keep 3 \
    --max_outputs 1 \
    --augmentation_method resize_random_scale_crop_flip_rotate \
    --max_epoch_from_restore 30000 \
    --lambda_alpha_loss 1 \
    --lambda_comp_loss 1 \
    --lambda_grad_loss 1 \
    --lambda_kd_loss 1 \
    --lambda_aux_loss 1 \
    MMNetModel \
    --width_multiplier 1.0 \
    --weight_decay 4e-7 \
