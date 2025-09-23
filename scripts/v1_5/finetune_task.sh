#!/bin/bash
export PYTHONPATH=./:$PYTHONPATH
model_name_or_path=ckpts/llava-v1.5-7b
data_path=/MSMU/train.parquet
vision_tower=/ckpts/clip-vit-large-patch14-336 # if changed, the corresponding path in LLava1.5 file need also to be changed

depth_path=/ckpts/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth

gt_depth=False
use_depth=True
include=localhost:0,1,2,3,4,5,6,7
deepspeed --include $include llava/train/train_xformers.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $model_name_or_path  \
    --depth_path $depth_path\
    --version v1 \
    --data_path $data_path \
    --vision_tower $vision_tower \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False\
    --fp16 True\
    --output_dir ./checkpoints/llava-v1.5-7b-task \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False\
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
