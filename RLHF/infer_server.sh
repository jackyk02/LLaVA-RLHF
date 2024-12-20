#!/bin/bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=0
export MODEL_DIR="/root/LLaVA-RLHF/model_dir"
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=1

# MODEL CONFIG
VISION_TOWER=openai/clip-vit-large-patch14-336
LM_MODEL_NAME=LLaVA-RLHF-7b-v1.5-224/sft_model/

# SAVE CONFIG
MODEL_NAME=LLaVA-Fact-RM-LLaVA-RLHF-7b-v1.5-224

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    infer_server.py \
    --do_eval \
    --seed 42 \
    --model_name_or_path $MODEL_DIR/$LM_MODEL_NAME \
    --vision_tower $VISION_TOWER \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --model_max_length 2048 \
    --query_len 1280 \
    --response_len 768 \
    --bits 16 \
    --lora_r 64 \
    --lora_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --output_dir "$MODEL_DIR/$MODEL_NAME" \
    --checkpoint_dir $CHECKPOINT_DIR \
    --group_by_length False \
    --bf16 True \
    --reward_prompt_file "./prompts/robot_reward_prompt.txt" \
    --image_aspect_ratio 'pad' \