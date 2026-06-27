#!/bin/bash
# Finetune DeepSeek-Coder-6.7B-Instruct with LoRA on 4x 4090 (GPU 1,2,3,4)

export CUDA_VISIBLE_DEVICES=1,2,3,4

accelerate launch --num_processes 4 --mixed_precision bf16 \
    finetuning-deepseek.py \
    --model_name ~/models/deepseek-coder-6.7b-instruct \
    --lora_rank 64 \
    --lora_alpha 64 \
    --lora_target_layers_start 0 \
    --train_batch_size 1 \
    --learning_rate 5e-5 \
    --max_train_steps 12000 \
    --seq_length 1024 \
    --extrapolate_length 8192 \
    --save_checkpoint_steps 3000 \
    --output_dir ./results_deepseek \
    --project_name COREGEN-DeepSeek \
    --gradient_checkpointing \
    "$@"
