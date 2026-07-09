#!/bin/bash
# COREGEN two-stage training for DeepSeek-Coder-6.7B on 4x 4090
# Stage 1: Train DependencyEncoding dictionary (250M tokens, no mask, no LoRA)
# Stage 2: LoRA finetune with block mask (25M tokens, rank 32)
# Both stages: batch_size 8 (4 GPUs x 2), lr 5e-5, seq_length 900

export CUDA_VISIBLE_DEVICES=1,2,3,4

STAGE=${1:-1}
shift 2>/dev/null  # consume $1 so "$@" passes remaining args

if [ "$STAGE" = "1" ]; then
    echo "=== Stage 1: DependencyEncoding pre-training ==="
    accelerate launch --num_processes 4 --mixed_precision bf16 \
        finetuning-deepseek.py \
        --stage 1 \
        --model_name ~/models/deepseek-coder-6.7b-instruct \
        --train_batch_size 2 \
        --learning_rate 5e-5 \
        --max_train_steps 34722 \
        --seq_length 900 \
        --save_checkpoint_steps 5000 \
        --log_step 1000 \
        --output_dir ./results_deepseek_s1 \
        --project_name COREGEN-DeepSeek-S1 \
        --gradient_checkpointing \
        "$@"

elif [ "$STAGE" = "2" ]; then
    echo "=== Stage 2: LoRA finetune with block mask ==="
    accelerate launch --num_processes 4 --mixed_precision bf16 \
        finetuning-deepseek.py \
        --stage 2 \
        --model_name ./results_deepseek_s1/final_checkpoint \
        --lora_rank 32 \
        --lora_alpha 32 \
        --lora_target_layers_start 16 \
        --train_batch_size 2 \
        --learning_rate 5e-5 \
        --max_train_steps 3472 \
        --seq_length 900 \
        --save_checkpoint_steps 1000 \
        --log_step 500 \
        --output_dir ./results_deepseek_s2 \
        --project_name COREGEN-DeepSeek-S2 \
        --gradient_checkpointing \
        "$@"
else
    echo "Usage: $0 {1|2} [extra args...]"
    exit 1
fi
