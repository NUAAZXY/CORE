#!/bin/bash

# COPE Training Script for Language Models
# This script trains a language model using the COPE method

set -e

# Default parameters
MODEL_NAME="codellama-7b"
VOCAB_SIZE=32016
HIDDEN_SIZE=768
NUM_HIDDEN_LAYERS=12
NUM_ATTENTION_HEADS=8
INTERMEDIATE_SIZE=3072
MAX_POSITION_EMBEDDINGS=32768
TRAIN_BATCH_SIZE=16
VALID_BATCH_SIZE=1
WEIGHT_DECAY=0.1
LEARNING_RATE=5e-4
LR_SCHEDULER_TYPE="cosine"
NUM_WARMUP_STEPS=3000
GRADIENT_ACCUMULATION_STEPS=1
MAX_TRAIN_STEPS=150000
MAX_EVAL_STEPS=50
TRAIN_DATASET="starcoder_20B"
VALID_DATASET="starcoder_20Btokens_val"
SEQ_LENGTH=1024
EXTRAPOLATE_LENGTH=8192
SEED=42
SAVE_CHECKPOINT_STEPS=5000
LOG_STEP=5000
OUTPUT_DIR="./results"
PROJECT_NAME="NDCoDE"
USE_WANDB=false
GRADIENT_CHECKPOINTING=false

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  Model Parameters:"
    echo "    --model_name NAME                   Base model name (default: $MODEL_NAME)"
    echo "    --vocab_size SIZE                   Vocabulary size (default: $VOCAB_SIZE)"
    echo "    --hidden_size SIZE                  Hidden size (default: $HIDDEN_SIZE)"
    echo "    --num_hidden_layers NUM             Number of hidden layers (default: $NUM_HIDDEN_LAYERS)"
    echo "    --num_attention_heads NUM           Number of attention heads (default: $NUM_ATTENTION_HEADS)"
    echo "    --intermediate_size SIZE            Intermediate size (default: $INTERMEDIATE_SIZE)"
    echo "    --max_position_embeddings SIZE      Max position embeddings (default: $MAX_POSITION_EMBEDDINGS)"
    echo ""
    echo "  Training Parameters:"
    echo "    --train_batch_size SIZE             Training batch size (default: $TRAIN_BATCH_SIZE)"
    echo "    --valid_batch_size SIZE             Validation batch size (default: $VALID_BATCH_SIZE)"
    echo "    --weight_decay DECAY                Weight decay (default: $WEIGHT_DECAY)"
    echo "    --learning_rate RATE                Learning rate (default: $LEARNING_RATE)"
    echo "    --lr_scheduler_type TYPE            Learning rate scheduler (default: $LR_SCHEDULER_TYPE)"
    echo "    --num_warmup_steps STEPS            Number of warmup steps (default: $NUM_WARMUP_STEPS)"
    echo "    --gradient_accumulation_steps STEPS Gradient accumulation steps (default: $GRADIENT_ACCUMULATION_STEPS)"
    echo "    --max_train_steps STEPS             Maximum training steps (default: $MAX_TRAIN_STEPS)"
    echo "    --max_eval_steps STEPS              Maximum evaluation steps (default: $MAX_EVAL_STEPS)"
    echo ""
    echo "  Data Parameters:"
    echo "    --train_dataset DATASET             Training dataset path (default: $TRAIN_DATASET)"
    echo "    --valid_dataset DATASET             Validation dataset path (default: $VALID_DATASET)"
    echo "    --seq_length LENGTH                 Sequence length (default: $SEQ_LENGTH)"
    echo "    --extrapolate_length LENGTH         Extrapolation length (default: $EXTRAPOLATE_LENGTH)"
    echo ""
    echo "  Other Parameters:"
    echo "    --seed SEED                         Random seed (default: $SEED)"
    echo "    --save_checkpoint_steps STEPS       Save checkpoint steps (default: $SAVE_CHECKPOINT_STEPS)"
    echo "    --log_step STEPS                    Log steps (default: $LOG_STEP)"
    echo "    --output_dir DIR                    Output directory (default: $OUTPUT_DIR)"
    echo "    --project_name NAME                 Project name (default: $PROJECT_NAME)"
    echo "    --use_wandb                         Use wandb for logging"
    echo "    --gradient_checkpointing            Enable gradient checkpointing"
    echo "    -h, --help                          Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --max_train_steps 100000 --learning_rate 1e-4 --use_wandb"
    echo "  $0 --train_batch_size 32 --gradient_checkpointing --output_dir ./my_results"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --vocab_size)
            VOCAB_SIZE="$2"
            shift 2
            ;;
        --hidden_size)
            HIDDEN_SIZE="$2"
            shift 2
            ;;
        --num_hidden_layers)
            NUM_HIDDEN_LAYERS="$2"
            shift 2
            ;;
        --num_attention_heads)
            NUM_ATTENTION_HEADS="$2"
            shift 2
            ;;
        --intermediate_size)
            INTERMEDIATE_SIZE="$2"
            shift 2
            ;;
        --max_position_embeddings)
            MAX_POSITION_EMBEDDINGS="$2"
            shift 2
            ;;
        --train_batch_size)
            TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --valid_batch_size)
            VALID_BATCH_SIZE="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lr_scheduler_type)
            LR_SCHEDULER_TYPE="$2"
            shift 2
            ;;
        --num_warmup_steps)
            NUM_WARMUP_STEPS="$2"
            shift 2
            ;;
        --gradient_accumulation_steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift 2
            ;;
        --max_train_steps)
            MAX_TRAIN_STEPS="$2"
            shift 2
            ;;
        --max_eval_steps)
            MAX_EVAL_STEPS="$2"
            shift 2
            ;;
        --train_dataset)
            TRAIN_DATASET="$2"
            shift 2
            ;;
        --valid_dataset)
            VALID_DATASET="$2"
            shift 2
            ;;
        --seq_length)
            SEQ_LENGTH="$2"
            shift 2
            ;;
        --extrapolate_length)
            EXTRAPOLATE_LENGTH="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --save_checkpoint_steps)
            SAVE_CHECKPOINT_STEPS="$2"
            shift 2
            ;;
        --log_step)
            LOG_STEP="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --project_name)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --use_wandb)
            USE_WANDB=true
            shift
            ;;
        --gradient_checkpointing)
            GRADIENT_CHECKPOINTING=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Build command
CMD="python ../train_cope.py"
CMD="$CMD --model_name \"$MODEL_NAME\""
CMD="$CMD --vocab_size $VOCAB_SIZE"
CMD="$CMD --hidden_size $HIDDEN_SIZE"
CMD="$CMD --num_hidden_layers $NUM_HIDDEN_LAYERS"
CMD="$CMD --num_attention_heads $NUM_ATTENTION_HEADS"
CMD="$CMD --intermediate_size $INTERMEDIATE_SIZE"
CMD="$CMD --max_position_embeddings $MAX_POSITION_EMBEDDINGS"
CMD="$CMD --train_batch_size $TRAIN_BATCH_SIZE"
CMD="$CMD --valid_batch_size $VALID_BATCH_SIZE"
CMD="$CMD --weight_decay $WEIGHT_DECAY"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --lr_scheduler_type \"$LR_SCHEDULER_TYPE\""
CMD="$CMD --num_warmup_steps $NUM_WARMUP_STEPS"
CMD="$CMD --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS"
CMD="$CMD --max_train_steps $MAX_TRAIN_STEPS"
CMD="$CMD --max_eval_steps $MAX_EVAL_STEPS"
CMD="$CMD --train_dataset \"$TRAIN_DATASET\""
CMD="$CMD --valid_dataset \"$VALID_DATASET\""
CMD="$CMD --seq_length $SEQ_LENGTH"
CMD="$CMD --extrapolate_length $EXTRAPOLATE_LENGTH"
CMD="$CMD --seed $SEED"
CMD="$CMD --save_checkpoint_steps $SAVE_CHECKPOINT_STEPS"
CMD="$CMD --log_step $LOG_STEP"
CMD="$CMD --output_dir \"$OUTPUT_DIR\""
CMD="$CMD --project_name \"$PROJECT_NAME\""

if [[ "$USE_WANDB" == "true" ]]; then
    CMD="$CMD --use_wandb"
fi

if [[ "$GRADIENT_CHECKPOINTING" == "true" ]]; then
    CMD="$CMD --gradient_checkpointing"
fi

# Display configuration
echo "================================"
echo "COPE Training Configuration"
echo "================================"
echo "Model Configuration:"
echo "  Model Name: $MODEL_NAME"
echo "  Vocab Size: $VOCAB_SIZE"
echo "  Hidden Size: $HIDDEN_SIZE"
echo "  Hidden Layers: $NUM_HIDDEN_LAYERS"
echo "  Attention Heads: $NUM_ATTENTION_HEADS"
echo "  Intermediate Size: $INTERMEDIATE_SIZE"
echo "  Max Position Embeddings: $MAX_POSITION_EMBEDDINGS"
echo ""
echo "Training Configuration:"
echo "  Train Batch Size: $TRAIN_BATCH_SIZE"
echo "  Valid Batch Size: $VALID_BATCH_SIZE"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Learning Rate: $LEARNING_RATE"
echo "  LR Scheduler: $LR_SCHEDULER_TYPE"
echo "  Warmup Steps: $NUM_WARMUP_STEPS"
echo "  Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  Max Train Steps: $MAX_TRAIN_STEPS"
echo "  Max Eval Steps: $MAX_EVAL_STEPS"
echo ""
echo "Data Configuration:"
echo "  Train Dataset: $TRAIN_DATASET"
echo "  Valid Dataset: $VALID_DATASET"
echo "  Sequence Length: $SEQ_LENGTH"
echo "  Extrapolate Length: $EXTRAPOLATE_LENGTH"
echo ""
echo "Other Configuration:"
echo "  Seed: $SEED"
echo "  Save Checkpoint Steps: $SAVE_CHECKPOINT_STEPS"
echo "  Log Steps: $LOG_STEP"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Project Name: $PROJECT_NAME"
echo "  Use Wandb: $USE_WANDB"
echo "  Gradient Checkpointing: $GRADIENT_CHECKPOINTING"
echo "================================"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Execute the command
echo "Starting COPE training..."
eval $CMD

# Check if training was successful
if [[ $? -eq 0 ]]; then
    echo ""
    echo "================================"
    echo "Training completed successfully!"
    echo "Model saved to: $OUTPUT_DIR"
    echo "================================"
else
    echo ""
    echo "================================"
    echo "Training failed!"
    echo "================================"
    exit 1
fi