#!/bin/bash
set -euo pipefail

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a __CUDA_DEVICE_LIST <<< "${CUDA_VISIBLE_DEVICES// /}"
    NPROC_PER_NODE=${#__CUDA_DEVICE_LIST[@]}
else
    NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
fi
if [ "${NPROC_PER_NODE:-0}" -le 0 ]; then NPROC_PER_NODE=1; fi

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
WORLD_SIZE=$((NPROC_PER_NODE * NNODES))
export WORLD_SIZE NODE_RANK

# ======================
# Path Configuration
# ======================
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-facebook/VGGT-1B}"
OUTPUT_DIR="${OUTPUT_DIR:-./JanusVLN_Base}"
CACHE_DIR="${CACHE_DIR:-./cache}"
mkdir -p "$OUTPUT_DIR"

# ======================
# Training Hyperparameters
# ======================
LR="${LR:-2e-5}"
total_batch_size="${TOTAL_BATCH_SIZE:-64}"
if [ "$WORLD_SIZE" -gt 0 ]; then
    GRADIENT_ACCUMULATION_STEPS=$(( total_batch_size / WORLD_SIZE ))
else
    GRADIENT_ACCUMULATION_STEPS=$total_batch_size
fi
if [ "$GRADIENT_ACCUMULATION_STEPS" -le 0 ]; then GRADIENT_ACCUMULATION_STEPS=1; fi
echo ">>>>> grad accum = $GRADIENT_ACCUMULATION_STEPS"

# ======================
# Model / Fusion Configuration
# ======================
DATASETS="${DATASETS:-train_r2r_rxr}"
LAM="${LAM:-0.2}"
REFERENCE_FRAME="${REFERENCE_FRAME:-first}"

# Deepstack settings — set FEATURE_FUSION_METHOD=lam_add to use the original JanusVLN path.
FEATURE_FUSION_METHOD="${FEATURE_FUSION_METHOD:-deepstack_language_add}"
GEOMETRY_FUSION_LAYERS="${GEOMETRY_FUSION_LAYERS:-0 1 2}"       # e.g. "0 1 2"
GEOMETRY_ENCODER_LAYERS="${GEOMETRY_ENCODER_LAYERS:-11 17 23}"     # e.g. "11 17 23"
POS_ENCODING_TYPE="${POS_ENCODING_TYPE:-none}"
INCLUDE_CAMERA_TOKEN="${INCLUDE_CAMERA_TOKEN:-False}"
TUNE_MM_FUSION="${TUNE_MM_FUSION:-True}"

export NCCL_NVLS_ENABLE=0

train_args=(
        --model_name_or_path "$MODEL_PATH"
        --vggt_model_path "$VGGT_MODEL_PATH"
        --lam "$LAM"
        --reference_frame "$REFERENCE_FRAME"
        --tune_mm_llm True
        --tune_mm_vision False
        --tune_mm_mlp True
        --tune_mm_fusion "$TUNE_MM_FUSION"
        --dataset_use "$DATASETS"
        --output_dir "$OUTPUT_DIR"
        --cache_dir "$CACHE_DIR"
        --bf16
        --per_device_train_batch_size 1
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
        --learning_rate "$LR"
        --mm_projector_lr 1e-5
        --vision_tower_lr 1e-6
        --optim adamw_torch
        --model_max_length 163840
        --data_flatten False
        --max_pixels $((576*28*28))
        --min_pixels $((16*28*28))
        --base_interval 2
        --video_max_frames 8
        --video_min_frames 4
        --video_max_frame_pixels $((1664*28*28))
        --video_min_frame_pixels $((256*28*28))
        --num_train_epochs 1
        --warmup_ratio 0.03
        --lr_scheduler_type cosine
        --weight_decay 0.01
        --logging_steps 10
        --save_steps 1000
        --save_total_limit 1
        --deepspeed "scripts/zero3.json"
        --gradient_checkpointing
        --dataloader_num_workers 8
        --group_by_modality_length true
        --seed 42
        --report_to none
        --feature_fusion_method "$FEATURE_FUSION_METHOD"
        --pos_encoding_type "$POS_ENCODING_TYPE"
        --include_camera_token "$INCLUDE_CAMERA_TOKEN"
)

# Only pass layer indices when deepstack is active (non-empty strings avoid nargs parse errors).
if [[ -n "${GEOMETRY_FUSION_LAYERS}" ]]; then
    # shellcheck disable=SC2086
    train_args+=(--geometry_fusion_layers ${GEOMETRY_FUSION_LAYERS})
fi
if [[ -n "${GEOMETRY_ENCODER_LAYERS}" ]]; then
    # shellcheck disable=SC2086
    train_args+=(--geometry_encoder_layers ${GEOMETRY_ENCODER_LAYERS})
fi

torchrun --nproc_per_node="$NPROC_PER_NODE" \
         --nnodes="$NNODES" \
         --node_rank="$NODE_RANK" \
         --master_addr="$MASTER_ADDR" \
         --master_port="$MASTER_PORT" \
         src/qwen_vl/train/train_qwen.py \
         "${train_args[@]}" \
         2>&1 | tee "${OUTPUT_DIR}/train.rank${NODE_RANK}.log"
