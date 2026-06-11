#!/bin/bash -e
#SBATCH --job-name=janusvln-train
#SBATCH --output=logs/janusvln_train_%j.log
#SBATCH --error=logs/janusvln_train_%j.err
#SBATCH --nodelist=worker-2
#SBATCH --gpus=8
#SBATCH --cpus-per-task=120
#SBATCH --mem-per-cpu=8192
#
#SBATCH --container-image=/mnt/data/vmo-ai-task/dungpq6/ubuntu22-cuda128-conda-janusvln.sqsh
#SBATCH --container-mounts=/mnt/data/:/mnt/data/,/home/dungpq6/Project:/home/dungpq6/Project

set -euo pipefail

source /home/dungpq6/anaconda3/etc/profile.d/conda.sh
conda activate janusvln

# IMPORTANT: SLURM may execute a copied script under /var/spool/slurmd.
# Use explicit project root instead of deriving from script location.
PROJECT_ROOT="${PROJECT_ROOT:-/home/dungpq6/Project/JanusVLN}"
cd "${PROJECT_ROOT}"
mkdir -p logs

export PYTHONPATH="./:${PYTHONPATH:-}"
export NCCL_NVLS_ENABLE=1

# Per-node GPU count (NOT total across the job).
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi --list-gpus | wc -l)}}"
NUM_NODES="${NUM_NODES:-${SLURM_NNODES:-1}}"
MASTER_PORT="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 10000))}"

# NCCL: NVL is for intra-node NVLink; use SYS (or set NCCL_SOCKET_IFNAME) across nodes.
if [[ "${NUM_NODES}" -gt 1 ]]; then
    export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
    # Uncomment and set to your cluster NIC if cross-node hangs, e.g. eth0 or ib0:
    # export NCCL_SOCKET_IFNAME=eth0
fi

# Defaults aligned with scripts/train.sh
MODEL_PATH="${MODEL_PATH:-/mnt/data/vmo-ai-task/dungpq6/model-checkpoint/Qwen2.5-VL-3B-Instruct}"
VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-/mnt/data/vmo-ai-task/dungpq6/model-checkpoint/VGGT-1B}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data/vmo-ai-task/dungpq6/model-checkpoint/JanusVLN-3B-spatial-stack}"
CACHE_DIR="${CACHE_DIR:-./cache}"
DATASETS="${DATASETS:-train_r2r_rxr}"

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

# Master = first node in the allocation (torch.distributed rendezvous).
if [[ "${NUM_NODES}" -gt 1 ]]; then
    mapfile -t _slurm_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
    MASTER_ADDR="${MASTER_ADDR:-${_slurm_nodes[0]}}"
    if [[ "${RESOLVE_MASTER_IP:-1}" == "1" ]]; then
        _master_ip=$(srun --nodes=1 --ntasks=1 -w "${_slurm_nodes[0]}" hostname -I 2>/dev/null | awk '{print $1}' || true)
        if [[ -n "${_master_ip}" ]]; then
            MASTER_ADDR="${_master_ip}"
        fi
    fi
else
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "NUM_NODES=${NUM_NODES} NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "VGGT_MODEL_PATH=${VGGT_MODEL_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "DATASETS=${DATASETS}"

TORCHRUN_ARGS=(
    --nproc_per_node "${NPROC_PER_NODE}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

TRAIN_ARGS=(
    src/qwen_vl/train/train_qwen.py
    --model_name_or_path "${MODEL_PATH}"
    --vggt_model_path "${VGGT_MODEL_PATH}"
    --tune_mm_llm True
    --tune_mm_vision False
    --tune_mm_mlp True
    --dataset_use "${DATASETS}"
    --output_dir "${OUTPUT_DIR}"
    --cache_dir "${CACHE_DIR}"
    --bf16
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 64
    --learning_rate 2e-5
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
    --warmup_ratio 0.05
    --lr_scheduler_type "cosine"
    --weight_decay 0.01
    --logging_steps 10
    --save_steps 1000
    --save_total_limit 1
    --deepspeed "scripts/zero2.json"
    --gradient_checkpointing
    --dataloader_num_workers 8
    --group_by_modality_length true
    --seed 42
    --report_to "none"
    --reference_frame first
)

if [[ "${NUM_NODES}" -gt 1 ]]; then
    # SLURM_NODEID must be read inside each srun task, not in the submit shell.
    srun --nodes="${NUM_NODES}" --ntasks="${NUM_NODES}" --ntasks-per-node=1 bash -c '
        torchrun \
            --nnodes '"${NUM_NODES}"' \
            --nproc_per_node '"${NPROC_PER_NODE}"' \
            --node_rank "${SLURM_NODEID}" \
            --master_addr '"${MASTER_ADDR}"' \
            --master_port '"${MASTER_PORT}"' \
            '"$(printf '%q ' "${TRAIN_ARGS[@]}")"'
    ' > "${OUTPUT_DIR}/train.log" 2>&1
else
    if command -v srun >/dev/null 2>&1; then
        srun torchrun "${TORCHRUN_ARGS[@]}" "${TRAIN_ARGS[@]}" \
            > "${OUTPUT_DIR}/train.log" 2>&1
    else
        torchrun "${TORCHRUN_ARGS[@]}" "${TRAIN_ARGS[@]}" \
            > "${OUTPUT_DIR}/train.log" 2>&1
    fi
fi
