#!/usr/bin/env bash
# LocateAnything full SFT with MoonViT-V2 vision (Magi Attention).
#
# Init: hybrid LocAny LLM + MoonViT-V2 (+ fresh mlp1)
#   /workspace/models/CommonModels/LocateAnything-3B-MoonViTV2
#
# Usage:
#   conda activate locateanything
#   bash shell/locate-anything-streaming-synthdata-moonvitv2.sh [NNODES] [OUTPUT_DIR]
# Foreground:
#   TRAIN_BACKGROUND=0 bash shell/locate-anything-streaming-synthdata-moonvitv2.sh
#
# Continue on anno after this run finishes (example):
#   MODEL_PATH=.../locateanything-3b-moonvitv2-synthdata-v4-mix/checkpoint-6000 \
#   META_PATH=data/recipe/recipe_anno_mix.json \
#   WANDB_NAME=locateanything-3b-moonvitv2-synthdatav4mixckpt6000-annodata \
#   LR=5e-6 MAX_STEPS=3000 SAVE_STEPS=200 MILESTONE_INTERVAL=1000 \
#   bash shell/locate-anything-streaming-synthdata-moonvitv2.sh
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
cd "$ROOT"

# =============================================================================
# wandb
# =============================================================================
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_PROJECT="${WANDB_PROJECT:-pad_dimension_and_pad_hole}"
export WANDB_NAME="${WANDB_NAME:-locateanything-3b-moonvitv2-synthdata-v4-mix}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

# =============================================================================
# Paths / model / data  (MoonViT-V2 hybrid)
# =============================================================================
MODEL_PATH=${MODEL_PATH:-"/workspace/models/CommonModels/LocateAnything-3B-MoonViTV2"}
META_PATH=${META_PATH:-"data/recipe/recipe_synth_v4_mix.json"}
# Must omit DeepSpeed "optimizer" when using --lr_scale (HF creates param-group AdamW).
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"deepspeed_configs/zero_stage1_config_lr_scale.json"}

# =============================================================================
# Distributed
# =============================================================================
GPUS=${GPUS:-8}
NNODES=${1:-1}
NODE_RANK=${NODE_RANK:-0}
# Default different from v1 synth (29500) to avoid clashing if both launchers coexist.
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# =============================================================================
# Freeze  (True=freeze, False=train)
# =============================================================================
FREEZE_LLM=${FREEZE_LLM:-False}          # language_model
FREEZE_MLP=${FREEZE_MLP:-False}          # mlp1 / linear connector
FREEZE_BACKBONE=${FREEZE_BACKBONE:-False} # vision_model (MoonViT-V2)

# =============================================================================
# Learning rate
# actual_lr = LR * scale
# default @ LR=1e-5 → ViT 5e-6 | MLP 1e-5 | LLM 1e-5
# =============================================================================
LR=${LR:-1e-5}
LR_SCALE=${LR_SCALE:-"vision_model: 0.5, mlp: 1.0, llm: 1.0"}

# =============================================================================
# Schedule  (aligned with locate-anything-streaming-synthdata.sh)
# =============================================================================
MAX_STEPS=${MAX_STEPS:-6000}
WARMUP_RATIO=${WARMUP_RATIO:-0.01}
if [ -z "${WARMUP_STEPS:-}" ]; then
  WARMUP_STEPS="$(python -c "print(max(1, int(${MAX_STEPS} * ${WARMUP_RATIO})))")"
fi
SAVE_STEPS=${SAVE_STEPS:-250}
SAVE_LIMIT=${SAVE_LIMIT:-3}
MILESTONE_INTERVAL=${MILESTONE_INTERVAL:-2000}

# =============================================================================
# Batch / packing / seq length (Magi long-context defaults)
# =============================================================================
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=${GRADIENT_ACC:-2}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-magi}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
MAX_NUM_TOKENS_PER_SAMPLE=${MAX_NUM_TOKENS_PER_SAMPLE:-4096}
MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-4096}
PACKING_BUFFER_SIZE=${PACKING_BUFFER_SIZE:-32}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}


# =============================================================================
# Runtime env
# =============================================================================
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN
fi
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# =============================================================================
# Output / logs
# =============================================================================
OUTPUT_DIR=${2:-"/workspace/models/CheckPoints/${WANDB_PROJECT}/${WANDB_NAME}"}
mkdir -p "$OUTPUT_DIR"

mkdir -p "${ROOT}/logs"
LOG_FILE="${LOG_FILE:-${ROOT}/logs/${WANDB_NAME}.log}"

# Default: nohup background. Set TRAIN_BACKGROUND=0 for foreground.
if [ "${TRAIN_BACKGROUND:-1}" != 0 ] && [ "${_MAGI_TRAIN_INNER:-0}" != 1 ]; then
  export _MAGI_TRAIN_INNER=1
  nohup bash "$SCRIPT_PATH" "$@" >"${LOG_FILE}" 2>&1 &
  echo "nohup pid=$! log=${LOG_FILE}"
  exit 0
fi

echo "============================================================"
echo " LocateAnything Magi SFT (MoonViT-V2)"
echo "------------------------------------------------------------"
echo " MODEL_PATH          = ${MODEL_PATH}"
echo " META_PATH           = ${META_PATH}"
echo " OUTPUT_DIR          = ${OUTPUT_DIR}"
echo " LOG_FILE            = ${LOG_FILE}"
echo " GPUS/NNODES         = ${GPUS} / ${NNODES}  (rank=${NODE_RANK})"
echo " ATTN                = ${ATTN_IMPLEMENTATION}  MAX_SEQ=${MAX_SEQ_LENGTH}"
echo " FREEZE              = llm=${FREEZE_LLM}  mlp=${FREEZE_MLP}  vit=${FREEZE_BACKBONE}"
echo " LR / LR_SCALE       = ${LR} / ${LR_SCALE}"
echo " MAX_STEPS           = ${MAX_STEPS}"
echo " WARMUP              = steps=${WARMUP_STEPS}  (ratio=${WARMUP_RATIO})"
echo " SAVE                = steps=${SAVE_STEPS}  limit=${SAVE_LIMIT}  milestone=${MILESTONE_INTERVAL}"
echo " BATCH               = per_device=${PER_DEVICE_BATCH_SIZE}  grad_acc=${GRADIENT_ACC}"
echo " WANDB               = project=${WANDB_PROJECT}  name=${WANDB_NAME}"
echo " python              = $(command -v python)"
echo "============================================================"

run_train() {
  LAUNCHER=pytorch python -m torch.distributed.run \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    eaglevl/train/locany_finetune_magi_stream.py \
    --model_name_or_path "$MODEL_PATH" \
    --meta_path "$META_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir False \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
    --causal_attn False \
    --block_size 6 \
    --mlp_connector_layers 2 \
    --vision_select_layer -1 \
    --freeze_llm "$FREEZE_LLM" \
    --freeze_mlp "$FREEZE_MLP" \
    --freeze_backbone "$FREEZE_BACKBONE" \
    --learning_rate "$LR" \
    --lr_scale "$LR_SCALE" \
    --weight_decay 0.01 \
    --max_steps "$MAX_STEPS" \
    --warmup_steps "$WARMUP_STEPS" \
    --lr_scheduler_type "cosine" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACC" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --max_num_tokens_per_sample "$MAX_NUM_TOKENS_PER_SAMPLE" \
    --max_num_tokens "$MAX_NUM_TOKENS" \
    --packing_buffer_size "$PACKING_BUFFER_SIZE" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --save_strategy "steps" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_LIMIT" \
    --milestone_interval "$MILESTONE_INTERVAL" \
    --logging_steps 1 \
    --sample_log_interval 1 \
    --video_total_pixels 8192 \
    --bf16 True \
    --do_train True \
    --grad_checkpoint True \
    --group_by_length False \
    --report_to "wandb" \
    --run_name "$WANDB_NAME" \
    --use_onelogger True
}

if [ "${_MAGI_TRAIN_INNER:-0}" = 1 ] || [ "${TRAIN_LAUNCHER_MANAGED:-0}" = 1 ]; then
  run_train
else
  run_train 2>&1 | tee "${LOG_FILE}"
fi
