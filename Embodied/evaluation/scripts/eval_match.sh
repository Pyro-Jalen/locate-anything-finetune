#!/usr/bin/env bash
# LocateAnything - PCB dimension line↔value match evaluation
# Steps: DDP Inference → (optional) pad_detect reward eval
#
# Default: background via nohup, log under Embodied/logs/<save_stem>.log
# Foreground:
#   EVAL_BACKGROUND=0 bash evaluation/scripts/eval_match.sh
set -euo pipefail

GPUS=${GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PORT=${PORT:-29510}
GENERATION_MODE=${GENERATION_MODE:-"slow"} # hybrid/ fast/ slow
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
LIMIT=${LIMIT:-}

MODEL_PATH=${MODEL_PATH:-"/workspace/models/CheckPoints/size_line_value_match/locateanything-3b-full-synthdatackpt5000-annodata-unfreezevit/milestones/checkpoint-1500"}
TEST_JSONL=${TEST_JSONL:-"/workspace/PROJECTS/github/Eagle/Embodied/data/test/test.jsonl"}
IMAGE_ROOT=${IMAGE_ROOT:-""}
SAVE_PATH=${SAVE_PATH:-"/workspace/PROJECTS/pad_detect/results/size_line_value_match/locate_anything/locateanything-3b-full-${GENERATION_MODE}-synthdatackpt5000-annodatackpt1500-unfrezevit.jsonl"}
PROMPT_PATH=${PROMPT_PATH:-""}
RUN_REWARD=${RUN_REWARD:-1}
PRINT_HISTORY=${PRINT_HISTORY:-0}
PRINT_SAMPLE=${PRINT_SAMPLE:-1}
EXTRA_ARGS=()

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EAGLE_EMBODIED="$(cd "${EVAL_DIR}/.." && pwd)"
DEFAULT_PROMPT="${EAGLE_EMBODIED}/prompts/pcb_dimension_locate.txt"
PROMPT_PATH="${PROMPT_PATH:-$DEFAULT_PROMPT}"
LOG_DIR="${LOG_DIR:-${EAGLE_EMBODIED}/logs}"

print_help() {
    cat <<EOF
Usage: $0 [OPTIONS]

  --model_path PATH       Finetuned LocateAnything checkpoint
  --test_jsonl PATH       test JSONL (ID/image_path/dimension_label[/pad_hole_label])
  --image_root DIR        Optional image root for relative paths
  --save_path PATH        Output prediction JSONL
  --prompt_path PATH      Training prompt file
  --generation_mode M     fast | slow | hybrid (default: hybrid)
  --gpus N                GPUs per node (default: 8)
  --limit N               Only first N samples
  --no-reward             Skip pad_detect reward evaluation
  --print-history         Print MTP/AR step chunks per sample
  --no-print-sample       Disable per-sample terminal dump
  --foreground            Run in foreground (also: EVAL_BACKGROUND=0)
  -h|--help               Show help

Log (background default):
  ${LOG_DIR}/<basename(SAVE_PATH) with .log>

Example:
  GPUS=8 bash \$0 --model_path /path/to/ckpt --limit 3
  EVAL_BACKGROUND=0 bash \$0 --limit 1
EOF
}

FOREGROUND_FLAG=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)       MODEL_PATH="$2"; shift 2;;
        --test_jsonl)       TEST_JSONL="$2"; shift 2;;
        --image_root)       IMAGE_ROOT="$2"; shift 2;;
        --save_path)        SAVE_PATH="$2"; shift 2;;
        --prompt_path)      PROMPT_PATH="$2"; shift 2;;
        --generation_mode)  GENERATION_MODE="$2"; shift 2;;
        --gpus)             GPUS="$2"; shift 2;;
        --limit)            LIMIT="$2"; shift 2;;
        --print-history)    PRINT_HISTORY=1; shift;;
        --no-print-sample)  PRINT_SAMPLE=0; shift;;
        --no-reward)        RUN_REWARD=0; shift;;
        --foreground)       FOREGROUND_FLAG=1; shift;;
        -h|--help)          print_help; exit 0;;
        *)                  echo "Unknown option: $1"; print_help; exit 1;;
    esac
done

# Log name = SAVE_PATH filename stem + .log
SAVE_BASENAME="$(basename "$SAVE_PATH")"
SAVE_STEM="${SAVE_BASENAME%.*}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/eval_${SAVE_STEM}.log}"
mkdir -p "$LOG_DIR" "$(dirname "$SAVE_PATH")"

# Default: nohup background. Set EVAL_BACKGROUND=0 or --foreground for fg.
if [[ "${EVAL_BACKGROUND:-1}" != 0 && "$FOREGROUND_FLAG" != 1 && "${_EVAL_MATCH_INNER:-0}" != 1 ]]; then
    REEXEC_ARGS=(
        --model_path "$MODEL_PATH"
        --test_jsonl "$TEST_JSONL"
        --save_path "$SAVE_PATH"
        --prompt_path "$PROMPT_PATH"
        --generation_mode "$GENERATION_MODE"
        --gpus "$GPUS"
    )
    if [[ -n "$IMAGE_ROOT" ]]; then
        REEXEC_ARGS+=(--image_root "$IMAGE_ROOT")
    fi
    if [[ -n "$LIMIT" ]]; then
        REEXEC_ARGS+=(--limit "$LIMIT")
    fi
    if [[ "$PRINT_HISTORY" == "1" ]]; then
        REEXEC_ARGS+=(--print-history)
    fi
    if [[ "$PRINT_SAMPLE" == "0" ]]; then
        REEXEC_ARGS+=(--no-print-sample)
    fi
    if [[ "$RUN_REWARD" == "0" ]]; then
        REEXEC_ARGS+=(--no-reward)
    fi
    _EVAL_MATCH_INNER=1 nohup bash "$0" "${REEXEC_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
    echo "nohup pid=$! log=${LOG_FILE}"
    echo "tail -f ${LOG_FILE}"
    exit 0
fi

if [[ -n "$IMAGE_ROOT" ]]; then
    EXTRA_ARGS+=(--image_root_dir "$IMAGE_ROOT")
fi
if [[ -n "$LIMIT" ]]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi
if [[ "$PRINT_HISTORY" == "1" ]]; then
    EXTRA_ARGS+=(--print_history)
fi
if [[ "$PRINT_SAMPLE" == "0" ]]; then
    EXTRA_ARGS+=(--no-print_sample)
fi

echo "=== PCB Match Inference ==="
echo "MODEL_PATH=$MODEL_PATH"
echo "TEST_JSONL=$TEST_JSONL"
echo "SAVE_PATH=$SAVE_PATH"
echo "LOG_FILE=$LOG_FILE"
echo "GPUS=$GPUS GENERATION_MODE=$GENERATION_MODE"

run_eval() {
torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$GPUS" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$PORT" \
    "$EVAL_DIR/inference_match_ddp.py" \
    --model_path "$MODEL_PATH" \
    --test_jsonl_path "$TEST_JSONL" \
    --save_path "$SAVE_PATH" \
    --prompt_path "$PROMPT_PATH" \
    --generation_mode "$GENERATION_MODE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    "${EXTRA_ARGS[@]}"

if [[ "$RUN_REWARD" == "1" ]]; then
    REWARD_SCRIPT="/workspace/PROJECTS/pad_detect/eval/reward_new/evaluate_dimension_line_value_reward.py"
    if [[ -f "$REWARD_SCRIPT" ]]; then
        echo "=== Running pad_detect reward eval ==="
        # LocAny raw tokens live in model_response; score structured model_result as JSON.
        REWARD_ARGS=(--input "$SAVE_PATH" --pred-key model_result --reward-format json)
        if command -v uv >/dev/null 2>&1; then
            (cd /workspace/PROJECTS/pad_detect && uv run python "$REWARD_SCRIPT" "${REWARD_ARGS[@]}") || true
        else
            python "$REWARD_SCRIPT" "${REWARD_ARGS[@]}" || true
        fi
    else
        echo "Reward script not found: $REWARD_SCRIPT (skip)"
    fi
fi

echo "Done. Predictions: $SAVE_PATH"
}

# Background (nohup already redirected to LOG_FILE). Foreground also tees to log.
if [[ "${_EVAL_MATCH_INNER:-0}" == 1 ]]; then
    run_eval
else
    run_eval 2>&1 | tee "${LOG_FILE}"
fi
