#!/usr/bin/env bash
# LocateAnything - PCB eval (dimension match and/or pad/hole detection)
# Steps: DDP Inference → (optional) pad_detect dimension reward / pad_hole YOLO metrics
#
# --task dimension|pad_hole|both
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
TASK=${TASK:-dimension}

MODEL_PATH=${MODEL_PATH:-"/workspace/models/CheckPoints/pad_dimension_and_pad_hole/locateanything-3b-moonvitv2-annodata/milestones/checkpoint-3000/"}
TEST_JSONL=${TEST_JSONL:-"/workspace/PROJECTS/github/Eagle/Embodied/data/test/test.jsonl"}
IMAGE_ROOT=${IMAGE_ROOT:-""}
# SAVE_PATH=${SAVE_PATH:-"/workspace/PROJECTS/pad_detect/results/size_line_value_match/locate_anything/locateanything-3b-moonvitv2-full-${GENERATION_MODE}-${TASK}-synthdatav5ckpt6000.jsonl"}
SAVE_PATH=${SAVE_PATH:-"/workspace/PROJECTS/pad_detect/results/size_line_value_match/locate_anything/locateanything-3b-moonvitv2-full-${GENERATION_MODE}-${TASK}-annodatackpt3000.jsonl"}
DIMENSION_PROMPT=${DIMENSION_PROMPT:-""}
PAD_HOLE_PROMPT=${PAD_HOLE_PROMPT:-""}
RUN_REWARD=${RUN_REWARD:-1}
PRINT_HISTORY=${PRINT_HISTORY:-0}
PRINT_SAMPLE=${PRINT_SAMPLE:-1}
EXTRA_ARGS=()

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EAGLE_EMBODIED="$(cd "${EVAL_DIR}/.." && pwd)"
DEFAULT_DIMENSION_PROMPT="${EAGLE_EMBODIED}/prompts/pcb_dimension_locate.txt"
DEFAULT_PAD_HOLE_PROMPT="${EAGLE_EMBODIED}/prompts/pcb_smd_hole_locate.txt"
DIMENSION_PROMPT="${DIMENSION_PROMPT:-$DEFAULT_DIMENSION_PROMPT}"
PAD_HOLE_PROMPT="${PAD_HOLE_PROMPT:-$DEFAULT_PAD_HOLE_PROMPT}"
LOG_DIR="${LOG_DIR:-${EAGLE_EMBODIED}/logs}"

print_help() {
    cat <<EOF
Usage: $0 [OPTIONS]

  --task T                dimension | pad_hole | both (default: both)
  --model_path PATH       Finetuned LocateAnything checkpoint
  --test_jsonl PATH       test JSONL (ID/image_path/dimension_label/pad_hole_label)
  --image_root DIR        Optional image root for relative paths
  --save_path PATH        Output prediction JSONL
  --dimension_prompt PATH Dimension prompt file
  --pad_hole_prompt PATH  Pad/hole prompt file
  --generation_mode M     fast | slow | hybrid (default: hybrid)
  --gpus N                GPUs per node (default: 8)
  --limit N               Only first N samples
  --no-reward             Skip pad_detect dimension reward evaluation
  --print-history         Print MTP/AR step chunks per sample
  --no-print-sample       Disable per-sample terminal dump
  --foreground            Run in foreground (also: EVAL_BACKGROUND=0)
  -h|--help               Show help

Output JSONL keeps GT fields from test_dimension_pad_hole.jsonl and adds:
  model_response / model_result
  pad_hole_response / pad_hole_result

Pad/hole prints YOLO-style Box(P)/R/mAP50/mAP50-95 (circle=hole, rect=pad).

Example:
  TASK=both GPUS=8 bash \$0 --model_path /path/to/ckpt --limit 3
  EVAL_BACKGROUND=0 bash \$0 --task pad_hole --limit 1
EOF
}

FOREGROUND_FLAG=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)                 TASK="$2"; shift 2;;
        --model_path)           MODEL_PATH="$2"; shift 2;;
        --test_jsonl)           TEST_JSONL="$2"; shift 2;;
        --image_root)           IMAGE_ROOT="$2"; shift 2;;
        --save_path)            SAVE_PATH="$2"; shift 2;;
        --dimension_prompt)     DIMENSION_PROMPT="$2"; shift 2;;
        --pad_hole_prompt)      PAD_HOLE_PROMPT="$2"; shift 2;;
        # legacy aliases
        --prompt_path)          DIMENSION_PROMPT="$2"; shift 2;;
        --pad_hole_prompt_path) PAD_HOLE_PROMPT="$2"; shift 2;;
        --generation_mode)      GENERATION_MODE="$2"; shift 2;;
        --gpus)                 GPUS="$2"; shift 2;;
        --limit)                LIMIT="$2"; shift 2;;
        --print-history)        PRINT_HISTORY=1; shift;;
        --no-print-sample)      PRINT_SAMPLE=0; shift;;
        --no-reward)            RUN_REWARD=0; shift;;
        --foreground)           FOREGROUND_FLAG=1; shift;;
        -h|--help)              print_help; exit 0;;
        *)                      echo "Unknown option: $1"; print_help; exit 1;;
    esac
done

case "$TASK" in
    dimension|pad_hole|both) ;;
    *) echo "Invalid --task: $TASK (expected dimension|pad_hole|both)"; exit 1;;
esac

# Log name = SAVE_PATH filename stem + .log
SAVE_BASENAME="$(basename "$SAVE_PATH")"
SAVE_STEM="${SAVE_BASENAME%.*}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/eval_${SAVE_STEM}.log}"
mkdir -p "$LOG_DIR" "$(dirname "$SAVE_PATH")"

# Default: nohup background. Set EVAL_BACKGROUND=0 or --foreground for fg.
if [[ "${EVAL_BACKGROUND:-1}" != 0 && "$FOREGROUND_FLAG" != 1 && "${_EVAL_MATCH_INNER:-0}" != 1 ]]; then
    REEXEC_ARGS=(
        --task "$TASK"
        --model_path "$MODEL_PATH"
        --test_jsonl "$TEST_JSONL"
        --save_path "$SAVE_PATH"
        --dimension_prompt "$DIMENSION_PROMPT"
        --pad_hole_prompt "$PAD_HOLE_PROMPT"
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

echo "=== PCB LocAny Inference (task=$TASK) ==="
echo "MODEL_PATH=$MODEL_PATH"
echo "TEST_JSONL=$TEST_JSONL"
echo "SAVE_PATH=$SAVE_PATH"
echo "LOG_FILE=$LOG_FILE"
echo "GPUS=$GPUS GENERATION_MODE=$GENERATION_MODE TASK=$TASK"

run_eval() {
torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$GPUS" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$PORT" \
    "$EVAL_DIR/inference_match_ddp.py" \
    --task "$TASK" \
    --model_path "$MODEL_PATH" \
    --test_jsonl_path "$TEST_JSONL" \
    --save_path "$SAVE_PATH" \
    --prompt_path "$DIMENSION_PROMPT" \
    --pad_hole_prompt_path "$PAD_HOLE_PROMPT" \
    --generation_mode "$GENERATION_MODE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    "${EXTRA_ARGS[@]}"

# Dimension reward (pad_detect) when dimension predictions exist.
if [[ "$RUN_REWARD" == "1" && ("$TASK" == "dimension" || "$TASK" == "both") ]]; then
    REWARD_SCRIPT="/workspace/PROJECTS/pad_detect/eval/reward_new/evaluate_dimension_line_value_reward.py"
    if [[ -f "$REWARD_SCRIPT" ]]; then
        echo "=== Running pad_detect dimension reward eval ==="
        REWARD_ARGS=(--input "$SAVE_PATH" --pred-key model_result --gt-key dimension_label --reward-format json)
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
