#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$ROOT/local_env.sh"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

DEFAULT_LOCAL_MODEL="/home1/irteam/models/Qwen2.5-Math-7B-16k-think"
if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -d "$DEFAULT_LOCAL_MODEL" ]]; then
    MODEL_PATH="$DEFAULT_LOCAL_MODEL"
  else
    MODEL_PATH="Elliott/Qwen2.5-Math-7B-16k-think"
  fi
fi

DATA_DIR="${DATA_DIR:-$ROOT/data}"
TRAIN_PARQUET="${TRAIN_PARQUET:-$DATA_DIR/openr1.parquet}"
SFT_JSONL_DIR="${SFT_JSONL_DIR:-$DATA_DIR/openrlhf_sft}"
SFT_JSONL_FILE="${SFT_JSONL_FILE:-$SFT_JSONL_DIR/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/luffy_sft_base}"
LOG_FILE="${LOG_FILE:-}"
NUM_GPUS="${NUM_GPUS:-8}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
ZERO_STAGE="${ZERO_STAGE:-2}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
MAX_SAMPLES="${MAX_SAMPLES:-500000}"
MAX_LEN="${MAX_LEN:-16384}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.1}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:--1}"
PACKING_SAMPLES="${PACKING_SAMPLES:-1}"

WANDB_PROJECT="${WANDB_PROJECT:-r1_sft_distill}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen-7b-base-sft}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_KEY="${WANDB_KEY:-}"
if [[ -z "$WANDB_KEY" ]]; then
  WANDB_KEY="$(
    python - <<'PY'
from pathlib import Path
import netrc

try:
    auth = netrc.netrc(Path.home() / ".netrc").authenticators("api.wandb.ai")
except FileNotFoundError:
    auth = None

print(auth[2] if auth else "")
PY
  )"
fi
if [[ -z "$WANDB_KEY" ]]; then
  echo "Missing WANDB key. Set WANDB_KEY or configure ~/.netrc for api.wandb.ai." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

python - <<'PY'
import importlib.util
import sys

missing = [name for name in ("openrlhf", "deepspeed", "wandb") if importlib.util.find_spec(name) is None]
if missing:
    print("Missing Python packages:", ", ".join(missing), file=sys.stderr)
    sys.exit(1)
PY

if [[ ! -f "$TRAIN_PARQUET" ]]; then
  echo "Missing training parquet: $TRAIN_PARQUET" >&2
  echo "Run: cd $ROOT/data && python prepare_train.py" >&2
  exit 1
fi

mkdir -p "$SFT_JSONL_DIR"
pushd "$DATA_DIR" >/dev/null
LUFFY_DATA_DIR="$DATA_DIR" \
LUFFY_SFT_INPUT="$TRAIN_PARQUET" \
LUFFY_SFT_OUTPUT_DIR="$SFT_JSONL_DIR" \
python prepare_sft.py
popd >/dev/null

if [[ ! -f "$SFT_JSONL_FILE" ]]; then
  echo "Failed to create SFT JSONL: $SFT_JSONL_FILE" >&2
  exit 1
fi

cd "$ROOT"

DEEPSPEED_BIN="${DEEPSPEED_BIN:-$(command -v deepspeed)}"
if [[ -z "$DEEPSPEED_BIN" ]]; then
  echo "deepspeed command not found in current environment" >&2
  exit 1
fi

cmd=(
  "$DEEPSPEED_BIN"
  --num_gpus "$NUM_GPUS"
  --module openrlhf.cli.train_sft
  --model.model_name_or_path "$MODEL_PATH"
  --data.dataset "$SFT_JSONL_FILE"
  --data.input_key prompt
  --data.output_key target
  --data.apply_chat_template
  --data.max_samples "$MAX_SAMPLES"
  --data.max_len "$MAX_LEN"
  --train.batch_size "$TRAIN_BATCH_SIZE"
  --train.micro_batch_size "$MICRO_BATCH_SIZE"
  --train.max_epochs "$MAX_EPOCHS"
  --adam.lr "$LEARNING_RATE"
  --lr_warmup_ratio "$LR_WARMUP_RATIO"
  --ds.zero_stage "$ZERO_STAGE"
  --ds.param_dtype bf16
  --ds.adam_offload
  --ds.attn_implementation "$ATTN_IMPL"
  --model.gradient_checkpointing_enable
  --ckpt.output_dir "$OUTPUT_DIR"
  --ckpt.save_steps "$SAVE_STEPS"
  --logger.logging_steps "$LOGGING_STEPS"
  --eval.steps "$EVAL_STEPS"
  --logger.wandb.key "$WANDB_KEY"
  --logger.wandb.project "$WANDB_PROJECT"
  --logger.wandb.run_name "$WANDB_RUN_NAME"
)

if [[ -n "$WANDB_ENTITY" ]]; then
  cmd+=(--logger.wandb.org "$WANDB_ENTITY")
fi

if [[ "$PACKING_SAMPLES" == "1" ]]; then
  cmd+=(--ds.packing_samples)
fi

if [[ -n "$LOG_FILE" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  nohup "${cmd[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
  echo $!
else
  "${cmd[@]}"
fi
