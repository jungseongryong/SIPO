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
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/luffy_sft_lora}"
WANDB_PROJECT="${WANDB_PROJECT:-luffy-sft-lora}"
WANDB_KEY="${WANDB_KEY:-}"
ZERO_STAGE="${ZERO_STAGE:-2}"
NUM_GPUS="${NUM_GPUS:-8}"
ATTN_IMPL="${ATTN_IMPL:-eager}"
PACKING_SAMPLES="${PACKING_SAMPLES:-0}"
LOG_FILE="${LOG_FILE:-}"

LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-3}"
MAX_SAMPLES="${MAX_SAMPLES:-500000}"
MAX_LEN="${MAX_LEN:-16384}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:--1}"

mkdir -p "$OUTPUT_DIR"

python - <<'PY'
import importlib.util
import sys

missing = [name for name in ("openrlhf", "deepspeed") if importlib.util.find_spec(name) is None]
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
  --ds.zero_stage "$ZERO_STAGE"
  --ds.param_dtype bf16
  --ds.attn_implementation "$ATTN_IMPL"
  --ds.lora.rank "$LORA_RANK"
  --ds.lora.alpha "$LORA_ALPHA"
  --model.gradient_checkpointing_enable
  --ckpt.output_dir "$OUTPUT_DIR"
  --ckpt.save_steps "$SAVE_STEPS"
  --logger.logging_steps "$LOGGING_STEPS"
  --eval.steps "$EVAL_STEPS"
)

if [[ "$PACKING_SAMPLES" == "1" ]]; then
  cmd+=(--ds.packing_samples)
fi

if [[ -n "$WANDB_KEY" ]]; then
  cmd+=(--logger.wandb.key "$WANDB_KEY")
fi

if [[ -n "$LOG_FILE" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  nohup "${cmd[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
  echo $!
else
  "${cmd[@]}"
fi
