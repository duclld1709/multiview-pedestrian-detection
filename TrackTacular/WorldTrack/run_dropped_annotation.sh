#!/usr/bin/env bash
# Run a sequence of TrackTacular experiments.
# Lists are space-separated and can be overridden through environment variables.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU="${GPU:-0}"
DATASETS="${DATASETS:-wildtrack}"
MODELS="${MODELS:-mvdet segnet liftnet}"
DROP_RATIOS="${DROP_RATIOS:-0 20 45 60}"
RUNS_ROOT="${RUNS_ROOT:-experiments}"
RUN_TRAIN="${RUN_TRAIN:-1}"
# By default, every completed training run is immediately evaluated with
# the best val_center checkpoint. Set RUN_TESTS=0 to train without testing.
RUN_TESTS="${RUN_TESTS:-1}"

WILDTRACK_DIR="${WILDTRACK_DIR:-/workspace/Wildtrack}"
MULTIVIEWX_DIR="${MULTIVIEWX_DIR:-/workspace/MultiviewX}"

read -r -a dataset_list <<< "$DATASETS"
read -r -a model_list <<< "$MODELS"
read -r -a drop_ratio_list <<< "$DROP_RATIOS"

if [[ "$RUN_TRAIN" != "0" && "$RUN_TRAIN" != "1" ]]; then
  echo "RUN_TRAIN must be 0 or 1, got: $RUN_TRAIN" >&2
  exit 2
fi
if [[ "$RUN_TESTS" != "0" && "$RUN_TESTS" != "1" ]]; then
  echo "RUN_TESTS must be 0 or 1, got: $RUN_TESTS" >&2
  exit 2
fi
if [[ "$RUN_TRAIN" == "0" && "$RUN_TESTS" == "0" ]]; then
  echo "Nothing to do: both RUN_TRAIN and RUN_TESTS are 0." >&2
  exit 2
fi

find_latest_training_log_dir() {
  local run_dir="$1"
  local candidate
  local latest=""
  local version_dirs=()

  shopt -s nullglob
  version_dirs=("$run_dir"/lightning_logs/version_*)
  shopt -u nullglob

  if [[ "${#version_dirs[@]}" -eq 0 ]]; then
    return 1
  fi

  while IFS= read -r candidate; do
    if compgen -G "$candidate/checkpoints/*.ckpt" > /dev/null; then
      latest="$candidate"
    fi
  done < <(printf '%s\n' "${version_dirs[@]}" | sort -V)

  if [[ -z "$latest" ]]; then
    return 1
  fi
  printf '%s\n' "$latest"
}

ensure_best_checkpoint() {
  local log_dir="$1"
  local checkpoint_dir="$log_dir/checkpoints"
  local destination="$checkpoint_dir/best.ckpt"
  local candidates=()
  local source
  local score

  if [[ -f "$destination" ]]; then
    printf '%s\n' "$destination"
    return 0
  fi

  shopt -s nullglob
  candidates=("$checkpoint_dir"/model-epoch=*-val_center=*.ckpt)
  shopt -u nullglob
  if [[ "${#candidates[@]}" -eq 0 ]]; then
    return 1
  fi

  # Filename format ends in val_center=<score>.ckpt; field 4 is the score.
  source="$(printf '%s\n' "${candidates[@]}" | sort -t= -k4,4n | head -n 1)"
  score="${source##*val_center=}"
  score="${score%.ckpt}"
  cp -f "$source" "$destination"
  printf 'source=%s\nval_center=%s\n' "$(basename "$source")" "$score" \
    > "$checkpoint_dir/best_checkpoint_source.txt"
  printf '%s\n' "$destination"
}

for dataset in "${dataset_list[@]}"; do
  case "$dataset" in
    wildtrack) data_dir="$WILDTRACK_DIR" ;;
    multiviewx) data_dir="$MULTIVIEWX_DIR" ;;
    *) echo "Unsupported dataset: $dataset" >&2; exit 2 ;;
  esac

  if [[ ! -d "$data_dir" ]]; then
    echo "Dataset directory not found: $data_dir" >&2
    exit 2
  fi

  for model in "${model_list[@]}"; do
    case "$model" in
      mvdet|segnet|liftnet|bevformer) ;;
      *) echo "Unsupported model: $model" >&2; exit 2 ;;
    esac

    for drop_ratio in "${drop_ratio_list[@]}"; do
      case "$drop_ratio" in
        0) annotation_dir="$data_dir/annotations_positions" ;;
        20|45|60) annotation_dir="$data_dir/drop_annotations/drop_${drop_ratio}/annotations_positions" ;;
        *) echo "Unsupported drop ratio: $drop_ratio" >&2; exit 2 ;;
      esac

      if [[ ! -d "$annotation_dir" ]]; then
        echo "Annotation directory not found: $annotation_dir" >&2
        exit 2
      fi

      run_dir="$RUNS_ROOT/$dataset/$model/drop_${drop_ratio}"
      echo "===== dataset=$dataset model=$model drop_ratio=$drop_ratio output=$run_dir ====="

      if [[ "$RUN_TRAIN" == "1" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" world_track.py fit \
          -c configs/t_fit.yml \
          -c "configs/d_${dataset}.yml" \
          -c "configs/m_${model}.yml" \
          --data.init_args.data_dir "$data_dir" \
          --data.init_args.drop_ratio "$drop_ratio" \
          --trainer.default_root_dir "$run_dir"
      fi

      if [[ "$RUN_TESTS" == "1" ]]; then
        if ! latest_log_dir="$(find_latest_training_log_dir "$run_dir")"; then
          echo "No training version with checkpoints found under: $run_dir/lightning_logs" >&2
          exit 2
        fi
        if ! checkpoint="$(ensure_best_checkpoint "$latest_log_dir")"; then
          echo "No val_center checkpoint found under: $latest_log_dir/checkpoints" >&2
          exit 2
        fi

        test_dir="$latest_log_dir/test"
        mkdir -p "$test_dir"

        echo "===== Testing version=$latest_log_dir checkpoint=$checkpoint ====="
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" world_track.py test \
          -c "$latest_log_dir/config.yaml" \
          --ckpt_path "$checkpoint" \
          --trainer.devices 1 \
          --trainer.default_root_dir "$test_dir" \
          2>&1 | tee "$test_dir/test_best.log"
      fi
    done
  done
done
