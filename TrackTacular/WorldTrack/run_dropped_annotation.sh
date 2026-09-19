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
RUN_TESTS="${RUN_TESTS:-0}"

WILDTRACK_DIR="${WILDTRACK_DIR:-/workspace/Wildtrack}"
MULTIVIEWX_DIR="${MULTIVIEWX_DIR:-/workspace/MultiviewX}"

read -r -a dataset_list <<< "$DATASETS"
read -r -a model_list <<< "$MODELS"
read -r -a drop_ratio_list <<< "$DROP_RATIOS"

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

      CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" world_track.py fit \
        -c configs/t_fit.yml \
        -c "configs/d_${dataset}.yml" \
        -c "configs/m_${model}.yml" \
        --data.init_args.data_dir "$data_dir" \
        --data.init_args.drop_ratio "$drop_ratio" \
        --trainer.default_root_dir "$run_dir"

      if [[ "$RUN_TESTS" == "1" ]]; then
        log_dirs=("$run_dir"/lightning_logs/version_*)
        latest_log_dir="$(printf '%s\n' "${log_dirs[@]}" | sort -V | tail -n 1)"
        checkpoint="$latest_log_dir/checkpoints/last.ckpt"

        if [[ ! -f "$checkpoint" ]]; then
          echo "Checkpoint not found after training: $checkpoint" >&2
          exit 2
        fi

        echo "===== Testing checkpoint=$checkpoint ====="
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" world_track.py test \
          -c "$latest_log_dir/config.yaml" \
          --ckpt_path "$checkpoint" \
          --trainer.devices 1 \
          --trainer.default_root_dir "$run_dir/test"
      fi
    done
  done
done
