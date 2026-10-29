#!/usr/bin/env bash
# Chạy model bằng venv của nó, từ đúng thư mục của model (đường dẫn tương đối tính theo thư mục đó).
#   bash run.sh mvdet -d wildtrack --data_path $WT --drop_ratio 60 --loss brl   # MVDet/main.py
#   bash run.sh shot main_sep.py ...                                            # script khác trong shot/
#   bash run.sh earlybird fit -c configs/t_fit.yml -c configs/d_wildtrack.yml   # EarlyBird/EarlyBird/main.py
#   bash run.sh tracktacular test -c lightning_logs/version_0/config.yaml ...   # TrackTacular/WorldTrack/world_track.py
#   bash run.sh tools data_tools/generate_pseudo_cache.py ...                   # chạy từ original/
set -euo pipefail
source "$(dirname "$0")/envs/common.sh"

m=${1:-}
dir=$(model_dir "$m") || die "cách dùng: bash run.sh <${MODELS[*]}> [script.py] [tham số...]"
shift
py=$(venv_python "$m")
[ -x "$py" ] || die "$m chưa có môi trường, chạy: bash setup_env.sh $m"

entry=$(model_entry "$m")
if [[ "${1:-}" == *.py ]]; then entry=$1; shift; fi
cd "$ROOT/$dir"
exec "$py" ${entry:+"$entry"} "$@"
