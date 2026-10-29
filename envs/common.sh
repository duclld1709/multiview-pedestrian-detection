# Cấu hình dùng chung cho setup_env.sh, run.sh, envs/lock.sh.
# Đổi phiên bản Python/torch ở đây rồi chạy lại envs/lock.sh.
PYTHON_VERSION=3.11
TORCH_VERSION=2.7.1
TORCHVISION_VERSION=0.22.1
MODELS=(mvdet mvdetr shot earlybird tracktacular tools)

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Nơi đặt venv: MVPD_ENV_ROOT > Kaggle (/tmp, không bị lưu vào output) > original/.envs
if [ -n "${MVPD_ENV_ROOT:-}" ]; then
    ENV_ROOT=$MVPD_ENV_ROOT
elif [ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]; then
    ENV_ROOT=/tmp/mvpd_envs
else
    ENV_ROOT=$ROOT/.envs
fi

die() { echo "[mvpd] LỖI: $*" >&2; exit 1; }
log() { echo "[mvpd] $*"; }

model_dir() {
    case $1 in
        mvdet) echo models/MVDet ;;
        mvdetr) echo models/MVDeTr ;;
        shot) echo models/shot ;;
        earlybird) echo models/EarlyBird/EarlyBird ;;
        tracktacular) echo models/TrackTacular/WorldTrack ;;
        tools) echo . ;;
        *) return 1 ;;
    esac
}

model_entry() {
    case $1 in
        tracktacular) echo world_track.py ;;
        tools) echo "" ;;
        *) echo main.py ;;
    esac
}

venv_python() {
    if [ -x "$ENV_ROOT/$1/Scripts/python.exe" ]; then
        echo "$ENV_ROOT/$1/Scripts/python.exe"
    else
        echo "$ENV_ROOT/$1/bin/python"
    fi
}
