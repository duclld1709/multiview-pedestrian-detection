#!/usr/bin/env bash
# Cài venv cho từng model (mỗi model 1 venv riêng), dùng được trên Kaggle, server GPU thuê, PC có GPU.
#   bash setup_env.sh mvdet                 # 1 model
#   bash setup_env.sh mvdet shot earlybird  # nhiều model
#   bash setup_env.sh all                   # mvdet mvdetr shot earlybird tracktacular tools
# uv tự tải Python (envs/common.sh) nên không phụ thuộc Python có sẵn trên máy.
# torch được chọn theo driver + GPU; các gói còn lại cài đúng version trong envs/lock/<model>.txt.
#
# Biến tuỳ chọn:
#   TORCH_CUDA=cu118|cu126|cu128   ép bản wheel torch thay vì tự chọn
#   MVPD_ENV_ROOT=<dir>            nơi đặt venv (mặc định original/.envs, Kaggle: /tmp/mvpd_envs)
#   MVPD_FRESH=1                   xoá venv cũ và cài lại từ đầu
set -euo pipefail
source "$(dirname "$0")/envs/common.sh"

[ $# -gt 0 ] || die "cần tên model: ${MODELS[*]} | all"
if [ "$1" = all ]; then set -- "${MODELS[@]}"; fi
for m in "$@"; do model_dir "$m" >/dev/null || die "model không hợp lệ: $m (chọn: ${MODELS[*]})"; done

# ---------- uv ----------
if command -v uv >/dev/null 2>&1; then
    UV=(uv)
else
    log "cài uv"
    SYS_PY=$(command -v python3 || command -v python || true)
    if [ -n "$SYS_PY" ] && "$SYS_PY" -m pip install -q uv 2>/dev/null; then
        UV=("$SYS_PY" -m uv)
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        UV=(uv)
    fi
fi
export UV_PYTHON_PREFERENCE=only-managed  # cùng 1 bản Python trên mọi máy

# ---------- GPU -> bản CUDA của torch ----------
command -v nvidia-smi >/dev/null 2>&1 || die "không thấy nvidia-smi: cần GPU NVIDIA và driver"
ver2int() { awk -F. '{printf "%d", $1 * 100 + $2}' <<<"$1"; }
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | paste -sd ',' -)
GPU_CCS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d ' \r' | sort -t. -k1,1n -k2,2n -u)
DRV_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -1)
[ -n "$GPU_CCS" ] && [ -n "$DRV_CUDA" ] || die "không đọc được compute capability / CUDA version từ nvidia-smi"
CC_MIN=$(ver2int "$(head -1 <<<"$GPU_CCS")")
CC_MAX=$(ver2int "$(tail -1 <<<"$GPU_CCS")")
DRV=$(ver2int "$DRV_CUDA")
ARCH_LIST=$(paste -sd ';' - <<<"$GPU_CCS")

if [ -z "${TORCH_CUDA:-}" ]; then
    if (( DRV >= 1208 && CC_MIN >= 705 )); then TORCH_CUDA=cu128
    elif (( DRV >= 1206 )); then TORCH_CUDA=cu126
    else TORCH_CUDA=cu118
    fi
fi
(( DRV >= 1100 )) || die "driver quá cũ (CUDA $DRV_CUDA), cần >= 11.x"
if (( CC_MAX >= 1000 )) && [ "$TORCH_CUDA" != cu128 ]; then
    die "GPU Blackwell ($GPU_NAMES) cần torch cu128, driver phải hỗ trợ CUDA >= 12.8 (đang là $DRV_CUDA)"
fi
log "GPU: $GPU_NAMES | compute capability: $ARCH_LIST | driver CUDA $DRV_CUDA -> torch $TORCH_VERSION+$TORCH_CUDA"

# ---------- MVDeTr: build CUDA op MultiScaleDeformableAttention ----------
# Luôn build. Build lỗi thì MVDeTr tự dùng MSDeformAttn bản PyTorch (cùng phép tính, chậm hơn) và in cảnh báo.
msda_fallback() {
    log "=================================================================="
    log "CẢNH BÁO: KHÔNG build được CUDA op MultiScaleDeformableAttention: $1"
    log "MVDeTr vẫn chạy được nhưng dùng MSDeformAttn bản PyTorch (chậm hơn)."
    log "Cần nvcc cùng major CUDA với torch ($TORCH_CUDA), ví dụ image nvidia/cuda:12.x-devel,"
    log "rồi chạy lại: bash setup_env.sh mvdetr"
    log "=================================================================="
}

build_msda() {
    local py=$1 ops=$ROOT/models/MVDeTr/multiview_detector/models/ops nvcc nvcc_major torch_major=${TORCH_CUDA:2:2} tmp
    "${UV[@]}" pip uninstall -q --python "$py" MultiScaleDeformableAttention 2>/dev/null || true
    nvcc=$(command -v nvcc || true)
    [ -n "$nvcc" ] || nvcc=$(ls "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" 2>/dev/null || true)
    if [ -z "$nvcc" ]; then msda_fallback "không tìm thấy nvcc"; return; fi
    nvcc_major=$("$nvcc" --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
    if [ "$nvcc_major" != "$torch_major" ]; then
        msda_fallback "nvcc CUDA $nvcc_major khác torch CUDA $torch_major"; return
    fi
    log "build MultiScaleDeformableAttention bằng $nvcc (arch $ARCH_LIST)"
    # build trên bản copy để không sinh build/ và *.egg-info trong thư mục MVDeTr
    tmp=$(mktemp -d)
    cp -r "$ops/." "$tmp/"
    if ! CUDA_HOME=$(dirname "$(dirname "$nvcc")") TORCH_CUDA_ARCH_LIST="$ARCH_LIST" \
        "${UV[@]}" pip install -q --python "$py" --no-build-isolation --no-cache "$tmp"; then
        msda_fallback "build lỗi (xem log ở trên)"
    fi
    rm -rf "$tmp"
}

# ---------- cài từng model ----------
for m in "$@"; do
    venv=$ENV_ROOT/$m
    log "===== $m -> $venv"
    if [ "${MVPD_FRESH:-0}" = 1 ]; then rm -rf "$venv"; fi
    py=$(venv_python "$m")
    [ -x "$py" ] || "${UV[@]}" venv -q --python "$PYTHON_VERSION" "$venv"
    py=$(venv_python "$m")

    printf 'torch==%s\ntorchvision==%s\n' "$TORCH_VERSION" "$TORCHVISION_VERSION" > "$venv/constraints.txt"
    "${UV[@]}" pip install -q --python "$py" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
        --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"

    req=$ROOT/envs/lock/$m.txt
    [ -f "$req" ] || { log "chưa có lock, dùng envs/$m.txt"; req=$ROOT/envs/$m.txt; }
    "${UV[@]}" pip install -q --python "$py" -r "$req" -c "$venv/constraints.txt"
    if [ -f "$ROOT/envs/$m.nodeps.txt" ]; then
        "${UV[@]}" pip install -q --python "$py" --no-deps -r "$ROOT/envs/$m.nodeps.txt"
    fi
    if [ "$m" = mvdetr ]; then build_msda "$py"; fi

    "$py" "$ROOT/envs/check_env.py" "$m" || die "$m: kiểm tra môi trường thất bại"
done
log "xong. Chạy: bash run.sh <model> <tham số>"
