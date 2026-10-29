#!/usr/bin/env bash
# Sinh lại envs/lock/<model>.txt từ envs/<model>.txt (chạy khi sửa dependency, rồi commit lock).
# Lock cho Linux x86_64 + Python 3.11 + torch 2.7.1; torch/torchvision và các gói CUDA của chúng
# không ghi vào lock vì setup_env.sh cài theo GPU từng máy.
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh
mkdir -p lock
tmp=$(mktemp)
printf 'torch==%s\ntorchvision==%s\n' "$TORCH_VERSION" "$TORCHVISION_VERSION" > "$tmp"
models=("$@"); [ ${#models[@]} -eq 0 ] && models=(mvdet mvdetr shot earlybird tracktacular tools)
for m in "${models[@]}"; do
    echo "[lock] $m"
    uv pip compile "$m.txt" "$tmp" -q -o "lock/$m.txt" \
        --python-version "$PYTHON_VERSION" --python-platform x86_64-manylinux_2_28 \
        --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match \
        --no-emit-package torch --no-emit-package torchvision --no-emit-package triton \
        --no-header --no-annotate
    # các gói nvidia-* là dependency của torch, đi theo bản CUDA của từng máy
    sed -i '/^nvidia-/d' "lock/$m.txt"
done
rm -f "$tmp"
