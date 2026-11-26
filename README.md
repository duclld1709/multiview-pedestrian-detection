# Hướng dẫn chạy thực nghiệm

5 mô hình: **MVDet**, **MVDeTr**, **SHOT**, **EarlyBird**, **TrackTacular**.
Các lệnh chạy từ thư mục gốc của repo. Mỗi khối ghi rõ Bash hoặc PowerShell.

## 0. Cài môi trường (Kaggle / server GPU thuê / PC có GPU)

Mỗi model có một venv riêng do [uv](https://docs.astral.sh/uv/) quản lý. Quy trình giống nhau trên mọi máy:

```bash
bash setup_env.sh mvdet          # hoặc nhiều model: mvdet mvdetr shot earlybird tracktacular tools | all
bash run.sh mvdet <tham số>      # chạy main.py của model bằng đúng venv đó (tự cd vào thư mục model)
```

| venv | Dùng cho |
|---|---|
| `mvdet`, `mvdetr`, `shot`, `earlybird`, `tracktacular` | train / test từng mô hình |
| `tools` | `data_tools/` (tạo nhãn drop, pseudo cache bằng YOLO) |

`setup_env.sh` tự làm:
- cài `uv` nếu chưa có, tạo venv với **Python 3.11 do uv tải về**, nên Python có sẵn trên máy là bản nào cũng được;
- đọc GPU + driver qua `nvidia-smi` rồi chọn torch 2.7.1 bản `cu128` (driver CUDA ≥ 12.8; bắt buộc với GPU Blackwell như
  RTX 50xx/B200), `cu126` (driver ≥ 12.6, kể cả P100/V100) hoặc `cu118` (driver cũ hơn);
- cài các gói còn lại theo đúng version trong `envs/lock/<model>.txt`, nên mọi máy ra cùng một môi trường;
- MVDeTr: build CUDA op `MultiScaleDeformableAttention` (cần `nvcc` cùng major CUDA với torch). Nếu build lỗi, script in
  cảnh báo và MVDeTr tự dùng MSDeformAttn bản PyTorch (cùng phép tính, chậm hơn);
- chạy `envs/check_env.py`: torch thấy GPU, có kernel cho kiến trúc GPU đó, entrypoint của model import đủ thư viện;
  với MVDeTr còn so CUDA op với bản PyTorch.

| Biến | Tác dụng |
|---|---|
| `TORCH_CUDA=cu126` | ép bản wheel torch thay vì tự chọn |
| `MVPD_ENV_ROOT=/path` | nơi đặt venv (mặc định `original/.envs`; trên Kaggle là `/tmp/mvpd_envs`) |
| `MVPD_FRESH=1` | xoá venv cũ, cài lại từ đầu |

**Kaggle**: bật Internet + GPU, trong notebook chạy `!cd original && bash setup_env.sh mvdet` rồi
`!cd original && bash run.sh mvdet ...`. `/kaggle/input` là read-only mà bước 2 cần ghi vào thư mục dataset, nên copy
dataset sang `/kaggle/working/original/Data/` trước.

**Server GPU thuê**: nên chọn image loại `devel` (ví dụ `nvidia/cuda:12.x-devel`, hoặc image PyTorch có `devel`) để có
`nvcc` build CUDA op cho MVDeTr.

**Sửa dependency**: sửa `envs/<model>.txt` → `bash envs/lock.sh <model>` → lưu cả `envs/lock/`.
Phiên bản Python/torch dùng chung khai báo ở `envs/common.sh`. Gói cài `--no-deps` (nuscenes-devkit) nằm ở
`envs/<model>.nodeps.txt`.

## 1. Đặt dataset

```
original/
├── Data/
│   ├── Wildtrack/            # tên thư mục phải chứa "wildtrack" (EarlyBird/TrackTacular dựa vào tên này)
│   │   ├── Image_subsets/
│   │   ├── annotations_positions/
│   │   └── calibrations/
│   ├── MultiviewX/           # (tuỳ chọn) cấu trúc tương tự
│   └── WaveTrack/            # dùng với MVDet và MVDeTr
├── data_tools/
└── models/
    └── MVDet/  MVDeTr/  shot/  EarlyBird/  TrackTacular/
```

### WaveTrack full label trên Windows

WaveTrack dùng 5 camera thật và lưới BEV 241 × 313 với bước 2,5 cm. Repo chia 90/10 theo
thứ tự ID frame. Với bản 870 frame, train dùng 783 frame đầu và test dùng 87 frame cuối.
MVDet dùng MSE, MVDeTr dùng focal loss. Tải ZIP gồm `Image_subsets`,
`annotations_positions` và `calibrations` với `grid.json`, `intrinsic_zero` và `extrinsic`.

Trong PowerShell, điền link chia sẻ của file ZIP rồi chạy từ thư mục repo.
Nếu đã đặt dataset trong `Data\WaveTrack`, chạy thẳng khối train.

```powershell
$driveUrl = 'DAN_LINK_ZIP_WAVETRACK'
$downloads = Join-Path $PWD.Path 'Data\_downloads'
$wave = Join-Path $PWD.Path 'Data\WaveTrack'
if (Test-Path -LiteralPath $wave) { throw "Đã có $wave" }
New-Item -ItemType Directory -Path $downloads -Force -ErrorAction Stop | Out-Null
$zip = Join-Path $downloads 'WaveTrack.zip'
uvx --python 3.11 gdown --continue --retries 3 $driveUrl -O $zip
if ($LASTEXITCODE -ne 0) { throw 'Tải WaveTrack thất bại' }
$staging = Join-Path $downloads ("WaveTrack-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $staging -ErrorAction Stop | Out-Null
tar.exe -xf $zip -C $staging
if ($LASTEXITCODE -ne 0) { throw 'Giải nén WaveTrack thất bại' }
$roots = @(Get-ChildItem -LiteralPath $staging -Directory -Recurse -Filter 'annotations_positions' |
    ForEach-Object { $_.Parent.FullName } |
    Where-Object {
        (Test-Path -LiteralPath (Join-Path $_ 'Image_subsets') -PathType Container) -and
        (Test-Path -LiteralPath (Join-Path $_ 'calibrations\grid.json') -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $_ 'calibrations\intrinsic_zero') -PathType Container) -and
        (Test-Path -LiteralPath (Join-Path $_ 'calibrations\extrinsic') -PathType Container)
    })
if ($roots.Count -ne 1) { throw "Cần đúng một thư mục WaveTrack trong $staging" }
Move-Item -LiteralPath $roots[0] -Destination $wave -ErrorAction Stop
```

```powershell
$gitRoot = Split-Path (Split-Path (Get-Command git.exe -ErrorAction Stop).Source)
$gitBash = Join-Path $gitRoot 'bin\bash.exe'
$wave = Join-Path $PWD.Path 'Data\WaveTrack'
$env:CUDA_VISIBLE_DEVICES = '0'

# Nếu chưa có môi trường, chạy: & $gitBash setup_env.sh mvdet mvdetr
& $gitBash run.sh mvdet -d wavetrack --data_path "$wave" --drop_ratio 0 --loss mse -j 0
& $gitBash run.sh mvdetr -d wavetrack --data_path "$wave" --drop_ratio 0 --loss focal -j 0
```

Thêm `--epochs 1` để kiểm tra trên GPU trước khi chạy mặc định 10 epoch. Hai model tự
đánh giá sau khi train. Log nằm ở `models/MVDet/logs/wavetrack_frame/full/mse/` và
`models/MVDeTr/logs/wavetrack/full/focal/`.

## 2. Chuẩn bị dữ liệu (chạy 1 lần)

Cần venv `tools`: `bash setup_env.sh tools`.

**a. Tạo nhãn bị drop** (partial supervision) → ghi vào `Data/Wildtrack/drop_annotations/drop_XX/`:

```bash
bash run.sh tools data_tools/simulate_dropped_annotations.py -d Wildtrack -r ./Data -s drop20 drop45 drop60 --seed 1
```

**a'. (Tuỳ chọn) Tạo nhãn drop theo frame** (dùng cho cụm 6). Frame train bị drop ngẫu nhiên 20/45/60% *số frame*;
frame còn lại giữ đủ nhãn → ghi vào `Data/Wildtrack/drop_annotations/frame_drop_XX/`:

```bash
bash run.sh tools data_tools/simulate_dropped_annotations.py -d Wildtrack -r ./Data -s drop20 drop45 drop60 --mode frame --seed 1
```

Frame bị drop không có file JSON trong `annotations_positions/`, nên loader bỏ hẳn chúng khỏi tập train (bản đầy đủ nằm
ở `hidden_annotations_positions/`, danh sách frame trong `stats.json` → `dropped_frame_ids`). Số frame giữ lại luôn
chẵn (sampler của EarlyBird cần), với Wildtrack/MultiviewX là 288 / 198 / 144 trên 360 frame train.

**b. Tạo pseudo cache** từ YOLO, dùng chung cho cả 5 mô hình:

```bash
bash run.sh tools data_tools/generate_pseudo_cache.py --dataset wildtrack --data_root ./Data/Wildtrack \
    --output ./Data/pseudo_cache/wildtrack_yolo26s.json --weights yolo26s.pt --device 0
```

## 3. Biến dùng chung

Khai báo một lần trong terminal (đứng ở `original/`) trước khi chạy các cụm bên dưới:

```bash
export CUDA_VISIBLE_DEVICES=0
WT=$PWD/Data/Wildtrack
CACHE=$PWD/Data/pseudo_cache/wildtrack_yolo26s.json
DROP=60          # mức drop nhãn: 20 / 45 / 60
```

`--drop_mode` (EarlyBird/TrackTacular: `--data.init_args.drop_mode`) chọn kiểu drop khi `drop_ratio > 0`:
`instance` (mặc định, cụm 2–5) đọc `drop_XX/`, `frame` (cụm 6) đọc `frame_drop_XX/`.

Ghi chú:
- **AABL** trong code có tên là `brl`. Loss gốc: MVDet/SHOT = `mse`, MVDeTr/EarlyBird/TrackTacular = `focal`.
- Ở chế độ `instance`, cụm 4 và 5 mặc định chỉ giảm phạt nền theo pseudo label. Peak loss mặc định tắt và `pseudo_loss_weight` bằng 0.
- Để bật lại peak loss với trọng số cũ, thêm `--pseudo_peak_loss --pseudo_loss_weight 0.01` (MVDet/MVDeTr/SHOT) hoặc `--model.pseudo_no_peak_loss false --model.pseudo_loss_weight 0.01` (EarlyBird/TrackTacular). Khi dùng pseudo labels cho frame bị drop, cũng cần truyền hai tham số này vì trọng số áp dụng cho toàn bộ pseudo BEV loss.
- MVDet, MVDeTr, SHOT **tự test sau khi train**; kết quả nằm trong `logs/.../log.txt` (trong thư mục của model).
- EarlyBird, TrackTacular cần **chạy lệnh test riêng** (mục 5).

## 4. Lệnh train theo cụm ablation

### Cụm 1 — Full supervision + loss gốc

```bash
bash run.sh mvdet  -d wildtrack --data_path $WT --drop_ratio 0 --loss mse
bash run.sh mvdetr -d wildtrack --data_path $WT --drop_ratio 0 --loss focal
bash run.sh shot   -d wildtrack --data_path $WT --drop_ratio 0 --loss mse
bash run.sh earlybird fit -c configs/t_fit.yml -c configs/d_wildtrack.yml \
    --data.init_args.data_dir $WT --data.init_args.drop_ratio 0 --model.heatmap_loss focal
bash run.sh tracktacular fit -c configs/t_fit.yml -c configs/d_wildtrack.yml -c configs/m_mvdet.yml \
    --data.init_args.data_dir $WT --data.init_args.drop_ratio 0 --model.heatmap_loss focal
```

### Cụm 2 — Partial supervision + loss gốc

```bash
bash run.sh mvdet  -d wildtrack --data_path $WT --drop_ratio $DROP --loss mse
bash run.sh mvdetr -d wildtrack --data_path $WT --drop_ratio $DROP --loss focal
bash run.sh shot   -d wildtrack --data_path $WT --drop_ratio $DROP --loss mse
bash run.sh earlybird fit -c configs/t_fit.yml -c configs/d_wildtrack.yml \
    --data.init_args.data_dir $WT --data.init_args.drop_ratio $DROP --model.heatmap_loss focal
bash run.sh tracktacular fit -c configs/t_fit.yml -c configs/d_wildtrack.yml -c configs/m_mvdet.yml \
    --data.init_args.data_dir $WT --data.init_args.drop_ratio $DROP --model.heatmap_loss focal
```

### Cụm 3 — Partial supervision + AABL

```bash
bash run.sh mvdet  -d wildtrack --data_path $WT --drop_ratio $DROP --loss brl
bash run.sh mvdetr -d wildtrack --data_path $WT --drop_ratio $DROP --loss brl
bash run.sh shot   -d wildtrack --data_path $WT --drop_ratio $DROP --loss brl
bash run.sh earlybird fit -c configs/t_fit.yml -c configs/d_wildtrack.yml \
    --data.init_args.data_dir $WT --data.init_args.drop_ratio $DROP --model.heatmap_loss brl
bash run.sh tracktacular fit -c configs/t_fit.yml -c configs/d_wildtrack.yml -c configs/m_mvdet.yml \
    --data.init_args.data_dir $WT --data.init_args.drop_ratio $DROP --model.heatmap_loss brl
```

### Cụm 4 — Partial supervision + pseudo (loss gốc)

```bash
bash run.sh mvdet  -d wildtrack --data_path $WT --drop_ratio $DROP --loss mse \
    --use_pseudo_labels --pseudo_cache $CACHE
bash run.sh mvdetr -d wildtrack --data_path $WT --drop_ratio $DROP --loss focal \
    --use_pseudo_labels --pseudo_cache $CACHE
bash run.sh shot   -d wildtrack --data_path $WT --drop_ratio $DROP --loss mse \
    --use_pseudo_labels --pseudo_cache $CACHE
bash run.sh earlybird fit -c configs/t_fit.yml -c configs/d_wildtrack.yml \
    --data.init_args.data_dir $WT --data.init_args.drop_ratio $DROP --model.heatmap_loss focal \
    --data.init_args.pseudo_cache $CACHE --model.use_pseudo_labels true
bash run.sh tracktacular fit -c configs/t_fit.yml -c configs/d_wildtrack.yml -c configs/m_mvdet.yml \
    --data.init_args.data_dir $WT --data.init_args.drop_ratio $DROP --model.heatmap_loss focal \
    --data.init_args.pseudo_cache $CACHE --model.use_pseudo_labels true
```

### Cụm 5 — Partial supervision + pseudo + AABL

```bash
bash run.sh mvdet  -d wildtrack --data_path $WT --drop_ratio $DROP --loss brl \
    --use_pseudo_labels --pseudo_cache $CACHE
bash run.sh mvdetr -d wildtrack --data_path $WT --drop_ratio $DROP --loss brl \
    --use_pseudo_labels --pseudo_cache $CACHE
bash run.sh shot   -d wildtrack --data_path $WT --drop_ratio $DROP --loss brl \
    --use_pseudo_labels --pseudo_cache $CACHE
bash run.sh earlybird fit -c configs/t_fit.yml -c configs/d_wildtrack.yml \
    --data.init_args.data_dir $WT --data.init_args.drop_ratio $DROP --model.heatmap_loss brl \
    --data.init_args.pseudo_cache $CACHE --model.use_pseudo_labels true
bash run.sh tracktacular fit -c configs/t_fit.yml -c configs/d_wildtrack.yml -c configs/m_mvdet.yml \
    --data.init_args.data_dir $WT --data.init_args.drop_ratio $DROP --model.heatmap_loss brl \
    --data.init_args.pseudo_cache $CACHE --model.use_pseudo_labels true
```

### Cụm 6 — Frame-drop + loss gốc

Cần chạy bước 2a' trước. Mỗi frame còn lại có đủ nhãn (không có người bị thiếu nhãn), chỉ có tập train nhỏ đi.

```bash
bash run.sh mvdet  -d wildtrack --data_path $WT --drop_ratio $DROP --drop_mode frame --loss mse
bash run.sh mvdetr -d wildtrack --data_path $WT --drop_ratio $DROP --drop_mode frame --loss focal
bash run.sh shot   -d wildtrack --data_path $WT --drop_ratio $DROP --drop_mode frame --loss mse
bash run.sh earlybird fit -c configs/t_fit.yml -c configs/d_wildtrack.yml \
    --data.init_args.data_dir $WT --data.init_args.drop_ratio $DROP --data.init_args.drop_mode frame \
    --model.heatmap_loss focal
bash run.sh tracktacular fit -c configs/t_fit.yml -c configs/d_wildtrack.yml -c configs/m_mvdet.yml \
    --data.init_args.data_dir $WT --data.init_args.drop_ratio $DROP --data.init_args.drop_mode frame \
    --model.heatmap_loss focal
```

Ghi chú:
- Log của MVDet/MVDeTr/SHOT nằm ở `logs/.../frame_drop_XX/...`. EarlyBird/TrackTacular ghi `drop_mode: frame` trong
  `config.yaml`.
- Số iteration mỗi epoch giảm theo tỉ lệ drop (cùng số epoch thì model được train ít bước hơn).
- TrackTacular dùng frame liền trước làm temporal context. Nếu frame đó bị drop, model coi frame hiện tại như frame
  đầu chuỗi (`pre_frame = frame`).

## 5. Test EarlyBird và TrackTacular

Mỗi lần train tạo một thư mục `lightning_logs/version_N/` (N tăng dần, nằm trong `models/EarlyBird/EarlyBird/` hoặc
`models/TrackTacular/WorldTrack/`) chứa `config.yaml` và `checkpoints/last.ckpt`.
Luôn test **`last.ckpt`** (không dùng checkpoint có `val_loss` tốt nhất, vì tập val trùng tập test).
Đường dẫn bên dưới tính từ thư mục của model, vì `run.sh` tự `cd` vào đó.

```bash
N=0   # số version của lần train cần test

bash run.sh earlybird test \
    -c lightning_logs/version_$N/config.yaml \
    --ckpt_path lightning_logs/version_$N/checkpoints/last.ckpt

bash run.sh tracktacular test \
    -c lightning_logs/version_$N/config.yaml \
    --ckpt_path lightning_logs/version_$N/checkpoints/last.ckpt
```

Mẹo: ghi lại version ứng với từng cụm ngay sau khi train, hoặc mở `config.yaml` để xem `drop_ratio`, `heatmap_loss`, `use_pseudo_labels`.

## 6. Dùng MultiviewX

Thay `wildtrack` → `multiviewx` ở mọi chỗ:
- Bước 2: `-d MultiviewX`, `--dataset multiviewx`, `--data_root ./Data/MultiviewX`, cache `multiviewx_yolo26s.json`.
- Bước 3: `WT=$PWD/Data/MultiviewX`, `CACHE=$PWD/Data/pseudo_cache/multiviewx_yolo26s.json`.
- MVDet/MVDeTr/SHOT: `-d multiviewx`. EarlyBird/TrackTacular: `-c configs/d_multiviewx.yml`.
