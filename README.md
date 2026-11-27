# MVPD — Realtime demo

## Chạy trên Windows hoặc Linux

Giải nén `MVPD_Demo.zip`, cài `uv` và mở terminal trong thư mục vừa giải nén. Trong bước setup, `uv` tạo môi trường riêng `.venv-mvpd` với Python 3.11 và chọn bản PyTorch theo GPU, driver và hệ điều hành. Setup chạy thử các phép toán inference để xác nhận backend đã cài. Nếu backend GPU được chọn tự động chạy thử thất bại, setup chuyển sang bản CPU. [Cài uv](https://docs.astral.sh/uv/getting-started/installation/) · [Cách uv chọn backend PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)

**1. Setup môi trường (Windows và Linux)**

```sh
uv run --no-project --python 3.11 python setup_demo.py
```

Thêm `--backend cpu` vào cuối lệnh setup nếu muốn dùng CPU.

**2. Kích hoạt môi trường**

**Windows (PowerShell)**

```powershell
.\.venv-mvpd\Scripts\Activate.ps1
```

**Linux**

```sh
. .venv-mvpd/bin/activate
```

**3. Tải và chuẩn bị dataset (nếu máy chưa có)**

Chạy tại thư mục demo. Lệnh sau tự cài `gdown` trong môi trường tạm của `uv`, tải [thư mục dataset trên Google Drive](https://drive.google.com/drive/folders/1k4qLz_JNe6sHdKfp_rqpPhl8zgZ1ENSf?usp=sharing) và chuẩn bị `./Dataset/`:

```sh
uv run --no-project --python 3.11 --with "gdown>=5.2,<6" python setup_dataset.py
```

Để chọn nơi lưu, thêm `--dest` (thư mục cha của `Dataset/`, mặc định là thư mục hiện tại `.`):

```powershell
uv run --no-project --python 3.11 --with "gdown>=5.2,<6" python setup_dataset.py --dest "D:\Duc_Data\AIP491"
```

```sh
uv run --no-project --python 3.11 --with "gdown>=5.2,<6" python setup_dataset.py --dest "/home/user/data"
```

Script dùng `gdown --folder --continue` để tải thư mục. Google Drive hiện chứa ba ZIP riêng (`WaveTrack.zip`, `Wildtrack.zip`, `MultiviewX.zip`), không cần tạo ZIP chung trên Drive. Script cũng hỗ trợ ZIP ngoài chứa các ZIP con: giải nén lần lượt các lớp và xóa từng ZIP **sau khi giải nén thành công**, rồi chuẩn hóa thành:

```text
<dest>/Dataset/
  WaveTrack/
  Wildtrack/
  MultiviewX/
```

Nếu cả ba dataset đã có calibration và dữ liệu camera, script bỏ qua tải. Dataset đã có được giữ nguyên; thư mục đích tồn tại nhưng chưa đầy đủ sẽ báo lỗi để tránh ghi đè. Nếu tải hoặc giải nén bị gián đoạn, chạy lại cùng lệnh; dữ liệu tải tạm nằm trong `<dest>/.mvpd-dataset-download/`. Cuối cùng script in đường dẫn tuyệt đối tới **Dataset** để nhập vào web. Ví dụ Windows phía trên cho kết quả `D:\Duc_Data\AIP491\Dataset`.

**4. Chạy server web**

```sh
python -m realtime_demo.server
```

Mở [http://127.0.0.1:8770](http://127.0.0.1:8770/) và nhập đường dẫn tuyệt đối tới thư mục **Dataset**, ví dụ `D:\Duc_Data\AIP491\Dataset` hoặc `/home/user/Dataset`. Web tự quét lại đường dẫn đã lưu mỗi khi mở trang. **Settings → Dataset → Scan** dùng để đổi thư mục hoặc quét lại. Menu mang tên dataset đang chạy trên thanh công cụ liệt kê các bộ đã phát hiện. Có thể đổi trong lúc inference đang chạy. Demo đợi lượt xử lý hiện tại hoàn tất, tải dataset mới và tiếp tục từ đầu dataset đó.

```text
Dataset/
  WaveTrack/
    calibrations/calibration.json
    cam1.mp4 … cam5.mp4
  Wildtrack/
    calibrations/intrinsic_zero/ · calibrations/extrinsic/
    Image_subsets/C1/ … Image_subsets/C7/
  MultiviewX/
    calibrations/intrinsic/ · calibrations/extrinsic/
    Image_subsets/C1/ … Image_subsets/C6/
```

Chỉ cần có một trong ba bộ. Với Wildtrack và MultiviewX, dùng **Direct images** để chạy bộ ảnh gốc. Để dùng **Direct MP4**, thêm đủ MP4 đã đồng bộ theo thứ tự camera với tên `cam1.mp4`, `cam2.mp4`, v.v. tại thư mục dataset hoặc `videos/`. Video dùng kích thước ảnh theo calibration và FPS bằng nhau. MP4 mặc định dùng hệ ảnh đã khử méo. Với WaveTrack raw, đặt đủ XML gốc trong `calibrations/intrinsic_original/` để demo khử méo tại chỗ. MP4 trong `videos_rectified/` luôn được đọc theo calibration đã khử méo. [Wildtrack](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) · [MultiviewX và MVDet](https://github.com/hou-yz/MVDet)

## Tương tác dashboard

- Double click vào một view để chỉ xem camera đó và BEV bên cạnh; double click lần nữa để trở về layout trước đó.
- Trong **Display**, chọn **Shape → Cylinders** để thay bbox bằng hình trụ cao **1,8 m**, bán kính **0,25 m**, đặt tại điểm BEV của mỗi người. Hình trụ dùng calibration của camera và màu của bbox. **People overlays** bật/tắt nhãn ở các view; **Occluded people** cho phép hiện cả người bị che khuất.
- Khi đã **Pause** và frame đang xem có kết quả inference, click vào view rồi nhấn **Tab** để lần lượt chọn từng người; **Shift+Tab** chọn ngược lại. Người được chọn hiện xanh lá trên BEV và bbox/hình trụ ở các view. Chạy tiếp, đổi frame hoặc đổi dataset sẽ xóa lựa chọn. Xem và chọn người khi dừng không chạy inference mới.

## Checkpoint

| Dataset | Tên file trong `checkpoints/` |
|---|---|
| WaveTrack | `wavetrack_best.pt` (có trong ZIP) |
| Wildtrack | `bevnet_wildtrack.pt` |
| MultiviewX | `bevnet_multiviewx.pt` |

Demo cũng tự nhận checkpoint `wavetrack_lite.pt`, `wildtrack_lite.pt`, `multiviewx_lite.pt` (hoặc `bevnet_<dataset>_lite.pt`). Khi đổi dataset, ưu tiên checkpoint riêng của bộ đó; nếu chưa có thì dùng checkpoint WaveTrack đang có và chiếu theo calibration của dataset được chọn.

Khi một bộ chưa có trọng số riêng, demo dùng trọng số WaveTrack và chiếu theo calibration của bộ đang xem. Kết quả suy luận chéo phụ thuộc mức độ khác biệt giữa hai bộ. Khi có checkpoint BEVNet riêng, đặt đúng tên trong `checkpoints/` rồi quét lại hoặc khởi động lại. Các checkpoint BEVNet cũ vẫn đọc được từ **Settings → Checkpoint**, kể cả các checkpoint đã huấn luyện thêm nhánh phụ. Demo chỉ chạy đầu ra BEV. Checkpoint concat đời cũ dùng số camera cố định theo trọng số.

## Tạo cache toàn bộ dataset

Tạm dừng inference rồi vào **Settings → FP16 cache**. Nhập đường dẫn thư mục, chọn FPS và bấm **Generate**. Demo lấy mẫu từ đầu đến hết bản ghi đồng bộ ở FPS đã chọn, dựa trên nguồn và kích thước đầu vào đang dùng. FPS có thể là số thập phân từ trên 0 đến FPS của nguồn. Phần ước tính dung lượng hiển thị kích thước tensor FP16, cộng thêm dung lượng JPEG xem trước khi ghi cache.

Theo dõi tiến trình ngay trong menu. Nếu bấm **Cancel**, bấm **Generate** lại với cùng thư mục, nguồn và FPS để tiếp tục. Khi hoàn tất, chọn **FP16 cache** ở **Video source** để phát cache vừa tạo. Nếu đang phát từ một cache, demo dùng bản vừa tạo cho lượt inference tiếp theo.

**Advanced → Input source → Prefetch frames during inference** áp dụng cho cả **FP16 cache**, **Direct MP4** và **Direct images**, ở cả **Realtime** và **Sequential**. Prefetch mặc định bật, đọc trước tối đa một batch vào RAM trong lúc inference batch hiện tại. Tạm dừng để bật/tắt; lựa chọn được giữ khi đổi nguồn trong phiên chạy. Không cần tạo lại cache. FPS phát vẫn phụ thuộc FPS đã chọn khi tạo cache.

**Benchmark** dùng đúng trạng thái prefetch đang chọn: bật thì đọc batch tiếp theo song song với inference/hậu xử lý, tắt thì đọc và xử lý tuần tự. Mỗi batch size có một lượt warm-up và ba lượt đo, tính cả thời gian chờ dữ liệu và chuyển lên GPU, không chờ theo nhịp FPS phát. Median/P95 là thời gian mỗi chu kỳ xử lý trong pipeline; khi bật prefetch, chúng không phải độ trễ toàn trình từ lúc bắt đầu đọc một batch. Kết quả hiển thị nguồn và trạng thái prefetch đã dùng.

Dùng **Scan** nếu muốn tìm cache có sẵn trong một thư mục. Nếu có nhiều cache tương thích, demo ưu tiên cache của nguồn chính và FPS cao nhất. Có thể nhập thư mục chứa trực tiếp `manifest.json` để chọn một cache cụ thể. Các thiết lập mặc định nằm tại `%LOCALAPPDATA%\BEVLab\realtime.json` trên Windows và `~/.config/bev-lab/realtime.json` trên Linux.

## Điều khiển

- **Realtime** phát theo thời gian thực và lấy frame mới nhất. **Sequential** đi qua từng frame và cho phép chọn batch 1/2/4/8.
- **Benchmark** đo thông lượng và độ trễ theo nguồn và camera đang chọn. Khi tạm dừng có thể tua, bật/tắt camera, đổi nguồn và đổi chế độ phát.
- **Scenarios** lưu các khoảng phát theo dataset trong `outputs/realtime_demo_scenarios.json`. Chọn một khoảng để giới hạn tua và suy luận trong khoảng đó.
- **Display** điều khiển bbox, ROI và lưới. Camera preview và bbox thuộc cùng frame inference. WaveTrack dùng sơ đồ và bố cục gốc, Wildtrack có bảy camera quanh bản đồ, MultiviewX có sáu camera bên cạnh bản đồ.

Để đọc bản ghi WaveTrack và cache cũ từ một checkout trước đây, dùng `python -m realtime_demo.server --session data/20261107 --prepare`.

## Đóng gói lại

Sau khi thêm checkpoint riêng vào `checkpoints/`, chạy `python -m realtime_demo.package` để cập nhật `dist/MVPD_Demo.zip`. ZIP chứa mã ứng dụng, hướng dẫn, script setup bằng uv và các checkpoint theo tên ở bảng trên. Dataset và cache được cung cấp bằng đường dẫn trên máy chạy.
