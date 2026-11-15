# Chạy Wave Tool trên máy khác

1. Cài Python 3.10 trở lên, bật Add Python to PATH và giữ thành phần Tkinter.
2. Giải nén ZIP vào thư mục có quyền ghi. Không chạy bên trong ZIP.
3. Nhấp đúp SETUP.bat để tạo môi trường .venv và cài thư viện. Lần đầu cần Internet.
4. Nhấp đúp RUN_WAVE_TOOL.bat để mở tool. Những lần sau chỉ cần chạy file này.

## Chọn dữ liệu

Dataset được gửi riêng. Trong hộp thoại khởi động, chọn:

- Images: thư mục gốc dataset hoặc Image_subsets chứa C1, C2, ...
- Calibration: thư mục chứa intrinsic_zero/ và extrinsic/. WaveTrack cần thêm grid.json trong thư mục này; Wildtrack chính thức không cần.
- Annotations: thư mục JSON theo frame. Wildtrack thường dùng annotations_positions; WaveTrack dùng annotations hoặc thư mục nhãn xuất từ QA.

Ảnh phải được khử méo và đúng kích thước calibration. Wildtrack chính thức dùng bảy camera và ảnh 1920 x 1080.

Tool tự nhận diện Wildtrack từ intrinsic_zero/intr_CVLab1.xml. Wildtrack chỉ xem. Với WaveTrack, Lock / Enter lưu đè JSON của frame trong thư mục nhãn đã chọn; dùng bản sao nhãn nếu cần giữ bản gốc.

Tool lưu đường dẫn vào .settings.json. Xóa file này trước khi gửi lại cho người khác.

## Chạy bằng terminal

Nếu dùng môi trường Python riêng, mở terminal trong thư mục tool:

    python -m pip install -r requirements.txt
    python wave_tool.py

Ép chế độ Wildtrack khi cần:

    python wave_tool.py --dataset wildtrack --source "D:\Data\WILDTRACK"

Nhấn h trong ứng dụng để xem phím tắt, hoặc chạy python wave_tool.py --help. Chi tiết có trong README.md; khi đứng trong thư mục tool, rút đường dẫn wave_tool/wave_tool.py thành wave_tool.py.

Gói này chứa mã chạy, bản đồ BEV, requirements và hướng dẫn/script khởi động. Không chứa tests, script migration, cache, môi trường ảo, cài đặt cá nhân hay dataset. Cần cài Python và thư viện trước khi chạy.