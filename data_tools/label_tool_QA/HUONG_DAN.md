# Chạy QA trên máy khác

## Cài đặt và mở tool (Windows)

1. Cài Python 3.11 64-bit, bật Add Python to PATH. Khuyến nghị dùng đúng bản 3.11 như môi trường QA hiện tại.
2. Giải nén ZIP vào thư mục có quyền ghi, không chạy trực tiếp bên trong ZIP.
3. Nhấp đúp SETUP.bat. Cần Internet để tải thư viện; script tạo .venv và cài PyTorch bản CPU cùng các thư viện QA.
4. Nhấp đúp RUN_QA.bat, giữ cửa sổ terminal mở.
5. Khi server đã chạy, mở http://127.0.0.1:8770 trong trình duyệt. Nhấn Ctrl+C tại terminal để dừng.

Những lần sau chỉ cần chạy RUN_QA.bat.

Máy có GPU NVIDIA và driver phù hợp CUDA 12.8 có thể cài bản GPU bằng terminal trong thư mục tool:

    .\SETUP.bat -Cuda

Nếu không dùng được CUDA, cài lại bằng SETUP.bat không có tham số -Cuda. Bản CPU vẫn chạy được, auto-label sẽ chậm hơn.

## Dữ liệu và thao tác

- Dataset, calibration và project nhãn được gửi riêng; gói này không chứa dữ liệu cá nhân hoặc project đang làm.
- Trong Setup của giao diện web, chọn thư mục dataset trên máy đích, calibration, camera, frame và thư mục project mới.
- Với calibration XML của WaveTrack, giữ grid.json cùng intrinsic_zero/ và extrinsic/. Bản đóng gói có sẵn bộ đọc XML, không cần thư mục label_tools bên ngoài.
- Dùng Import annotations để nhập nhãn từ wave_tool vào project QA riêng, rồi review và export. Nhãn nguồn không bị thay đổi bởi thao tác import.
- Project QA tự lưu các chỉnh sửa. Khi chuyển project từ máy khác, các đường dẫn dataset, calibration hoặc weight cũ cần được cập nhật theo máy đích; có thể import nhãn vào project mới nếu phù hợp.
- Giữ dữ liệu ảnh đúng biến thể raw/đã khử méo và calibration tương ứng. Xem README.md để biết đầy đủ thao tác và định dạng export.

## YOLO và model BEV

Gói đã kèm yolo26s.pt (khoảng 20 MB) và hiển thị weight này trong danh sách YOLO. Không cần tải lại weight này khi auto-label. Chọn không dùng model BEV để dùng đề xuất YOLO đa view, hoặc import nhãn có sẵn để review.

Gói gọn không kèm repository/checkpoint BEV MVDet, MVDeTr hoặc thư viện CUDA biên dịch cho MVDeTr. Nếu cần các model đó, bổ sung repository tương ứng vào label_tool_QA/MVDet hoặc label_tool_QA/MVDeTr, cùng checkpoint và các phụ thuộc tương thích. Chỉ chép file checkpoint là chưa đủ. Không có các thành phần này thì vẫn dùng được giao diện, import/review/export và YOLO đa view.

## Chạy bằng terminal

    .\.venv\Scripts\python.exe server.py
    .\.venv\Scripts\python.exe server.py --open "D:\QA_project\label_project.json"

Nếu cổng mặc định đang bận:

    .\RUN_QA.bat --port 8771

Sau đó mở http://127.0.0.1:8771. Tùy chọn --help hiển thị các tham số của server.

## Thành phần đã loại bỏ

Không kèm .venv, .wheels, cache, tests, projects, export, báo cáo, project cá nhân, script audit/chia tập dữ liệu hoặc giao diện demo không phục vụ QA. Giữ các module demo_multiview cần thiết cho calibration và inference; đây là phụ thuộc thực sự của QA.

Bản nguồn gốc không bị sửa. Trong bản đóng gói, bộ đọc XML được chuyển thành _xml_dataset.py, danh sách YOLO trỏ vào thư mục tool, và bỏ gợi ý đường dẫn dataset của máy cũ. requirements.txt chỉ giữ thư viện chạy; setup hỗ trợ CPU mặc định hoặc CUDA tùy chọn.