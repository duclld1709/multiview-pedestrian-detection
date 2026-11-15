# ChArUco Calibrator — hướng dẫn sử dụng

Ứng dụng có hai chế độ (nút ở đầu cửa sổ): **Intrinsic** — thông số trong của từng camera bằng bảng ChArUco; **Extrinsic** — vị trí các camera trên mặt sàn lát gạch.

Công cụ có giao diện đồ họa để hiệu chỉnh nội tại camera (ma trận K và hệ số méo) bằng bảng ChArUco, chạy trên Windows.

## Cài đặt (một lần)

1. Cài **Python 3.10 – 3.13** từ https://www.python.org/downloads/windows/
   Khi cài, nhớ tick **“Add python.exe to PATH”**.
2. Giải nén thư mục `charuco_tool` vào nơi bất kỳ (đường dẫn có tiếng Việt cũng được).
3. Nháy đúp **`run.bat`**. Lần đầu sẽ mất vài phút để tải thư viện (cần mạng), các lần sau mở ngay.

Nếu chương trình không mở, chạy `run_debug.bat` để xem thông báo lỗi.

**Muốn có file .exe để mang sang máy khác không cài Python:** chạy `run.bat` một lần, sau đó chạy `build_exe.bat`. Kết quả nằm ở `dist\ChArUcoCalibrator\` — nén cả thư mục này lại để mang đi.

## Quy trình 5 bước (5 tab)

### 1. Bảng hiệu chỉnh
- Chọn số ô, cạnh ô, cạnh marker, từ điển, khổ giấy. Bấm **Tối đa hóa cạnh ô theo khổ giấy** để dùng hết tờ giấy.
- **Xuất PDF để in…** → mang ra tiệm, yêu cầu in **100% / Actual size**, không “Fit to page”.
- Đo vạch kiểm tra dưới bảng phải đúng **100 mm**.
- Dán tờ in **thật phẳng** lên tấm cứng (kính, mica, gỗ ép), phủ keo kín mặt sau.
- **Đo lại cạnh ô và cạnh marker bằng thước, nhập số đo thực tế** vào tab này trước khi hiệu chỉnh.
- Cùng thư mục với PDF có file `.board.json` lưu thông số bảng — dùng **Mở cấu hình bảng…** để nạp lại lần sau.

### 2. Ảnh
- **Thêm ảnh…**, **Thêm thư mục…** hoặc kéo-thả ảnh vào cửa sổ.
- Bảng liệt kê từng ảnh: số góc nhận được, **độ nét** (đo trên vùng bảng, dưới 60 là mờ), diện tích bảng chiếm, góc ngả, sai số, trạng thái.
- Bấm một dòng để xem điểm nhận diện trên ảnh.
- **Bỏ chọn ảnh xấu** bỏ tích tất cả ảnh mờ / thiếu góc. Có thể tự tích/bỏ tích ở cột “Dùng”.

### Nhập từ video (nút **Thêm video…**)
Thay vì chụp từng ảnh, có thể quay **một video 1–3 phút** cầm bảng đi khắp khung hình và ngả nhiều góc:
- Quay ở **đúng chế độ sẽ dùng cho dataset** (cùng độ phân giải, fps, khóa nét, tắt chống rung EIS).
- **Di chuyển chậm và dừng khoảng 1 giây ở mỗi tư thế.** Khung hình lúc bảng đang di chuyển bị nhòe và bị
  méo do rolling shutter của điện thoại, tool sẽ tự loại.
- Lộ trình gợi ý: chia khung hình thành 9 vùng; ở mỗi vùng giữ thẳng 1 giây, rồi ngả mép trên ra sau,
  mép dưới ra sau, mép trái ra xa, mép phải ra xa — mỗi tư thế dừng 1 giây.

Tool quét video (~6 khung/giây), loại khung không thấy bảng, mờ, hoặc bảng đang di chuyển, rồi chọn
**khoảng 40 khung khác nhau nhiều nhất** về vị trí, kích thước và góc ngả. Khung được lưu thành ảnh JPG
vào thư mục `<tên video>_frames` cạnh file video và tự thêm vào danh sách ảnh.

Cửa sổ kết quả có **biểu đồ dòng thời gian** (tốc độ bảng theo thời gian). Nếu quá ít khung được chọn:
- tay rung nhiều → tăng **Ngưỡng đứng yên** (ví dụ 1.5–2 px/khung);
- quay lại video, dừng lâu hơn ở mỗi tư thế.

### 3. Độ phủ
- **Bản đồ 3×3**: số góc bàn cờ rơi vào mỗi vùng của khung hình. Vùng ghi “thiếu” cần chụp thêm — đặc biệt 4 góc.
- **Biểu đồ góc ngả**: mỗi chấm là một ảnh. Mục tiêu: chấm tỏa ra mọi hướng, phần lớn nằm giữa vòng 12° và vòng 40°.
- **Gợi ý chụp thêm**: tool nói cụ thể cần thêm ảnh ở vùng nào, ngả theo hướng nào.

### 4. Hiệu chỉnh
- Mô hình khuyên dùng: **Pinhole k3 = 0** (điện thoại, webcam). “Rational” cho ống kính méo mạnh, “Fisheye” cho ống kính mắt cá.
- Bấm **Chạy hiệu chỉnh**. Kết quả gồm:
  - **Kết luận**: ĐẠT / ĐẠT có lưu ý / CẦN CHỤP THÊM / KHÔNG ĐẠT.
  - **Thông số** fx, fy, cx, cy, hệ số méo kèm **± σ** (độ bất định).
  - **Các mục kiểm tra**: RMS, số ảnh, ảnh mờ, độ phủ, góc ngả, fx ≈ fy, tâm ảnh, mô hình méo hợp lệ, **kiểm tra chéo** (hiệu chỉnh lại trên nhiều tập con xem K có ổn định không), **sai số trên ảnh để riêng**.
  - Biểu đồ sai số từng ảnh, vector sai số, bản đồ méo ống kính.
- **Xuất**: `.yaml` (đọc được bằng `cv2.FileStorage`), `.json`, `camera_info` cho ROS, **báo cáo HTML** đầy đủ.

> RMS thấp **chưa đủ** để kết luận. Ví dụ: RMS 0.2 px nhưng toàn bộ ảnh nằm ở giữa khung hình → tool sẽ báo KHÔNG ĐẠT vì thiếu dữ liệu ở góc ảnh và K không ổn định khi kiểm tra chéo.

### 5. Khử méo & kiểm tra
- Dùng kết quả vừa hiệu chỉnh hoặc **mở file .yaml/.json** đã lưu.
- **Mở ảnh…** để xem ảnh gốc và ảnh đã khử méo; thanh trượt chọn cắt viền đen hay giữ toàn bộ điểm ảnh. Lưu ảnh đã khử méo.
- **Kiểm tra trên ảnh mới…**: chọn một ảnh bảng **không dùng để hiệu chỉnh** → tool đo sai số chiếu lại, khoảng cách tới bảng, vẽ trục toạ độ. Sai số gần bằng RMS lúc hiệu chỉnh = kết quả tốt.

## Cách chụp ảnh hiệu chỉnh

1. **Khóa lấy nét** (chế độ Pro / Manual) ở khoảng cách sẽ dùng thật, giữ nguyên suốt quá trình. Không zoom, không đổi ống kính, không đổi độ phân giải. Tắt chống rung điện tử (EIS / Super steady) nếu có.
2. Giữ bảng trong **vùng nét** — không mang bảng lại gần hơn khoảng cách đã lấy nét. Nếu bảng quá nhỏ trong ảnh, **in khổ lớn hơn** thay vì đến gần.
3. Đưa bảng đến **khắp 9 vùng** của khung hình, đặc biệt **4 góc**.
4. Ở mỗi vùng, **ngả bảng ra khỏi mặt phẳng** khoảng 20–40°: mép trên ra sau, mép dưới ra sau, mép trái ra xa, mép phải ra xa. (Xoay bảng như kim đồng hồ không thay thế được.)
5. Đủ sáng để ảnh không nhòe; đặt máy lên giá và di chuyển bảng là tốt nhất.
6. 20–30 ảnh, hoặc một video (xem mục *Nhập từ video*). Ảnh/khung phải cùng **chế độ sẽ dùng thật** — nếu dataset là video thì hiệu chỉnh bằng video ở đúng chế độ đó; không trộn ảnh chụp tĩnh với khung video.

## Dùng kết quả trong code

```python
import cv2
fs = cv2.FileStorage("calib.yaml", cv2.FILE_STORAGE_READ)
K = fs.getNode("camera_matrix").mat()
dist = fs.getNode("dist_coeffs").mat()
fs.release()
img = cv2.imread("anh.jpg")
und = cv2.undistort(img, K, dist)
```

Kết quả chỉ đúng cho **cùng camera, cùng độ phân giải, cùng chế độ chụp và cùng trạng thái lấy nét** như lúc chụp ảnh hiệu chỉnh.

## Các file trong thư mục

| File | Vai trò |
|---|---|
| `run.bat` | Cài đặt lần đầu và mở chương trình |
| `run_debug.bat` | Mở chương trình kèm cửa sổ hiển thị lỗi |
| `build_exe.bat` | Đóng gói thành file .exe |
| `app.py` | Điểm khởi chạy |
| `gui.py` | Giao diện |
| `core.py` | Xử lý: bảng, PDF, nhận diện, hiệu chỉnh, đánh giá (dùng riêng trong code được) |
| `report.py` | Biểu đồ và báo cáo HTML |
| `video.py` | Quét video và chọn khung hình hiệu chỉnh |
| `extrinsic_core.py` | Extrinsic: marker sàn, lưới gạch, giải tư thế, độ phủ, ảnh ghép, xuất |
| `extrinsic_gui.py`, `extrinsic_viz.py` | Giao diện và hiển thị / báo cáo của chế độ Extrinsic |

---

# Chế độ Extrinsic — vị trí các camera trên mặt sàn

Bấm nút **“Extrinsic”** ở đầu cửa sổ. Mục tiêu: đưa mọi camera vào **một hệ toạ độ chung trên mặt sàn**, để một điểm trên sàn
cho cùng toạ độ (cm) ở mọi camera — yêu cầu cốt lõi của multiview pedestrian detection.

**Cần có trước:** mỗi camera một file intrinsic (từ chế độ Intrinsic, hoặc file XML có `camera_matrix` + `distortion_coefficients`).

## Hệ toạ độ thế giới
- Gốc (0, 0, 0) tại **một góc gạch** — dán băng keo màu để đánh dấu.
- X và Y chạy theo đường ron, Z hướng lên, mặt sàn Z = 0, đơn vị mét.
- **Đứng ở gốc nhìn theo +X thì +Y ở bên TRÁI.** Nếu đặt ngược, tool sẽ báo “camera nằm dưới mặt sàn”.
- Góc gạch (i, j) có toạ độ X = i × kích thước gạch X, Y = j × kích thước gạch Y.

## Chuẩn bị tại hiện trường
1. Gắn camera cố định lên giá, đúng vị trí và chế độ quay dataset (khóa nét, tắt EIS).
2. Đo **10 viên gạch liên tiếp** (tính cả ron) rồi chia 10 → kích thước 1 viên. Thông số này quyết định thang mét.
3. Tab **2. Hệ toạ độ**: thêm 4–6 marker, **Xuất PDF marker**, in 100%. Đặt mỗi marker sao cho **tâm** trùng một góc gạch và
   hai đường nét đứt trùng đường ron; ghi lại vị trí (i, j) và hướng mũi tên (+X / +Y / −X / −Y). Trải marker khắp vùng quay;
   mỗi camera nên thấy ≥ 2 marker. Sơ đồ bên phải tab giúp đối chiếu với sàn thật.
4. Quay vài giây khi cảnh trống (hoặc chụp ảnh) từ mỗi camera. Sau đó có thể cất marker và quay dataset — **không chạm vào camera**.

## Quy trình trong tool
1. **Tab 1 — Camera & dự án:** thêm camera, chọn file intrinsic, chọn ảnh tham chiếu hoặc **Lấy khung từ video**
   (có tùy chọn lấy trung vị nhiều khung để xóa người đi qua). Lưu dự án (.json) để mở lại sau.
   Hoặc **Chọn video (duyệt khung khi gán điểm)**: dùng cả video làm nguồn, chọn khung ngay ở tab 3.
2. **Tab 2 — Hệ toạ độ:** kích thước gạch, từ điển & cạnh marker (đo lại sau khi in), bảng vị trí marker.
3. **Tab 3 — Gán điểm & giải** (từng camera):
   - **Nguồn là video:** thanh khung dưới ảnh; **← →** chuyển 1 khung, **Shift + ← →** 10 khung, **Ctrl + ← →** 1 giây.
     Chọn khung thấy rõ mốc rồi click gán điểm như bình thường — mỗi điểm có thể click ở một khung khác nhau
     (camera đứng yên nên toạ độ pixel không đổi). Nhận diện marker / tìm lưới gạch / bắt điểm luôn chạy trên **khung đang xem**.
   - **① Tự nhận diện marker** → tự giải tư thế ban đầu.
   - **② Tìm lưới gạch** → tool dự đoán toàn bộ giao điểm ron trong tầm nhìn, tinh chỉnh từng điểm dưới 1 pixel
     (nắn thẳng vùng nhỏ quanh góc gạch về góc nhìn từ trên xuống rồi tìm đường ron), lặp vài vòng → thường hàng chục
     đến hàng trăm điểm. Điểm bị che / ron mờ tự bị bỏ; điểm lệch bất thường bị RANSAC loại.
   - Không có marker: bật **＋ Thêm điểm (click)**, click ≥ 4 góc gạch trải rộng (lăn chuột để phóng to), nhập (i, j)
     — tool tự bắt điểm vào giao điểm ron — rồi bấm **Giải**, sau đó **Tìm lưới gạch**.
   - Có thể thêm điểm có độ cao (góc khung cửa, cạnh cột) với toạ độ XYZ đã đo.
   - **Kiểm tra bằng mắt:** lưới trắng vẽ theo tư thế đã giải phải **trùng ron thật** ở cả vùng gần và xa.
   - Bảng kết quả: RMS (px), sai số trên sàn (cm), vị trí và độ cao camera ± độ bất định, các cảnh báo.
4. **Tab 4 — Kiểm tra đa camera:**
   - *Nhìn từ trên xuống:* vị trí, hướng camera, vùng sàn mỗi camera thấy, **bản đồ số camera thấy trọn một người** (chân + đầu 1.7 m).
     Dùng để bố trí camera: nên ≥ 2 camera trên toàn vùng quay.
   - *Ảnh ghép mặt sàn:* ảnh mọi camera chiếu xuống sàn rồi chồng lên — ron và marker phải trùng khít, không có đường đôi.
   - *Độ bất định trên sàn:* sai số vị trí (cm) tại mỗi chỗ trên sàn do sai số tư thế camera.
   - *Điểm kiểm tra chéo:* click cùng một điểm thật (chân người đứng yên, vết băng keo) ở từng camera → tool báo độ lệch
     giữa các camera (mục tiêu < 5–10 cm) và vẽ hình trụ cao 1.7 m tại đó trên mọi camera.
5. **Tab 5 — Vùng lưới & xuất:** đặt vùng quay và cỡ ô lưới (ví dụ 2.5 cm), xem kích thước lưới; xuất kết quả và báo cáo HTML.

## File xuất ra
| File | Nội dung |
|---|---|
| `intr_<camera>.xml` | `camera_matrix`, `distortion_coefficients` (định dạng OpenCV XML) |
| `extr_<camera>.xml` | `rvec`, `tvec` (thế giới → camera; đơn vị tvec chọn m hoặc cm) |
| `calibration.json` | Mọi camera: K, hệ số méo, rvec, tvec, R, vị trí camera, ma trận chiếu P = K[R\|t], homography mặt sàn → ảnh đã khử méo, sai số; hệ toạ độ; vùng quay và kích thước lưới |

Chiếu một điểm thế giới vào ảnh: `cv2.projectPoints(X, rvec, tvec, K, dist)`.

## Lưu ý
- Sai kích thước gạch → cả hệ thống co giãn đều (các camera vẫn khớp nhau), không phát hiện được bằng sai số. Hãy đo kỹ.
- Camera bị chạm / xê dịch sau khi hiệu chỉnh → phải làm lại extrinsic của camera đó (intrinsic giữ nguyên nếu không đổi nét/zoom).
- Ảnh tham chiếu phải cùng độ phân giải với lúc hiệu chỉnh intrinsic (tool sẽ cảnh báo nếu khác).
