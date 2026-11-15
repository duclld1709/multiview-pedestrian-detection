# MV Label: tool gán nhãn multi-view bằng pseudo label

- **Kéo bounding box trên camera:** giữ chuột trái bên trong box để chọn và kéo người; điểm BEV, box các camera khác và dải crop cập nhật cùng lúc. Tắt **Chỉnh riêng camera (`R`)** để dùng chế độ liên kết. Box YOLO của người được di chuyển chuyển sang box chiếu; box chỉnh tay ở các camera dịch theo vị trí mặt sàn mới, camera đã tắt vẫn giữ tắt. Click không kéo chỉ chọn người; góc/cạnh box vẫn dùng để resize riêng camera. Thả chuột để tự lưu, `Ctrl+Z` để hoàn tác.

- **Crops (`J`)** trên thanh trên cùng: mở dải crop người đang chọn bên trái vùng camera/BEV. Các camera xếp dọc, lấy ảnh trực tiếp từ cùng nguồn ảnh đang hiển thị, giữ tỷ lệ và cập nhật khi đổi người, kéo BEV, sửa box, đổi chiều cao hoặc chuyển frame. Crop dùng box nhãn hiện tại (chiếu/YOLO/chỉnh tay); camera không có box hoặc hết video sẽ có thông báo. Kéo mép phải dải crop để đổi độ rộng, hoặc focus mép và dùng phím trái/phải; trạng thái và độ rộng được nhớ trong trình duyệt.

**WaveTrackNew:** xem [quy tắc gán nhãn chung, ảnh tình huống và bản chia việc](projects/wavetracknew_label_guide/QUY_TAC_GAN_NHAN.md), cùng [kiểm tra raw/ảnh đã khử méo](projects/wavetracknew_label_guide/DISTORTION_AUDIT.md).

Tool riêng (không phụ thuộc giao diện demo) để tạo nhãn dạng Wildtrack cho một rig nhiều camera:

1. **Setup**: chọn dataset (`Image_subsets/C1..CN`), calibration, các camera cần label, số frame và khoảng frame.
2. **Auto-label** (chạy nền): checkpoint BEV tốt nhất cho rig đó (Wildtrack: MVDeTr drop 20% brl_detic, MODA 89.9) cho ra các điểm trên mặt sàn. Mỗi điểm được chiếu thành box trên từng camera, theo cách demo trong khoá luận chiếu box. YOLO26s detect người trên từng ảnh để kiểm tra các box chiếu đó và tìm người mà model bỏ sót.
3. **Review**: xem từng frame, dời người bằng phím mũi tên hoặc kéo chuột (box ở mọi camera đi theo), so box chiếu với box YOLO, chấp nhận hoặc xoá đề xuất, thêm người bị sót.
4. **Export**: `annotations_positions/*.json` (định dạng Wildtrack, personID được nối giữa các frame), `labels_yolo/C*/` (YOLO txt), kèm ảnh và calibration.

## Chạy

```powershell
.\label_tool\run_label_tool.ps1          # http://127.0.0.1:8770
.\label_tool\run_label_tool.ps1 -Open D:\Thesis_Experiments\label_tool\projects\wildtrack
```

Tool dùng cùng môi trường Python với demo (torch CUDA, ultralytics, opencv, scipy). Mã model được nạp qua `demo_multiview/inference.py`, đọc calibration qua `demo_multiview/scene.py`.

Kiểm tra không cần trình duyệt:

```powershell
python label_tool\eval_autolabel.py --frames 1800:2000:5           # chấm auto label với GT Wildtrack
python -m pytest label_tool\test_labeling.py -q
python -m pytest label_tool\test_labeling.py label_tool\test_sources.py -q  # kèm WaveTrack/video + hồi quy tải ảnh
```

## Pseudo label được tạo thế nào

- **Box chiếu**: box của một hình trụ bán kính 0,20 m, cao 1,8 m, đặt tại điểm trên mặt sàn. Box GT của Wildtrack thực chất là hình chiếu của đúng hình trụ này: IoU trung vị với box gốc là 0,91, trong khi r = 0,16 m (rig MultiviewX) chỉ đạt 0,79. Phím `[` `]` chỉnh chiều cao cho từng người.
- **Khớp với YOLO**: trên mỗi camera, box chiếu và box YOLO được ghép một-một theo IoU giảm dần (ngưỡng 0,3). Trình duyệt chạy lại đúng thuật toán này (`app.js`, có test so khớp với Python), nên thứ bạn thấy cũng là thứ được export.
- **Trạng thái** (màu):
  - xanh lá `ổn`: model chắc chắn, YOLO xác nhận ở ≥ 2 view.
  - cam `cần kiểm tra`: model chắc chắn nhưng YOLO xác nhận ở ≤ 1 view.
  - tím `đề xuất`: một trong hai nguồn sau, chỉ được ghi vào nhãn khi bấm Enter.
    - điểm model có score thấp (0,4 đến ngưỡng checkpoint) mà YOLO xác nhận ở ≥ 2 view.
    - người chỉ YOLO thấy. Các cặp box YOLO chưa khớp từ 2 camera được tam giác hoá, và chỉ giữ khi:
      - hình trụ tại điểm đó có IoU ≥ 0,5 với box ở ≥ 2 view;
      - hai view lệch nhau ≥ 15°. Wildtrack C1 và C4 gắn cùng một chỗ nên không tính là hai view;
      - số view nhìn rõ chỗ đó mà không có box (bằng chứng phủ định) ít hơn số view ủng hộ.
  - xanh dương `thêm tay`: người do bạn thêm.
- **Đo trên mặt sàn bằng box YOLO** (`view_rows`): tâm box cho phương vị rất chính xác (lệch ngang khoảng 8 cm trên Wildtrack). Pixel chân thì không: sai số dọc tia nhìn khoảng 0,5 m và thiếu khoảng 0,15 m về phía camera, vì điểm thấp nhất của box là mũi chân gần camera. Vì vậy **Snap YOLO** (`S`) giải bình phương tối thiểu với ràng buộc phương vị mạnh và ràng buộc độ sâu yếu.

### Đo trên Wildtrack (40 frame test 1800–1995, cả 7 camera, so với GT ở 0,5 m)

| | MODA | Precision | Recall | Sai số vị trí | IoU box với GT |
|---|---|---|---|---|---|
| Auto label (= điểm model) | 89,7 | 93,9 | 95,9 | 9,2 cm | 0,765 |
| Nếu chấp nhận mọi đề xuất | 85,7 | 89,2 | 97,6 | 9,2 cm | |

- Trạng thái của các người tìm được:

  | Trạng thái | Số người | Là người thật |
  |---|---|---|
  | `ổn` | 939 | 887 (94 %) |
  | `cần kiểm tra` | 33 | 24 (73 %) |
  | `đề xuất` | 70 | 18 (16 người model bỏ sót được tìm lại) |

  Nên review theo thứ tự `Tab`: đề xuất và cần kiểm tra trước.
- Phần lớn đề xuất "sai" là người thật đứng sát mép vùng 12 × 36 m mà GT không gán, hoặc người quay phim đứng yên cạnh chân máy (cùng một vị trí lặp lại ở mọi frame). Khi review, xoá các đề xuất này bằng `Del`.
- Tự động dời điểm model theo YOLO **không** làm vị trí tốt hơn (9,2 → 9,5 cm), nên mặc định tắt (`auto_refine`). Snap chỉ chạy khi bạn bấm.
- Mỗi frame mất khoảng 1,2 s trên RTX 4050 (MVDeTr fp16 + YOLO26s 1280). Lần đầu mất thêm khoảng 15 s để nạp model.
- Checkpoint Wildtrack được train trên frame 0–1795. Label lại các frame đó sẽ cho kết quả lạc quan, nên Setup mặc định chọn đoạn test.

## Giao diện review

- **Camera** (mỗi camera đã chọn một ô; double-click để phóng to):
  - với nguồn video, thanh tên camera hiện clip và thời gian đang đọc; rê chuột lên thanh tên để xem thư mục nguồn;
  - bấm vào box để chọn người; khi box chồng nhau, người gần camera nhất được chọn;
  - kéo box để dời người: chân đi theo chuột trên mặt sàn;
  - lăn chuột để zoom, kéo chuột phải để pan;
  - `Z` bật zoom theo người đang chọn trên mọi camera.
  - `L` (**Lưới sàn**) chiếu lưới mặt sàn lên từng camera:
    - vùng BEV được label tô xanh, viền trắng; người đứng ngoài vùng này không được export, và bảng bên phải cảnh báo khi người đang chọn nằm ngoài;
    - lưới 1 m, đậm mỗi 5 m, có ghi toạ độ x/y ở mép vùng;
    - ô 25 cm màu hồng quanh người đang chọn để căn chân cho chính xác.
    - Lưới được cắt ở mặt phẳng trước camera, nên không bị lật khi đường lưới đi ra sau lưng camera. Trạng thái bật/tắt được trình duyệt nhớ.
- **Ký hiệu trên camera**:
  - đỏ đứt nét: box YOLO chưa khớp người nào, có thể là người bị bỏ sót. Bấm vào box rồi nhấn `A` để thêm người từ box đó;
  - trắng mờ: box YOLO đã khớp;
  - xanh ngọc đứt nét: GT (`G`).
- **BEV** (dải dưới, ô lưới 1 m):
  - chấm là người, chấm đặc là người đã duyệt;
  - vạch mờ nối tới vị trí auto ban đầu, để thấy mình đã dời bao xa;
  - double-click để thêm người.
  - Với WaveTrack, bấm **✎ Vẽ hiện trường** trên thanh trên cùng, ngay cạnh **Setup**, để mở bản đồ lớn và công cụ vẽ. Sơ đồ mở rộng tới vị trí của cả 5 camera. Khung nét đứt là ROI của model; phần ngoài ROI chỉ giúp quan sát hiện trường, không mở rộng vùng inference/export.
  - **Vẽ hiện trường**: chọn *vách* rồi đặt ít nhất hai điểm tại chân vách; chọn *cột* bằng tâm–mép để đo bán kính hoặc hai mép đối diện để đo đường kính. Có thể click trên bản đồ hoặc vào chân vật thể trong ảnh camera, nơi tia nhìn cắt mặt sàn. Chọn *Đo giữa hai điểm* để xem khoảng cách theo mét. Chọn vật thể để sửa tọa độ điểm, bề dày vách, bán kính/đường kính cột hoặc kéo điểm trên bản đồ. Khi sửa, đường vách/vòng cột được chiếu lại lên camera để đối chiếu; lớp này chỉ là hình học mặt sàn, không tự tính che khuất.
  - Bản đồ được lưu tại `projects/site_maps/` và chia sẻ cho các project dùng cùng calibration trên máy này; mỗi project cũng giữ bản sao trong `label_project.json`. **Tải sơ đồ JSON / Nạp sơ đồ JSON** để chuyển sang máy khác. Các tọa độ cột/vách cần được người gán nhãn đối chiếu với ảnh hoặc số đo hiện trường. Calibration hiện có chỉ cho camera và ROI, không chứa tọa độ vách/cột.
  - [Ảnh giao diện bản đồ mở rộng và bảng vẽ](projects/wavetracknew_label_guide/map_editor_preview.png).
- **Mũi tên**: nếu đang chọn ô camera, chân người dời 2 px trên ảnh đó rồi chiếu lại xuống mặt sàn, nên so được với box YOLO ngay trên view đó. Nếu đang chọn BEV, người dời 2,5 cm (đúng một ô lưới positionID). `Shift` nhân bước lên 5.
- **Bảng người đang chọn**:
  - IoU giữa box chiếu và box YOLO trên từng camera, cập nhật ngay khi dời;
  - chọn box ghi vào nhãn cho từng camera: chiếu / YOLO / tắt (phím `P` `Y` `O`);
  - nút "Về vị trí auto".
- **Box ghi vào nhãn** có hai chế độ:
  - box chiếu (mặc định, cùng kiểu với Wildtrack, hợp để train MVDet/MVDeTr);
  - box YOLO khi khớp (box sát người, hợp để train detector 2D).
- `F`: đánh dấu đã review frame và sang frame chưa review kế tiếp. Mọi người không phải đề xuất được coi là đã duyệt.
- `Ctrl+Z` hoàn tác.
- Mọi chỉnh sửa tự lưu vào `<project>/label_project.json`. Mở lại project bằng **Setup → Mở project có sẵn…** hoặc `-Open`.
- Nút **So với GT** (khi dataset có `annotations_positions`) so sánh model, auto label và nhãn sau review.

## Camera và calibration

**Danh sách và trạng thái model:** mọi project dùng chung danh sách checkpoint đã đăng ký, kể cả model chưa tương thích với dataset hiện tại. Setup và mục **Tình trạng model** ở bên phải hiển thị dataset/số camera yêu cầu, đường dẫn weight và lý do chưa chạy được. Model phù hợp được ưu tiên; model không tương thích vẫn có thể chọn để xem thông tin, nhưng không thể áp dụng/chạy. Model đã chọn bị thiếu hoặc không hợp lệ sẽ báo lỗi thay vì âm thầm chuyển sang YOLO.

Trong project có sẵn, mở **Model đã đăng ký**, chọn model tương thích rồi bấm **Dùng model này**. Lựa chọn được lưu cho lần auto-label tiếp theo; nhãn đang có được giữ. BEV và YOLO có trạng thái riêng: chưa chọn, chưa nạp, đang chạy, đã chạy hoặc lỗi; kèm frame, số điểm/box và thời gian của lần suy luận gần nhất trong phiên. Mở lại server bắt đầu một phiên mới, nên trạng thái chưa nạp không có nghĩa các kết quả đã lưu bị mất. Đăng ký model trong danh sách không tự chuyển weight Wildtrack/MultiviewX thành weight tương thích rig WaveTrack 5 camera.

- Hỗ trợ thêm **WaveTrack** với video trực tiếp trong `Cam01_GoPro`, `Cam02_DinhDuc`, …; xem cách dùng bên dưới. Cách tải ảnh của Wildtrack, MultiviewX và các rig khác giữ nguyên.
- Calibration đọc mọi layout mà tab Scene của demo đọc được: Wildtrack, MultiviewX, `intr_*/extr_*` bất kỳ, hoặc JSON K/R/t của CalibFree.
  - Với Wildtrack, `Image_subsets` đã được khử méo nên tool dùng `intrinsic_zero`.
  - Với `WaveTrackNew/Image_subsets`, ảnh đã khử méo nên tool giữ nguyên pixel và dùng `intrinsic_zero`; với `Image_subsets_raw`, tool dùng `intrinsic_original` rồi khử méo một lần. Giao diện hiển thị trạng thái xử lý ảnh. [Kiểm tra hai folder và project cũ](projects/wavetracknew_label_guide/DISTORTION_AUDIT.md).
  - Với nguồn ảnh raw có hệ số méo, ảnh được khử méo khi hiển thị và khi export.
- **Chọn ít camera hơn**: checkpoint vẫn nhận đủ số camera mà nó được train (lớp fusion cố định số camera). YOLO, box và nhãn chỉ làm trên các camera đã chọn; camera không chọn ghi `-1` trong JSON.
- Rig không có checkpoint cùng số camera và cùng layout: chọn "Không dùng model" để chạy YOLO đa view. Mọi người khi đó là đề xuất. Trên Wildtrack, chế độ này cho precision 79 %, recall 63 %.

### WaveTrack (video theo camera)

**Rig BEV WaveTrack:** Tool đọc `camera_matrix`, `R`, `tvec_m` của `Cam1`…`Cam5` từ `calibrations/calibration.json`, khử méo ảnh raw một lần hoặc giữ nguyên ảnh đã khử méo, rồi dùng lưới 2,5 cm trong ROI 0…7,82 m × 0…6,02 m. Với weight MVDeTr yêu cầu 6 hoặc 7 camera, chế độ thử nghiệm thêm tương ứng 1 hoặc 2 ảnh đen sau 5 camera thật. Lưới BEV được đệm tới 320 × 248 ô để phù hợp tầng transformer; điểm ngoài ROI bị bỏ. Có thể chọn weight MultiviewX/Wildtrack trong danh sách model rồi auto-label, nhưng tool không tự chọn weight này. Kết quả là dự đoán thử nghiệm của weight huấn luyện trên rig khác, cần review và chưa có chỉ số chính xác trên WaveTrack. Project mẫu 3 frame ở `projects/wavetrack_mvdetr_black6_trial/label_project.json` dùng weight MultiviewX 6 camera và trả 1, 2, 4 điểm BEV trên frame 240–242. Weight WaveTrack 5 camera được huấn luyện riêng vẫn có thể chạy trực tiếp qua scene `wavetrack`.

Để trích riêng bộ ảnh chưa gán nhãn từ cấu hình đồng bộ đã đo, chạy trong thư mục `label_tool`:

```powershell
python extract_synced_subset.py --out F:\thesis_dataset\WaveTrack_synced_2fps
```

Công cụ tạo thư mục mới với `Image_subsets/C1..C5`, ảnh JPEG chất lượng 95 ở độ phân giải gốc, calibration gốc, bản sao cấu hình và `timestamps/C*.csv`. Ảnh đã xoay đúng chiều video, chưa khử méo; label tool sẽ khử méo khi tải. Cùng số ảnh là cùng mốc timeline đã căn (2 ảnh/giây theo cấu hình hiện tại). Không tạo nhãn giả hay sao chép nhãn từ timeline cũ. Kiểm tra toàn bộ ảnh và đối chiếu mẫu với video gốc bằng:

```powershell
python verify_synced_subset.py F:\thesis_dataset\WaveTrack_synced_2fps --report projects\wavetrack_sync\subset_verification.json
```

**Đồng bộ thời gian đã đo cho bộ dữ liệu hiện tại:** sau khi đọc dataset, bấm **Áp dụng đoạn đồng bộ 5 camera** trong Setup. Cấu hình ở `projects/wavetrack_sync/sync_profile.json` chọn take trùng nhau bằng âm thanh, bù mốc bắt đầu và tốc độ trôi đồng hồ, đọc theo timestamp thật của video (kể cả video điện thoại FPS biến đổi), rồi giới hạn vào đoạn chung đủ mọi camera. Xem số đo, độ tin cậy và ảnh đối chiếu trong `projects/wavetrack_sync/SYNC_REPORT.md`. Tạo project mới để label trên các frame đã đồng bộ; nhãn từ các mốc cũ không tự chuyển sang frame mới.

Cấu hình này chỉ được đề xuất khi đường dẫn, kích thước và thời gian sửa các video vẫn khớp với dữ liệu đã đo. Công cụ phân tích dùng PyAV (`av`), NumPy và SciPy. Có thể đo lại bằng:

```powershell
python label_tool\sync_audio.py --root F:\thesis_dataset\WaveTrack --out label_tool\projects\wavetrack_sync --reference-files GH010309_1.MP4 GH020309_2.MP4
```

Âm thanh giúp tìm đúng sự kiện chung nhưng còn độ trễ do truyền âm, xử lý microphone và độ phân giải frame. Kết quả hiện tại cần xem là đồng bộ phần mềm có sai số, không bảo đảm khóa phần cứng chính xác từng frame.

Trong **Setup**, chọn `F:\thesis_dataset\WaveTrack` rồi bấm **Đọc dataset + calibration**. Tool nhận cấu trúc:

```text
WaveTrack/
  calibrations/
    calibration.json
    extrinsic/extr_Cam1.xml … extr_Cam5.xml
    intrinsic_original/intr_Cam1.xml … intr_Cam5.xml
    intrinsic_zero/intr_Cam1.xml … intr_Cam5.xml
  Cam01_GoPro/*.MP4
  Cam02_DinhDuc/*.mp4
  Cam03_QuocAnh/*.MOV
  Cam04_NhutAn/*.MOV
  Cam05_HoaiPhuc/*.MOV
```

1. Mỗi camera có thể **Giữ một video** (chọn clip cần dùng) hoặc **Ghép nhiều video theo thứ tự** nếu thư mục có từ hai clip trở lên. Dùng **↑ ↓** chọn clip trước/sau; có thể thêm/bỏ clip nếu có nhiều hơn hai. Ví dụ GoPro: `GH010309_1.MP4 → GH020309_2.MP4`. Video/ảnh trong thư mục con `intrinsic` không được dùng làm dữ liệu label.
2. Các clip được nối liên tục trên timeline: clip sau bắt đầu đúng tại `số_frame / FPS` của clip trước, không chèn khung đen ở giữa. Từng clip dùng FPS riêng; clip khác độ phân giải được đưa về độ phân giải của clip đầu để giữ đúng calibration. **Xem điểm nối** hiển thị frame cuối và frame đầu của hai clip để bạn kiểm tra nội dung quay có liền nhau. Việc nối thực hiện khi đọc frame, không sửa hoặc tạo lại video gốc.
3. Điền **Mốc đầu (giây)** của từng camera sao cho các mốc cùng một thời điểm thực tế. Mốc này tính trên toàn chuỗi clip đã chọn (có thể nằm trong clip sau). Giá trị 0 mặc định chỉ là điểm bắt đầu chuỗi, chưa xác nhận các camera đồng bộ. Bấm **Xem ảnh tại các mốc bắt đầu** để kiểm tra.
4. Chọn tần suất lấy frame (mặc định **2 frame/giây**), rồi bấm lại **Đọc dataset + calibration** sau khi đổi clip, thứ tự, mốc đầu hoặc tần suất. Frame `n` đọc tại `mốc_đầu + n / tần_suất` giây của mỗi camera. Timeline chạy đến **camera có thời lượng còn lại dài nhất**. Camera hết video hiển thị **nền đen**, không chạy YOLO và không ghi box cho view đó; các camera còn video vẫn tiếp tục. Export vẫn có ảnh đen, JSON view `-1` và YOLO txt rỗng ở camera đã hết video.
5. Chọn camera, khoảng/số frame và thư mục project rồi auto-label như trước. Có thể chọn weight MVDeTr 6/7 camera để thử nghiệm với view đen; nếu không chọn model BEV, WaveTrack dùng YOLO đa view.

Calibration lấy từ `calibrations/calibration.json` (hoặc chọn chính file JSON): ghép `Cam1` với `Cam01_GoPro`, … theo số camera, kể cả khi JSON sắp xếp khác thứ tự. Tool dùng `tvec_m` theo mét, `camera_matrix`, `dist_coeffs`, `image_size` để chỉnh đúng độ phân giải, và vùng `roi` làm BEV. Các XML của bộ này có `tvec` theo cm; JSON giúp tránh suy đoán sai đơn vị.

Video, thứ tự các clip, mốc đầu và tần suất được lưu trong project để mở lại. Export giữ layout `annotations_positions`, `labels_yolo/C*`, `Image_subsets/C*`: chỉ trích ảnh của các frame đã chọn, không sao chép toàn bộ video. Nhãn WaveTrack dùng `position_m` trong hệ tọa độ calibration; người ngoài ROI không được export. Calibration export được cập nhật theo ảnh đã khử méo để tải lại không khử méo thêm lần nữa. Nếu dataset có sẵn `Image_subsets`, tool ưu tiên cách tải ảnh như cũ.

## Định dạng export

```
export/
  annotations_positions/00001800.json   [{personID, positionID, views: [{viewNum, xmin, ymin, xmax, ymax}] * N}]
  labels_yolo/C1/00001800.txt           "0 cx cy w h" (chuẩn hoá), chỉ camera đã chọn
  Image_subsets/C1..CN/                 ảnh của các frame đã label (tuỳ chọn)
  calibrations/                         bản sao calibration
  export_summary.json                   số lượng, tham số, điểm so với GT nếu có
  label_project.json                    toàn bộ trạng thái review
```

- `positionID` theo lưới 2,5 cm của dataset: Wildtrack 480 × 1440, MultiviewX 1000 × 640.
- Rig khác không có lưới: JSON ghi `position_m` thay cho `positionID`.
- personID được nối giữa các frame bằng ghép Hungarian. Cổng ghép là 0,5 m + 0,75 m cho mỗi bước frame của dataset; hai frame cách nhau quá 6 bước thì không nối.

## File

| File | Nội dung |
|---|---|
| `labeling.py` | hình học (hình trụ, chiếu xuống mặt sàn), ghép với YOLO, đề xuất, chấm điểm, export |
| `project.py` | project: chọn frame, job auto-label, trạng thái review, lưu và mở lại |
| `sources.py` | nguồn ảnh/video, chọn clip và quy đổi timeline chung theo FPS từng camera |
| `server.py` | HTTP server (cổng 8770) |
| `static/` | giao diện (`app.js` lặp lại `project_box` / `match_views`) |
| `eval_autolabel.py` | chấm auto label với GT, không cần trình duyệt |
| `test_labeling.py` | test trên rig tổng hợp, so khớp JS với Python, kiểm tra box hình trụ với GT Wildtrack |
| `test_sources.py` | test WaveTrack, đồng bộ timeline video, lưu/mở/export và tương thích dataset ảnh |

## Import label_tool annotations

Use the import button in the top bar or Setup. Choose the dataset containing
`Image_subsets`, the calibration folder containing `extrinsic/`, `intrinsic_zero/`
and `grid.json`, the source annotation folder and a new QA project output folder.
Blank calibration/annotation paths default to `dataset/calibrations` and `dataset/annotations`.

The import creates a separate project with annotated frames only, without auto-labeling.
Position IDs decode using grid.json's origin, columns and cell size, with no half-cell offset.
Cylinder height and radius come from `.editor/<frame>.json` and remain per person.
Legacy annotations must first be migrated with `label_tools/scripts/migrate_grid_labels.py`.

Edit positions normally; press **F** to review a frame. Export with only reviewed frames
selected. Export includes the WaveTrack `positionID` and `position_m`, preserving the
precise edited metre coordinates. IDs use the containing cell (floor from grid origin).
Boxes are projected again using QA's clipping/visibility rules so they follow edits;
source boxes are not frozen as manual overrides. Source annotations remain unchanged.
Reopen `label_project.json` to continue the review.
