---
name: performance-rating-logging-and-checkpointing
overview: Thiết lập logging hiệu năng với WandB cho train_modal.py và triển khai best-checkpoint strategy thủ công thay cho việc lưu model mỗi epoch.
todos:
  - id: analyze-train-and-modal
    content: Phân tích cấu trúc và logic train trong baseline/train.py và baseline/train_modal.py, cùng cấu hình backbone
    status: completed
  - id: design-wandb-logging
    content: Thiết kế chiến lược logging WandB (metrics, config, naming) cho train_modal.py
    status: completed
  - id: implement-wandb-in-train-modal
    content: Tích hợp WandB init/log/finish vào train_modal.py và đồng bộ metrics với train.py
    status: completed
  - id: implement-best-checkpoint-strategy
    content: Thay thế việc lưu model mỗi epoch trong train.py bằng best-checkpoint strategy thủ công, giữ lại code cũ dưới dạng comment
    status: completed
  - id: sync-metrics-and-config
    content: Đảm bảo logic loss/metrics và cấu hình được chia sẻ giữa train.py và train_modal.py càng nhiều càng tốt
    status: completed
  - id: testing-and-verification
    content: Chạy các bài test cục bộ và trên Modal để xác minh WandB logging và lưu best model một lần khi kết thúc training hoạt động đúng
    status: completed
  - id: update-docs
    content: Cập nhật tài liệu để hướng dẫn sử dụng WandB và best-checkpoint strategy trong workflow huấn luyện
    status: completed
isProject: false
---

## Mục tiêu tổng quát

- **Mục tiêu 1**: Thiết lập logging hiệu năng (tất cả loss functions liên quan) trong `baseline/train_modal.py` tương đương với logic đang có trong `baseline/train.py`, sử dụng Weights & Biases (WandB) để trực quan hóa.
- **Mục tiêu 2**: Thay cơ chế lưu model thủ công mỗi epoch trong `baseline/train.py` (line 237) bằng **best-checkpoint strategy tự triển khai**: sau mỗi epoch nếu metric tốt hơn best hiện tại thì cập nhật best model trong RAM, và chỉ lưu ra storage (Modal Volume hoặc local) một lần khi training kết thúc.

## Bước 1: Phân tích cấu trúc train hiện tại

- **Xem logic train gốc**: Đọc và hiểu luồng huấn luyện trong `[baseline/train.py](baseline/train.py)`:
  - Cách tổ chức vòng lặp epoch/batch.
  - Các loss functions đang được tính và log (log ra console/file, hoặc biến thống kê).
  - Cách model và optimizer được khởi tạo từ opts/config.
- **Xem train với Modal AI**: Đọc `[baseline/train_modal.py](baseline/train_modal.py)`:
  - Cách Modal AI được tích hợp (wrapper job, function, hay CLI).
  - Những tham số nào được truyền vào (dataset, model, hyperparams).
  - Vị trí thích hợp để thêm WandB init, log metrics và kết thúc run, cũng như áp dụng best-checkpoint strategy nếu cần.
- **Xem cấu hình model/backbone**: Đọc `[baseline/models/Backbone/backbone.py](baseline/models/Backbone/backbone.py)`:
  - Danh sách backbone được cấu hình.
  - Xác định cách tên/backbone id sẽ được log vào WandB (config).

## Bước 2: Thiết kế chiến lược logging với WandB

- **Xác định metrics cần log**:
  - Các loss chính và phụ (training loss, validation loss, từng thành phần loss nếu có).
  - Các metric hiệu năng khác (accuracy, IoU, mAP, v.v. nếu có).
  - Hyperparameters chính (learning rate, batch size, backbone, số epoch, v.v.).
- **Thiết kế cấu trúc WandB run**:
  - Chọn `project` và `group` mặc định, nhưng cho phép cấu hình qua CLI/`opts.yaml`.
  - Thiết kế cách đặt `run.name` (ví dụ: `"{backbone}_{timestamp}"`).
  - Xác định `config` dictionary: backbone, dataset, lr, optimizer, scheduler, số epoch, seed.
- **Tích hợp WandB vào train loop Modal**:
  - Chèn `wandb.init(...)` ở phần đầu trong `train_modal.py`, sau khi parse config.
  - Log `wandb.config.update(...)` với toàn bộ hyperparameters.
  - Trong vòng lặp train/val, thêm `wandb.log({...}, step=global_step)` cho mỗi batch hoặc mỗi epoch.
  - Đảm bảo gọi `wandb.finish()` khi huấn luyện kết thúc hoặc khi có exception (try/finally).

## Bước 3: Điều chỉnh train_modal.py để log đầy đủ performance

- **Đồng bộ metrics với train.py**:
  - So sánh nơi tính các loss/metrics trong `train.py` với `train_modal.py`.
  - Nếu `train_modal.py` chưa tính/ghi đủ các giá trị, bổ sung tính toán tương tự.
- **Thêm logging chi tiết**:
  - Log **per-epoch**: tổng loss train, tổng loss val, các metric tổng hợp.
  - Nếu cần, log **per-step/batch** cho loss để support debugging (có thể bật/tắt bằng flag).
  - Log thông tin backbone hiện tại, số tham số, chế độ huấn luyện (Modal/local) vào WandB.
- **Tùy chọn lưu artifacts** (nếu phù hợp với thiết kế sau này):
  - Chuẩn bị hook để sau này có thể `wandb.save()` hoặc log model checkpoint/best weights.

## Bước 4: Triển khai best-checkpoint strategy trong train.py

- **Xác định vị trí lưu model hiện tại**:
  - Tìm đoạn code ở line ~237 trong `[baseline/train.py](baseline/train.py)` đang lưu model mỗi epoch.
  - Ghi chú lại đường dẫn, pattern tên file, và điều kiện lưu (mỗi epoch, best val loss, v.v.).
- **Thêm biến theo dõi best**:
  - Trước vòng lặp epoch:
    - Khởi tạo `best_metric` (ví dụ `+inf` nếu monitor `val_loss`, hoặc `-inf` nếu monitor metric càng lớn càng tốt).
    - Khởi tạo `best_state_dict = None`.
- **Cập nhật best sau mỗi epoch**:
  - Sau khi có kết quả validation trong mỗi epoch:
    - Tính `current_metric` (ví dụ: `val_loss` hoặc metric mà bạn quan tâm).
    - Nếu `current_metric` tốt hơn `best_metric`:
      - Cập nhật `best_metric = current_metric`.
      - Sao chép `best_state_dict = deepcopy(model.state_dict())`.
- **Chỉ save một lần khi training kết thúc**:
  - Sau vòng lặp epoch:
    - Gọi `model.load_state_dict(best_state_dict)` để đảm bảo model đang ở trạng thái tốt nhất.
    - Gọi **một lần duy nhất** hàm `torch.save(...)` để lưu model best này ra file (local hoặc đường dẫn mà sau đó Modal Volume sẽ sử dụng).
- **Thay thế logic save thủ công**:
  - Giữ nguyên đoạn code cũ đang save mỗi epoch nhưng **comment lại** theo rule của bạn (kèm comment `"old code"`).
  - Thêm đoạn code mới triển khai best-checkpoint strategy, kèm comment `"new code"`.

## Bước 5: Áp dụng best-checkpoint strategy tương tự trong train_modal.py (nếu cần)

- **Thêm best-checkpoint vào train_modal.py**:
  - Thêm `best_metric` và `best_state_dict` tương tự như trong `train.py`.
  - Cập nhật sau mỗi epoch dựa trên metric validation (hoặc metric khác phù hợp).
- **Save best model vào Modal Volume một lần cuối**:
  - Sau khi training trong Modal kết thúc:
    - Load lại `best_state_dict` vào model.
    - Gọi đúng API/đường dẫn để lưu **một file model duy nhất** (best model) vào Modal Volume.
- **Đảm bảo tương thích với WandB**:
  - Có thể log thêm `best_metric` cuối cùng lên WandB (ví dụ `wandb.log({"best_val_loss": best_metric})` khi kết thúc).

## Bước 6: Đồng bộ giữa train.py và train_modal.py

- **Chia sẻ chung logic loss/metrics**:
  - Nếu logic loss/metrics trùng lặp giữa `train.py` và `train_modal.py`, cân nhắc trích ra utils chung (ví dụ: `baseline/losses/utils.py`).
  - Đảm bảo cả hai nơi tính toán cùng một cách để số liệu trên WandB và log console nhất quán.
- **Chia sẻ config**:
  - Đảm bảo các tham số như `max_epochs`, `val_interval`... được lấy từ cùng nguồn (`opts.yaml`) để so sánh dễ dàng.

## Bước 7: Kiểm thử và xác minh

- **Kiểm thử cục bộ (không Modal)**:
  - Chạy `train.py` với số epoch nhỏ:
    - Kiểm tra không còn lưu model mỗi epoch.
    - Kiểm tra chỉ có một file model best sau khi kết thúc training.
    - Xác nhận model best đúng là epoch có metric tốt nhất (đối chiếu log).
- **Kiểm thử với Modal + WandB**:
  - Chạy `train_modal.py` thông qua Modal (ví dụ: CLI hoặc entrypoint có sẵn) với cấu hình nhẹ.
  - Vào dashboard WandB, kiểm tra:
    - Run được tạo với đúng `project`/`name`.
    - Các loss/metrics hiển thị đúng, tương ứng với log trong console.
    - Hyperparameters được ghi đầy đủ.
  - Kiểm tra Modal Volume:
    - Chỉ có một file model best được lưu sau khi training kết thúc.
- **Kiểm tra edge cases**:
  - Huấn luyện dừng sớm (error hoặc KeyboardInterrupt): WandB vẫn được `finish`, không tạo run bị kẹt.
  - Trường hợp không có validation: thiết kế fallback metric (hoặc chỉ lưu best theo training loss) và đảm bảo code không crash.

## Bước 8: Dọn dẹp, tài liệu hóa

- **Cập nhật tài liệu**:
  - Mở rộng `docs/Performance_rating_processing.md` (hoặc docs khác) để mô tả:
    - Cách chạy `train_modal.py` để có WandB logging.
    - Cách xem run, metric trên WandB.
    - Cách hoạt động của best-checkpoint strategy và cách lấy best model (local và Modal Volume).
- **Tuân thủ rule giữ code cũ**:
  - Đảm bảo mọi chỗ thay thế logic lưu model hoặc logging đều:
    - Comment code cũ lại với chú thích rõ ràng (old code).
    - Đánh dấu đoạn code mới bằng comment new code để bạn dễ review.
- **Chuẩn bị cho mở rộng sau**:
  - Ghi chú các điểm mở (extensibility) để sau này thêm metric khác, log hình ảnh, video, hoặc artifacts lên WandB, hoặc thay đổi tiêu chí chọn best model.
